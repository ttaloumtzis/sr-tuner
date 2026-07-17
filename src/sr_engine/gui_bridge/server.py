"""Two accept loops (GUI clients + job control), ClientHandler, ControlHandler."""

import json
import shutil
import socket
import threading
from pathlib import Path

from sr_engine.gui_bridge.jobs import JobManager
from sr_engine.gui_bridge.command_schema import command_schema_dict
from sr_engine.gui_bridge.config_schema import CONFIG_SECTIONS, all_params
from sr_engine.gui_bridge.config_utils import validate_config_values


class Server:
    """TCP socket server managing GUI client and job control connections."""

    def __init__(self, host: str, port: int, workspace: Path) -> None:
        """Initialize server sockets and job manager.

        Args:
            host: Host address to bind the GUI listener.
            port: Port to bind the GUI listener.
            workspace: Workspace root path for job management.
        """
        self.host = host
        self.port = port
        self._workspace = workspace
        self._running = threading.Event()
        self._running.set()

        self._clients: list["ClientHandler"] = []
        self._clients_lock = threading.Lock()

        self.gui_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gui_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.gui_listener.bind((host, port))
        self.gui_listener.listen(5)

        self.job_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.job_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.job_listener.bind(("127.0.0.1", 0))
        self.job_listener.listen(5)
        self.job_listener.settimeout(1.0)

        self._job_manager = JobManager(workspace, self.broadcast)
        self._job_manager._job_listener_port = self.job_listener.getsockname()[1]

    def run(self) -> None:
        """Start accept loops and block until the server is stopped."""
        threads = [
            threading.Thread(target=self._gui_accept_loop, daemon=True),
            threading.Thread(target=self._job_accept_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            self._running.wait()
        except KeyboardInterrupt:
            self.stop()

    def broadcast(self, msg: dict) -> None:
        """Send a JSON message to every connected GUI client.

        Dead clients are automatically pruned.

        Args:
            msg: Dictionary to serialise and broadcast.
        """
        payload = (json.dumps(msg, default=str) + "\n").encode("utf-8")
        with self._clients_lock:
            dead: list[ClientHandler] = []
            for client in self._clients:
                if not client.send_raw(payload):
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)

    def stop(self) -> None:
        """Signal shutdown and close both listener sockets."""
        self._running.clear()
        self.gui_listener.close()
        self.job_listener.close()

    def _gui_accept_loop(self) -> None:
        """Accept incoming GUI client connections in a loop."""
        self.gui_listener.settimeout(1.0)
        while self._running.is_set():
            try:
                conn, addr = self.gui_listener.accept()
                handler = ClientHandler(conn, addr, self)
                handler.daemon = True
                with self._clients_lock:
                    self._clients.append(handler)
                handler.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _job_accept_loop(self) -> None:
        """Accept incoming job subprocess control connections in a loop."""
        while self._running.is_set():
            try:
                conn, addr = self.job_listener.accept()
                handler = ControlHandler(conn, addr, self)
                handler.daemon = True
                handler.start()
            except socket.timeout:
                continue
            except OSError:
                break


class ClientHandler(threading.Thread):
    """Handles a single GUI client connection — receives commands, sends responses."""

    def __init__(self, conn: socket.socket, addr: tuple, server: Server) -> None:
        """Wrap a connected socket with per-client send lock and buffer.

        Args:
            conn: The accepted client socket.
            addr: Client address tuple.
            server: Parent Server instance.
        """
        super().__init__()
        self._conn = conn
        self._addr = addr
        self._server = server
        self._send_lock = threading.Lock()
        self._buf = b""

    def send_raw(self, data: bytes) -> bool:
        """Send raw bytes to the client.

        Args:
            data: Bytes to send.

        Returns:
            True on success, False on connection error.
        """
        with self._send_lock:
            try:
                self._conn.sendall(data)
                return True
            except OSError:
                return False

    def run(self) -> None:
        """Read and dispatch lines from the client until disconnection."""
        self._conn.settimeout(None)
        try:
            while self._server._running.is_set():
                data = self._conn.recv(65536)
                if not data:
                    break
                self._buf += data
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    self._handle_line(line.decode("utf-8", errors="replace"))
        except (OSError, ConnectionError):
            pass
        finally:
            try:
                self._conn.close()
            except OSError:
                pass

    def _handle_line(self, line: str) -> None:
        """Parse a JSON line and dispatch to the appropriate handler.

        Args:
            line: A raw JSON string from the client.
        """
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self._send_error("0", "Malformed JSON")
            return

        request_id = msg.get("id", "")
        command = msg.get("command", "")
        params = msg.get("params", {})

        if not command:
            self._send_error(request_id, "Missing 'command' field")
            return

        if command == "hello":
            self._send_result(request_id, {
                "schema_version": 1,
                "server_version": "0.1.0",
            })
            return

        if command in _SYNC_HANDLERS:
            try:
                result = _SYNC_HANDLERS[command](self._server, params)
                self._send_result(request_id, result)
            except Exception as e:
                self._send_error(request_id, str(e), type(e).__name__)
        elif command in _ASYNC_HANDLERS:
            try:
                job_id, result = _ASYNC_HANDLERS[command](self._server, params)
                self._send_accepted(request_id, result)
            except Exception as e:
                self._send_error(request_id, str(e), type(e).__name__)
        else:
            self._send_error(request_id, f"Unknown command: {command}")

    def _send_result(self, request_id: str, data: dict) -> None:
        """Send a synchronous command result to the client.

        Args:
            request_id: The request identifier from the client.
            data: Result payload.
        """
        self.send_raw(json.dumps({"id": request_id, "type": "result", "data": data}).encode("utf-8") + b"\n")

    def _send_error(self, request_id: str, message: str, error_type: str = "Error") -> None:
        """Send an error response to the client.

        Args:
            request_id: The request identifier from the client.
            message: Human-readable error description.
            error_type: Machine-readable error type classifier.
        """
        self.send_raw(json.dumps({
            "id": request_id, "type": "error", "message": message, "error_type": error_type,
        }).encode("utf-8") + b"\n")

    def _send_accepted(self, request_id: str, data: dict) -> None:
        """Send an accepted response for an async command.

        Args:
            request_id: The request identifier from the client.
            data: Accepted payload (typically includes job_id).
        """
        self.send_raw(json.dumps({"id": request_id, "type": "accepted", "data": data}).encode("utf-8") + b"\n")


def _handle_config_schema(server: Server, params: dict) -> dict:
    """Return the full command and config schema for dynamic UI building."""
    result = command_schema_dict()
    result["config_sections"] = CONFIG_SECTIONS
    return result


def _handle_workspace_info(server: Server, params: dict) -> dict:
    """Return the workspace path."""
    return {"workspace": str(server._workspace)}


def _handle_workspace_check(server: Server, params: dict) -> dict:
    """Check whether the workspace directory exists."""
    exists = server._workspace.exists()
    return {"exists": exists, "workspace": str(server._workspace)}


def _handle_workspace_init(server: Server, params: dict) -> dict:
    """Initialize a workspace directory tree."""
    from sr_engine.workspace import Workspace
    path = Path(params.get("path", "."))
    reset_configs = params.get("reset_configs", False)
    ws = Workspace(path)
    ws.init(reset_configs=reset_configs)
    return {"path": str(ws.path), "status": "created"}


def _handle_project_list(server: Server, params: dict) -> dict:
    """List all model instances (workspace IS the project)."""
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    instances = ws.list_model_instances() if hasattr(ws, "list_model_instances") else []
    return {"instances": [i.name for i in instances]}


def _handle_project_create(server: Server, params: dict) -> dict:
    """Create a model instance in the workspace.

    Args:
        params: Must include ``name`` key.
    """
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    name = params.get("name")
    arch = params.get("arch", "swinir")
    if not name:
        return {"status": "error", "message": "missing 'name' parameter"}
    from sr_engine.utils.config import DefaultConfigs
    cfg = DefaultConfigs(workspace=ws)
    arch_config = cfg.get_model_config(arch)
    ws.create_model_instance(name, arch_config)
    return {"instance": name, "status": "created"}


def _handle_dataset_validate(server: Server, params: dict) -> dict:
    """Validate a dataset directory."""
    from sr_engine.data.dataset_validator import validate
    from sr_engine.utils.progress import ProgressReporter
    path = Path(params["path"])
    report = validate(path, reporter=ProgressReporter())
    return {
        "ok": report.ok,
        "num_pairs": report.num_pairs,
        "problems": list(report.problems) if hasattr(report, "problems") else [],
    }


def _handle_dataset_health(server: Server, params: dict) -> dict:
    """Run a health check on a dataset directory."""
    from sr_engine.data.dataset_health import check_dataset_health
    from sr_engine.utils.progress import ProgressReporter
    path = Path(params["path"])
    report = check_dataset_health(path, reporter=ProgressReporter())
    return report


def _handle_dataset_merge(server: Server, params: dict) -> dict:
    """Merge multiple datasets grouped by scale."""
    from sr_engine.data.dataset_merge import merge_datasets
    from sr_engine.utils.progress import ProgressReporter
    input_path = Path(params["input"])
    out_path = Path(params["out"]) if params.get("out") else input_path / "merged"
    results = merge_datasets(
        datasets_root=input_path,
        out_dir=out_path,
        scale=params.get("scale"),
        output_name=params.get("name"),
        reporter=ProgressReporter(),
    )
    return {
        "merged": [
            {"scale": r.scale, "path": str(r.output_path)}
            for r in results
        ]
    }


def _read_yaml(path: Path) -> dict:
    """Read a YAML file (lazy import)."""
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _handle_model_info(server: Server, params: dict) -> dict:
    """Return available model architectures."""
    from sr_engine.utils.config import DefaultConfigs
    cfg = DefaultConfigs()
    return {"models": list(cfg.models.keys())}


def _handle_model_instance_list(server: Server, params: dict) -> dict:
    """List model instances in a project."""
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    project = params.get("project", "")
    instances = ws.list_model_instances(project)
    return {
        "instances": [
            {
                "name": inst.name,
                "checkpoints": len(list(inst.path.glob("checkpoints/*.pt"))),
                "runs": len(list(inst.path.glob("runs/run_*"))),
            }
            for inst in instances
        ]
    }


def _handle_model_instance_info(server: Server, params: dict) -> dict:
    """Return details for one model instance."""
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    inst = ws.get_model_instance(params["project"], params["instance"])
    return {
        "name": inst.name,
        "config": _read_yaml(inst.path / "config.yaml"),
        "checkpoints": sorted(p.name for p in inst.path.glob("checkpoints/*.pt")),
        "runs": sorted(
            ({"run_id": d.name, "has_metrics": (d / "metrics.jsonl").exists()}
             for d in inst.path.glob("runs/run_*")),
            key=lambda r: r["run_id"], reverse=True,
        ),
    }


def _handle_model_list_runs(server: Server, params: dict) -> dict:
    """List training runs for a model instance."""
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    run_dirs = ws.list_runs(params["instance"])
    return {
        "runs": [
            {
                "run_id": d.name,
                "has_metrics": (d / "metrics.jsonl").exists(),
                "has_config": (d / "train_config.yaml").exists(),
            }
            for d in run_dirs
        ]
    }


def _handle_model_export(server: Server, params: dict) -> dict:
    """Export a model checkpoint."""
    import torch
    import yaml
    from sr_engine.models.registry import build_model
    from sr_engine.models.checkpoint import (
        load_checkpoint, export_to_safetensors, export_to_onnx, export_to_torchscript,
    )
    from sr_engine.workspace import Workspace
    from sr_engine.utils.config import DefaultConfigs
    from .helpers import resolve_model_config

    instance = params.get("instance")
    model_name = params.get("model_name")
    ckpt = Path(params["ckpt"]) if params.get("ckpt") else None
    version = params.get("version")
    fmt = params["format"]
    out = Path(params["out"])

    if instance:
        ws = Workspace(server._workspace)
        model_inst = ws.get_model_instance(instance)
        inst_cfg = yaml.safe_load(
            (model_inst.path / "config.yaml").read_text(encoding="utf-8")
        )
        model_name = inst_cfg["name"]
        v_path = ws.resolve_version(instance, version)
        if not v_path:
            raise ValueError(f"No versions found for instance '{instance}'")
        state_dict = torch.load(v_path, weights_only=True, map_location="cpu")
        model = build_model(model_name, inst_cfg)
        model.load_state_dict(state_dict)
        model.eval()
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            from safetensors.torch import save_file
            cpu_sd = {k: v.contiguous().cpu() for k, v in state_dict.items()}
            save_file(cpu_sd, str(out))
        elif fmt == "onnx":
            dummy = torch.randn(1, 3, 256, 256)
            torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["output"], opset_version=17)
        elif fmt == "torchscript":
            traced = torch.jit.trace(model, torch.randn(1, 3, 256, 256))
            traced.save(str(out))
    elif model_name and ckpt:
        cfg = DefaultConfigs()
        resolve_model_config(cfg, model_name)
        export_map = {
            "safetensors": export_to_safetensors,
            "onnx": export_to_onnx,
            "torchscript": export_to_torchscript,
        }
        export_map[fmt](ckpt, out)
    else:
        raise ValueError("Provide --instance or --model-name + --ckpt")

    return {"path": str(out), "format": fmt, "model": model_name}


def _handle_job_cancel(server: Server, params: dict) -> dict:
    """Request cancellation of a running job."""
    return server._job_manager.cancel_job(params["job_id"])


def _handle_job_list(server: Server, params: dict) -> dict:
    """List all completed jobs."""
    return {"jobs": server._job_manager.list_jobs()}


def _handle_job_status(server: Server, params: dict) -> dict:
    """Return the manifest for a specific job."""
    job = server._job_manager.get_job(params["job_id"])
    if job is None:
        return {"job_id": params["job_id"], "status": "not_found"}
    return job


def _handle_env_check(server: Server, params: dict) -> dict:
    """Check the current environment and return a report."""
    import torch
    from sr_engine.device.backend import get_device, is_rocm, autocast_dtype, supports_flash_attn
    device = get_device()
    is_cuda = torch.cuda.is_available()
    info = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": is_cuda,
        "rocm": is_rocm(),
        "autocast_dtype": str(autocast_dtype()),
        "flash_attention": supports_flash_attn(),
    }
    if is_cuda:
        dev_idx = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(dev_idx)
        info["vram_total_mb"] = torch.cuda.get_device_properties(dev_idx).total_memory // 1024 ** 2
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def _handle_env_bench(server: Server, params: dict) -> dict:
    """Run a micro-benchmark and return throughput statistics."""
    import torch
    import statistics
    import time
    from sr_engine.models.registry import build_model
    from sr_engine.utils.config import DefaultConfigs
    from sr_engine.device.backend import get_device

    model = params.get("model", "rrdb_esrgan")
    iterations = params.get("iterations", 10)
    device = get_device()

    cfg = DefaultConfigs()
    model_cfg = cfg.get_model_config(model)
    if not model_cfg:
        raise ValueError(f"Unknown model: {model}")
    net = build_model(model, model_cfg).to(device).train()
    dummy = torch.randn(1, 3, 128, 128, device=device)

    for _ in range(3):
        _ = net(dummy)

    times = []
    for _ in range(iterations):
        net.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = net(dummy)
        loss = out.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "device": str(device),
        "model": model,
        "iterations": iterations,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _handle_train_start(server: Server, params: dict) -> tuple[str, dict]:
    """Start a training job asynchronously."""
    config_json = params.get("config")
    if config_json:
        errors = validate_config_values(config_json, all_params())
        if errors:
            raise ValueError(f"Config validation failed: {'; '.join(errors)}")
    return server._job_manager.start_job("train", params, workspace_path=server._workspace)


def _handle_infer_start(server: Server, params: dict) -> tuple[str, dict]:
    """Start an inference job asynchronously."""
    return server._job_manager.start_job("infer", params, workspace_path=server._workspace)


def _handle_dataset_build(server: Server, params: dict) -> tuple[str, dict]:
    """Start a dataset build job asynchronously."""
    config_json = params.get("config")
    if config_json:
        errors = validate_config_values(config_json, all_params())
        if errors:
            raise ValueError(f"Config validation failed: {'; '.join(errors)}")
    return server._job_manager.start_job("dataset.build", params, workspace_path=server._workspace)


_SYNC_HANDLERS = {
    "config.schema": _handle_config_schema,
    "workspace.info": _handle_workspace_info,
    "workspace.check": _handle_workspace_check,
    "workspace.init": _handle_workspace_init,
    "project.list": _handle_project_list,
    "project.create": _handle_project_create,
    "dataset.validate": _handle_dataset_validate,
    "dataset.health": _handle_dataset_health,
    "dataset.merge": _handle_dataset_merge,
    "model.info": _handle_model_info,
    "model.instance_list": _handle_model_instance_list,
    "model.instance_info": _handle_model_instance_info,
    "model.list_runs": _handle_model_list_runs,
    "model.export": _handle_model_export,
    "env.check": _handle_env_check,
    "env.bench": _handle_env_bench,
    "job.cancel": _handle_job_cancel,
    "job.list": _handle_job_list,
    "job.status": _handle_job_status,
}
"""Map of synchronous command names to their handler functions."""

_ASYNC_HANDLERS = {
    "train.start": _handle_train_start,
    "infer.start": _handle_infer_start,
    "dataset.build": _handle_dataset_build,
}
"""Map of asynchronous command names to their handler functions."""


class ControlHandler(threading.Thread):
    """Handles a control connection from a job subprocess."""

    def __init__(self, conn: socket.socket, addr: tuple, server: Server) -> None:
        """Wrap a control-socket connection.

        Args:
            conn: The accepted job subprocess socket.
            addr: Client address tuple.
            server: Parent Server instance.
        """
        super().__init__()
        self._conn = conn
        self._addr = addr
        self._server = server
        self._buf = b""

    def run(self) -> None:
        """Read events from the subprocess and forward them via broadcast."""
        try:
            data = self._conn.recv(65536)
            if not data:
                return
            self._buf += data

            job_id = ""
            lines_processed = 0
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                msg = json.loads(line.decode("utf-8", errors="replace"))

                if lines_processed == 0:
                    if msg.get("type") != "hello":
                        self._send_ack("rejected", "First message must be 'hello'")
                        return

                    job_id = msg.get("job_id", "")
                    token = msg.get("token", "")
                    ok = self._server._job_manager._on_hello(job_id, token)

                    if ok:
                        self._send_ack("ok")
                    else:
                        self._send_ack("rejected", "Token mismatch")
                        return
                else:
                    event = msg
                    event.setdefault("job_id", job_id)
                    self._server.broadcast(event)

                lines_processed += 1

            while self._server._running.is_set():
                data = self._conn.recv(65536)
                if not data:
                    break
                self._buf += data
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    event = json.loads(line.decode("utf-8", errors="replace"))
                    event.setdefault("job_id", job_id)
                    self._server.broadcast(event)
        except (OSError, ConnectionError, json.JSONDecodeError):
            pass
        finally:
            try:
                self._conn.close()
            except OSError:
                pass

    def _send_ack(self, status: str, message: str | None = None) -> None:
        """Send a hello-acknowledgement to the subprocess.

        Args:
            status: ``ok`` or ``rejected``.
            message: Optional human-readable rejection reason.
        """
        ack = {"type": "hello_ack", "status": status}
        if message:
            ack["message"] = message
        try:
            self._conn.sendall((json.dumps(ack) + "\n").encode("utf-8"))
        except OSError:
            pass
