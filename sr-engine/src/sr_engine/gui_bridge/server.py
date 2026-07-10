"""Two accept loops (GUI clients + job control), ClientHandler, ControlHandler."""

import json
import socket
import threading
from pathlib import Path

from sr_engine.gui_bridge.jobs import JobManager


class Server:
    def __init__(self, host: str, port: int, workspace: Path) -> None:
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
        payload = (json.dumps(msg, default=str) + "\n").encode("utf-8")
        with self._clients_lock:
            dead: list[ClientHandler] = []
            for client in self._clients:
                if not client.send_raw(payload):
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)

    def stop(self) -> None:
        self._running.clear()
        self.gui_listener.close()
        self.job_listener.close()

    def _gui_accept_loop(self) -> None:
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
    def __init__(self, conn: socket.socket, addr: tuple, server: Server) -> None:
        super().__init__()
        self._conn = conn
        self._addr = addr
        self._server = server
        self._send_lock = threading.Lock()
        self._buf = b""

    def send_raw(self, data: bytes) -> bool:
        with self._send_lock:
            try:
                self._conn.sendall(data)
                return True
            except OSError:
                return False

    def run(self) -> None:
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
        self.send_raw(json.dumps({"id": request_id, "type": "result", "data": data}).encode("utf-8") + b"\n")

    def _send_error(self, request_id: str, message: str, error_type: str = "Error") -> None:
        self.send_raw(json.dumps({
            "id": request_id, "type": "error", "message": message, "error_type": error_type,
        }).encode("utf-8") + b"\n")

    def _send_accepted(self, request_id: str, data: dict) -> None:
        self.send_raw(json.dumps({"id": request_id, "type": "accepted", "data": data}).encode("utf-8") + b"\n")


def _handle_workspace_info(server: Server, params: dict) -> dict:
    return {"workspace": str(server._workspace)}


def _handle_workspace_check(server: Server, params: dict) -> dict:
    exists = server._workspace.exists()
    return {"exists": exists, "workspace": str(server._workspace)}


def _handle_project_list(server: Server, params: dict) -> dict:
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    projects = ws.list_projects() if hasattr(ws, "list_projects") else []
    return {"projects": projects}


def _handle_project_create(server: Server, params: dict) -> dict:
    from sr_engine.workspace import Workspace
    ws = Workspace(server._workspace)
    name = params.get("name")
    if not name:
        return {"status": "error", "message": "missing 'name' parameter"}
    ws.create_project(name)
    return {"project": name, "status": "created"}


def _handle_dataset_validate(server: Server, params: dict) -> dict:
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
    from sr_engine.data.dataset_health import check_dataset_health
    from sr_engine.utils.progress import ProgressReporter
    path = Path(params["path"])
    report = check_dataset_health(path, reporter=ProgressReporter())
    return report


def _handle_model_info(server: Server, params: dict) -> dict:
    return {"model": params.get("model", "")}


def _handle_job_cancel(server: Server, params: dict) -> dict:
    return server._job_manager.cancel_job(params["job_id"])


def _handle_job_list(server: Server, params: dict) -> dict:
    return {"jobs": server._job_manager.list_jobs()}


def _handle_job_status(server: Server, params: dict) -> dict:
    job = server._job_manager.get_job(params["job_id"])
    if job is None:
        return {"job_id": params["job_id"], "status": "not_found"}
    return job


def _handle_train_start(server: Server, params: dict) -> tuple[str, dict]:
    return server._job_manager.start_job("train", params, workspace_path=server._workspace)


def _handle_infer_start(server: Server, params: dict) -> tuple[str, dict]:
    return server._job_manager.start_job("infer", params, workspace_path=server._workspace)


def _handle_dataset_build(server: Server, params: dict) -> tuple[str, dict]:
    return server._job_manager.start_job("dataset.build", params, workspace_path=server._workspace)


_SYNC_HANDLERS = {
    "workspace.info": _handle_workspace_info,
    "workspace.check": _handle_workspace_check,
    "project.list": _handle_project_list,
    "project.create": _handle_project_create,
    "dataset.validate": _handle_dataset_validate,
    "dataset.health": _handle_dataset_health,
    "model.info": _handle_model_info,
    "job.cancel": _handle_job_cancel,
    "job.list": _handle_job_list,
    "job.status": _handle_job_status,
}

_ASYNC_HANDLERS = {
    "train.start": _handle_train_start,
    "infer.start": _handle_infer_start,
    "dataset.build": _handle_dataset_build,
}


class ControlHandler(threading.Thread):
    def __init__(self, conn: socket.socket, addr: tuple, server: Server) -> None:
        super().__init__()
        self._conn = conn
        self._addr = addr
        self._server = server
        self._buf = b""

    def run(self) -> None:
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
        ack = {"type": "hello_ack", "status": status}
        if message:
            ack["message"] = message
        try:
            self._conn.sendall((json.dumps(ack) + "\n").encode("utf-8"))
        except OSError:
            pass
