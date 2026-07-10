"""Job manifest I/O, subprocess lifecycle, signal handling, CLI-arg builders."""

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sr_engine.gui_bridge.protocol import EXIT_SUCCESS, EXIT_CANCELLED

SRENGINE_GUI_SOCKET = "SRENGINE_GUI_SOCKET"


@dataclass
class JobManifest:
    job_id: str
    job_type: str
    status: str
    pid: int | None
    started_at: str
    finished_at: str | None
    exit_code: int | None
    project: str | None
    log_path: str | None
    error_message: str | None


_cancelled = False


def _handle_sigterm(signum, frame):
    global _cancelled
    _cancelled = True


def install_cancel_handler():
    signal.signal(signal.SIGTERM, _handle_sigterm)

def was_cancelled() -> bool:
    return _cancelled


def cli_args_for_train(params: dict) -> list[str]:
    args = ["train", "run"]
    if "model_name" in params:
        args.extend(["--model", params["model_name"]])
    if "dataset" in params:
        args.extend(["--dataset", params["dataset"]])
    if "config" in params:
        args.extend(["--config", params["config"]])
    if "resume" in params:
        args.extend(["--resume", params["resume"]])
    if "device" in params:
        args.extend(["--device", params["device"]])
    if "batch_size" in params:
        args.extend(["--batch-size", str(params["batch_size"])])
    if "learning_rate" in params:
        args.extend(["--learning-rate", str(params["learning_rate"])])
    if "max_epochs" in params:
        args.extend(["--max-epochs", str(params["max_epochs"])])
    if "project" in params:
        args.extend(["--project", params["project"]])
    if params.get("machine"):
        args.append("--machine")
    if params.get("experiment_id"):
        args.extend(["--experiment-id", params["experiment_id"]])
    return args


def cli_args_for_infer(params: dict) -> list[str]:
    args = ["infer", "run"]
    if "model" in params:
        args.extend(["--model", params["model"]])
    if "input_path" in params:
        args.extend(["--input", params["input_path"]])
    if "output" in params:
        args.extend(["--output", params["output"]])
    if "tile" in params:
        args.extend(["--tile", str(params["tile"])])
    if "overlap" in params:
        args.extend(["--overlap", str(params["overlap"])])
    if "device" in params:
        args.extend(["--device", params["device"]])
    return args


def cli_args_for_dataset_build(params: dict) -> list[str]:
    args = ["dataset", "build"]
    if "input" in params:
        args.extend(["--input", params["input"]])
    if "out" in params:
        args.extend(["--out", params["out"]])
    if "config" in params:
        args.extend(["--config", params["config"]])
    return args


_ARG_BUILDERS = {
    "train": cli_args_for_train,
    "infer": cli_args_for_infer,
    "dataset.build": cli_args_for_dataset_build,
}


class JobManager:
    def __init__(self, workspace: Path, broadcast_fn: Callable[[dict], None]) -> None:
        self._lock = threading.Lock()
        self._pending_hello: dict[str, dict] = {}
        self._hello_timers: dict[str, threading.Timer] = {}
        self._active_jobs: dict[str, dict] = {}
        self._workspace = workspace
        self._broadcast = broadcast_fn
        self._job_listener_port: int | None = None

    @property
    def jobs_dir(self) -> Path:
        path = self._workspace / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start_job(self, job_type: str, params: dict, *, workspace_path: Path | None = None) -> tuple[str, dict]:
        ws_path = workspace_path or self._workspace
        timestamp = int(time.time())
        job_id = f"{job_type.replace('.', '_')}_{timestamp}_{secrets.token_hex(4)}"
        token = secrets.token_hex(32)

        cli_args = _ARG_BUILDERS[job_type](params)

        control_info = json.dumps({
            "job_id": job_id,
            "token": token,
            "control_host": "127.0.0.1",
            "control_port": self._job_listener_port,
        })
        env = os.environ.copy()
        env[SRENGINE_GUI_SOCKET] = control_info

        srengine_cmd = shutil.which("srengine")
        if srengine_cmd:
            cmd = [srengine_cmd, "--workspace", str(ws_path), *cli_args]
        else:
            cmd = [sys.executable, "-m", "sr_engine.cli.main", "--workspace", str(ws_path), *cli_args]

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        manifest_entry = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "running",
            "pid": proc.pid,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": None,
            "exit_code": None,
            "project": params.get("project"),
            "log_path": None,
            "error_message": None,
        }

        with self._lock:
            self._pending_hello[job_id] = {"token": token, "spawned_at": time.time()}
            timer = threading.Timer(10.0, self._fail_if_no_hello, args=[job_id])
            timer.daemon = True
            self._hello_timers[job_id] = timer
            timer.start()

        threading.Thread(
            target=self._relay_stdout, args=(proc, job_id), daemon=True,
        ).start()
        threading.Thread(
            target=self._wait_and_finalize, args=(proc, job_id, manifest_entry), daemon=True,
        ).start()

        return job_id, {"status": "accepted", "job_id": job_id}

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            if job_id in self._pending_hello:
                entry = self._pending_hello[job_id]
            elif job_id in self._active_jobs:
                entry = self._active_jobs[job_id]
            else:
                return {"status": "not_found", "job_id": job_id}

            pid = entry.get("pid")
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return {"status": "cancelling", "job_id": job_id}

    def list_jobs(self) -> list[dict]:
        jobs = []
        if not self.jobs_dir.exists():
            return jobs
        for p in sorted(self.jobs_dir.iterdir()):
            if p.suffix == ".json":
                try:
                    jobs.append(json.loads(p.read_text("utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
        return jobs

    def get_job(self, job_id: str) -> dict | None:
        path = self.jobs_dir / f"{job_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _fail_if_no_hello(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._pending_hello:
                self._pending_hello.pop(job_id, None)
                self._hello_timers.pop(job_id, None)
                self._broadcast({
                    "job_id": job_id,
                    "type": "log",
                    "level": "warning",
                    "message": f"Job {job_id}: subprocess did not connect control socket within 10s",
                })

    def _on_hello(self, job_id: str, token: str) -> bool:
        with self._lock:
            if job_id not in self._pending_hello:
                return False
            expected = self._pending_hello[job_id]
            if token != expected["token"]:
                return False
            self._hello_timers.pop(job_id, None)
            entry = self._pending_hello.pop(job_id)
            self._active_jobs[job_id] = entry
            return True

    def _relay_stdout(self, proc: subprocess.Popen, job_id: str) -> None:
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    self._broadcast({
                        "job_id": job_id,
                        "type": "log",
                        "level": "info",
                        "message": text,
                    })
        except OSError:
            pass

    def _write_manifest(self, job_id: str, manifest: dict) -> None:
        path = self.jobs_dir / f"{job_id}.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def _wait_and_finalize(self, proc: subprocess.Popen, job_id: str, manifest: dict) -> None:
        proc.wait()
        exit_code = proc.returncode

        if exit_code == EXIT_SUCCESS:
            status = "completed"
        elif exit_code == EXIT_CANCELLED:
            status = "cancelled"
        else:
            status = "failed"

        manifest["status"] = status
        manifest["exit_code"] = exit_code
        manifest["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if exit_code not in (0, 130):
            manifest["error_message"] = f"Process exited with code {exit_code}"

        with self._lock:
            self._active_jobs.pop(job_id, None)

        self._write_manifest(job_id, manifest)

        self._broadcast({
            "job_id": job_id,
            "type": "done",
            "exit_code": exit_code,
            "elapsed_seconds": None,
        })
