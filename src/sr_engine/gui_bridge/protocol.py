"""Schema constants, SocketReporter, SocketCallback, connect_control_socket()."""

import json
import socket
import threading
from typing import Any, Callable

from sr_engine.engine.trainer import TrainerCallback
from sr_engine.utils.progress import ProgressReporter


class SchemaVersion:
    """Defines the current protocol schema version."""
    CURRENT = 1


EXIT_SUCCESS = 0
"""Subprocess exit code for successful completion."""

EXIT_ERROR = 1
"""Subprocess exit code for generic error."""

EXIT_CANCELLED = 130
"""Subprocess exit code for SIGTERM cancellation."""


_HELLO_TIMEOUT = 10.0


class SocketReporter(ProgressReporter):
    """Progress reporter that sends progress events over a control socket."""

    def __init__(self, send_fn: Callable[[dict], None], job_id: str) -> None:
        """Wrap a socket send function as a progress reporter.

        Args:
            send_fn: Function that serialises and sends a dict.
            job_id: Job identifier attached to every event.
        """
        self._send = send_fn
        self._job_id = job_id

    def start(self, total: int | None = None, desc: str = "") -> None:
        """Broadcast a progress-start event."""
        self._send({"job_id": self._job_id, "type": "progress_start", "total": total, "desc": desc})

    def update(self, n: int = 1) -> None:
        """Broadcast a progress-update event."""
        self._send({"job_id": self._job_id, "type": "progress_update", "n": n})

    def finish(self) -> None:
        """Broadcast a progress-end event."""
        self._send({"job_id": self._job_id, "type": "progress_end"})

    def set_description(self, desc: str) -> None:
        """Broadcast a description update."""
        self._send({"job_id": self._job_id, "type": "postfix", "desc": desc})

    def set_postfix(self, **kwargs: Any) -> None:
        """Broadcast postfix key-value pairs."""
        self._send({"job_id": self._job_id, "type": "postfix", **kwargs})


class SocketCallback(TrainerCallback):
    """Trainer callback that sends training events over a control socket."""

    def __init__(self, send_fn: Callable[[dict], None], job_id: str) -> None:
        """Wrap a socket send function as a trainer callback.

        Args:
            send_fn: Function that serialises and sends a dict.
            job_id: Job identifier attached to every event.
        """
        self._send = send_fn
        self._job_id = job_id

    def on_phase(self, phase: str, **data: Any) -> None:
        """Send a phase-change event."""
        self._send({"job_id": self._job_id, "type": "phase", "phase": phase, **data})

    def on_step(self, epoch: int, batch: int, total_batches: int, **losses: float) -> None:
        """Send a training-step progress event."""
        self._send({
            "job_id": self._job_id, "type": "step", "epoch": epoch,
            "batch": batch, "total_batches": total_batches, **losses,
        })

    def on_validate(self, epoch: int, **metrics: float) -> None:
        """Send a validation-metrics event."""
        self._send({"job_id": self._job_id, "type": "validate", "epoch": epoch, **metrics})

    def on_done(self, elapsed_seconds: float) -> None:
        """Send a training-done event with elapsed time."""
        self._send({"job_id": self._job_id, "type": "done", "elapsed_seconds": elapsed_seconds})


def make_json_sender(writer: Callable[[str], None]) -> Callable[[dict], None]:
    """Create a dict-to-JSON-line sender from a string writer.

    Args:
        writer: A callable that accepts a JSON string.

    Returns:
        A function that accepts a dict, serialises it, and passes it to ``writer``.
    """
    def sender(msg: dict) -> None:
        writer(json.dumps(msg, default=str) + "\n")
    return sender


def parse_message(line: str) -> dict | None:
    """Parse a single JSON line from the protocol.

    Args:
        line: Raw string line.

    Returns:
        Parsed dict or None if the line is empty or malformed.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def connect_control_socket(env_value: str) -> tuple[str, Callable[[dict], None], Callable[[], None]]:
    """Connect to the GUI server's control socket using an environment variable.

    Performs the hello handshake and returns send/close functions.

    Args:
        env_value: Value of the ``SRENGINE_GUI_SOCKET`` env variable (JSON).

    Returns:
        ``(job_id, send_fn, close_fn)`` where ``send_fn`` sends a dict as a JSON
        line and ``close_fn`` cleanly shuts the socket.

    Raises:
        ConnectionRefusedError: If the handshake is rejected.
    """
    info = json.loads(env_value)
    job_id: str = info["job_id"]
    token: str = info["token"]
    host: str = info.get("control_host", "127.0.0.1")
    port: int = info["control_port"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_HELLO_TIMEOUT)
    sock.connect((host, port))

    hello = json.dumps({"type": "hello", "job_id": job_id, "token": token}) + "\n"
    sock.sendall(hello.encode("utf-8"))

    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    line, *_ = buf.split(b"\n", 1)
    ack = json.loads(line.decode("utf-8"))

    if ack.get("status") != "ok":
        sock.close()
        raise ConnectionRefusedError(f"Handshake rejected: {ack.get('message', 'unknown')}")

    sock.settimeout(None)

    _send_lock = threading.Lock()

    def send_fn(msg: dict) -> None:
        data = json.dumps(msg, default=str) + "\n"
        with _send_lock:
            sock.sendall(data.encode("utf-8"))

    def close_fn() -> None:
        try:
            sock.close()
        except OSError:
            pass

    return job_id, send_fn, close_fn
