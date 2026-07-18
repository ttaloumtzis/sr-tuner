from sr_engine.utils.progress import ProgressReporter
from sr_engine.api.event_manager import SSEEventManager


class SSEProgressReporter(ProgressReporter):
    """ProgressReporter that pushes events to an SSE event manager."""

    def __init__(self, events: SSEEventManager, job_id: str) -> None:
        self._events = events
        self._job_id = job_id

    def start(self, total: int | None = None, desc: str = "") -> None:
        self._events.publish(self._job_id, {
            "type": "progress_start", "total": total, "desc": desc,
        })

    def update(self, n: int = 1) -> None:
        self._events.publish(self._job_id, {
            "type": "progress_update", "n": n,
        })

    def finish(self) -> None:
        self._events.publish(self._job_id, {"type": "progress_end"})

    def set_description(self, desc: str) -> None:
        self._events.publish(self._job_id, {
            "type": "postfix", "desc": desc,
        })

    def set_postfix(self, **kwargs) -> None:
        self._events.publish(self._job_id, {
            "type": "postfix", **kwargs,
        })