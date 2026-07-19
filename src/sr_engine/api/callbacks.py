from typing import Any

from sr_engine.engine.trainer import TrainerCallback
from sr_engine.api.event_manager import SSEEventManager


class SSECallback(TrainerCallback):
    """TrainerCallback that pushes training events to an SSE event manager."""

    def __init__(self, events: SSEEventManager, job_id: str) -> None:
        self._events = events
        self._job_id = job_id

    def on_phase(self, phase: str, **data: Any) -> None:
        self._events.publish(self._job_id, {
            "type": "phase", "phase": phase, **data,
        })

    def on_step(self, epoch: int, batch: int, total_batches: int, **losses: Any) -> None:
        self._events.publish(self._job_id, {
            "type": "step", "epoch": epoch, "batch": batch,
            "total_batches": total_batches, **losses,
        })

    def on_validate(self, epoch: int, frames: dict | None = None, **metrics: Any) -> None:
        event: dict[str, Any] = {"type": "validate", "epoch": epoch, **metrics}
        if frames:
            event["frames"] = frames
        self._events.publish(self._job_id, event)

    def on_done(self, elapsed_seconds: float) -> None:
        self._events.publish(self._job_id, {
            "type": "done", "elapsed_seconds": elapsed_seconds,
        })