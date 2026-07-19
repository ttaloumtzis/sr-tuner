import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class TaskRecord:
    job_id: str
    job_type: str
    status: str = "pending"
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


_training_mutex = threading.Lock()


def acquire_training_slot() -> bool:
    return _training_mutex.acquire(blocking=False)


def release_training_slot() -> None:
    _training_mutex.release()


class BackgroundTaskManager:
    """Registry for long-running background operations.

    Provides job creation, status tracking, cancellation, and cleanup.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create_job(self, job_type: str) -> str:
        job_id = f"{job_type}_{int(time.time())}_{uuid4().hex[:8]}"
        with self._lock:
            self._tasks[job_id] = TaskRecord(
                job_id=job_id,
                job_type=job_type,
                status="pending",
                created_at=time.time(),
            )
        return job_id

    def start_job(self, job_id: str) -> None:
        with self._lock:
            rec = self._tasks.get(job_id)
            if rec:
                rec.status = "running"
                rec.started_at = time.time()

    def complete_job(self, job_id: str, result: dict | None = None) -> None:
        with self._lock:
            rec = self._tasks.get(job_id)
            if rec:
                rec.status = "completed"
                rec.completed_at = time.time()
                rec.result = result

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            rec = self._tasks.get(job_id)
            if rec:
                rec.status = "failed"
                rec.completed_at = time.time()
                rec.error = error

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            rec = self._tasks.get(job_id)
            if not rec:
                return False
            if rec.status in ("completed", "failed", "cancelled"):
                return True
            rec.cancel_event.set()
            rec.status = "cancelled"
            rec.completed_at = time.time()
        return True

    def get_job(self, job_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(job_id)

    def list_jobs(self) -> list[TaskRecord]:
        with self._lock:
            return list(self._tasks.values())

    def cleanup_old(self, max_age_hours: int = 24) -> None:
        cutoff = time.time() - max_age_hours * 3600
        with self._lock:
            self._tasks = {
                k: v for k, v in self._tasks.items()
                if v.created_at > cutoff
            }