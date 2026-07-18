import asyncio
import json
import threading


class SSEEventManager:
    """Per-job event bus for SSE streaming.

    Thread-safe ``publish()`` from background workers, async ``subscribe()``
    for FastAPI SSE endpoints.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, job_id: str) -> asyncio.Queue:
        with self._lock:
            if job_id not in self._queues:
                self._queues[job_id] = asyncio.Queue()
            return self._queues[job_id]

    def publish(self, job_id: str, event: dict) -> None:
        """Push an event to the job's event queue (thread-safe)."""
        q = self._get_or_create(job_id)
        q.put_nowait(event)

    async def subscribe(self, job_id: str):
        """Async generator yielding SSE-formatted events for *job_id*.

        Yields ``data: <json>\\n\\n`` strings.  A ``None`` sentinel signals
        the end of the stream.
        """
        q = self._get_or_create(job_id)
        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    def cleanup(self, job_id: str) -> None:
        """Remove the event queue for *job_id* (called on disconnect)."""
        with self._lock:
            self._queues.pop(job_id, None)