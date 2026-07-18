import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from sr_engine.api.event_manager import SSEEventManager
from sr_engine.api.task_manager import BackgroundTaskManager
from sr_engine.api.routes import workspace, models, training, inference, datasets, jobs, env

log = logging.getLogger(__name__)

# Global singletons (initialised at startup)
events = SSEEventManager()
tasks = BackgroundTaskManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API server starting up")
    yield
    log.info("API server shutting down")


app = FastAPI(
    title="sr-engine API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SSE event stream ───────────────────────────────────────────────────

@app.get("/api/events")
async def event_stream(job_id: str):
    async def generate():
        try:
            async for event in events.subscribe(job_id):
                yield event
        except Exception:
            pass
        finally:
            events.cleanup(job_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Health check ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    from sr_engine.api.deps import _workspace

    return {
        "status": "ok",
        "workspace": str(_workspace.path) if _workspace else None,
    }


# ── Register routes ────────────────────────────────────────────────────

app.include_router(workspace.router)
app.include_router(models.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(env.router)