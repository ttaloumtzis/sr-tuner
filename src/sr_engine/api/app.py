import logging
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from sr_engine.api.event_manager import SSEEventManager
from sr_engine.api.middleware import RequestLogMiddleware
from sr_engine.api.task_manager import BackgroundTaskManager
from sr_engine.api.routes import workspace, models, training, inference, datasets, jobs, env
from sr_engine.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

# Global singletons (initialised at startup)
events = SSEEventManager()
tasks = BackgroundTaskManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    log.info("API server starting up")
    yield
    log.info("API server shutting down")


app = FastAPI(
    title="sr-engine API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLogMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log = structlog.get_logger("sr_engine.api.exception")
    log.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── SSE event stream ───────────────────────────────────────────────────

@app.get("/api/events")
async def event_stream(job_id: str):
    async def generate():
        try:
            async for event in events.subscribe(job_id):
                yield event
        except Exception:
            log.warning("SSE client disconnected for job %s", job_id)
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