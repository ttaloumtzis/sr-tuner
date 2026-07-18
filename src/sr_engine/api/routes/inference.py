import threading

from fastapi import APIRouter, Depends

from sr_engine.api.deps import get_workspace
from sr_engine.api.schemas import InferParams
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/infer", tags=["inference"])


@router.post("/start")
async def infer_start(params: InferParams, ws: Workspace = Depends(get_workspace)):
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_inference

    job_id = tasks.create_job("infer")
    thread = threading.Thread(
        target=run_inference,
        args=(job_id, {
            "model": params.model,
            "instance": params.instance,
            "version": params.version,
            "input": params.input,
            "output": params.output,
            "tile": params.tile,
            "overlap": params.overlap,
            "device": params.device,
        }, ws, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}