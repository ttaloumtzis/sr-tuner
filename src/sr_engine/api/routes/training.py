import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException

from sr_engine.api.deps import get_configs, get_workspace
from sr_engine.api.schemas import TrainParams, VramProbeParams
from sr_engine.data.dataset_validator import validate
from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/train", tags=["training"])
_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/start")
async def train_start(params: TrainParams, ws: Workspace = Depends(get_workspace), cfg: DefaultConfigs = Depends(get_configs)):
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_training

    overrides = params.to_overrides()

    # Allocate the run directory up-front so the client gets a run_id
    # immediately (and the Runs UI can show it as "running" with no race).
    run_dir = None
    if params.instance:
        try:
            ws.get_model_instance(params.instance)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        run_dir = ws.get_run_path(params.instance)

    job_id = tasks.create_job("train", params={**overrides, "run_dir": str(run_dir) if run_dir else None})
    thread = threading.Thread(
        target=run_training,
        args=(job_id, {
            "model_name": params.model_name,
            "instance": params.instance,
            "dataset": params.dataset,
            "config": params.config,
            "resume": params.resume,
            "overrides": overrides,
            "write_metrics_file": params.write_metrics_file,
            "run_dir": str(run_dir) if run_dir else None,
        }, ws, cfg, tasks, events),
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job_id,
        "status": "accepted",
        "run_id": run_dir.name if run_dir else None,
    }


@router.post("/validate-dataset")
async def validate_dataset(params: TrainParams, ws: Workspace = Depends(get_workspace)):
    dataset_path = ws.resolve_dataset(ws.path / "datasets" / params.dataset) if params.dataset else None
    if not dataset_path or not dataset_path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Dataset not found: {params.dataset}")
    report = validate(dataset_path)
    return {"valid": report.ok, "problems": report.problems}


@router.post("/estimate-vram")
async def estimate_vram(params: VramProbeParams):
    """Run a dry forward+loss+backward in a subprocess and return peak GPU memory."""
    from sr_engine.api.vram_probe import run_vram_probe
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, run_vram_probe, params.model_dump())
    return result