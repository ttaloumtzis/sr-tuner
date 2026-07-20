import threading

from fastapi import APIRouter, Depends

from sr_engine.api.deps import get_configs, get_workspace
from sr_engine.api.schemas import TrainParams
from sr_engine.data.dataset_validator import validate
from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("/start")
async def train_start(params: TrainParams, ws: Workspace = Depends(get_workspace), cfg: DefaultConfigs = Depends(get_configs)):
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_training

    job_id = tasks.create_job("train")
    overrides = params.to_overrides()
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
        }, ws, cfg, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/validate-dataset")
async def validate_dataset(params: TrainParams, ws: Workspace = Depends(get_workspace)):
    dataset_path = ws.resolve_dataset(ws.path / "datasets" / params.dataset) if params.dataset else None
    if not dataset_path or not dataset_path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Dataset not found: {params.dataset}")
    report = validate(dataset_path)
    return {"valid": report.ok, "problems": report.problems}