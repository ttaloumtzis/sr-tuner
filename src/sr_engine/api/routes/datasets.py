import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from sr_engine.api.deps import get_configs, get_workspace
from sr_engine.api.schemas import DatasetBuildParams, DatasetHealthParams, DatasetMergeParams, DatasetPruneParams, DatasetValidateParams
from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/build")
async def build_dataset(params: DatasetBuildParams, ws: Workspace = Depends(get_workspace), cfg: DefaultConfigs = Depends(get_configs)):
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_build

    job_id = tasks.create_job("dataset.build")
    build_params = {
        "input": params.input,
        "out": params.out,
        "config": params.config,
        "degradations": params.degradations,
    }
    if params.config_overrides:
        build_params["config_overrides"] = params.config_overrides
    thread = threading.Thread(
        target=run_dataset_build,
        args=(job_id, build_params, ws, cfg, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/validate")
async def validate_dataset(params: DatasetValidateParams, ws: Workspace = Depends(get_workspace)):
    path = Path(params.path).resolve()
    if not str(path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    from sr_engine.data.dataset_validator import validate
    report = validate(path)
    return {"valid": report.ok, "problems": report.problems, "num_pairs": report.num_pairs}


@router.post("/validate-async")
async def validate_dataset_async(params: DatasetValidateParams, ws: Workspace = Depends(get_workspace)):
    path = Path(params.path).resolve()
    if not str(path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_validate

    job_id = tasks.create_job("dataset.validate")
    thread = threading.Thread(
        target=run_dataset_validate,
        args=(job_id, {"path": params.path}, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/health")
async def health_check(params: DatasetHealthParams, ws: Workspace = Depends(get_workspace)):
    path = Path(params.path).resolve()
    if not str(path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_health

    job_id = tasks.create_job("dataset.health")
    thread = threading.Thread(
        target=run_dataset_health,
        args=(job_id, {"path": params.path, "yes": params.yes}, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/merge")
async def merge(params: DatasetMergeParams, ws: Workspace = Depends(get_workspace)):
    input_path = Path(params.input).resolve()
    if not str(input_path).startswith(str(ws.path)):
        raise HTTPException(403, "Input path is outside the workspace")
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_merge

    job_id = tasks.create_job("dataset.merge")
    merge_params = {
        "input": params.input,
        "out": params.out,
        "scale": params.scale,
        "name": params.name,
        "keep_sources": params.keep_sources,
        "input_datasets": params.input_datasets,
    }
    thread = threading.Thread(
        target=run_dataset_merge,
        args=(job_id, merge_params, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/prune")
async def prune_dataset(params: DatasetPruneParams, ws: Workspace = Depends(get_workspace)):
    path = Path(params.path).resolve()
    if not str(path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_prune

    job_id = tasks.create_job("dataset.prune")
    thread = threading.Thread(
        target=run_dataset_prune,
        args=(job_id, params.path, params.black_frames, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}