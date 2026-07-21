import json
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

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


@router.get("/health")
async def get_health_report(
    path: str,
    ws: Workspace = Depends(get_workspace),
):
    dataset_path = Path(path).resolve()
    if not str(dataset_path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    from sr_engine.data.dataset_health import load_health_report

    report = load_health_report(dataset_path)
    if report is None:
        return None
    return report


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


@router.delete("/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    ws: Workspace = Depends(get_workspace),
):
    dataset_path = ws.path / "datasets" / dataset_name
    if not dataset_path.is_dir():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    if not str(dataset_path).startswith(str(ws.path)):
        raise HTTPException(403, "Path is outside the workspace")
    shutil.rmtree(dataset_path)
    return {"deleted": dataset_name}


@router.get("/{dataset_name}/image")
async def serve_dataset_image(
    dataset_name: str,
    kind: str,
    index: int,
    ws: Workspace = Depends(get_workspace),
):
    dataset_path = ws.path / "datasets" / dataset_name
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "Manifest not found for dataset")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise HTTPException(500, "Failed to read manifest")

    pairs = data.get("pairs", [])

    if pairs:
        # Normal path: use manifest pairs
        if index < 0 or index >= len(pairs):
            raise HTTPException(404, f"Pair index {index} out of range (0-{len(pairs) - 1})")
        rel_path = pairs[index].get(kind, "")
        if not rel_path:
            raise HTTPException(404, f"No '{kind}' path for pair {index}")
        full_path = (dataset_path / rel_path).resolve()
    else:
        # Fallback: scan HR/ and LR/ directories (covers merged datasets
        # and any other case where the manifest pairs list is empty)
        kind_dir = dataset_path / ("HR" if kind == "hr" else "LR")
        if not kind_dir.is_dir():
            raise HTTPException(404, f"No '{kind.upper()}/' directory for dataset")
        files = sorted(kind_dir.glob("*.png"))
        if index < 0 or index >= len(files):
            raise HTTPException(404, f"Pair index {index} out of range (0-{len(files) - 1})")
        full_path = files[index].resolve()

    if not str(full_path).startswith(str(ws.path)):
        raise HTTPException(403, "Path outside workspace")
    if not full_path.is_file():
        raise HTTPException(404, f"Image file not found: {full_path}")
    return FileResponse(str(full_path), filename=full_path.name)


@router.get("")
async def list_datasets(
    scale: int | None = None,
    ws: Workspace = Depends(get_workspace),
):
    datasets_dir = ws.path / "datasets"
    if not datasets_dir.is_dir():
        return []
    result = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ds_scale = int(data.get("config", {}).get("scale", 4))
        if scale is not None and ds_scale != scale:
            continue
        pairs = data.get("pairs", [])
        if pairs:
            num_pairs = len(pairs)
        elif (d / "HR").is_dir() and (d / "LR").is_dir():
            hr_files = list((d / "HR").glob("*.png"))
            lr_files = list((d / "LR").glob("*.png"))
            num_pairs = min(len(hr_files), len(lr_files))
        else:
            num_pairs = 0
        result.append({
            "name": d.name,
            "path": str(d),
            "scale": ds_scale,
            "num_pairs": num_pairs,
        })
    return result