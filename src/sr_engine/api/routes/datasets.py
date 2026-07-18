import threading
from pathlib import Path

from fastapi import APIRouter, Depends

from sr_engine.api.deps import get_workspace
from sr_engine.api.schemas import DatasetBuildParams, DatasetHealthParams, DatasetMergeParams, DatasetValidateParams
from sr_engine.data.dataset_health import check_dataset_health
from sr_engine.data.dataset_merge import merge_datasets
from sr_engine.data.dataset_validator import validate
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/build")
async def build_dataset(params: DatasetBuildParams, ws: Workspace = Depends(get_workspace)):
    from sr_engine.api.app import events, tasks
    from sr_engine.api.workers import run_dataset_build

    job_id = tasks.create_job("dataset.build")
    thread = threading.Thread(
        target=run_dataset_build,
        args=(job_id, {
            "input": params.input,
            "out": params.out,
            "config": params.config,
            "degradations": params.degradations,
        }, ws, tasks, events),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "accepted"}


@router.post("/validate")
async def validate_dataset(params: DatasetValidateParams):
    report = validate(Path(params.path))
    return {"valid": report.ok, "problems": report.problems, "num_pairs": report.num_pairs}


@router.post("/health")
async def health_check(params: DatasetHealthParams):
    report = check_dataset_health(Path(params.path))
    return report


@router.post("/merge")
async def merge(params: DatasetMergeParams):
    results = merge_datasets(
        datasets_root=Path(params.input),
        out_dir=Path(params.out) if params.out else None,
        scale=params.scale,
        output_name=params.name,
    )
    return [{"scale": r.scale, "output_path": str(r.output_path), "source_datasets": [str(s) for s in r.source_datasets]} for r in results]