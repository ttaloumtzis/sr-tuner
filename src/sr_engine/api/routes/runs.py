"""Runs API — disk-derived run listings, checkpoints, and deletion.

Run status is ground truth from each run directory's ``run_status.json``
(see :meth:`sr_engine.workspace.Workspace.infer_run_status`); the ``.srproj``
file is a frontend cache reconciled against this endpoint.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from sr_engine.api.deps import get_workspace
from sr_engine.api.schemas import (
    CheckpointMetrics,
    ModelRuns,
    RunCheckpointEntry,
    RunInfo,
)
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _instance_meta(inst) -> dict:
    """Read architecture/scale from an instance's config.yaml (best effort)."""
    meta: dict = {"architecture": None, "scale": None}
    cfg_path = Path(inst.path) / "config.yaml"
    if cfg_path.is_file():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            meta["architecture"] = cfg.get("architecture") or cfg.get("name")
            scale = cfg.get("scale") or cfg.get("scale_factor")
            meta["scale"] = scale if isinstance(scale, int) else None
        except Exception:
            pass
    return meta


def _active_jobs():
    """Task records currently backing a live training process."""
    from sr_engine.api.app import tasks

    return [r for r in tasks.list_jobs() if r.status == "running"]


def _active_job_ids() -> set[str]:
    """Job ids currently backing a live training process."""
    return {r.job_id for r in _active_jobs()}


@router.get("", response_model=list[ModelRuns])
async def list_runs(ws: Workspace = Depends(get_workspace)):
    active = _active_job_ids()
    result: list[ModelRuns] = []
    for inst in ws.list_model_instances():
        runs = [
            RunInfo(**ws.run_summary(run_dir, active))
            for run_dir in ws.list_runs(inst.name)
        ]
        runs.sort(key=lambda r: r.run_id, reverse=True)
        meta = _instance_meta(inst)
        result.append(ModelRuns(
            name=inst.name,
            architecture=meta["architecture"],
            scale=meta["scale"],
            runs=runs,
        ))
    return result


def _parse_epoch_metrics_sidecar(ckpt_path: Path) -> dict | None:
    """Prefer the per-checkpoint metrics sidecar written by the trainer."""
    m = re.fullmatch(r"(epoch_\d+)\.pt", ckpt_path.name)
    if not m:
        return None
    sidecar = ckpt_path.with_name(f"{m.group(1)}_metrics.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "loss": data.get("val_loss"),
        "psnr": data.get("psnr"),
        "ssim": data.get("ssim"),
    }


def _load_jsonl_validate_metrics(run_dir: Path) -> dict[int, dict]:
    """Map epoch -> last validate row from metrics.jsonl (fallback)."""
    f = run_dir / "metrics.jsonl"
    if not f.is_file():
        return {}
    out: dict[int, dict] = {}
    try:
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lstrip().startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "validate":
                continue
            epoch = obj.get("epoch")
            if isinstance(epoch, int):
                out[epoch] = {
                    "loss": obj.get("val_loss"),
                    "psnr": obj.get("psnr"),
                    "ssim": obj.get("ssim"),
                }
    except OSError:
        pass
    return out


@router.get("/{instance}/{run_id}/checkpoints", response_model=list[RunCheckpointEntry])
async def run_checkpoints(instance: str, run_id: str, ws: Workspace = Depends(get_workspace)):
    try:
        inst_path = ws.path / "models" / instance
        if not inst_path.is_dir():
            raise FileNotFoundError(f"Model instance '{instance}' not found")
        if not re.fullmatch(r"run_[A-Za-z0-9_]+", run_id):
            raise HTTPException(400, f"Invalid run id: {run_id!r}")
        run_dir = (inst_path / "runs" / run_id).resolve()
        if run_dir.parent != (inst_path / "runs").resolve() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {instance}/{run_id}")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    jsonl_metrics = _load_jsonl_validate_metrics(run_dir)
    entries: list[RunCheckpointEntry] = []
    for ckpt in sorted(run_dir.glob("epoch_*.pt")):
        m = re.fullmatch(r"epoch_(\d+)\.pt", ckpt.name)
        if not m:
            continue
        epoch = int(m.group(1))
        metrics = _parse_epoch_metrics_sidecar(ckpt) or jsonl_metrics.get(epoch, {})
        try:
            st = ckpt.stat()
            size_mb = st.st_size / (1024 * 1024)
            created_at = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            size_mb = 0.0
            created_at = ""
        entries.append(RunCheckpointEntry(
            epoch=epoch,
            filename=ckpt.name,
            path=str(ckpt),
            created_at=created_at,
            file_size_mb=round(size_mb, 2),
            metrics=CheckpointMetrics(
                loss=metrics.get("loss"),
                psnr=metrics.get("psnr"),
                ssim=metrics.get("ssim"),
            ),
        ))
    return entries


@router.delete("/{instance}/{run_id}")
async def delete_run(instance: str, run_id: str, ws: Workspace = Depends(get_workspace)):
    # Refuse to delete the run backing a live training job.
    active_run_ids = {
        str(p.get("run_dir", "")).split("/")[-1]
        for p in (r.params or {} for r in _active_jobs())
        if p.get("run_dir")
    }
    if run_id in active_run_ids:
        raise HTTPException(409, "RUN_ACTIVE: cannot delete a run that is currently training")
    try:
        ws.delete_run(instance, run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"deleted": run_id}
