from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from sr_engine.api.deps import get_configs, get_workspace
from sr_engine.api.schemas import CreateInstanceParams, ExportParams, ModelInfo, ModelInstance, ModelVersion
from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/models", tags=["models"])


def _enrich_instance(ws: Workspace, inst: ModelInstance) -> ModelInstance:
    """Populate architecture, scale, and latest_version from config.yaml / versions dir."""
    import yaml
    cfg_path = Path(inst.path) / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        inst.architecture = cfg.get("architecture") or inst.architecture
        inst.scale = cfg.get("scale")
        inst.config = cfg
    inst.latest_version = ws.latest_model_version(inst.name)
    return inst


@router.get("", response_model=list[ModelInfo])
async def list_models(cfg: DefaultConfigs = Depends(get_configs)):
    return [
        ModelInfo(name=name, display_name=name, description="")
        for name in cfg.models
    ]


@router.get("/instances", response_model=list[ModelInstance])
async def list_instances(ws: Workspace = Depends(get_workspace)):
    instances = ws.list_model_instances()
    result = []
    for inst in instances:
        mi = ModelInstance(name=inst.name, path=str(inst.path))
        _enrich_instance(ws, mi)
        result.append(mi)
    return result


@router.get("/instances/{name}", response_model=ModelInstance)
async def instance_info(name: str, ws: Workspace = Depends(get_workspace)):
    try:
        inst = ws.get_model_instance(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    mi = ModelInstance(name=inst.name, path=str(inst.path))
    _enrich_instance(ws, mi)
    return mi


@router.post("/instances", response_model=ModelInstance, status_code=201)
async def create_instance(params: CreateInstanceParams, ws: Workspace = Depends(get_workspace)):
    if not params.name.strip():
        raise HTTPException(400, "name is required")
    config_with_meta = {**params.config, "architecture": params.architecture}
    try:
        inst = ws.create_model_instance(params.name.strip(), config_with_meta)
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    mi = ModelInstance(name=inst.name, path=str(inst.path))
    _enrich_instance(ws, mi)
    return mi


@router.post("/instances/{name}/export")
async def export_model(name: str, params: ExportParams, ws: Workspace = Depends(get_workspace)):
    from sr_engine.models.checkpoint import export_checkpoint

    try:
        inst = ws.get_model_instance(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    ckpt_path = params.output or str(inst.path / "checkpoints" / "latest.pt")
    out_path = export_checkpoint(
        Path(ckpt_path),
        export_format=params.format,
        output_path=Path(params.output) if params.output else None,
    )
    return {"output": str(out_path), "format": params.format}


@router.get("/instances/{name}/versions", response_model=list[ModelVersion])
async def instance_versions(name: str, ws: Workspace = Depends(get_workspace)):
    try:
        ws.get_model_instance(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return ws.list_model_versions(name)


@router.delete("/instances/{name}")
async def delete_instance(name: str, ws: Workspace = Depends(get_workspace)):
    try:
        ws.delete_model_instance(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"deleted": name}