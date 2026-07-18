from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from sr_engine.api.deps import get_configs, get_workspace
from sr_engine.api.schemas import ExportParams, ModelInfo, ModelInstance
from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/models", tags=["models"])


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
        ckpts = ws.get_instance_checkpoints(inst.name)
        result.append(ModelInstance(
            name=inst.name,
            path=str(inst.path),
            checkpoints=[str(p) for p in ckpts],
        ))
    return result


@router.get("/instances/{name}", response_model=ModelInstance)
async def instance_info(name: str, ws: Workspace = Depends(get_workspace)):
    try:
        inst = ws.get_model_instance(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    ckpts = ws.get_instance_checkpoints(name)
    return ModelInstance(
        name=inst.name,
        path=str(inst.path),
        architecture=inst.name,
        checkpoints=[str(p) for p in ckpts],
    )


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