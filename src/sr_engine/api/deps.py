from pathlib import Path

from fastapi import HTTPException

from sr_engine.utils.config import DefaultConfigs
from sr_engine.utils.logging import set_log_file
from sr_engine.workspace import Workspace


_workspace: Workspace | None = None
_configs: DefaultConfigs | None = None


def init_workspace(path: str) -> Workspace:
    global _workspace, _configs
    _workspace = Workspace(Path(path))
    _configs = DefaultConfigs(workspace=_workspace)
    set_log_file(_workspace.path)
    return _workspace


async def get_workspace() -> Workspace:
    if _workspace is None:
        raise HTTPException(503, "Workspace not initialised — call /api/workspace/init first")
    return _workspace


async def get_configs() -> DefaultConfigs:
    if _configs is None:
        raise HTTPException(503, "Config loader not initialised")
    return _configs


async def resolve_model_config(model_name: str) -> dict:
    cfg = _configs.get_model_config(model_name)
    if not cfg:
        available = list(_configs.models.keys())
        raise HTTPException(404, f"Model '{model_name}' not found. Available: {available}")
    return cfg