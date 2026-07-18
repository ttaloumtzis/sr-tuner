from fastapi import APIRouter, Depends

from sr_engine.api.deps import get_workspace
from sr_engine.api.schemas import WorkspaceInfo, WorkspaceInitParams
from sr_engine.workspace import Workspace

router = APIRouter(prefix="/api/workspace", tags=["workspace"])


@router.get("", response_model=WorkspaceInfo)
async def workspace_info(ws: Workspace = Depends(get_workspace)):
    return WorkspaceInfo(
        path=str(ws.path),
        exists=ws.path.exists(),
        models=[{"name": m.name, "path": str(m.path)} for m in ws.list_model_instances()],
    )


@router.post("/init", response_model=WorkspaceInfo)
async def workspace_init(params: WorkspaceInitParams):
    from sr_engine.api.deps import init_workspace

    ws = init_workspace(params.path)
    ws.init()
    return WorkspaceInfo(
        path=str(ws.path),
        exists=True,
    )


@router.get("/check")
async def workspace_check(ws: Workspace = Depends(get_workspace)):
    check = ws.check()
    return {"healthy": check.healthy, "issues": check.issues}