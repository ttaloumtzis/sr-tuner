import os
from pathlib import Path

import click

from sr_engine.utils.config import DefaultConfigs
from sr_engine.workspace import Workspace


def resolve_workspace(ctx) -> Workspace | None:
    if ctx.obj and "workspace" in ctx.obj:
        return ctx.obj["workspace"]
    explicit = os.environ.get("SRENGINE_WORKSPACE")
    if explicit:
        return Workspace(Path(explicit))
    return Workspace.discover()


def require_workspace(ctx) -> Workspace:
    ws = resolve_workspace(ctx)
    if not ws:
        raise click.ClickException(
            "No workspace found. Use 'workspace init' to create one."
        )
    return ws


def make_workspace_config_loader(ctx, no_workspace_config=False, *, ws=None
                                 ) -> tuple[Workspace | None, DefaultConfigs]:
    if ws is None:
        ws = resolve_workspace(ctx)
    return ws, DefaultConfigs(workspace=None if no_workspace_config else ws)


def resolve_model_config(cfg_loader: DefaultConfigs, name: str) -> dict | None:
    cfg = cfg_loader.get_model_config(name)
    if not cfg:
        available = list(cfg_loader.models.keys())
        raise click.ClickException(
            f"Model '{name}' not found. Available: {available}"
        )
    return cfg


no_workspace_config_option = click.option(
    "--no-workspace-config", is_flag=True, default=False,
    help="Skip workspace config auto-discovery, use package defaults only.",
)
