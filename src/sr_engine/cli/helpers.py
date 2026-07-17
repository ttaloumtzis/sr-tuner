"""CLI helper functions — workspace resolution, config loading, progress resolution."""

import os
from pathlib import Path
from typing import Any, Callable

import click

from sr_engine.utils.config import DefaultConfigs
from sr_engine.utils.progress import ProgressReporter, TqdmReporter
from sr_engine.workspace import Workspace


def resolve_workspace(ctx) -> Workspace | None:
    """Resolve the workspace from click context, env var, or CWD discovery.

    Args:
        ctx: Click context (may contain a cached workspace).

    Returns:
        Workspace instance or None if not found.
    """
    if ctx.obj and "workspace" in ctx.obj:
        return ctx.obj["workspace"]
    explicit = os.environ.get("SRENGINE_WORKSPACE")
    if explicit:
        return Workspace(Path(explicit))
    return Workspace.discover()


def require_workspace(ctx) -> Workspace:
    """Resolve the workspace or raise a ClickException.

    Args:
        ctx: Click context.

    Returns:
        Workspace instance.

    Raises:
        click.ClickException: If no workspace is found.
    """
    ws = resolve_workspace(ctx)
    if not ws:
        raise click.ClickException(
            "No workspace found. Use 'workspace init' to create one."
        )
    return ws


def make_workspace_config_loader(ctx, no_workspace_config=False, *, ws=None
                                 ) -> tuple[Workspace | None, DefaultConfigs]:
    """Return a ``(workspace, config_loader)`` pair.

    Args:
        ctx: Click context.
        no_workspace_config: If True, skip workspace config overrides.
        ws: Optional pre-resolved workspace (otherwise resolved from ctx).

    Returns:
        ``(workspace_or_None, DefaultConfigs)``.
    """
    if ws is None:
        ws = resolve_workspace(ctx)
    return ws, DefaultConfigs(workspace=None if no_workspace_config else ws)


def resolve_model_config(cfg_loader: DefaultConfigs, name: str) -> dict | None:
    """Look up a model config by name, raising on unknown models.

    Args:
        cfg_loader: DefaultConfigs instance.
        name: Model name.

    Returns:
        Model configuration dict.

    Raises:
        click.ClickException: If the model name is not found.
    """
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

def resolve_reporter(**tqdm_kwargs: Any) -> ProgressReporter:
    """Return a ``TqdmReporter`` for terminal progress output.

    Args:
        tqdm_kwargs: Forwarded to ``TqdmReporter``.

    Returns:
        A ``ProgressReporter`` instance.
    """
    return TqdmReporter(**tqdm_kwargs)


def resolve_callbacks() -> list:
    """Return an empty callback list.

    Returns:
        Empty list (no GUI callbacks in CLI mode).
    """
    return []


def resolve_cancel_check() -> Callable[[], bool]:
    """Return a no-op cancellation check.

    Returns:
        Callable that always returns ``False``.
    """
    return lambda: False
