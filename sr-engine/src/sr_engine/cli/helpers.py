import os
from pathlib import Path
from typing import Any, Callable

import click

from sr_engine.utils.config import DefaultConfigs
from sr_engine.utils.progress import ProgressReporter
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

_SRENGINE_GUI_SOCKET = "SRENGINE_GUI_SOCKET"
_control_connection: tuple | None = None


def invalidate_control_connection() -> None:
    global _control_connection
    if _control_connection is not None:
        _, _, close_fn = _control_connection
        close_fn()
    _control_connection = None


def _get_control_connection():
    global _control_connection
    if _control_connection is None:
        env_value = os.environ.get(_SRENGINE_GUI_SOCKET)
        if env_value:
            from sr_engine.gui_bridge.protocol import connect_control_socket
            job_id, send_fn, close_fn = connect_control_socket(env_value)

            def _reconnecting_send(msg: dict) -> None:
                try:
                    send_fn(msg)
                except OSError:
                    invalidate_control_connection()
                    raise

            _control_connection = (job_id, _reconnecting_send, close_fn)
    return _control_connection


def resolve_reporter(**tqdm_kwargs: Any) -> ProgressReporter:
    conn = _get_control_connection()
    if conn:
        from sr_engine.gui_bridge.protocol import SocketReporter
        return SocketReporter(send_fn=conn[1], job_id=conn[0])
    from sr_engine.utils.progress import TqdmReporter
    return TqdmReporter(**tqdm_kwargs, disable=True)


def resolve_callbacks() -> list:
    conn = _get_control_connection()
    if conn:
        from sr_engine.gui_bridge.protocol import SocketCallback
        return [SocketCallback(send_fn=conn[1], job_id=conn[0])]
    return []


def resolve_cancel_check() -> Callable[[], bool]:
    if os.environ.get(_SRENGINE_GUI_SOCKET):
        from sr_engine.gui_bridge.jobs import was_cancelled, install_cancel_handler
        install_cancel_handler()
        return was_cancelled
    return lambda: False
