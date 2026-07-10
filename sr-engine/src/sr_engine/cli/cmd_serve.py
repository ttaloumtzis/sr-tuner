"""CLI command for the GUI socket server."""

import click

from .helpers import require_workspace


@click.group()
def serve() -> None:
    """Long-lived backend server for GUI clients."""


@serve.command()
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.pass_context
def start(ctx, host, port) -> None:
    """Start the socket server. Blocks until killed."""
    ws = require_workspace(ctx)
    from sr_engine.gui_bridge.server import Server
    server = Server(host=host, port=port, workspace=ws.path)
    click.secho(f"GUI server listening on {host}:{port} (workspace: {ws.path})",
                fg="green", bold=True)
    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        server.stop()
