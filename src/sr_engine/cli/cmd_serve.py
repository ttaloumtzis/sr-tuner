"""CLI command for the HTTP API server."""

import click

from .helpers import require_workspace


@click.group()
def serve() -> None:
    """Long-lived HTTP API server (FastAPI) for frontend clients."""


@serve.command()
@click.option("--port", type=int, default=8765, show_default=True,
              help="HTTP port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Network interface to bind to (use 0.0.0.0 for all).")
@click.pass_context
def start(ctx, host, port) -> None:
    """Start the FastAPI HTTP server. Blocks until killed."""
    require_workspace(ctx)
    import uvicorn
    click.secho(f"API server listening on http://{host}:{port}",
                fg="green", bold=True)
    uvicorn.run(
        "sr_engine.api.app:app",
        host=host,
        port=port,
        log_level="info",
    )