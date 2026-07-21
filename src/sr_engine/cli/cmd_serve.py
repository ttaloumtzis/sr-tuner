"""CLI command for the HTTP API server."""

import logging

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
@click.option("--log-level", default="info", show_default=True,
              type=click.Choice(["debug", "info", "warning", "error", "critical"]),
              help="Uvicorn log level.")
@click.pass_context
def start(ctx, host, port, log_level) -> None:
    """Start the FastAPI HTTP server. Blocks until killed."""
    require_workspace(ctx)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    import uvicorn
    click.secho(f"API server listening on http://{host}:{port}",
                fg="green", bold=True)
    uvicorn.run(
        "sr_engine.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
    )