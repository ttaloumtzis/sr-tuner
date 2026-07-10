from pathlib import Path
import click

from sr_engine.workspace import Workspace
from .helpers import require_workspace


@click.group()
def workspace() -> None:
    """Workspace management commands."""


@workspace.command()
@click.option("--path", "-p", default=".", type=click.Path(path_type=Path),
              help="Path to initialize workspace in (default: CWD).")
@click.option("--reset-configs", is_flag=True, default=False,
              help="Overwrite workspace configs with fresh copies from package defaults.")
def init(path: Path, reset_configs: bool) -> None:
    """Initialize a workspace directory tree."""
    ws = Workspace(path)
    ws.init(reset_configs=reset_configs)
    click.secho(f"Workspace initialized at {ws.path}", fg="green", bold=True)


@workspace.command()
@click.pass_context
def info(ctx) -> None:
    """Show workspace summary."""
    ws = require_workspace(ctx)
    d = ws.info()
    click.echo(f"Workspace: {d['path']}")
    click.echo(f"Projects:  {len(d['projects'])}")
    for name in d['projects']:
        click.echo(f"  - {name}")
    click.echo(f"Datasets:  {len(d['datasets'])}")
    for name in d['datasets']:
        click.echo(f"  - {name}")


@workspace.command()
@click.pass_context
def check(ctx) -> None:
    """Validate workspace health."""
    ws = require_workspace(ctx)
    report = ws.check()
    click.echo(f"Workspace: {report['path']}")
    click.echo(f"Structure:  {'OK' if report['status'] == 'ok' else report['status'].upper()}")
    click.echo(f"Projects:   {len(report['projects'])}")
    click.echo(f"Datasets:   {len(report['datasets'])}")
    if report["issues"]:
        click.secho("Issues:", fg="yellow", bold=True)
        for issue in report["issues"]:
            click.echo(f"  - {issue}")
        ctx.exit(1 if report["status"] == "error" else 0)
    else:
        click.secho("No issues found.", fg="green")


@click.group()
def project() -> None:
    """Project management commands."""


@project.command()
@click.argument("name")
@click.pass_context
def create(ctx, name: str) -> None:
    """Create a new project in the workspace."""
    ws = require_workspace(ctx)
    try:
        proj = ws.create_project(name)
        click.secho(f"Project '{proj.name}' created at {proj.path}", fg="green", bold=True)
    except FileExistsError as e:
        raise click.ClickException(str(e))


@project.command(name="list")
@click.pass_context
def list_projects(ctx) -> None:
    """List projects in the workspace."""
    ws: Workspace | None = require_workspace(ctx)
    projects = ws.list_projects()
    if not projects:
        click.echo("No projects yet.")
        return
    click.echo(f"Projects ({len(projects)}):")
    for p in projects:
        click.echo(f"  - {p.name}")
