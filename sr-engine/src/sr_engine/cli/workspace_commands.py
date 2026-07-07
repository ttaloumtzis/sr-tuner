from pathlib import Path
import click

from sr_engine.workspace import Workspace


@click.group()
def workspace() -> None:
    """Workspace management commands."""


@workspace.command()
@click.option("--path", "-p", default=".", type=click.Path(path_type=Path),
              help="Path to initialize workspace in (default: CWD).")
def init(path: Path) -> None:
    """Initialize a workspace directory tree."""
    ws = Workspace(path)
    ws.init()
    click.secho(f"Workspace initialized at {ws.path}", fg="green", bold=True)


@workspace.command()
@click.pass_context
def info(ctx) -> None:
    """Show workspace summary."""
    ws: Workspace | None = ctx.obj.get("workspace") if ctx.obj else Workspace.discover()
    if not ws:
        raise click.ClickException("No workspace found. Use 'workspace init' to create one.")
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
    ws: Workspace | None = ctx.obj.get("workspace") if ctx.obj else Workspace.discover()
    if not ws:
        raise click.ClickException("No workspace found. Use 'workspace init' to create one.")
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
    ws: Workspace | None = ctx.obj.get("workspace") if ctx.obj else Workspace.discover()
    if not ws:
        raise click.ClickException("No workspace found. Use 'workspace init' to create one.")
    try:
        proj = ws.create_project(name)
        click.secho(f"Project '{proj.name}' created at {proj.path}", fg="green", bold=True)
    except FileExistsError as e:
        raise click.ClickException(str(e))


@project.command(name="list")
@click.pass_context
def list_projects(ctx) -> None:
    """List projects in the workspace."""
    ws: Workspace | None = ctx.obj.get("workspace") if ctx.obj else Workspace.discover()
    if not ws:
        click.echo("No workspace found.")
        return
    projects = ws.list_projects()
    if not projects:
        click.echo("No projects yet.")
        return
    click.echo(f"Projects ({len(projects)}):")
    for p in projects:
        click.echo(f"  - {p.name}")
