"""sr-engine CLI entry point."""

from pathlib import Path
import click

from sr_engine.workspace import Workspace

from .cmd_dataset import dataset
from .cmd_train import train
from .cmd_infer import infer
from .cmd_model import model
from .cmd_env import env
from .cmd_serve import serve
from .workspace_commands import workspace


@click.group()
@click.option("--workspace", envvar="SRENGINE_WORKSPACE", type=click.Path(path_type=Path),
              default=None, help="Explicit workspace path.")
@click.version_option()
@click.pass_context
def cli(ctx, workspace: Path | None) -> None:
    """sr-engine: super-resolution training and inference toolkit."""
    ctx.ensure_object(dict)
    if workspace:
        ctx.obj["workspace"] = Workspace(workspace)
    else:
        ctx.obj["workspace"] = Workspace.discover()


cli.add_command(dataset)
cli.add_command(train)
cli.add_command(infer)
cli.add_command(model)
cli.add_command(env)
cli.add_command(workspace)
cli.add_command(serve)
