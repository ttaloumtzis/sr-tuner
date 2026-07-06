"""sr-engine CLI entry point."""

import click

from .cmd_dataset import dataset
from .cmd_train import train
from .cmd_infer import infer
from .cmd_model import model
from .cmd_env import env


@click.group()
@click.version_option()
def cli() -> None:
    """sr-engine: super-resolution training and inference toolkit."""


cli.add_command(dataset)
cli.add_command(train)
cli.add_command(infer)
cli.add_command(model)
cli.add_command(env)
