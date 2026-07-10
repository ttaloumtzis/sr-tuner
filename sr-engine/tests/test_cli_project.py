"""Tests for project CLI commands."""

from pathlib import Path

from click.testing import CliRunner
from sr_engine.cli.main import cli
from sr_engine.workspace import Workspace


def test_project_create_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["project", "create", "--help"])
    assert r.exit_code == 0


def test_project_list_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["project", "list", "--help"])
    assert r.exit_code == 0


def test_project_create_needs_workspace(cli_runner):
    r = cli_runner.invoke(cli, ["project", "create", "my_proj"])
    assert r.exit_code != 0
    assert "No workspace found" in r.output


def test_project_create_and_list(cli_runner, tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.init()

    r = cli_runner.invoke(cli, [
        "project", "create", "my_proj",
    ], env={"SRENGINE_WORKSPACE": str(tmp_path / "ws")})
    assert r.exit_code == 0
    assert (tmp_path / "ws" / "projects" / "my_proj").is_dir()

    r = cli_runner.invoke(cli, [
        "project", "list",
    ], env={"SRENGINE_WORKSPACE": str(tmp_path / "ws")})
    assert r.exit_code == 0
    assert "my_proj" in r.output


def test_project_create_duplicate(cli_runner, tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.create_project("dup")

    r = cli_runner.invoke(cli, [
        "project", "create", "dup",
    ], env={"SRENGINE_WORKSPACE": str(tmp_path / "ws")})
    assert r.exit_code != 0
    assert "already exists" in r.output


def test_project_list_empty(cli_runner, tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.init()

    r = cli_runner.invoke(cli, [
        "project", "list",
    ], env={"SRENGINE_WORKSPACE": str(tmp_path / "ws")})
    assert r.exit_code == 0
    assert "No projects yet" in r.output
