"""Tests for workspace CLI commands."""

from pathlib import Path

from click.testing import CliRunner
from sr_engine.cli.main import cli
from sr_engine.workspace import Workspace


def test_workspace_init_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "init", "--help"])
    assert r.exit_code == 0


def test_workspace_info_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "info", "--help"])
    assert r.exit_code == 0


def test_workspace_check_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "check", "--help"])
    assert r.exit_code == 0


def test_workspace_init_creates_marker(cli_runner, tmp_path):
    r = cli_runner.invoke(cli, ["workspace", "init", "--path", str(tmp_path / "ws")])
    assert r.exit_code == 0
    assert (tmp_path / "ws" / ".sr_workspace").is_file()


def test_workspace_init_creates_dirs(cli_runner, tmp_path):
    r = cli_runner.invoke(cli, ["workspace", "init", "--path", str(tmp_path / "ws")])
    assert r.exit_code == 0
    for name in ("datasets", "projects", "configs"):
        assert (tmp_path / "ws" / name).is_dir()


def test_workspace_info_on_uninitialized_fails(cli_runner, tmp_path):
    r = cli_runner.invoke(cli, ["workspace", "info"])
    assert r.exit_code != 0
    assert "No workspace found" in r.output


def test_workspace_check_on_uninitialized_fails(cli_runner, tmp_path):
    r = cli_runner.invoke(cli, ["workspace", "check"])
    assert r.exit_code != 0
    assert "No workspace found" in r.output
