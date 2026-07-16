"""Tests for workspace CLI commands."""

from click.testing import CliRunner
from sr_engine.cli.main import cli


def test_workspace_init_help():
    """``workspace init --help`` should succeed."""
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "init", "--help"])
    assert r.exit_code == 0


def test_workspace_info_help():
    """``workspace info --help`` should succeed."""
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "info", "--help"])
    assert r.exit_code == 0


def test_workspace_check_help():
    """``workspace check --help`` should succeed."""
    runner = CliRunner()
    r = runner.invoke(cli, ["workspace", "check", "--help"])
    assert r.exit_code == 0


def test_workspace_init_creates_marker(cli_runner, tmp_path):
    """``workspace init`` should create the .sr_workspace marker file."""
    r = cli_runner.invoke(cli, ["workspace", "init", "--path", str(tmp_path / "ws")])
    assert r.exit_code == 0
    assert (tmp_path / "ws" / ".sr_workspace").is_file()


def test_workspace_init_creates_dirs(cli_runner, tmp_path):
    """``workspace init`` should create datasets, projects, and configs dirs."""
    r = cli_runner.invoke(cli, ["workspace", "init", "--path", str(tmp_path / "ws")])
    assert r.exit_code == 0
    for name in ("datasets", "projects", "configs"):
        assert (tmp_path / "ws" / name).is_dir()


def test_workspace_info_on_uninitialized_fails(cli_runner, tmp_path):
    """``workspace info`` without a workspace should fail."""
    r = cli_runner.invoke(cli, ["workspace", "info"])
    assert r.exit_code != 0
    assert "No workspace found" in r.output


def test_workspace_check_on_uninitialized_fails(cli_runner, tmp_path):
    """``workspace check`` without a workspace should fail."""
    r = cli_runner.invoke(cli, ["workspace", "check"])
    assert r.exit_code != 0
    assert "No workspace found" in r.output
