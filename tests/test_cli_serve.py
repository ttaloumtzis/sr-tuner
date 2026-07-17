"""Tests for CLI serve command."""


def test_serve_help(cli_invoker):
    """``serve --help`` should succeed."""
    r = cli_invoker(["serve", "--help"])
    assert r.exit_code == 0


def test_serve_start_help(cli_invoker):
    """``serve start --help`` should succeed and show port/host options."""
    r = cli_invoker(["serve", "start", "--help"])
    assert r.exit_code == 0
    assert "--port" in r.output
    assert "--host" in r.output


def test_serve_start_no_workspace_fails(cli_invoker, tmp_path, monkeypatch):
    """``serve start`` without a workspace should fail."""
    monkeypatch.chdir(tmp_path)
    r = cli_invoker(["serve", "start"])
    assert r.exit_code != 0