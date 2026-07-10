"""Smoke tests for env CLI commands."""


def test_env_check_help(cli_invoker):
    """``env check --help`` should succeed."""
    r = cli_invoker(["env", "check", "--help"])
    assert r.exit_code == 0


def test_env_bench_help(cli_invoker):
    """``env bench --help`` should succeed."""
    r = cli_invoker(["env", "bench", "--help"])
    assert r.exit_code == 0
