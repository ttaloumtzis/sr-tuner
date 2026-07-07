"""Smoke tests for model CLI commands."""


def test_model_export_help(cli_invoker):
    r = cli_invoker(["model", "export", "--help"])
    assert r.exit_code == 0
    assert "model-name" in r.output


def test_model_info_missing_ckpt_fails(cli_invoker, tmp_path):
    r = cli_invoker([
        "model", "info",
        "--model", str(tmp_path / "nonexistent.pt"),
    ])
    assert r.exit_code != 0
