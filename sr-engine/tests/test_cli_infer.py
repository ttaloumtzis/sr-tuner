"""Smoke tests for infer CLI command."""

from tests.conftest import _make_image


def test_infer_image_exists(cli_invoker, tmp_path):
    """Verify infer run --help works and command is registered."""
    r = cli_invoker(["infer", "run", "--help"])
    assert r.exit_code == 0
    assert "model" in r.output
    assert "input" in r.output
    assert "output" in r.output


def test_infer_missing_model_fails(cli_invoker, tmp_path):
    r = cli_invoker([
        "infer", "run",
        "--model", str(tmp_path / "nonexistent.pt"),
        "--input", str(tmp_path / "input.png"),
        "--output", str(tmp_path / "output.png"),
    ])
    assert r.exit_code != 0


def test_infer_missing_input_fails(cli_invoker, tmp_path):
    r = cli_invoker([
        "infer", "run",
        "--model", str(tmp_path / "model.pt"),
        "--input", str(tmp_path / "nonexistent.png"),
        "--output", str(tmp_path / "output.png"),
    ])
    assert r.exit_code != 0
