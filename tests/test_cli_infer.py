"""Smoke tests for infer CLI command."""

from sr_engine.models.registry import build_model
from sr_engine.workspace import Workspace

_TINY_RRDBNET = {
    "num_in_ch": 3,
    "num_out_ch": 3,
    "num_feat": 8,
    "num_block": 1,
    "num_grow_ch": 8,
    "scale": 4,
}


def _instance_with_version(ws: Workspace, name: str, arch: dict) -> None:
    """Create a model instance plus one saved version (state_dict from a real model)."""
    ws.create_model_instance(name, arch)
    model = build_model("rrdb_esrgan", _TINY_RRDBNET)
    ws.save_model_version(name, "v1", model.state_dict())


def test_infer_image_exists(cli_invoker, tmp_path):
    """Verify infer run --help works and command is registered."""
    r = cli_invoker(["infer", "run", "--help"])
    assert r.exit_code == 0
    assert "model" in r.output
    assert "input" in r.output
    assert "output" in r.output


def test_infer_missing_model_fails(cli_invoker, tmp_path):
    """Infer with a nonexistent model checkpoint should fail."""
    r = cli_invoker([
        "infer", "run",
        "--model", str(tmp_path / "nonexistent.pt"),
        "--input", str(tmp_path / "input.png"),
        "--output", str(tmp_path / "output.png"),
    ])
    assert r.exit_code != 0


def test_infer_missing_input_fails(cli_invoker, tmp_path):
    """Infer with a nonexistent input image should fail."""
    r = cli_invoker([
        "infer", "run",
        "--model", str(tmp_path / "model.pt"),
        "--input", str(tmp_path / "nonexistent.png"),
        "--output", str(tmp_path / "output.png"),
    ])
    assert r.exit_code != 0


def test_infer_with_instance_resolves_architecture_key(cli_invoker, tmp_path, sample_image):
    """infer run --instance should resolve arch from the 'architecture' config key."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    arch = {"architecture": "rrdb_esrgan", **_TINY_RRDBNET}
    _instance_with_version(ws, "my_model", arch)

    out = tmp_path / "out.png"
    r = cli_invoker([
        "--workspace", str(ws.path),
        "infer", "run",
        "--instance", "my_model",
        "--input", str(sample_image),
        "--output", str(out),
        "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output
    assert out.is_file()


def test_infer_instance_missing_arch_fails(cli_invoker, tmp_path, sample_image):
    """infer run --instance should fail clearly when config has neither arch nor name."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    _instance_with_version(ws, "my_model", {"scale": 4})

    out = tmp_path / "out.png"
    r = cli_invoker([
        "--workspace", str(ws.path),
        "infer", "run",
        "--instance", "my_model",
        "--input", str(sample_image),
        "--output", str(out),
        "--device", "cpu",
    ])
    assert r.exit_code != 0
    assert "neither 'architecture' nor 'name'" in r.output
