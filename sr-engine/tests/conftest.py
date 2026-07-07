from pathlib import Path
import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sr_engine.cli.main import cli
from sr_engine.workspace import Workspace
from sr_engine.utils.config import save_config


def _make_image(path: Path, w: int = 64, h: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_dataset_dir(tmp_path: Path, num_pairs: int = 5) -> Path:
    d = tmp_path / "dataset"
    for i in range(num_pairs):
        _make_image(d / "HR" / f"frame_{i:04d}.png", w=256, h=256)
        _make_image(d / "LR" / f"frame_{i:04d}.png", w=64, h=64)
    return d


def _create_manifest(dataset_dir: Path, scale: int = 4) -> None:
    """Create a minimal manifest.json for a dataset directory."""
    import json
    hr_dir = dataset_dir / "HR"
    lr_dir = dataset_dir / "LR"
    pairs = []
    for hr_path in sorted(hr_dir.glob("*.png")):
        name = hr_path.name
        if (lr_dir / name).exists():
            pairs.append({
                "hr": f"HR/{name}",
                "lr": f"LR/{name}",
            })
    manifest = {
        "config": {"scale": scale},
        "pairs": pairs,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.fixture
def cli_runner():
    yield CliRunner()


@pytest.fixture
def cli_invoker(cli_runner):
    """Invoke the sr-engine CLI with args, returns Result."""
    def invoke(args: list[str], **kwargs):
        return cli_runner.invoke(cli, args, **kwargs)
    return invoke


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    ws.create_project("test_proj")
    return ws


@pytest.fixture
def custom_train_config(tmp_path):
    cfg = {"max_epochs": 3, "batch_size": 2}
    path = tmp_path / "custom_train.yaml"
    save_config(cfg, path)
    return path


@pytest.fixture
def minimal_dataset(tmp_path):
    return _create_dataset_dir(tmp_path, num_pairs=3)
