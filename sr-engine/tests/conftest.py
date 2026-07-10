"""Shared fixtures, helpers, and mocks for the sr-engine test suite."""

from pathlib import Path
import struct
import json
import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch

from sr_engine.cli.main import cli
from sr_engine.workspace import Workspace
from sr_engine.utils.config import save_config


def _make_image(path: Path, w: int = 64, h: int = 64, seed: int | None = None) -> None:
    """Write a random RGB image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_grayscale_image(path: Path, w: int = 64, h: int = 64) -> None:
    """Write a random grayscale image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_corrupt_image(path: Path) -> None:
    """Write invalid bytes to *path* (not a valid image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\x02\x03not-a-png-or-jpg")


def _make_video(path: Path, num_frames: int = 10, fps: int = 30, w: int = 64, h: int = 64) -> None:
    """Write a random MP4 video to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


def _create_dataset_dir(tmp_path: Path, num_pairs: int = 5) -> Path:
    """Create a temporary HR/LR dataset directory with *num_pairs* random images."""
    d = tmp_path / "dataset"
    for i in range(num_pairs):
        _make_image(d / "HR" / f"frame_{i:04d}.png", w=256, h=256)
        _make_image(d / "LR" / f"frame_{i:04d}.png", w=64, h=64)
    return d


def _create_manifest(dataset_dir: Path, scale: int = 4) -> None:
    """Write a ``manifest.json`` for the given dataset directory."""
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


def _create_dataset_with_manifest(tmp_path: Path, num_pairs: int = 5, scale: int = 4) -> Path:
    """Create a dataset directory with both images and a manifest."""
    d = _create_dataset_dir(tmp_path, num_pairs)
    _create_manifest(d, scale)
    return d


@pytest.fixture
def cli_runner():
    """Yield a Click ``CliRunner`` instance."""
    yield CliRunner()


@pytest.fixture
def cli_invoker(cli_runner):
    """Return a callable that invokes the CLI with given args."""
    def invoke(args: list[str], **kwargs):
        return cli_runner.invoke(cli, args, **kwargs)
    return invoke


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create an initialised workspace with a ``test_proj`` project."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    ws.create_project("test_proj")
    return ws


@pytest.fixture
def empty_workspace(tmp_path):
    """Create an initialised workspace with no projects."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    return ws


@pytest.fixture
def custom_train_config(tmp_path):
    """Save and return a path to a minimal custom training config YAML."""
    cfg = {"max_epochs": 3, "batch_size": 2}
    path = tmp_path / "custom_train.yaml"
    save_config(cfg, path)
    return path


@pytest.fixture
def minimal_dataset(tmp_path):
    """Return a dataset directory with 3 HR/LR pairs (no manifest)."""
    return _create_dataset_dir(tmp_path, num_pairs=3)


@pytest.fixture
def minimal_dataset_with_manifest(tmp_path):
    """Return a dataset directory with 3 HR/LR pairs and a manifest."""
    return _create_dataset_with_manifest(tmp_path, num_pairs=3)


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda.is_available() and related functions.

    Usage::

        def test_foo(mock_torch_cuda):
            with mock_torch_cuda(available=True, bf16=True, hip=False):
                ...
    """
    def _mock(available: bool = False, bf16: bool = False, hip: bool = False):
        patches = [
            patch("torch.cuda.is_available", return_value=available),
        ]
        if available:
            patches.append(patch("torch.cuda.is_bf16_supported", return_value=bf16))
            if hip:
                patches.append(patch("torch.version.hip", "1.0"))
            else:
                patches.append(patch("torch.version.hip", None))
        return _patch_context(patches)

    class _patch_context:
        """Context manager that starts/stops a list of patches."""
        def __init__(self, patches):
            self._patches = patches
        def __enter__(self):
            for p in self._patches:
                p.start()
        def __exit__(self, *args):
            for p in reversed(self._patches):
                p.stop()

    return _mock


@pytest.fixture
def mock_socket():
    """Create a pair of connected mock sockets for protocol/server tests."""
    import socket as _socket

    def _make_connected_pair():
        a, b = _socket.socketpair()
        return a, b

    return _make_connected_pair


@pytest.fixture
def mock_subprocess_popen():
    """Mock subprocess.Popen for JobManager tests."""
    def _mock(stdout_lines: list[str] | None = None, returncode: int = 0):
        mock_proc = MagicMock()
        mock_proc.pid = 12345

        class _MockStream:
            def __init__(self, lines):
                self._lines = lines or []
                self._idx = 0

            def readline(self, limit=-1):
                if self._idx >= len(self._lines):
                    return b""
                line = self._lines[self._idx]
                self._idx += 1
                if isinstance(line, str):
                    line = line.encode("utf-8")
                return line + b"\n" if not line.endswith(b"\n") else line

            def __iter__(self):
                return iter(self._lines)

            def close(self):
                pass

        mock_proc.stdout = _MockStream(stdout_lines or [])
        return mock_proc

    return _mock


@pytest.fixture
def tqdm_mock():
    """Fixture that prevents tqdm output during tests and returns the mock."""
    with patch("tqdm.tqdm") as mock_tqdm:
        mock_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_bar
        yield mock_bar


@pytest.fixture
def mock_progress_reporter():
    """Return a MagicMock conforming to ProgressReporter interface."""
    return MagicMock()


@pytest.fixture
def sample_config_dict():
    """A comprehensive config dict with all typical keys."""
    return {
        "model": {
            "name": "rrdb",
            "scale": 4,
            "num_in_ch": 3,
            "num_out_ch": 3,
        },
        "train": {
            "max_epochs": 10,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "validation_split": 0.1,
            "device": "cpu",
        },
        "dataset": {
            "scale": 4,
            "patch_size": 64,
            "augment": True,
        },
        "degradation": {
            "blur": {"kernel_size": 7, "sigma": [0.1, 3.0], "prob": 0.5},
            "noise": {
                "gaussian": {"sigma_range": [1, 30], "prob": 0.5},
                "poisson": {"scale_range": [0.05, 0.5], "prob": 0.5},
            },
            "jpeg": {"quality_range": [60, 95], "prob": 0.5},
        },
    }


@pytest.fixture
def sample_image(tmp_path):
    """Return a path to a generated sample PNG."""
    path = tmp_path / "sample.png"
    _make_image(path)
    return path


@pytest.fixture
def corrupt_image(tmp_path):
    """Return a path to a corrupt image file."""
    path = tmp_path / "corrupt.png"
    _make_corrupt_image(path)
    return path


@pytest.fixture
def sample_video(tmp_path):
    """Return a path to a generated sample MP4."""
    path = tmp_path / "sample.mp4"
    _make_video(path)
    return path
