"""Tests for PairedImageFolderDataset."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sr_engine.data.datasets import PairedImageFolderDataset


def _make_image(path: Path, w: int = 64, h: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_dataset_dir(tmp_path: Path, num_pairs: int = 3, hr_w: int = 256, hr_h: int = 256, lr_w: int = 64, lr_h: int = 64) -> Path:
    d = tmp_path / "dataset"
    hr_dir = d / "HR"
    lr_dir = d / "LR"
    for i in range(num_pairs):
        _make_image(hr_dir / f"frame_{i:04d}.png", w=hr_w, h=hr_h)
        _make_image(lr_dir / f"frame_{i:04d}.png", w=lr_w, h=lr_h)
    return d


class TestPairedImageFolderDataset:
    def test_len_matches_file_count(self, tmp_path: Path):
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        ds = PairedImageFolderDataset(d)
        assert len(ds) == 5

    def test_getitem_returns_tuple_of_tensors(self, tmp_path: Path):
        d = _create_dataset_dir(tmp_path, num_pairs=1)
        ds = PairedImageFolderDataset(d)
        lr, hr = ds[0]
        assert isinstance(lr, torch.Tensor)
        assert isinstance(hr, torch.Tensor)
        assert lr.shape[0] == 3  # RGB channels
        assert hr.shape[0] == 3

    def test_getitem_values_in_range(self, tmp_path: Path):
        d = _create_dataset_dir(tmp_path, num_pairs=1)
        ds = PairedImageFolderDataset(d)
        lr, hr = ds[0]
        assert lr.min() >= 0.0
        assert lr.max() <= 1.0
        assert hr.min() >= 0.0
        assert hr.max() <= 1.0

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PairedImageFolderDataset(tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path: Path):
        d = tmp_path / "empty"
        (d / "HR").mkdir(parents=True)
        (d / "LR").mkdir(parents=True)
        with pytest.raises(ValueError, match="No HR/LR pairs found"):
            PairedImageFolderDataset(d)

    def test_with_manifest(self, tmp_path: Path):
        import json
        d = _create_dataset_dir(tmp_path, num_pairs=3)
        manifest = {
            "config": {"scale": 4},
            "pairs": [
                {"hr": "HR/frame_0000.png", "lr": "LR/frame_0000.png"},
                {"hr": "HR/frame_0001.png", "lr": "LR/frame_0001.png"},
            ],
        }
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f)
        ds = PairedImageFolderDataset(d)
        assert len(ds) == 2  # manifest has 2 pairs, not 3

    def test_with_transform(self, tmp_path: Path):
        from sr_engine.data.transforms import RandomCrop
        d = _create_dataset_dir(tmp_path, num_pairs=1)
        transform = RandomCrop(patch_size=32, scale=4)
        ds = PairedImageFolderDataset(d, transform=transform)
        lr, hr = ds[0]
        assert lr.shape == (3, 32, 32)
        assert hr.shape == (3, 128, 128)

    def test_module_importable(self):
        assert PairedImageFolderDataset is not None
