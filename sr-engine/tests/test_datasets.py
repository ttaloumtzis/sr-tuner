"""Tests for data/datasets.py — PairedImageFolderDataset."""

import pytest
import torch

from sr_engine.data.datasets import PairedImageFolderDataset


class TestPairedImageFolderDataset:
    """Tests for PairedImageFolderDataset."""

    def test_len(self, minimal_dataset_with_manifest):
        """__len__ should return the number of pairs."""
        ds = PairedImageFolderDataset(minimal_dataset_with_manifest)
        assert len(ds) == 3

    def test_getitem_returns_tensors(self, minimal_dataset_with_manifest):
        """__getitem__ should return HR and LR tensors."""
        ds = PairedImageFolderDataset(minimal_dataset_with_manifest)
        hr, lr = ds[0]
        assert isinstance(hr, torch.Tensor)
        assert isinstance(lr, torch.Tensor)

    def test_getitem_valid_shapes(self, minimal_dataset_with_manifest):
        """Tensors should have shape [C, H, W]."""
        ds = PairedImageFolderDataset(minimal_dataset_with_manifest)
        hr, lr = ds[0]
        assert hr.ndim == 3
        assert lr.ndim == 3

    def test_transform_applied(self, minimal_dataset_with_manifest):
        """A transform should be applied to both HR and LR."""
        transform = lambda lr, hr: (lr * 0.5, hr * 0.5)
        ds = PairedImageFolderDataset(
            minimal_dataset_with_manifest,
            transform=transform,
        )
        hr, lr = ds[0]
        assert hr.max() <= 0.5

    def test_missing_manifest_raises(self, tmp_path):
        """Missing manifest.json should raise FileNotFoundError."""
        (tmp_path / "HR").mkdir(parents=True)
        (tmp_path / "LR").mkdir(parents=True)
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            PairedImageFolderDataset(tmp_path)
