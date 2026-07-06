"""Tests for PairedImageFolderDataset."""

from pathlib import Path

import pytest

from sr_engine.data.datasets import PairedImageFolderDataset


class TestPairedImageFolderDataset:
    """Tests for the dataset class. Stub testing only — real logic TBD."""

    def test_init_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            PairedImageFolderDataset(tmp_path)

    def test_module_importable(self) -> None:
        assert PairedImageFolderDataset is not None
