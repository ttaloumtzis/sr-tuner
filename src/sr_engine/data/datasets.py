"""Torch Dataset classes for paired HR/LR image folders."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load an image from disk as a float32 CHW tensor in [0, 1], RGB order.

    Args:
        path: Path to the image file.

    Returns:
        Tensor of shape ``(3, H, W)``.

    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image (missing or corrupt): {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).contiguous()

    if any(d == 0 for d in tensor.shape):
        raise ValueError(f"Image loaded with zero-dimension tensor {tuple(tensor.shape)}: {path}")
    return tensor


class PairedImageFolderDataset(Dataset):
    """Reads paired HR/LR images from a dataset directory.

    Expects the directory structure::

        <dataset_dir>/
            HR/<filename>.png
            LR/<filename>.png
            manifest.json   (optional, produced by dataset_builder)

    If ``manifest.json`` is present, pairs are read from its ``pairs`` list
    (this is the same manifest ``dataset_builder``/``dataset_validator``
    produce and check, so it's treated as the source of truth). Otherwise,
    falls back to matching files by filename between HR/ and LR/, mirroring
    the logic in ``dataset_validator.validate``.
    """

    def __init__(self, dataset_dir: Path, transform=None) -> None:
        """Scan the dataset directory and build the list of HR/LR pairs.

        Args:
            dataset_dir: Path to the dataset directory.
            transform: Optional callable ``(lr_tensor, hr_tensor) -> (lr, hr)``.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
            ValueError: If no pairs are found.
        """
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        manifest_path = self.dataset_dir / "manifest.json"
        self.pairs = []
        if manifest_path.is_file():
            self.pairs = self._pairs_from_manifest(manifest_path)

        if not self.pairs:
            self.pairs = self._pairs_from_directory_scan()

        if not self.pairs:
            raise ValueError(
                f"No HR/LR pairs found in dataset directory: {self.dataset_dir}"
            )

    def _pairs_from_manifest(self, manifest_path: Path) -> list[tuple[Path, Path]]:
        """Read HR/LR pairs from a ``manifest.json`` file.

        Args:
            manifest_path: Path to the manifest file.

        Returns:
            List of ``(hr_path, lr_path)`` tuples.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        pairs: list[tuple[Path, Path]] = []
        for entry in manifest_data.get("pairs", []):
            hr_rel = entry.get("hr") or entry.get("HR")
            lr_rel = entry.get("lr") or entry.get("LR")
            if not hr_rel or not lr_rel:
                continue

            hr_path = self.dataset_dir / hr_rel
            lr_path = self.dataset_dir / lr_rel
            if hr_path.is_file() and lr_path.is_file():
                pairs.append((hr_path, lr_path))

        return pairs

    def _pairs_from_directory_scan(self) -> list[tuple[Path, Path]]:
        """Match HR and LR files by filename within ``HR/`` and ``LR/`` subdirectories.

        Returns:
            List of ``(hr_path, lr_path)`` tuples.
        """
        hr_dir = self.dataset_dir / "HR"
        lr_dir = self.dataset_dir / "LR"

        if not hr_dir.is_dir() or not lr_dir.is_dir():
            raise FileNotFoundError(
                f"No manifest.json found and 'HR/'/'LR/' subdirectories are "
                f"missing under: {self.dataset_dir}"
            )

        pairs: list[tuple[Path, Path]] = []
        for hr_path in sorted(hr_dir.glob("*.png")):
            lr_path = lr_dir / hr_path.name
            if lr_path.is_file():
                pairs.append((hr_path, lr_path))

        return pairs

    def __len__(self) -> int:
        """Return the total number of HR/LR pairs."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple ``(lr_tensor, hr_tensor)``.

        Args:
            index: Pair index.

        Returns:
            ``(lr, hr)`` tensors of shape ``(C, H, W)``.
        """
        hr_path, lr_path = self.pairs[index]

        hr_tensor = _load_image_tensor(hr_path)
        lr_tensor = _load_image_tensor(lr_path)

        if self.transform is not None:
            lr_tensor, hr_tensor = self.transform(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor
