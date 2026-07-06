"""Torch Dataset classes for paired HR/LR image folders."""

from pathlib import Path
from torch.utils.data import Dataset


class PairedImageFolderDataset(Dataset):
    """Reads paired HR/LR images from a dataset directory.

    Expects the directory structure::

        <dataset_dir>/
            HR/<filename>.png
            LR/<filename>.png
    """

    def __init__(self, dataset_dir: Path, transform=None) -> None:
        raise NotImplementedError(
            "TODO: implement __init__ — scan HR/ and LR/ for paired files"
        )

    def __len__(self) -> int:
        raise NotImplementedError("TODO: implement __len__")

    def __getitem__(self, index: int) -> tuple:
        """Return a tuple ``(lr_tensor, hr_tensor)``."""
        raise NotImplementedError(
            "TODO: implement __getitem__ — load and pair HR/LR images"
        )
