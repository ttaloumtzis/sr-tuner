"""Image and file I/O utilities."""

from pathlib import Path
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk.

    Returns a numpy array in ``(H, W, C)`` RGB uint8 format.
    """
    raise NotImplementedError("TODO: implement image reading")


def write_image(image: np.ndarray, path: Path) -> None:
    """Write a numpy array (*H*, *W*, *C*) to disk as a PNG image."""
    raise NotImplementedError("TODO: implement image writing")


def ensure_dir(path: Path) -> Path:
    """Create *path* if it doesn't exist, then return it."""
    raise NotImplementedError("TODO: implement ensure_dir")
