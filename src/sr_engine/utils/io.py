"""Image and file I/O utilities."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk.

    Returns a numpy array in ``(H, W, C)`` RGB uint8 format.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image (missing or corrupt): {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(image: np.ndarray, path: Path) -> None:
    """Write a numpy array (*H*, *W*, *C*) to disk as a PNG image.

    Expects *image* in RGB uint8 format (matching ``read_image``'s output);
    converts to BGR internally since that's what cv2 writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        # Grayscale — nothing to convert, cv2 handles single-channel fine.
        out_img = image
    elif image.ndim == 3 and image.shape[2] == 3:
        out_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(
            f"Expected a (H, W) grayscale or (H, W, 3) RGB array, got shape {image.shape}"
        )

    success = cv2.imwrite(str(path), out_img)
    if not success:
        raise IOError(f"Failed to write image to: {path}")


def ensure_dir(path: Path) -> Path:
    """Create *path* if it doesn't exist, then return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path