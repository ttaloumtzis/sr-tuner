# transforms.py
"""Image augmentations for training (crop, flip, rotate, etc.)."""

import random
import torch


class RandomCrop:
    """Randomly crop a pair of (LR, HR) images, keeping scale-factor alignment."""

    def __init__(self, patch_size: int, scale: int) -> None:
        """
        Args:
            patch_size: The target output size (spatial height/width) for the LR patch.
            scale: The scaling factor (HR size = LR size * scale).
        """
        self.patch_size = patch_size
        self.scale = scale

    def __call__(
            self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expects tensors with shape [C, H, W]
        _, lr_h, lr_w = lr.shape

        # FIXED: Removed the duplicate 'self.' here
        if lr_h < self.patch_size or lr_w < self.patch_size:
            raise ValueError(
                f"LR image dimensions ({lr_w}x{lr_h}) are smaller than "
                f"requested patch size ({self.patch_size}x{self.patch_size})."
            )

        # 1. Randomly sample the top-left corner in the LR coordinate space
        y_lr = random.randint(0, lr_h - self.patch_size)
        x_lr = random.randint(0, lr_w - self.patch_size)

        # 2. Project the sampled corner coordinates onto the HR space using scale mapping
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale
        hr_patch_size = self.patch_size * self.scale

        # 3. Perform identical spatial crops
        lr_patch = lr[:, y_lr: y_lr + self.patch_size, x_lr: x_lr + self.patch_size]
        hr_patch = hr[:, y_hr: y_hr + hr_patch_size, x_hr: x_hr + hr_patch_size]

        return lr_patch, hr_patch


class RandomFlip:
    """Randomly flip LR and HR images horizontally/vertically."""

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5) -> None:
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Horizontal Flip (dim -1 is Width)
        if random.random() < self.p_horizontal:
            lr = torch.flip(lr, dims=[-1])
            hr = torch.flip(hr, dims=[-1])

        # Vertical Flip (dim -2 is Height)
        if random.random() < self.p_vertical:
            lr = torch.flip(lr, dims=[-2])
            hr = torch.flip(hr, dims=[-2])

        return lr, hr


class RandomRotate:
    """Randomly rotate LR and HR images by multiples of 90 degrees."""

    def __init__(self, angles: list[int] | None = None) -> None:
        # Default options correspond to 0, 90, 180, and 270 degree configurations
        self.angles = angles if angles is not None else [0, 90, 180, 270]

        # Verify choices correspond entirely to multiples of 90 degrees
        if any(angle % 90 != 0 for angle in self.angles):
            raise ValueError("All rotation angle parameters must be clean multiples of 90.")

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angle = random.choice(self.angles)

        # Determine number of 90-degree steps (k parameter in torch.rot90)
        k = (angle // 90) % 4

        if k == 0:
            return lr, hr

        # Perform identical rotations on spatial layout dimensions [-2, -1]
        lr = torch.rot90(lr, k=k, dims=[-2, -1])
        hr = torch.rot90(hr, k=k, dims=[-2, -1])

        return lr, hr


class Compose:
    """Compose a sequence of augmentations."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr