"""Image augmentations for training (crop, flip, rotate, etc.)."""

import random
import torch


class RandomCrop:
    """Randomly crop a pair of (LR, HR) images, keeping scale-factor alignment."""

    def __init__(self, patch_size: int, scale: int) -> None:
        """Configure crop dimensions.

        Args:
            patch_size: Target LR patch size (spatial height/width).
            scale: Super-resolution scale factor (HR size = LR size * scale).
        """
        self.patch_size = patch_size
        self.scale = scale

    def __call__(
            self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a random aligned crop to both LR and HR tensors.

        Args:
            lr: LR tensor ``(C, H, W)``.
            hr: HR tensor ``(C, H*scale, W*scale)``.

        Returns:
            ``(lr_patch, hr_patch)`` with matching spatial alignment.
        """
        _, lr_h, lr_w = lr.shape

        if lr_h < self.patch_size or lr_w < self.patch_size:
            raise ValueError(
                f"LR image dimensions ({lr_w}x{lr_h}) are smaller than "
                f"requested patch size ({self.patch_size}x{self.patch_size})."
            )

        _, hr_h, hr_w = hr.shape
        min_hr = self.patch_size * self.scale
        if hr_h < min_hr or hr_w < min_hr:
            raise ValueError(
                f"HR image dimensions ({hr_w}x{hr_h}) are smaller than "
                f"required for patch size {self.patch_size} at scale {self.scale} "
                f"(need {min_hr}x{min_hr})."
            )

        y_lr = random.randint(0, lr_h - self.patch_size)
        x_lr = random.randint(0, lr_w - self.patch_size)

        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale
        hr_patch_size = self.patch_size * self.scale

        lr_patch = lr[:, y_lr: y_lr + self.patch_size, x_lr: x_lr + self.patch_size]
        hr_patch = hr[:, y_hr: y_hr + hr_patch_size, x_hr: x_hr + hr_patch_size]

        return lr_patch, hr_patch


class RandomFlip:
    """Randomly flip LR and HR images horizontally/vertically."""

    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5) -> None:
        """Configure flip probabilities.

        Args:
            p_horizontal: Probability of a horizontal flip.
            p_vertical: Probability of a vertical flip.
        """
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply flips independently to the pair.

        Args:
            lr: LR tensor ``(C, H, W)``.
            hr: HR tensor ``(C, H*scale, W*scale)``.

        Returns:
            Flipped ``(lr, hr)`` pair.
        """
        if random.random() < self.p_horizontal:
            lr = torch.flip(lr, dims=[-1])
            hr = torch.flip(hr, dims=[-1])

        if random.random() < self.p_vertical:
            lr = torch.flip(lr, dims=[-2])
            hr = torch.flip(hr, dims=[-2])

        return lr, hr


class RandomRotate:
    """Randomly rotate LR and HR images by multiples of 90 degrees."""

    def __init__(self, angles: list[int] | None = None) -> None:
        """Configure allowed rotation angles.

        Args:
            angles: List of rotation angles in degrees (must be multiples of 90).
                    Defaults to ``[0, 90, 180, 270]``.

        Raises:
            ValueError: If any angle is not a multiple of 90.
        """
        self.angles = angles if angles is not None else [0, 90, 180, 270]

        if any(angle % 90 != 0 for angle in self.angles):
            raise ValueError("All rotation angle parameters must be clean multiples of 90.")

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a random rotation to both tensors.

        Args:
            lr: LR tensor ``(C, H, W)``.
            hr: HR tensor ``(C, H*scale, W*scale)``.

        Returns:
            Rotated ``(lr, hr)`` pair.
        """
        angle = random.choice(self.angles)

        k = (angle // 90) % 4

        if k == 0:
            return lr, hr

        lr = torch.rot90(lr, k=k, dims=[-2, -1])
        hr = torch.rot90(hr, k=k, dims=[-2, -1])

        return lr, hr


class CenterCrop:
    """Deterministically center-crop a pair of (LR, HR) images."""

    def __init__(self, patch_size: int, scale: int) -> None:
        """Configure crop dimensions.

        Args:
            patch_size: Target LR patch size (spatial height/width).
            scale: Super-resolution scale factor.
        """
        self.patch_size = patch_size
        self.scale = scale

    def __call__(
            self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a deterministic center crop to both tensors.

        Args:
            lr: LR tensor ``(C, H, W)``.
            hr: HR tensor ``(C, H*scale, W*scale)``.

        Returns:
            Center-cropped ``(lr, hr)`` pair.
        """
        _, lr_h, lr_w = lr.shape
        if lr_h < self.patch_size or lr_w < self.patch_size:
            raise ValueError(
                f"LR image dimensions ({lr_w}x{lr_h}) are smaller than "
                f"requested patch size ({self.patch_size}x{self.patch_size})."
            )

        _, hr_h, hr_w = hr.shape
        min_hr = self.patch_size * self.scale
        if hr_h < min_hr or hr_w < min_hr:
            raise ValueError(
                f"HR image dimensions ({hr_w}x{hr_h}) are smaller than "
                f"required for patch size {self.patch_size} at scale {self.scale} "
                f"(need {min_hr}x{min_hr})."
            )

        y_lr = (lr_h - self.patch_size) // 2
        x_lr = (lr_w - self.patch_size) // 2
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale
        hr_patch_size = self.patch_size * self.scale
        lr_patch = lr[:, y_lr: y_lr + self.patch_size, x_lr: x_lr + self.patch_size]
        hr_patch = hr[:, y_hr: y_hr + hr_patch_size, x_hr: x_hr + hr_patch_size]
        return lr_patch, hr_patch


class Compose:
    """Compose a sequence of augmentations."""

    def __init__(self, transforms: list) -> None:
        """Store the transform chain.

        Args:
            transforms: Sequence of callables, each ``(lr, hr) -> (lr, hr)``.
        """
        self.transforms = transforms

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all transforms in sequence.

        Args:
            lr: LR tensor ``(C, H, W)``.
            hr: HR tensor ``(C, H*scale, W*scale)``.

        Returns:
            Transformed ``(lr, hr)`` pair.
        """
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr
