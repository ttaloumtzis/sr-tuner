"""Training engine — orchestrates model training, checkpointing, logging."""

from pathlib import Path


class Trainer:
    """Trainer for super-resolution models.

    Args:
        model_cfg: Model configuration dict.
        train_cfg: Training configuration dict.
        dataset_dir: Path to the paired HR/LR dataset.
        resume_from: Optional checkpoint path to resume from.
        device: Torch device string.
    """

    def __init__(
        self,
        model_cfg: dict,
        train_cfg: dict,
        dataset_dir: Path,
        resume_from: Path | None = None,
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError("TODO: implement Trainer.__init__")

    def train(self) -> None:
        """Run the full training loop."""
        raise NotImplementedError("TODO: implement training loop")
