"""Training engine — orchestrates model training, checkpointing, logging."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr_engine.data.datasets import PairedImageFolderDataset
from sr_engine.data.transforms import Compose, RandomCrop, RandomFlip, RandomRotate
from sr_engine.models.checkpoint import load_checkpoint, save_checkpoint
from sr_engine.models.losses import L1Loss, PerceptualLoss
from sr_engine.models.registry import build_model

class Trainer:
    """Trainer for super-resolution models using epoch-based logic."""

    def __init__(
        self,
        model_cfg: dict,
        train_cfg: dict,
        dataset_dir: Path,
        resume_from: Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_cfg = model_cfg
        self.device = torch.device(device)

        # Epoch settings
        self.max_epochs = int(train_cfg.get("max_epochs", 100))
        self.save_per_epoch = int(train_cfg.get("save_per_epoch", 5))
        self.current_epoch = 0

        # Optimization & Checkpointing
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-4))
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model, Optimizer, Losses
        self.model = build_model(model_cfg["name"], model_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.pixel_loss = L1Loss()

        loss_cfg = train_cfg.get("losses", {})
        self.perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
        self.perceptual_loss = PerceptualLoss(loss_cfg.get("perceptual_layers")).to(self.device) if self.perceptual_weight > 0 else None

        # Data
        transform = Compose([RandomCrop(patch_size=int(train_cfg.get("patch_size", 48)), scale=int(model_cfg.get("scale", 4))), RandomFlip(), RandomRotate()])
        dataset = PairedImageFolderDataset(dataset_dir, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=int(train_cfg.get("batch_size", 16)), shuffle=True,
                                     num_workers=int(train_cfg.get("num_workers", 4)), drop_last=True, pin_memory=(self.device.type == "cuda"))

        if resume_from:
            self._resume(resume_from)

    def _resume(self, checkpoint_path: Path) -> None:
        ckpt = load_checkpoint(checkpoint_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        self.current_epoch = int(ckpt.get("epoch", 0))
        print(f"[Trainer] Resumed from epoch {self.current_epoch}")

    def _save(self, epoch: int) -> None:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        # Using explicit keywords prevents argument misalignment
        save_checkpoint(
            path=path,
            state_dict=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=epoch,
            config=self.model_cfg,
            backend_info={"device": str(self.device)}
        )

    def _run_step(self, lr: torch.Tensor, hr: torch.Tensor) -> dict[str, float]:
        lr, hr = lr.to(self.device, non_blocking=True), hr.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(lr)
        loss = self.pixel_loss(pred, hr)
        comp = {"pixel": loss.item()}
        if self.perceptual_loss:
            ploss = self.perceptual_loss(pred, hr)
            loss += self.perceptual_weight * ploss
            comp["perceptual"] = ploss.item()
        loss.backward()
        self.optimizer.step()
        comp["total"] = loss.item()
        return comp

    def train(self) -> None:
        """Run the training loop per epoch."""
        self.model.train()
        for epoch in range(self.current_epoch, self.max_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            for lr, hr in pbar:
                losses = self._run_step(lr, hr)
                pbar.set_postfix(losses)

            if (epoch + 1) % self.save_per_epoch == 0 or (epoch + 1) == self.max_epochs:
                self._save(epoch + 1)
        print(f"[Trainer] Training complete.")