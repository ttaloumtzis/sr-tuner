"""Training engine — orchestrates model training, checkpointing, logging."""

import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from sr_engine.utils.progress import ProgressReporter
from sr_engine.device.backend import autocast_dtype

from sr_engine.data.datasets import PairedImageFolderDataset
from sr_engine.data.transforms import CenterCrop, Compose, RandomCrop, RandomFlip, RandomRotate
from sr_engine.engine.metrics import psnr, ssim
from sr_engine.engine.metrics_stream import MetricsStream
from sr_engine.models.checkpoint import load_checkpoint, save_checkpoint
from sr_engine.models.losses import L1Loss, PerceptualLoss
from sr_engine.models.registry import build_model
from sr_engine.utils.logging import get_logger

log = get_logger(__name__)


class _TransformSubset(Dataset):
    """Subset that applies transforms on-the-fly after retrieval from the base dataset."""

    def __init__(self, dataset: Dataset, indices: list[int], transform=None) -> None:
        """Wrap a dataset with index subsetting and optional transforms.

        Args:
            dataset: Base dataset to sample from.
            indices: List of valid indices into the base dataset.
            transform: Optional callable ``(lr, hr) -> (lr, hr)``.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Retrieve a sample by subset index, applying transforms.

        Args:
            idx: Index into the subset.

        Returns:
            ``(lr, hr)`` tensor pair.
        """
        lr, hr = self.dataset[self.indices[idx]]
        if self.transform:
            lr, hr = self.transform(lr, hr)
        return lr, hr


class TrainerCallback:
    """Base class for training lifecycle callbacks.

    Subclass and override the events you care about. Every method has a
    no-op default so you only implement what you need.
    """

    def on_phase(self, phase: str, **data: Any) -> None:
        """Called at phase transitions (training, saving, complete)."""

    def on_step(self, epoch: int, batch: int, total_batches: int, **losses: float) -> None:
        """Called every ``metrics_frequency`` batches with current loss values."""

    def on_validate(self, epoch: int, **metrics: float) -> None:
        """Called after each validation pass with PSNR, SSIM, etc."""

    def on_done(self, elapsed_seconds: float) -> None:
        """Called once when training finishes."""


class _MetricsStreamCallback(TrainerCallback):
    """Adapter that bridges a ``MetricsStream`` into the callback system."""

    def __init__(self, stream: MetricsStream) -> None:
        """Wrap a MetricsStream for callback-driven writes.

        Args:
            stream: MetricsStream instance to write to.
        """
        self._stream = stream

    def on_phase(self, phase: str, **data: Any) -> None:
        """Write a phase event to the metrics stream."""
        self._stream.write({"type": "phase", "phase": phase, **data})

    def on_step(self, epoch: int, batch: int, total_batches: int, **losses: float) -> None:
        """Write a step event to the metrics stream."""
        self._stream.write({
            "type": "step", "epoch": epoch, "batch": batch,
            "total_batches": total_batches, **losses,
        })

    def on_validate(self, epoch: int, **metrics: float) -> None:
        """Write a validation event to the metrics stream."""
        self._stream.write({"type": "validate", "epoch": epoch, **metrics})

    def on_done(self, elapsed_seconds: float) -> None:
        """Write a done event and close the metrics stream."""
        self._stream.write({"type": "done", "elapsed_seconds": elapsed_seconds})
        self._stream.close()


class Trainer:
    """Trainer for super-resolution models using epoch-based logic."""

    def __init__(
        self,
        model_cfg: dict,
        train_cfg: dict,
        dataset_dir: Path,
        resume_from: Path | None = None,
        device: str = "cuda",
        validation_enabled: bool = True,
        validation_split: float = 0.1,
        metrics_stream: MetricsStream | None = None,
        metrics_frequency: int = 1,
        progress_reporter: ProgressReporter | None = None,
        callbacks: list[TrainerCallback] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Configure model, optimizer, dataloaders, and schedule.

        Args:
            model_cfg: Model architecture configuration dict.
            train_cfg: Training hyperparameter dict.
            dataset_dir: Path to the paired HR/LR dataset directory.
            resume_from: Optional checkpoint path to resume from.
            device: Torch device string.
            validation_enabled: Whether to hold out a validation split.
            validation_split: Fraction of data used for validation.
            metrics_stream: Optional stream for writing metrics JSONL.
            metrics_frequency: Log metrics every N batches.
            progress_reporter: Progress reporter instance.
            callbacks: Additional lifecycle callbacks.
            cancel_check: Callable returning True when cancellation is requested.
        """
        self.model_cfg = model_cfg
        self.device = torch.device(device)
        self.train_cfg = train_cfg
        self.metrics_stream = metrics_stream
        self.metrics_frequency = metrics_frequency
        self._progress = progress_reporter or ProgressReporter()
        self._callbacks: list[TrainerCallback] = callbacks or []
        self._cancel_check = cancel_check or (lambda: False)
        if metrics_stream is not None:
            self._callbacks.append(_MetricsStreamCallback(metrics_stream))

        self.max_epochs = int(train_cfg.get("max_epochs", 100))
        self.save_per_epoch = int(train_cfg.get("save_per_epoch", 5))
        self.current_epoch = 0

        self.learning_rate = float(train_cfg.get("learning_rate", 1e-4))
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        seed = int(train_cfg.get("seed", 42))
        torch.manual_seed(seed)

        self.model = build_model(model_cfg["name"], model_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            betas=list(train_cfg.get("betas", [0.9, 0.99])),
        )

        dtype_str = str(train_cfg.get("dtype", "float32")).lower()
        if dtype_str == "bf16":
            if self.device.type == "cpu":
                log.info("BF16 requested but device is CPU — disabling AMP")
                self.amp_dtype = None
            elif not torch.cuda.is_bf16_supported():
                fallback = autocast_dtype()
                log.warning("bf16 requested but not supported — falling back to %s", fallback)
                self.amp_dtype = fallback
            else:
                self.amp_dtype = torch.bfloat16
        elif dtype_str == "float16":
            self.amp_dtype = torch.float16 if self.device.type != "cpu" else None
        else:
            self.amp_dtype = None

        self.amp_enabled = self.amp_dtype is not None
        self.grad_scaler = torch.amp.GradScaler() if self.amp_dtype == torch.float16 else None

        self.pixel_loss = L1Loss()
        loss_cfg = train_cfg.get("losses", {})
        self.perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
        self.perceptual_loss = (
            PerceptualLoss(loss_cfg.get("perceptual_layers", ["relu5_4"])).to(self.device)
            if self.perceptual_weight > 0 else None
        )

        transform = Compose([
            RandomCrop(
                patch_size=int(train_cfg.get("patch_size", 128)),
                scale=int(model_cfg.get("scale", 4)),
            ),
            RandomFlip(),
            RandomRotate(),
        ])

        batch_size = int(train_cfg.get("batch_size", 32))
        num_workers = int(train_cfg.get("num_workers", 4))
        pin = self.device.type == "cuda"

        full_dataset = PairedImageFolderDataset(dataset_dir, transform=None)

        if validation_enabled and validation_split > 0 and len(full_dataset) > 1:
            val_size = max(1, int(len(full_dataset) * validation_split))
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(seed)
            train_idx, val_idx = random_split(
                range(len(full_dataset)), [train_size, val_size], generator=generator,
            )
            val_transform = CenterCrop(
                patch_size=int(train_cfg.get("patch_size", 128)),
                scale=int(model_cfg.get("scale", 4)),
            )
            self.train_dataset = _TransformSubset(full_dataset, train_idx.indices, transform)
            self.val_dataset = _TransformSubset(full_dataset, val_idx.indices, val_transform)
            log.info(
                "Split dataset: %d train / %d val pairs (%.0f%% val)",
                train_size, val_size, validation_split * 100,
            )
        else:
            self.train_dataset = _TransformSubset(
                full_dataset, list(range(len(full_dataset))), transform,
            )
            self.val_dataset = None
            log.info("Validation disabled — using all %d pairs for training", len(full_dataset))

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, pin_memory=pin,
        )

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin,
            )
        else:
            self.val_dataloader = None

        self._build_scheduler()

        if resume_from:
            self._resume(resume_from)

    def _build_scheduler(self) -> None:
        """Build the LR scheduler with optional warmup."""
        min_lr = float(self.train_cfg.get("min_lr", 1e-7))
        warmup_steps = int(self.train_cfg.get("warmup_steps", 0))
        steps_per_epoch = len(self.train_dataloader)
        total_steps = self.max_epochs * steps_per_epoch

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-6, end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_steps - warmup_steps),
                eta_min=min_lr,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup, cosine], milestones=[warmup_steps],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=min_lr,
            )

    def _resume(self, checkpoint_path: Path) -> None:
        """Load model and optimizer state from a checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
        """
        ckpt = load_checkpoint(checkpoint_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            log.warning("No optimizer state in checkpoint — starting fresh optimizer")
        self.current_epoch = int(ckpt.get("epoch", ckpt.get("step", 0)))
        log.info("Resumed from epoch %d", self.current_epoch)

    def _emit(self, event: str, **payload: Any) -> None:
        """Dispatch an event to all registered callbacks.

        Args:
            event: Event name (e.g. ``"step"``, ``"phase"``).
            payload: Keyword arguments forwarded to the callback method.
        """
        for cb in self._callbacks:
            getattr(cb, f"on_{event}")(**payload)

    def _save(self, epoch: int) -> None:
        """Save a model checkpoint to disk.

        Args:
            epoch: Current epoch number (used in the filename).
        """
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        ckpt_config = {
            **self.model_cfg,
            "training_dtype": str(self.amp_dtype),
        }
        save_checkpoint(
            path=path,
            state_dict=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=epoch,
            config=ckpt_config,
            backend_info={"device": str(self.device)},
        )

    def _run_step(self, lr: torch.Tensor, hr: torch.Tensor) -> dict[str, float]:
        """Execute one training step: forward, loss, backward, optimiser step.

        Args:
            lr: Low-resolution input batch ``(B, C, H, W)``.
            hr: High-resolution target batch ``(B, C, H*scale, W*scale)``.

        Returns:
            Dict of loss components and current learning rate.
        """
        lr, hr = lr.to(self.device, non_blocking=True), hr.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            pred = self.model(lr)
            loss = self.pixel_loss(pred, hr)
            comp = {"pixel": loss.item()}
            if self.perceptual_loss:
                ploss = self.perceptual_loss(pred, hr)
                loss += self.perceptual_weight * ploss
                comp["perceptual"] = ploss.item()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        comp["total"] = loss.item()
        comp["lr"] = self.optimizer.param_groups[0]["lr"]
        return comp

    def _validate(self) -> dict[str, float]:
        """Run validation: compute average PSNR and SSIM over the validation set.

        Returns:
            Dict with ``"psnr"`` and ``"ssim"`` keys.
        """
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num = 0
        with torch.no_grad():
            for lr, hr in self.val_dataloader:
                lr, hr = lr.to(self.device), hr.to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    pred = self.model(lr)
                pred_clamped = pred.clamp(0.0, 1.0)
                total_psnr += psnr(pred_clamped, hr).item()
                total_ssim += ssim(pred_clamped, hr).item()
                num += 1
        self.model.train()
        avg_psnr = total_psnr / num if num > 0 else 0.0
        avg_ssim = total_ssim / num if num > 0 else 0.0
        return {"psnr": avg_psnr, "ssim": avg_ssim}

    def train(self) -> None:
        """Run the training loop per epoch."""
        self.model.train()
        start_time = time.time()

        self._emit("phase", phase="training", max_epochs=self.max_epochs)

        for epoch in range(self.current_epoch, self.max_epochs):
            if self._cancel_check():
                log.warning("Cancellation requested — saving checkpoint at epoch %d", epoch + 1)
                self._save(epoch + 1)
                self._emit("phase", phase="cancelled", epoch=epoch + 1)
                import sys
                sys.exit(130)

            self._progress.start(total=len(self.train_dataloader),
                                 desc=f"Epoch {epoch+1}/{self.max_epochs}")

            for batch_idx, (lr, hr) in enumerate(self.train_dataloader):
                losses = self._run_step(lr, hr)
                self._progress.update(1)
                self._progress.set_postfix(**losses)

                if batch_idx % self.metrics_frequency == 0:
                    self._emit("step", epoch=epoch + 1, batch=batch_idx + 1,
                               total_batches=len(self.train_dataloader), **losses)

            self._progress.finish()

            should_save = (
                (epoch + 1) % self.save_per_epoch == 0
                or (epoch + 1) == self.max_epochs
            )
            if should_save:
                val_metrics = self._validate() if self.val_dataloader is not None else {}
                if val_metrics:
                    log.info(
                        "Epoch %d/%d — PSNR: %.2f, SSIM: %.4f",
                        epoch + 1, self.max_epochs,
                        val_metrics["psnr"], val_metrics["ssim"],
                    )
                    self._emit("validate", epoch=epoch + 1, **val_metrics)
                self._emit("phase", phase="saving", epoch=epoch + 1)
                self._save(epoch + 1)

        elapsed = time.time() - start_time
        log.info("Training complete in %.1fs", elapsed)
        self._emit("phase", phase="complete")
        self._emit("done", elapsed_seconds=round(elapsed, 1))
