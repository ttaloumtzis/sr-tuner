from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator

# ── Architecture mapping ────────────────────────────────────────────────

FRONTEND_ARCH_MAP: dict[str, str] = {
    "Real-ESRGAN": "rrdb_esrgan",
    "SwinIR": "swinir",
}
ENGINE_ARCH_MAP: dict[str, str] = {v: k for k, v in FRONTEND_ARCH_MAP.items()}

# ── Workspace ───────────────────────────────────────────────────────────

class WorkspaceInfo(BaseModel):
    path: str
    exists: bool
    models: list[dict[str, Any]] = []

class WorkspaceInitParams(BaseModel):
    path: str | None = None

# ── Models ──────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    display_name: str | None = None
    description: str = ""

class ModelInstance(BaseModel):
    name: str
    path: str
    architecture: str | None = None
    scale: int | None = None
    checkpoints: list[str] = []
    latest_version: str | None = None
    config: dict = {}

class CreateInstanceParams(BaseModel):
    name: str
    architecture: str
    config: dict

class ModelVersion(BaseModel):
    tag: str
    path: str
    metadata: dict | None = None

class ExportParams(BaseModel):
    instance: str
    format: str = Field(pattern="^(pth|onnx|torchscript|safetensors)$")
    output: str | None = None

# ── Datasets ────────────────────────────────────────────────────────────

class DatasetBuildParams(BaseModel):
    input: str
    out: str | None = None
    config: str | None = None
    degradations: str | None = None
    config_overrides: dict | None = None

class DatasetValidateParams(BaseModel):
    path: str

class DatasetHealthParams(BaseModel):
    path: str
    yes: bool = False

class DatasetMergeParams(BaseModel):
    input: str
    out: str | None = None
    scale: int | None = None
    name: str | None = None
    keep_sources: bool = False
    input_datasets: list[str] | None = None

class DatasetPruneParams(BaseModel):
    path: str
    black_frames: list[str]

# ── Training ────────────────────────────────────────────────────────────

class TrainParams(BaseModel):
    model_name: str
    instance: str
    dataset: str
    config: str | None = None
    resume: str | None = None
    device: str = "auto"
    batch_size: int | None = None
    learning_rate: float | None = None
    max_epochs: int | None = None
    patch_size: int | None = None
    fp16: bool | None = None
    seed: int | None = None
    weight_decay: float | None = None
    betas: list[float] | None = None
    num_workers: int | None = None
    save_per_epoch: int | None = None
    validation_enabled: bool | None = None
    validation_split: float | None = None
    validation_dataset: str | None = None
    metrics_frequency: int | None = None
    perceptual_weight: float | None = None
    warmup_steps: int | None = None
    write_metrics_file: bool = True
    losses: dict[str, Any] | None = None

    @validator("losses")
    def validate_losses(cls, v):
        if v is None:
            return v
        valid_types = {"l1", "l2", "vgg", "edge", "style", "fft", "ssim", "lpips"}
        has_pixel = False
        for name, cfg in v.items():
            if not isinstance(cfg, dict) or "type" not in cfg:
                raise ValueError(f"Loss '{name}' must be a dict with a 'type' field")
            lt = cfg["type"]
            if lt not in valid_types:
                raise ValueError(
                    f"Loss '{name}': unknown type '{lt}'. Valid: {sorted(valid_types)}"
                )
            if cfg.get("weight", 1.0) < 0:
                raise ValueError(f"Loss '{name}': weight must be >= 0")
            if lt in ("l1", "l2"):
                has_pixel = True
        if not has_pixel:
            raise ValueError("At least one pixel loss (l1 or l2) is required")
        return v

    def to_overrides(self) -> dict:
        d: dict[str, Any] = {}
        for key in ("batch_size", "learning_rate", "max_epochs", "patch_size",
                     "seed", "weight_decay", "num_workers", "save_per_epoch",
                     "metrics_frequency", "warmup_steps"):
            val = getattr(self, key, None)
            if val is not None:
                d[key] = val
        if self.betas is not None:
            d["betas"] = self.betas
        if self.fp16 is not None:
            d["dtype"] = "bf16" if self.fp16 else "float32"
        if self.device and self.device != "auto":
            d["device"] = self.device
        if any(v is not None for v in (self.validation_enabled, self.validation_split, self.validation_dataset)):
            d.setdefault("validation", {})
            for k, v in (("enabled", self.validation_enabled),
                          ("split", self.validation_split),
                          ("dataset", self.validation_dataset)):
                if v is not None:
                    d["validation"][k] = v
        if self.losses is not None:
            d["losses"] = self.losses
        elif self.perceptual_weight is not None:
            d["losses"] = {
                "pixel": {"type": "l1", "weight": 1.0},
                "perceptual": {"type": "vgg", "weight": self.perceptual_weight},
            }
        return d

# ── Inference ───────────────────────────────────────────────────────────

class InferParams(BaseModel):
    model: str | None = None
    instance: str | None = None
    version: str | None = None
    input: str
    output: str
    tile: int = 512
    overlap: int = 64
    device: str = "auto"

# ── Jobs ────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: dict | None = None
    config: dict | None = None

class JobList(BaseModel):
    jobs: list[JobStatus]

class JobAccepted(BaseModel):
    job_id: str
    status: str = "accepted"

# ── Env ─────────────────────────────────────────────────────────────────

class EnvInfo(BaseModel):
    torch_version: str
    device: str
    cuda_available: bool
    rocm: bool
    bf16_supported: bool
    flash_attn: bool
    device_name: str | None = None
    vram_total_mb: int | None = None