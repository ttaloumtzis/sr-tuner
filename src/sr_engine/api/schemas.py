from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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

    def to_overrides(self) -> dict:
        d: dict[str, Any] = {}
        if self.batch_size is not None:
            d["train.batch_size"] = self.batch_size
        if self.learning_rate is not None:
            d["train.learning_rate"] = self.learning_rate
        if self.max_epochs is not None:
            d["train.max_epochs"] = self.max_epochs
        if self.patch_size is not None:
            d["train.patch_size"] = self.patch_size
        if self.fp16 is not None:
            d["train.dtype"] = "bf16" if self.fp16 else "float32"
        if self.device and self.device != "auto":
            d["train.device"] = self.device
        if self.seed is not None:
            d["train.seed"] = self.seed
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