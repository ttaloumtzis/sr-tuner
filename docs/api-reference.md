# API Reference

## Overview

sr-engine provides a FastAPI-based HTTP/REST API server that powers the desktop GUI and enables third-party integrations. The server runs on **port 8765** by default and is started via:

```bash
srengine serve start [--host 0.0.0.0] [--port 8765] [--log-level info]
```

The API is organized into route groups (`/api/workspace`, `/api/models`, `/api/train`, `/api/infer`, `/api/datasets`, `/api/jobs`, `/api/env`) with a top-level health check and a Server-Sent Events (SSE) endpoint for real-time job progress.

---

## Quick Reference

| Method | Path | Description | Sync/Async |
|--------|------|-------------|------------|
| GET | `/api/health` | Health check | Sync |
| GET | `/api/events` | SSE event stream | Stream |
| **Workspace** |||
| GET | `/api/workspace` | Workspace info | Sync |
| POST | `/api/workspace/init` | Initialize workspace | Sync |
| GET | `/api/workspace/check` | Workspace health check | Sync |
| **Models** |||
| GET | `/api/models` | List available architectures | Sync |
| GET | `/api/models/instances` | List model instances | Sync |
| GET | `/api/models/instances/{name}` | Instance details | Sync |
| POST | `/api/models/instances` | Create model instance | Sync |
| POST | `/api/models/instances/{name}/export` | Export checkpoint | Sync |
| GET | `/api/models/instances/{name}/versions` | List versions | Sync |
| DELETE | `/api/models/instances/{name}` | Delete instance | Sync |
| **Training** |||
| POST | `/api/train/start` | Start training job | Async |
| POST | `/api/train/validate-dataset` | Validate training dataset | Sync |
| **Inference** |||
| POST | `/api/infer/start` | Start inference job | Async |
| **Datasets** |||
| GET | `/api/datasets` | List datasets | Sync |
| POST | `/api/datasets/build` | Build dataset | Async |
| POST | `/api/datasets/validate` | Validate dataset (sync) | Sync |
| POST | `/api/datasets/validate-async` | Validate dataset (async) | Async |
| POST | `/api/datasets/health` | Dataset health check | Async |
| POST | `/api/datasets/merge` | Merge datasets | Async |
| POST | `/api/datasets/prune` | Prune black frames | Async |
| **Jobs** |||
| GET | `/api/jobs` | List all jobs | Sync |
| GET | `/api/jobs/{job_id}` | Get job status | Sync |
| POST | `/api/jobs/{job_id}/cancel` | Cancel job | Sync |
| **Environment** |||
| GET | `/api/env` | Environment diagnostics | Sync |

---

## Server Startup

```bash
# Start on default host/port (127.0.0.1:8765)
srengine serve start

# Expose to network
srengine serve start --host 0.0.0.0 --port 8080

# With debug logging
srengine serve start --log-level debug
```

The server can also be started directly via uvicorn:

```bash
uv run uvicorn sr_engine.api.app:app --host 127.0.0.1 --port 8765
```

### Startup Behaviour

1. CORS middleware is configured to allow all origins (for development flexibility)
2. Workspace is **not** auto-initialized — clients must call `POST /api/workspace/init` first
3. All background jobs run as daemon threads in the same process
4. SSE events are in-memory only (not persisted to disk)

---

## Base URL

All paths are relative to `http://<host>:<port>`. Example: `http://localhost:8765/api/health`

---

## Common Patterns

### Architecture Name Mapping

The frontend uses display names (`"Real-ESRGAN"`, `"SwinIR"`) which are mapped to engine registry keys (`"rrdb_esrgan"`, `"swinir"`) via:

```python
FRONTEND_ARCH_MAP = {
    "Real-ESRGAN": "rrdb_esrgan",
    "SwinIR": "swinir",
}
```

### Async Job Lifecycle

Operations that may take significant time (training, inference, dataset building) are handled asynchronously:

1. Client sends request → server validates → spawns daemon thread → returns `{job_id, status: "accepted"}`
2. Client subscribes to `GET /api/events?job_id=<id>` for SSE progress
3. Worker reports progress via SSE events
4. Worker completes → job status becomes `"completed"` or `"failed"`
5. Client can poll `GET /api/jobs/<job_id>` for status

### Synchronous vs Async Endpoints

| Operation | Sync Endpoint | Async Endpoint | Notes |
|-----------|---------------|----------------|-------|
| Dataset validate | `POST /api/datasets/validate` | `POST /api/datasets/validate-async` | Sync for small datasets, async for large |
| Dataset build | — | `POST /api/datasets/build` | Always async (video extraction + degradation) |
| Dataset merge | — | `POST /api/datasets/merge` | Always async |
| Dataset health | — | `POST /api/datasets/health` | Always async |
| Dataset prune | — | `POST /api/datasets/prune` | Always async |
| Training | — | `POST /api/train/start` | Always async (hours) |
| Inference | — | `POST /api/infer/start` | Always async (minutes) |

### Error Responses

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created (model instance) |
| 400 | Bad request (missing/invalid parameters) |
| 403 | Forbidden (path outside workspace) |
| 404 | Not found (instance, dataset, job) |
| 409 | Conflict (instance already exists) |
| 503 | Service unavailable (workspace not initialized) |

Error bodies follow FastAPI conventions:

```json
{"detail": "Model instance 'my_model' not found"}
```

---

## Endpoints

### GET /api/health

Health check returning server status and workspace path.

**Response:**
```json
{
  "status": "ok",
  "workspace": "/home/user/my_workspace"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if server is running |
| `workspace` | string\|null | Path to initialized workspace, or `null` |

---

### GET /api/events

Server-Sent Events stream for real-time job progress.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | Yes | Job ID returned from an async operation |

**Response:** `text/event-stream`

**Example:**
```bash
curl -N http://localhost:8765/api/events?job_id=dataset_build_1728000000_abc12345
```

**Event Format:**
```
data: {"type": "progress_start", "total": 100, "desc": "Extracting frames"}

data: {"type": "progress_update", "n": 1}

data: {"type": "done", "elapsed_seconds": 42.5}
```

**Event Types:**

| Event Type | Emitted By | Fields |
|------------|------------|--------|
| `progress_start` | Any worker | `total` (int\|null), `desc` (string) |
| `progress_update` | Any worker | `n` (int) |
| `progress_end` | Any worker | — |
| `postfix` | Any worker | `desc` (string) or key-value pairs |
| `phase` | Training | `phase` (string: `"training"`, `"validation"`, `"saving"`, `"complete"`) |
| `step` | Training | `epoch`, `batch`, `total_batches`, `loss_total`, `loss_pixel`, `loss_perceptual`, `lr` |
| `validate` | Training | `epoch`, `psnr`, `ssim`, `frames` (optional dict of validation images) |
| `done` | Any worker | `elapsed_seconds` (float), plus job-specific fields |
| `error` | Any worker | `code` (string), `message` (string) |
| `hardware` | HardwareMonitor | `gpu_util`, `vram_used`, `vram_total`, `cpu_percent`, `ram_percent`, `temperature` (per 3s interval) |

The stream ends when the worker sends a `None` sentinel.

---

### GET /api/workspace

Returns information about the current workspace.

**Requires:** Workspace initialized

**Response:**
```json
{
  "path": "/home/user/my_workspace",
  "exists": true,
  "models": [{"name": "my_model", "path": "/home/user/my_workspace/models/my_model"}]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Workspace directory path |
| `exists` | bool | Whether the directory exists |
| `models` | array | List of model instances with `name` and `path` |

---

### POST /api/workspace/init

Initialize a workspace at the given path (or auto-detect from CWD).

**Request Body:**
```json
{
  "path": "/home/user/my_workspace"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string\|null | No | Auto-detect | Path to initialize as workspace |

**Response:**
```json
{
  "path": "/home/user/my_workspace",
  "exists": true
}
```

Creates the workspace directory structure: `.sr_workspace`, `datasets/`, `models/`, `experiments/`, `configs/`.

---

### GET /api/workspace/check

Validate workspace structure health.

**Response:**
```json
{
  "healthy": true,
  "issues": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `healthy` | bool | True if no issues found |
| `issues` | array | List of issue descriptions (empty if healthy) |

---

### GET /api/models

List available model architectures from the registry.

**Response:**
```json
[
  {"name": "rrdb_esrgan", "display_name": "rrdb_esrgan", "description": ""},
  {"name": "swinir", "display_name": "swinir", "description": ""}
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Engine registry key (e.g. `"rrdb_esrgan"`, `"swinir"`) |
| `display_name` | string\|null | Human-readable name |
| `description` | string | Description text |

---

### GET /api/models/instances

List all model instances in the workspace.

**Response:**
```json
[
  {
    "name": "my_model",
    "path": "/home/user/workspace/models/my_model",
    "architecture": "rrdb_esrgan",
    "scale": 4,
    "checkpoints": [],
    "latest_version": "v3",
    "config": {"name": "rrdb_esrgan", "scale": 4, "num_feat": 64, ...}
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Instance name |
| `path` | string | Filesystem path |
| `architecture` | string\|null | Architecture name |
| `scale` | int\|null | Upscale factor |
| `checkpoints` | array | List of checkpoint filenames |
| `latest_version` | string\|null | Latest version tag (e.g. `"v3"`) |
| `config` | object | Full architecture configuration |

---

### GET /api/models/instances/{name}

Get details for a specific model instance.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Model instance name |

**Response:** Same as `GET /api/models/instances` entry

**Errors:** 404 if instance not found

---

### POST /api/models/instances

Create a new named model instance.

**Request Body:**
```json
{
  "name": "my_model",
  "architecture": "rrdb_esrgan",
  "config": {"scale": 4, "num_feat": 64, "num_block": 23}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique instance name |
| `architecture` | string | Yes | Registry key (`"rrdb_esrgan"` or `"swinir"`) |
| `config` | object | Yes | Architecture-specific params |

**Response:** `201 Created` with full instance details (same as GET response)

**Errors:** 400 if name is empty, 409 if instance already exists

---

### POST /api/models/instances/{name}/export

Export a model checkpoint to a specified format.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Model instance name |

**Request Body:**
```json
{
  "instance": "my_model",
  "format": "onnx",
  "output": "/path/to/output.onnx"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `instance` | string | Yes | — | Instance name (must match path param) |
| `format` | string | Yes | — | One of: `pth`, `onnx`, `torchscript`, `safetensors` |
| `output` | string\|null | No | `<instance>/checkpoints/latest.pt` | Output path |

**Response:**
```json
{
  "output": "/path/to/output.onnx",
  "format": "onnx"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `output` | string | Path to exported file |
| `format` | string | Export format used |

---

### GET /api/models/instances/{name}/versions

List all saved versions for a model instance.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Model instance name |

**Response:**
```json
[
  {"tag": "v1", "path": "/path/to/v1.pt", "metadata": {"run": "run_001", "timestamp": 1728000000.0}},
  {"tag": "v2", "path": "/path/to/v2.pt", "metadata": {"run": "run_002", "timestamp": 1728003600.0}}
]
```

| Field | Type | Description |
|-------|------|-------------|
| `tag` | string | Version tag (e.g., `"v1"`, `"v2"`) |
| `path` | string | Path to checkpoint file |
| `metadata` | dict\|null | Run metadata (timestamp, run name) |

---

### DELETE /api/models/instances/{name}

Delete a model instance and all its files.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Model instance name |

**Response:**
```json
{"deleted": "my_model"}
```

**Errors:** 404 if instance not found

---

### POST /api/train/start

Start a training job in the background.

**Request Body:**
```json
{
  "model_name": "rrdb_esrgan",
  "instance": "my_model",
  "dataset": "my_dataset",
  "config": null,
  "resume": null,
  "device": "auto",
  "batch_size": 4,
  "learning_rate": 0.0001,
  "max_epochs": 50,
  "patch_size": 128,
  "fp16": true,
  "seed": 42,
  "weight_decay": 0.0,
  "betas": [0.9, 0.999],
  "num_workers": 4,
  "save_per_epoch": 5,
  "validation_enabled": true,
  "validation_split": 0.1,
  "validation_dataset": null,
  "metrics_frequency": 1,
  "perceptual_weight": 0.1,
  "warmup_steps": 500
}
```

**TrainParams Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | Yes | — | Architecture key (`"rrdb_esrgan"` or `"swinir"`) |
| `instance` | string | Yes | — | Model instance name (for versioning) |
| `dataset` | string | Yes | — | Dataset name or path |
| `config` | string\|null | No | Built-in | Path to custom training config YAML |
| `resume` | string\|null | No | — | Version tag or checkpoint path to resume from |
| `device` | string | No | `"auto"` | `"cuda"`, `"cpu"`, `"auto"` |
| `batch_size` | int\|null | No | Config | Batch size |
| `learning_rate` | float\|null | No | Config | Learning rate |
| `max_epochs` | int\|null | No | Config | Maximum epochs |
| `patch_size` | int\|null | No | Config | Training patch size |
| `fp16` | bool\|null | No | Config | Enable bf16 mixed precision |
| `seed` | int\|null | No | Config | Random seed |
| `weight_decay` | float\|null | No | Config | Adam weight decay |
| `betas` | [float,float]\|null | No | Config | Adam betas |
| `num_workers` | int\|null | No | Config | Dataloader workers |
| `save_per_epoch` | int\|null | No | Config | Save checkpoint every N epochs |
| `validation_enabled` | bool\|null | No | Config | Enable validation |
| `validation_split` | float\|null | No | Config | Validation split ratio |
| `validation_dataset` | string\|null | No | — | Separate validation dataset name |
| `metrics_frequency` | int\|null | No | Config | Log metrics every N batches |
| `perceptual_weight` | float\|null | No | Config | Perceptual loss weight |
| `warmup_steps` | int\|null | No | Config | LR warmup steps |

**Behavior:**
- Only one training job can run at a time (mutex-protected)
- If another training is already running, the job fails immediately with `"Another training is already running"`
- Checkpoints are saved to `experiments/<instance>/run_<timestamp>/`
- On completion, the model state dict is saved as a new version (`v1`, `v2`, ...)
- Hardware monitoring (GPU/CPU/RAM) publishes `hardware` SSE events every 3 seconds

**Response:**
```json
{"job_id": "train_1728000000_abc12345", "status": "accepted"}
```

---

### POST /api/train/validate-dataset

Synchronously validate that a training dataset is usable.

**Request Body:** Same as `TrainParams` (only `dataset` field is used)

**Response:**
```json
{"valid": true, "problems": []}
```

| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | True if dataset passes validation |
| `problems` | array | List of validation issues |

---

### POST /api/infer/start

Start an inference job in the background.

**Request Body:**
```json
{
  "model": null,
  "instance": "my_model",
  "version": "v3",
  "input": "/path/to/input.png",
  "output": "/path/to/output.png",
  "tile": 512,
  "overlap": 64,
  "device": "auto"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string\|null | See note | — | Direct checkpoint path |
| `instance` | string\|null | See note | — | Model instance name |
| `version` | string\|null | No | Latest | Version tag (requires `instance`) |
| `input` | string | Yes | — | Input image or video path |
| `output` | string | Yes | — | Output path |
| `tile` | int | No | 512 | Tile size (0 = no tiling) |
| `overlap` | int | No | 64 | Tile overlap in pixels |
| `device` | string | No | `"auto"` | `"cuda"`, `"cpu"`, `"auto"` |

**Note:** Either `model` (direct checkpoint path) or `instance` (workspace-managed version) must be provided. If both are provided, `instance` takes precedence.

**Input type detection:**
- Image (`.png`, `.jpg`, `.bmp`, `.tiff`) → single image SR
- Video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.ts`) → frame-by-frame video SR

**Response:**
```json
{"job_id": "infer_1728000000_abc12345", "status": "accepted"}
```

---

### GET /api/datasets

List datasets in the workspace.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `scale` | int\|null | No | Filter by scale factor |

**Response:**
```json
[
  {
    "name": "my_dataset",
    "path": "/home/user/workspace/datasets/my_dataset",
    "scale": 4,
    "num_pairs": 1500
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Dataset directory name |
| `path` | string | Full filesystem path |
| `scale` | int | Scale factor from manifest |
| `num_pairs` | int | Number of HR/LR image pairs |

---

### POST /api/datasets/build

Build a dataset from a video file or preprocessed directory.

**Request Body:**
```json
{
  "input": "/path/to/video.mp4",
  "out": "/path/to/output",
  "config": null,
  "degradations": "blur,noise,jpeg",
  "config_overrides": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | Yes | — | Video file or preprocessed directory |
| `out` | string\|null | No | Auto | Output dataset directory |
| `config` | string\|null | No | Built-in | Dataset config YAML path |
| `degradations` | string\|null | No | Config | Comma-separated: `blur,noise,jpeg,jpeg2000,color-jitter` |
| `config_overrides` | dict\|null | No | — | Deep-merged overrides for degradation config |

**Input type behaviour:**
- Video file: extracts frames, applies degradation, writes HR/LR pairs
- Directory: re-validates and rebuilds manifest

**Response:**
```json
{"job_id": "dataset_build_1728000000_abc12345", "status": "accepted"}
```

---

### POST /api/datasets/validate

Synchronously validate a dataset's structural integrity.

**Request Body:**
```json
{"path": "/home/user/workspace/datasets/my_dataset"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Dataset directory path |

**Path restriction:** Must be within the workspace directory.

**Response:**
```json
{"valid": true, "problems": [], "num_pairs": 1500}
```

| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | True if dataset passes structural checks |
| `problems` | array | List of validation issues |
| `num_pairs` | int | Number of valid HR/LR pairs found |

---

### POST /api/datasets/validate-async

Asynchronously validate a dataset. Same request/response as `POST /api/datasets/validate` but returns a job ID for SSE progress tracking.

---

### POST /api/datasets/health

Run a health check on a dataset (resolution profile, black frame detection).

**Request Body:**
```json
{
  "path": "/home/user/workspace/datasets/my_dataset",
  "yes": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `path` | string | Yes | — | Dataset directory path |
| `yes` | bool | No | false | Skip confirmation prompt for black frame deletion |

**Response:**
```json
{"job_id": "dataset_health_1728000000_abc12345", "status": "accepted"}
```

---

### POST /api/datasets/merge

Merge multiple datasets grouped by scale factor.

**Request Body:**
```json
{
  "input": "/path/to/source/datasets",
  "out": "/path/to/merged",
  "scale": 4,
  "name": "merged_dataset",
  "keep_sources": false,
  "input_datasets": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | Yes | — | Directory containing source datasets |
| `out` | string\|null | Yes | — | Output directory for merged dataset |
| `scale` | int\|null | No | — | Filter by scale factor |
| `name` | string\|null | No | Auto | Output dataset name |
| `keep_sources` | bool | No | false | If false, source datasets are deleted after merge |
| `input_datasets` | array\|null | No | All | Specific dataset names to merge |

**Response:**
```json
{"job_id": "dataset_merge_1728000000_abc12345", "status": "accepted"}
```

---

### POST /api/datasets/prune

Remove black frame pairs from a dataset.

**Request Body:**
```json
{
  "path": "/home/user/workspace/datasets/my_dataset",
  "black_frames": ["frame_001.png", "frame_042.png"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Dataset directory path |
| `black_frames` | array | Yes | List of frame filenames to remove |

**Response:**
```json
{"job_id": "dataset_prune_1728000000_abc12345", "status": "accepted"}
```

---

### GET /api/jobs

List all background jobs (current and historical).

**Response:**
```json
[
  {
    "job_id": "train_1728000000_abc12345",
    "job_type": "train",
    "status": "completed",
    "created_at": 1728000000.0,
    "started_at": 1728000001.0,
    "completed_at": 1728003600.0,
    "error": null,
    "result": null
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `job_type` | string | Job type (`train`, `infer`, `dataset.build`, `dataset.validate`, `dataset.health`, `dataset.merge`, `dataset.prune`) |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `created_at` | float | Unix timestamp of creation |
| `started_at` | float\|null | Unix timestamp when execution started |
| `completed_at` | float\|null | Unix timestamp when execution ended |
| `error` | string\|null | Error message if failed |
| `result` | dict\|null | Job result data |

---

### GET /api/jobs/{job_id}

Get status of a specific job.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID |

**Response:** Same format as list entry

**Errors:** 404 if job not found

---

### POST /api/jobs/{job_id}/cancel

Cancel a running job.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID |

**Response:**
```json
{"status": "cancelled", "job_id": "train_1728000000_abc12345"}
```

**Errors:** 404 if job not found

---

### GET /api/env

Get environment diagnostics.

**Response:**
```json
{
  "torch_version": "2.5.0+cu124",
  "device": "cuda:0",
  "cuda_available": true,
  "rocm": false,
  "bf16_supported": true,
  "flash_attn": true,
  "device_name": "NVIDIA GeForce RTX 4090",
  "vram_total_mb": 24564
}
```

| Field | Type | Description |
|-------|------|-------------|
| `torch_version` | string | PyTorch version string |
| `device` | string | Detected device (`cuda:0`, `cpu`, `none`) |
| `cuda_available` | bool | Whether CUDA is available |
| `rocm` | bool | Whether ROCm is the backend |
| `bf16_supported` | bool | Whether bfloat16 is supported |
| `flash_attn` | bool | Whether flash attention is available |
| `device_name` | string\|null | GPU name (null if CPU) |
| `vram_total_mb` | int\|null | Total VRAM in MB (null if CPU) |

---

## Architecture Mapping

The frontend references models by display name. The API maps these:

| Frontend Display Name | Engine Registry Key |
|-----------------------|-------------------|
| `Real-ESRGAN` | `rrdb_esrgan` |
| `SwinIR` | `swinir` |

---

## Job Type Reference

| job_type | Created By | Status Lifecycle |
|----------|------------|-----------------|
| `train` | `POST /api/train/start` | pending → running → completed/failed/cancelled |
| `infer` | `POST /api/infer/start` | pending → running → completed/failed |
| `dataset.build` | `POST /api/datasets/build` | pending → running → completed/failed |
| `dataset.validate` | `POST /api/datasets/validate-async` | pending → running → completed/failed |
| `dataset.health` | `POST /api/datasets/health` | pending → running → completed/failed |
| `dataset.merge` | `POST /api/datasets/merge` | pending → running → completed/failed |
| `dataset.prune` | `POST /api/datasets/prune` | pending → running → completed/failed |

---

## Examples

### Health Check
```bash
curl http://localhost:8765/api/health
```

### Initialize Workspace
```bash
curl -X POST http://localhost:8765/api/workspace/init \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/sr-workspace"}'
```

### Create Model Instance
```bash
curl -X POST http://localhost:8765/api/models/instances \
  -H "Content-Type: application/json" \
  -d '{"name": "my_swinir", "architecture": "swinir", "config": {"scale": 4, "embed_dim": 180}}'
```

### Start Training
```bash
curl -X POST http://localhost:8765/api/train/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "swinir", "instance": "my_swinir", "dataset": "my_dataset", "max_epochs": 20}'
```

### Build Dataset from Video
```bash
curl -X POST http://localhost:8765/api/datasets/build \
  -H "Content-Type: application/json" \
  -d '{"input": "/path/to/video.mp4", "degradations": "blur,noise,jpeg"}'
```

### Subscribe to SSE Events
```bash
curl -N http://localhost:8765/api/events?job_id=dataset_build_1728000000_abc12345
```

### Check Job Status
```bash
curl http://localhost:8765/api/jobs/train_1728000000_abc12345
```

### Cancel a Job
```bash
curl -X POST http://localhost:8765/api/jobs/train_1728000000_abc12345/cancel
```

### Environment Diagnostics
```bash
curl http://localhost:8765/api/env
```

### Export Model
```bash
curl -X POST http://localhost:8765/api/models/instances/my_swinir/export \
  -H "Content-Type: application/json" \
  -d '{"instance": "my_swinir", "format": "onnx"}'
```

---

## Error Codes

| HTTP Status | Meaning | Example |
|-------------|---------|---------|
| 400 | Bad request — missing required fields | `{"detail": "name is required"}` |
| 403 | Forbidden — path outside workspace | `{"detail": "Path is outside the workspace"}` |
| 404 | Not found | `{"detail": "Job not found: invalid_id"}` |
| 409 | Conflict — resource already exists | `{"detail": "Instance 'my_model' already exists"}` |
| 503 | Service unavailable — workspace not initialized | `{"detail": "Workspace not initialised"}` |

---

## Internal Architecture

### Request Lifecycle

```
Client → FastAPI (uvicorn)
  → CORS middleware
  → Route handler (sync response or async job creation)
    → Sync: validate + respond immediately
    → Async: validate + create job + spawn daemon thread + respond {job_id}
      → Background thread executes:
        1. tasks.start_job(job_id)             → status: "running"
        2. Progress events via SSEEventManager  → SSE stream
        3. On success: tasks.complete_job()     → status: "completed"
        4. On error: tasks.fail_job()           → status: "failed"
        5. events.publish(job_id, None)         → stream ends
```

### Module Map

| Module | File | Responsibility |
|--------|------|----------------|
| `app.py` | `api/app.py` | FastAPI app, CORS, lifespan, health endpoint, SSE endpoint, route registration |
| `schemas.py` | `api/schemas.py` | Pydantic v2 models for request/response |
| `deps.py` | `api/deps.py` | FastAPI dependency injection (workspace, configs) |
| `task_manager.py` | `api/task_manager.py` | `BackgroundTaskManager` — job creation, status tracking, cancellation |
| `event_manager.py` | `api/event_manager.py` | `SSEEventManager` — per-job async event queues |
| `callbacks.py` | `api/callbacks.py` | `SSECallback` — `TrainerCallback` → SSE events |
| `progress.py` | `api/progress.py` | `SSEProgressReporter` — `ProgressReporter` → SSE events |
| `workers.py` | `api/workers.py` | Background worker functions (training, inference, dataset ops) |
| `routes/` | `api/routes/` | Route handlers organized by domain |

### Thread Safety

- `BackgroundTaskManager` uses `threading.Lock()` for all state mutations
- `SSEEventManager` uses `threading.Lock()` for queue creation, `asyncio.Queue` for thread-safe event publishing
- Training mutex (`_training_mutex`) ensures only one training job runs at a time
- All workers run as `daemon=True` threads (auto-terminate on server shutdown)

### SSE Event Bus

The `SSEEventManager` implements a per-job pub/sub pattern:

1. `publish(job_id, event)` — called from background workers (thread-safe)
2. `subscribe(job_id)` — async generator, called from FastAPI SSE endpoint
3. `cleanup(job_id)` — called on client disconnect

Events are in-memory only. If no client is subscribed, events are silently dropped (the queue buffers them, but cleanup removes the queue on disconnect).

---

## See Also

- [Architecture Overview](architecture.md)
- [CLI Reference](cli-reference.md) — `srengine serve start` command
- [Frontend Guide](frontend.md) — React GUI that consumes this API
- [Desktop Guide](desktop.md) — Tauri shell that manages the server process