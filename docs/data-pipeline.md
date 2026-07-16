# Data Pipeline

## Overview

The data pipeline converts raw video into paired HR/LR image datasets for super-resolution training. It supports two entry points:

1. **Raw video** → frame extraction → degradation → paired dataset
2. **Preprocessed directory** → re-validation → manifest rebuild

```
Video file (.mp4, .avi, .mov)
      │
      ▼
┌─────────────────┐
│ video_extract   │  Extract frames at target frame rate
│                 │  Skip duplicated frames (SSIM-based)
└────────┬────────┘
         │
         ▼  HR frames (original resolution)
┌─────────────────────────────────────┐
│ degrade                             │  Apply selectable degradation stages
│   color_jitter (opt)                │  to produce LR images. Each stage has
│   → blur (gaussian / motion)        │  its own `enabled` flag in config.
│   → downsample                      │
│   → noise (gaussian / poisson /     │
│       salt & pepper)                │
│   → jpeg (opt)                      │
│   → jpeg2000 (opt)                  │
└────────┬────────────────────────────┘
         │
         ▼  HR/ + LR/ directories
┌─────────────────┐
│ dataset_builder │  Write manifest.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ dataset_validator│  Structural integrity check
│                 │  Dimension ratios, file readability
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ dataset_health  │  Resolution profile, black frame detection
└─────────────────┘
```

## Video Extraction

`data/video_extract.py:extract_frames()`

Reads a video file using OpenCV and extracts frames at a configurable rate:

```python
extract_frames(
    video_path="/path/to/video.mp4",
    output_dir="/path/to/frames/",
    frame_rate=24,          # frames per second to extract
    skip_duplicates=True,   # SSIM-based duplicate detection
    duplicate_threshold=0.98
) → List[Path]
```

- Supports `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` formats
- Frame rate control: extracts at `min(video_fps, target_fps)` by default
- Duplicate detection: compares consecutive frames via SSIM; skips frames with SSIM > 0.98 to avoid feeding near-identical frames to the training set
- Outputs PNG files named `frame_000001.png`, `frame_000002.png`, ...

## Degradation Pipeline

`data/degrade.py` implements a synthetic degradation pipeline that transforms HR frames into LR counterparts. Every stage (except downsampling) can be independently enabled/disabled.

Each stage respects an `enabled: true/false` flag in the config section. Omit the flag or set it to `true` for the stage to participate (subject to per-image probability gating).

### Pipeline Order

```
HR → Crop to scale-multiple
    → Color Jitter          (optional, disabled by default)
    → Blur                  (optional, enabled by default)
        → Gaussian (prob)  OR  Motion (prob)   — coin-flip if both trigger
    → Downsample            (always — configurable method + optional antialias)
    → Noise                 (optional, enabled by default)
        → Gaussian (prob)  OR  Poisson (prob)  OR  Salt & Pepper (prob)  — pick one
    → JPEG                  (optional, enabled by default)
    → JPEG2000              (optional, disabled by default)
```

### 1. Color Jitter (optional)

Shifts hue, saturation, and value in HSV space — simulates color gamut variations across different cameras.

```yaml
color_jitter:
  enabled: false
  hue_range: [-0.05, 0.05]
  saturation_range: [-0.3, 0.3]
  value_range: [-0.3, 0.3]
  prob: 0.8
```

### 2. Blur (optional)

Two sub-types; if both trigger a coin-flip selects one:

| Type | Description | Key params |
|------|-------------|------------|
| **Gaussian** | Isotropic Gaussian blur (anti-aliasing) | `kernel_size`, `sigma` range |
| **Motion** | Linear motion blur at random angle | `max_kernel_size` |

```yaml
blur:
  enabled: true
  gaussian:
    kernel_size: 21
    sigma: [0.1, 3.0]
    prob: 1.0
  motion:
    max_kernel_size: 31
    prob: 0.5
```

### 3. Downsample (always applied)

Downscales the image to produce the LR counterpart. The default method is `area` (OpenCV `INTER_AREA`) for best anti-aliasing.

| Method | OpenCV flag | Info preservation |
|--------|-------------|-------------------|
| `area` | `INTER_AREA` | Best — avoids moiré patterns |
| `lanczos` | `INTER_LANCZOS4` | Sharp — 8×8 tap, may cause ringing |
| `bicubic` | `INTER_CUBIC` | Good balance — 4×4 |
| `bilinear` | `INTER_LINEAR` | Soft — 2×2, blurs detail |
| `nearest` | `INTER_NEAREST` | Pixelated — no interpolation |

```yaml
resize:
  method: area
  antialias: true
```

The `antialias` option applies a gentle Gaussian pre-filter (`sigma=0.5`) before downsampling when using bicubic, bilinear, lanczos, or nearest. `INTER_AREA` already does its own anti-aliasing and skips the extra filter.

### 4. Noise (optional)

Three sub-types; if multiple trigger one is selected randomly:

| Type | Description | Key params |
|------|-------------|------------|
| **Gaussian** | Additive white Gaussian noise | `sigma_range` |
| **Poisson** | Photon-like (shot) noise | `scale_range` |
| **Salt & Pepper** | Random black/white pixels | `amount`, `salt_vs_pepper` |

```yaml
noise:
  enabled: true
  gaussian:
    sigma_range: [1, 30]
    prob: 0.5
  poisson:
    scale_range: [0.05, 3.0]
    prob: 0.5
  salt_pepper:
    amount: 0.01
    salt_vs_pepper: 0.5
    prob: 0.3
```

### 5. JPEG Compression (optional)

Standard JPEG compression artifacts via OpenCV `imencode`/`imdecode`.

```yaml
jpeg:
  enabled: true
  quality_range: [30, 95]
  prob: 1.0
```

### 6. JPEG2000 Compression (optional)

JPEG2000 compression artifacts — complements JPEG with different block-free artifacts.

```yaml
jpeg2000:
  enabled: false
  quality_range: [30, 95]
  prob: 0.5
```

## Configuration

The full degradation config in `utils/configs/datasets/video_pairs.yaml`:

```yaml
scale: 4
degradation:
  color_jitter:
    enabled: false
    hue_range: [-0.05, 0.05]
    saturation_range: [-0.3, 0.3]
    value_range: [-0.3, 0.3]
    prob: 0.8
  blur:
    enabled: true
    gaussian:
      kernel_size: 21
      sigma: [0.1, 3.0]
      prob: 1.0
    motion:
      max_kernel_size: 31
      prob: 0.5
  resize:
    method: area
    antialias: true
  noise:
    enabled: true
    gaussian:
      sigma_range: [1, 30]
      prob: 0.5
    poisson:
      scale_range: [0.05, 3.0]
      prob: 0.5
    salt_pepper:
      amount: 0.01
      salt_vs_pepper: 0.5
      prob: 0.3
  jpeg:
    enabled: true
    quality_range: [30, 95]
    prob: 1.0
  jpeg2000:
    enabled: false
    quality_range: [30, 95]
    prob: 0.5
frame_rate: 10
start_time: 0.0
duration: null
```

## CLI Quick Select

For convenience, the `dataset build` command accepts `--degradations` to override `enabled` flags without editing a YAML file:

```bash
# Only JPEG compression (no blur, no noise)
srengine dataset build -i video.mp4 --degradations jpeg

# JPEG + blur, area downsampling
srengine dataset build -i video.mp4 --degradations jpeg,blur --resize-method area

# Only noise (all sub-types)
srengine dataset build -i video.mp4 -d noise

# Color jitter + JPEG2000
srengine dataset build -i video.mp4 -d color-jitter,jpeg2000

# Custom config + CLI override
srengine dataset build -i video.mp4 -c my_config.yaml -d jpeg
```

## Dataset Builder

`data/dataset_builder.py` orchestrates the full pipeline.

### `build_from_video()`

```python
build_from_video(video_path, output_dir, config) → Path
```

1. Create output directory structure: `output_dir/HR/`, `output_dir/LR/`
2. Extract frames from video → `HR/frame_*.png`
3. For each HR frame, run degradation pipeline → `LR/frame_*.png`
4. Write `manifest.json` with pairs index
5. Run validation

### `build_from_preprocessed()`

```python
build_from_preprocessed(input_dir) → Path
```

1. Scan `input_dir/HR/` and `input_dir/LR/` for matching files
2. Validate structure and dimension ratios
3. Rebuild or update `manifest.json`

## Dataset Validation

`data/dataset_validator.py:validate()` performs structural checks:

```python
validate(dataset_dir) → ValidationReport
```

Checks performed:
- `HR/` and `LR/` directories exist
- Every file in `HR/` has a matching `LR/` counterpart (by filename)
- Image dimensions satisfy `hr_dim / lr_dim = scale` (with tolerance)
- All image files are readable (no corruption)
- No missing or extra files

Returns a `ValidationReport` with:
- `ok: bool` — passed or failed
- `num_pairs: int` — total validated pairs
- `problems: list[str]` — descriptive error messages

## Dataset Health

`data/dataset_health.py:check_dataset_health()` profiles dataset quality:

```python
check_dataset_health(dataset_dir) → HealthReport
```

Profiles every HR frame for:
- Resolution distribution (`1920x1080`, `1280x720`, etc.)
- Aspect ratio distribution
- Color channel distribution (RGB, grayscale)
- Black frame detection (mean brightness below adaptive threshold)

### Black Frame Pruning

```python
prune_black_frames(dataset_dir, threshold=None, yes=False) → list[str]
```

- Computes adaptive threshold: `mean_brightness - 2 * std_brightness` across all frames
- Frames below threshold are deleted from `HR/` and `LR/` and removed from `manifest.json`
- With `--yes`, deletes without confirmation prompt

## Dataset Class

`data/datasets.py:PairedImageFolderDataset` is the PyTorch `Dataset` used during training:

```python
class PairedImageFolderDataset(Dataset):
    def __init__(self, root_dir, split="train", scale=4, transforms=None):
        # Loads manifest.json
        # Splits into train/val based on ratio
        # Returns (lr_tensor, hr_tensor) pairs
```

- Reads HR and LR image paths from `manifest.json`
- Supports train/validation split via `split` parameter
- Returns `(lr, hr)` tuples with shape `(C, H, W)` in range `[0, 1]`
- Uses OpenCV for image loading (BGR → RGB conversion)
- On-the-fly file I/O — no pre-caching to RAM

## Transforms

`data/transforms.py` provides augmentation transforms applied during training:

| Transform | Description |
|-----------|-------------|
| `RandomCrop(patch_size, scale)` | Randomly crops a `patch_size×patch_size` region from LR and the corresponding `patch_size*scale × patch_size*scale` region from HR |
| `CenterCrop(patch_size, scale)` | Center crop (used for validation) |
| `RandomFlip(direction)` | Horizontal or vertical flip (50% probability) |
| `RandomRotate(angles)` | Rotation by 90°, 180°, or 270° |
| `Compose(transforms)` | Chains multiple transforms together |

Transforms operate on `(lr, hr)` tuples simultaneously, ensuring aligned cropping and identical augmentation for both images.

### Usage in training

```python
from sr_engine.data.transforms import Compose, RandomCrop, RandomFlip

train_transforms = Compose([
    RandomCrop(patch_size=128, scale=4),
    RandomFlip(direction='horizontal'),
])

dataset = PairedImageFolderDataset(
    root_dir="./datasets/my_set",
    split="train",
    transforms=train_transforms
)
```
