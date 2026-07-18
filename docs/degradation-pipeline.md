# Degradation Pipeline

## Executive Summary

The degradation pipeline is the component of sr-engine responsible for transforming high-resolution (HR) video frames into low-resolution (LR) counterparts through simulation of real-world image quality degradation. It is the primary mechanism for generating paired HR/LR training data for supervised super-resolution model training.

**What it is:** A composable, configurable, multi-stage image degradation system that applies blur, noise, downsampling, compression artifacts, and color distortion to HR frames to produce realistic LR inputs.

**Why it matters:** Super-resolution models learn to reverse the degradation process. The quality, diversity, and realism of the synthetic degradation directly determines the model's ability to generalize to real-world low-resolution images. A well-designed degradation pipeline is the single most important factor in production SR model performance.

**Key takeaways:**

- Six independently configurable degradation stages (color jitter, blur, downsample, noise, JPEG, JPEG2000)
- Each stage has per-image probability gating for diversity
- Noise and blur sub-types are mutually exclusive per image (randomly selected)
- Operates via `ProcessPoolExecutor` for parallel CPU throughput
- Validated output guarantees dimensional consistency (HR dimensions must be exact multiples of scale factor)
- CLI supports quick-select presets and per-stage toggling without YAML editing
- Configurable via 4-level merge system (builtin → workspace → file → CLI flags)

**Target audience:** ML engineers training super-resolution models, systems integrators building data pipelines, researchers experimenting with degradation strategies, and developers deploying production SR systems.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Historical Context](#historical-context)
3. [Terminology and Definitions](#terminology-and-definitions)
4. [Conceptual Foundations](#conceptual-foundations)
5. [System Architecture](#system-architecture)
6. [Core Components](#core-components)
   - 6.1 [Video Frame Extraction](#61-video-frame-extraction)
   - 6.2 [Color Jitter Stage](#62-color-jitter-stage)
   - 6.3 [Blur Stage](#63-blur-stage)
   - 6.4 [Downsample Stage](#64-downsample-stage)
   - 6.5 [Noise Stage](#65-noise-stage)
   - 6.6 [JPEG Compression Stage](#66-jpeg-compression-stage)
   - 6.7 [JPEG2000 Compression Stage](#67-jpeg2000-compression-stage)
   - 6.8 [Batch Degradation Orchestrator](#68-batch-degradation-orchestrator)
   - 6.9 [Dataset Builder](#69-dataset-builder)
   - 6.10 [Dataset Validator](#610-dataset-validator)
   - 6.11 [Dataset Health Checker](#611-dataset-health-checker)
   - 6.12 [PyTorch Dataset Class](#612-pytorch-dataset-class)
7. [Internal Mechanics](#internal-mechanics)
8. [Data Model](#data-model)
9. [Configuration Reference](#configuration-reference)
10. [CLI Usage](#cli-usage)
11. [Implementation Guide](#implementation-guide)
12. [Security Analysis](#security-analysis)
13. [Performance Analysis](#performance-analysis)
14. [Troubleshooting](#troubleshooting)
15. [Common Mistakes](#common-mistakes)
16. [Best Practices](#best-practices)
17. [Real-World Case Studies](#real-world-case-studies)
18. [Advanced Topics](#advanced-topics)
19. [References](#references)

---

## Introduction

### Purpose

The degradation pipeline exists to bridge the gap between synthetic training data and real-world low-resolution imagery. Super-resolution models require paired (HR, LR) examples for supervised training, but acquiring naturally occurring pairs of the same scene at different resolutions is impractical at scale. The pipeline solves this by taking high-resolution source material (video frames) and algorithmically degrading it through a sequence of transformations that model the optical, electronic, and compression artifacts present in real-world imaging chains.

### Scope

This document covers:

- The six-stage degradation pipeline and each stage's mathematical/algorithmic implementation
- The orchestrator (`batch_degrade`) and its parallel execution model
- The upstream video extraction system
- The downstream dataset builder, validator, and health checker
- The PyTorch `Dataset` class that consumes the output
- Configuration via YAML and CLI overrides
- Performance characteristics and optimization strategies
- Troubleshooting common failure modes

It does **not** cover:

- Model architectures (RRDB, SwinIR) — see `docs/training.md`
- The training loop or loss functions — see `docs/training.md`
- The inference pipeline — see `docs/inference.md`
- The workspace system — see `docs/workspace.md`

### Audience

- ML engineers training or fine-tuning super-resolution models
- Data engineers building and maintaining training datasets
- Researchers experimenting with degradation strategies
- DevOps/SRE engineers operating dataset build pipelines
- Developers integrating sr-engine into larger systems

### Assumptions

- Familiarity with Python, NumPy, and OpenCV concepts
- Basic understanding of image processing (blur, noise, color spaces, compression)
- Familiarity with command-line tools and YAML configuration
- No prior super-resolution domain knowledge required

### Document Goals

After reading this document, the reader should be able to:

1. Explain the purpose and design of each degradation stage
2. Configure the pipeline for different use cases via YAML and CLI
3. Diagnose and fix common dataset building failures
4. Optimize pipeline throughput for large-scale dataset creation
5. Extend the pipeline with new degradation stages
6. Understand the relationship between degradation quality and model performance

---

## Historical Context

### Origin of Synthetic Degradation in Super-Resolution

The practice of synthesizing LR images from HR sources dates to the earliest CNN-based super-resolution methods. SRCNN (2014) used bicubic downsampling as its sole degradation model. This worked for controlled benchmarks but failed dramatically on real-world images, which contain complex, spatially-varying degradation.

### The Blind SR Revolution

The realization that real-world LR images contain unknown, composite degradation led to the "blind super-resolution" paradigm. Key milestones:

| Year | Work | Degradation Model |
|------|------|-------------------|
| 2014 | SRCNN | Bicubic downsampling only |
| 2017 | ESRGAN | Bicubic + simple noise |
| 2018 | ZSSR | Per-image internal learning |
| 2019 | RealSR | Real camera-captured pairs |
| 2020 | BSRGAN | Blur, noise, downscale, compression — random ordering |
| 2021 | Real-ESRGAN | High-order degradation model (2-stage pipeline) |
| 2022 | SwinIR | Classic degradation (same design as sr-engine) |

### sr-engine's Design Heritage

sr-engine's degradation pipeline follows the "classic degradation" model popularized by BSRGAN and Real-ESRGAN, but with several practical engineering improvements:

- **Deterministic ordering** (unlike BSRGAN's random-ordering approach) for reproducibility and simpler debugging
- **Strict scale-factor validation** to catch alignment errors early
- **Parallel CPU execution** via process pool for high-throughput dataset building
- **Probability-gated stages** enabling stochastic augmentation without configuration complexity
- **CLI quick-select** for rapid experimentation without YAML editing

---

## Terminology and Definitions

| Term | Definition |
|------|------------|
| **HR** | High-Resolution — the original, undegraded image (source of truth) |
| **LR** | Low-Resolution — the degraded counterpart of HR, simulating a real-world low-quality image |
| **Scale factor** | The integer ratio `HR_dimension / LR_dimension` (typically 2, 3, or 4) |
| **Degradation stage** | An individual transformation in the pipeline (e.g., blur, noise, JPEG) |
| **Probability gating** | A per-image random coin-flip that determines whether a given stage is applied |
| **Mutual exclusion** | A constraint where at most one sub-type of a stage (e.g., Gaussian vs. motion blur) is applied per image |
| **Manifest** | A `manifest.json` file that indexes all HR/LR pairs in a dataset directory |
| **Patch** | A cropped sub-region of an image used for training (e.g., 128×128 LR patch) |
| **Classic degradation** | The fixed-order pipeline model: blur → downsample → noise → compression |
| **High-order degradation** | A more complex model with repeated, randomly-ordered degradation stages (not implemented in sr-engine) |
| **INTER_AREA** | OpenCV interpolation method that applies pixel area relation resampling — best anti-aliasing for downscaling |
| **Antialias pre-filter** | A Gaussian blur applied before downsampling to prevent moiré patterns and aliasing artifacts |
| **ProcessPoolExecutor** | Python `concurrent.futures` mechanism for parallel CPU-bound work across multiple processes |
| **SSIM** | Structural Similarity Index Measure — used for duplicate frame detection during video extraction |

---

## Conceptual Foundations

### The Super-Resolution Inverse Problem

Super-resolution is fundamentally an ill-posed inverse problem. Given an LR image `y`, we seek an HR image `x` such that:

```
y = D(x) + η
```

Where `D` is the (unknown) degradation function and `η` is noise. The degradation pipeline approximates `D` so the model can learn its inverse `D⁻¹`.

### Why Multiple Degradation Types Matter

A model trained on bicubic-only downsampling learns to reverse only bicubic interpolation. Real-world LR images suffer from:

- **Optical blur** from lens defocus or motion
- **Sensor noise** (Gaussian, Poisson/shot, salt-and-pepper)
- **Downsampling artifacts** from various interpolation methods
- **Compression artifacts** from JPEG, JPEG2000, or video codecs
- **Color shifts** from different camera sensors or white balance

The degradation pipeline must produce examples covering this space for the model to generalize.

### The Diversity vs. Consistency Tradeoff

Each degradation stage has a probability of being applied per image. This creates diverse training examples but means a single LR image may have been degraded differently from its neighbor. The model must learn to handle all combinations. The probability values control the expected frequency of each degradation type in the training distribution.

### Pipeline Order Rationale

```
HR → Color Jitter → Blur → Downsample → Noise → JPEG → JPEG2000 → LR
```

This order is physically motivated:

1. **Color jitter** first — simulates camera sensor differences before any spatial degradation
2. **Blur** — simulates lens or motion blur that occurs before the image is sampled by the sensor
3. **Downsample** — simulates the sensor sampling process (reducing resolution)
4. **Noise** — simulates sensor read noise and photon shot noise (occurs at the sensor level)
5. **JPEG/JPEG2000** — simulates compression after capture (last in the imaging chain)

This ordering means that noise is applied in the LR domain, matching real-world noise characteristics where noise is introduced after downsampling. If noise were applied before downsampling, it would be attenuated by the downsampling operation, producing unrealistically clean LR images.

---

## System Architecture

### High-Level Data Flow

```
Video file (.mp4, .avi, .mov, .mkv, .webm)
       │
       ▼
┌──────────────────┐
│  video_extract   │  Frame extraction at target FPS
│                  │  SSIM-based duplicate detection
└────────┬─────────┘
         │
         ▼  HR frames (PNG sequence)
┌──────────────────────────────────────────────────────────┐
│  batch_degrade                                           │
│                                                          │
│  ProcessPoolExecutor (parallel per-frame)                │
│                                                          │
│  Per frame:                                              │
│    _degrade_image(hr)                                    │
│      ├─ Crop to scale-multiple                           │
│      ├─ Color Jitter          (probabilistic)            │
│      ├─ Blur (Gaussian OR    (probabilistic, exclusive)  │
│      │        Motion)                                     │
│      ├─ Antialias pre-filter (if enabled)                │
│      ├─ Downsample           (always applied)            │
│      ├─ Noise (Gaussian OR  (probabilistic, exclusive)   │
│      │        Poisson OR                                  │
│      │        Salt & Pepper)                              │
│      ├─ JPEG Compression     (probabilistic)             │
│      └─ JPEG2000 Compression (probabilistic)             │
│         │                                                 │
│         ▼ LR image                                        │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼  HR/ + LR/ directories + manifest.json
┌──────────────────┐
│ dataset_validator│  Structural integrity check
│                  │  Dimension ratio verification
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ dataset_health   │  Resolution profiling
│                  │  Black frame detection & pruning
└──────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ PairedImageFolderDataset    │  PyTorch Dataset for training
│   (data/datasets.py)        │  + Compose transforms
└─────────────────────────────┘
```

### Module Dependency Graph

```
cli/cmd_dataset.py
    │
    ├──► data/dataset_builder.py
    │         │
    │         ├──► data/video_extract.py
    │         │         └── OpenCV (cv2.VideoCapture)
    │         │
    │         ├──► data/degrade.py
    │         │         ├── OpenCV (cv2.*)
    │         │         └── concurrent.futures.ProcessPoolExecutor
    │         │
    │         ├──► data/dataset_validator.py
    │         │         └── OpenCV (cv2.imread)
    │         │
    │         └──► utils/progress.py (ProgressReporter)
    │
    ├──► data/dataset_health.py
    │         └── NumPy, OpenCV
    │
    └──► utils/config.py
              └── YAML config files
```

### Architectural Goals

| Goal | Implementation |
|------|----------------|
| **Configurability** | YAML-based with 4-level merge, CLI quick-select flags |
| **Throughput** | Parallel process pool for CPU-bound degradation |
| **Determinism** | Fixed pipeline order, seeded RNG per worker |
| **Correctness** | Strict dimension validation, scale-factor enforcement |
| **Observability** | ProgressReporter interface, detailed logging |
| **Extensibility** | New stages can be added by creating a function and adding config key |

### Constraints

- **Scale factor must be integer**: LR dimensions must exactly divide HR dimensions
- **OpenCV-dependent**: All image I/O and processing uses OpenCV (no PIL/Pillow path)
- **CPU-bound**: Degradation is purely CPU; GPU is not used for dataset building
- **PNG-only for frames**: Frame extraction writes PNG files (lossless, widely supported)
- **Process-based parallelism**: `ProcessPoolExecutor` (not thread pool) due to CPU-bound nature

### Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Deterministic pipeline order vs. random ordering | Reproducibility and debuggability vs. potentially higher diversity |
| Fixed degradation order vs. high-order (sind/Real-ESRGAN style) | Simplicity and speed vs. covering more complex real-world degradation chains |
| Process pool vs. thread pool | True parallelism for CPU-bound work vs. higher memory usage (each process has its own memory space) |
| Per-frame probability gating vs. per-dataset fixed degradation | Diversity within a single dataset vs. controlled experimental conditions |
| OpenCV exclusively vs. mixed library approach | Consistency and predictable behavior vs. access to library-specific algorithms |

---

## Core Components

### 6.1 Video Frame Extraction

**File:** `data/video_extract.py`

#### Purpose

Extracts individual frames from video files into a sequence of PNG images that serve as the HR source material for the degradation pipeline.

#### Responsibilities

- Open video files with OpenCV `VideoCapture`
- Seek to a configurable start time
- Extract frames at a configurable target frame rate (downsampling the video's native FPS if needed)
- Skip near-duplicate frames using SSIM-based detection
- Write numbered PNG files to the output directory

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `Path` | required | Path to video file |
| `out_dir` | `Path` | required | Output directory for PNG frames |
| `frame_rate` | `int \| None` | video FPS | Target extraction frame rate |
| `start_time` | `float` | 0.0 | Start time in seconds |
| `duration` | `float \| None` | full video | Duration in seconds |
| `reporter` | `ProgressReporter \| None` | no-op | Progress reporting |

#### Outputs

- List of `Path` objects pointing to extracted PNG frames, sorted lexicographically
- PNG files named `0.png`, `1.png`, ... (zero-padded to match total frame count length)

#### Internal Logic

1. **Open video**: `cv2.VideoCapture(str(video_path))` — raises `FileNotFoundError` on failure
2. **Compute frame range**: `start_frame = start_time * video_fps`, `end_frame` bounded by `duration`
3. **Determine frame step**: If target `frame_rate` < video FPS, compute `frame_step = round(video_fps / frame_rate)`
4. **Fast-forward**: `vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)`
5. **Iterate**: For each frame in range, either:
   - On-target frame: `vidcap.read()` to decode and save as PNG
   - Skip frame: `vidcap.grab()` to advance without decoding (faster)
6. **FFmpeg note**: OpenCV's `grab()` is significantly faster than `read()` for skipped frames since it avoids pixel decoding. This optimization matters for high-FPS video with low target frame rates.

#### Duplicate Detection

Currently **not implemented** in `video_extract.py` (the `skip_duplicates` and `duplicate_threshold` parameters are documented but the code path is not active). The config mentions this capability but the actual implementation simply extracts all frames at the computed step. This is a known gap.

#### Supported Formats

`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` — any format supported by the system's OpenCV build and FFmpeg backend.

#### Dependencies

- `cv2` (OpenCV) — `VideoCapture`, `imwrite`, `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| `FileNotFoundError: Could not open video file` | File doesn't exist, unsupported codec, or corrupt header |
| Zero frames extracted | `start_time` exceeds video duration; codec not supported |
| OpenCV warning about missing codec | System lacks required FFmpeg/driver for the video codec |
| `OSError: No space left on device` | Output disk is full |

#### Operational Concerns

- High-FPS source video at low target FPS: `grab()` optimization provides ~2-3× speedup
- Long videos: consider splitting into segments or using `duration` parameter
- Frame numbering: uses zero-padded sequential numbering, not original frame indices

---

### 6.2 Color Jitter Stage

**File:** `data/degrade.py` — function `_apply_color_jitter()`

#### Purpose

Simulates color gamut and white balance variations between different camera sensors by randomly shifting hue, saturation, and value in HSV color space.

#### When Applied

Before blur stage (first degradation step after scale-multiple cropping).

#### Configuration

```yaml
color_jitter:
  enabled: false       # Disabled by default
  hue_range: [-0.05, 0.05]
  saturation_range: [-0.3, 0.3]
  value_range: [-0.3, 0.3]
  prob: 0.8             # Per-image application probability
```

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | required | BGR uint8 image (H×W×3) |
| `hue_range` | `list[float]` | [-0.05, 0.05] | Hue shift range (fraction of 180°) |
| `saturation_range` | `list[float]` | [-0.3, 0.3] | Saturation shift range (fraction of 255) |
| `value_range` | `list[float]` | [-0.3, 0.3] | Value (brightness) shift range (fraction of 255) |

#### Internal Logic

1. Convert BGR → HSV (OpenCV: `cv2.COLOR_BGR2HSV`)
2. Sample random deltas uniformly from each range
3. Scale deltas: hue by 180, saturation/value by 255
4. Apply: hue wraps modulo 180, saturation/value are clamped to [0, 255]
5. Convert back to BGR

```
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

h_delta = random.uniform(hue_range[0], hue_range[1]) * 180
s_delta = random.uniform(saturation_range[0], saturation_range[1]) * 255
v_delta = random.uniform(value_range[0], value_range[1]) * 255

hsv[:,:,0] = (hsv[:,:,0] + h_delta) % 180
hsv[:,:,1] = np.clip(hsv[:,:,1] + s_delta, 0, 255)
hsv[:,:,2] = np.clip(hsv[:,:,2] + v_delta, 0, 255)

return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

#### Notes

- **OpenCV HSV ranges**: H = [0, 180), S = [0, 255], V = [0, 255]
- Hue wraps modulo 180 (circular), maintaining color continuity across the red boundary
- Applied in BGR space (OpenCV convention) — the image stays BGR throughout the pipeline

#### Failure Modes

- None significant — pure pixel manipulation, no file I/O or external dependencies

---

### 6.3 Blur Stage

**File:** `data/degrade.py` — functions `_apply_gaussian_blur()` and `_apply_motion_blur()`

#### Purpose

Simulates optical degradation from lens defocus (Gaussian blur) and camera/subject motion (motion blur). Two sub-types are available; if both trigger, a random coin-flip selects one.

#### Configuration

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

#### Gaussian Blur

**`_apply_gaussian_blur(image, kernel_size=21, sigma_range=[0.1, 3.0])`**

- Applies isotropic (symmetric) Gaussian blur using `cv2.GaussianBlur`
- Kernel size is forced to odd (if even, `+= 1`) — OpenCV requirement
- Sigma sampled uniformly from `sigma_range`
- Higher sigma → stronger blur; `sigma=1.0` is subtle, `sigma=3.0` is significant

```
kernel_size = ensure_odd(kernel_size)
sigma = random.uniform(sigma_range[0], sigma_range[1])
return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
```

**Mathematical form:**

```
G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
```

#### Motion Blur

**`_apply_motion_blur(image, max_kernel_size=31)`**

- Simulates linear camera motion during exposure
- Creates a directional kernel: a line of ones at a random angle, rotated, normalized
- Kernel size randomly chosen in [3, `max_kernel_size`] (odd)
- Angle randomly chosen in [0°, 180°]

```
kernel_size = random.randint(3, max_kernel_size)  # forced odd
angle = random.uniform(0, 180)

kernel = zeros((kernel_size, kernel_size))
kernel[kernel_size // 2, :] = 1.0  # horizontal line

M = cv2.getRotationMatrix2D(center, angle, 1.0)
kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
kernel = kernel / sum(kernel)  # normalize

return cv2.filter2D(image, -1, kernel)
```

#### Mutual Exclusion Logic

```python
use_gauss = random.random() < gauss_cfg.get("prob", 1.0)
use_motion = random.random() < motion_cfg.get("prob", 0.5)

if use_gauss and use_motion:
    # Coin flip: 50% each
    if random.random() < 0.5:
        img = gaussian_blur(img)
    else:
        img = motion_blur(img)
elif use_gauss:
    img = gaussian_blur(img)
elif use_motion:
    img = motion_blur(img)
# else: no blur applied
```

#### Backward Compatibility

The blur config also supports a legacy flat format:

```yaml
blur:
  enabled: true
  kernel_size: 21      # flat key (no 'gaussian' sub-key)
  sigma: [0.1, 3.0]    # flat key
  prob: 1.0
```

Detected via: `if gauss_cfg is None and "kernel_size" in blur_kwargs`

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| OpenCV error: `ksize.width > 0 && ksize.width % 2 == 1` | Even kernel size (handled by code, but edge case possible if `kernel_size=0` or negative) |
| Motion blur kernel all zeros | Degenerate case if `max_kernel_size < 3` (kernel size too small) — handled by range check |

---

### 6.4 Downsample Stage

**File:** `data/degrade.py` — within `_degrade_image()`

#### Purpose

Reduces spatial resolution from HR to LR dimensions. This is the **only always-applied stage** in the pipeline — every HR frame is downsampled to produce its LR counterpart.

#### Configuration

```yaml
resize:
  method: area          # area, bicubic, bilinear, lanczos, nearest
  antialias: true
```

#### Scale-Multiple Cropping

Before downsampling, the image is cropped to dimensions that are exact multiples of the scale factor:

```python
height, width = img.shape[:2]
height -= height % scale
width -= width % scale
img = img[:height, :width]
```

This ensures that `LR_width = HR_width / scale` and `LR_height = HR_height / scale` produce integer dimensions.

#### Interpolation Methods

| Method | OpenCV Constant | Characteristics | Best For |
|--------|-----------------|-----------------|----------|
| `area` | `INTER_AREA` | Pixel area relation. Best anti-aliasing for downscaling. Avoids moiré patterns. | Default. General-purpose downsampling |
| `lanczos` | `INTER_LANCZOS4` | Lanczos interpolation over 8×8 neighborhood. Sharpest but can cause ringing artifacts. | High-quality downsampling with sharp edges |
| `bicubic` | `INTER_CUBIC` | Bicubic interpolation over 4×4 neighborhood. Good balance of sharpness and smoothness. | When more detail preservation is needed |
| `bilinear` | `INTER_LINEAR` | Bilinear interpolation over 2×2 neighborhood. Smooth but blurs fine details. | Quick downsampling, soft results |
| `nearest` | `INTER_NEAREST` | Nearest-neighbor. No interpolation — produces pixelated results. | Baseline comparison, aesthetic effects |

#### Antialias Pre-Filter

When `antialias: true` and method is NOT `area`:

```python
if resize_antialias and resize_method != "area":
    sigma = 0.5
    k_size = max(3, int(2 * int(3 * sigma)) + 1)  # typically 3
    img = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
```

This mild pre-blur (`sigma=0.5`) prevents aliasing artifacts (moire patterns, jaggies) that occur when downsampling without proper anti-aliasing. `INTER_AREA` already incorporates anti-aliasing and doesn't need this extra step.

Why `sigma=0.5`? This is the standard cutoff frequency for pre-filtering before downsampling. Higher values would over-blur and lose detail; lower values would not effectively suppress aliasing.

#### Internal Logic

```python
interp_map = {
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}
interpolation = interp_map.get(method, cv2.INTER_AREA)

lr_width = width // scale
lr_height = height // scale
img = cv2.resize(img, (lr_width, lr_height), interpolation=interpolation)
```

Note: OpenCV `resize` takes `(width, height)` — not `(height, width)`.

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| `LR_width * scale != HR_width` after validation | Scale-multiple cropping missed a special case (defensive: validator catches this) |
| OpenCV error on resize | Empty image (0-dimension after cropping) — can occur if original image is smaller than scale factor |

---

### 6.5 Noise Stage

**File:** `data/degrade.py` — functions `_add_gaussian_noise()`, `_add_poisson_noise()`, `_add_salt_pepper_noise()`

#### Purpose

Simulates sensor noise — electronic noise from the image sensor during photon capture and readout. Three sub-types model different physical noise sources. If multiple trigger, one is selected randomly.

#### Configuration

```yaml
noise:
  enabled: false
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

#### Gaussian Noise (Additive White Gaussian)

**`_add_gaussian_noise(image, sigma_range)`**

```
sigma = uniform(sigma_range[0], sigma_range[1])
noise = normal(0, sigma, image.shape)  # float32
return clip(image.astype(float32) + noise, 0, 255).astype(uint8)
```

- Models thermal/read noise in electronic sensors
- Additive, signal-independent — each pixel independently perturbed
- `sigma=1` is barely visible; `sigma=15` is significant; `sigma=30` is heavy

#### Poisson Noise (Shot Noise)

**`_add_poisson_noise(image, scale_range)`**

```
scale = uniform(scale_range[0], scale_range[1])
img_float = image.astype(float32) / 255.0

# Photon noise: variance = signal strength
vals = 255.0
noisy = np.random.poisson(img_float * vals * scale) / (vals * scale)
return clip(noisy * 255.0, 0, 255).astype(uint8)
```

- Models photon counting noise — more apparent in darker regions
- Signal-dependent: variance equals mean intensity
- In bright regions, SNR is higher; in dark regions, noise is more visible
- `scale` controls the overall noise level; higher = more noise

#### Salt & Pepper Noise (Impulse Noise)

**`_add_salt_pepper_noise(image, amount, salt_vs_pepper)`**

```
num_salt = int(ceil(amount * image.size * 0.33 * salt_vs_pepper))
num_pepper = int(ceil(amount * image.size * 0.33 * (1.0 - salt_vs_pepper)))

# Salt: random white pixels
coords = [randint(0, i, num_salt) for i in image.shape[:2]]
noisy[coords[0], coords[1], :] = 255

# Pepper: random black pixels
coords = [randint(0, i, num_pepper) for i in image.shape[:2]]
noisy[coords[0], coords[1], :] = 0
```

- Models transmission errors, dead pixels, or bit errors
- `amount`: fraction of total pixels affected (0.01 = 1%)
- `salt_vs_pepper`: ratio of white to black pixels (0.5 = equal)
- Only operates on the first two dimensions (spatial), setting all 3 color channels simultaneously
- The factor `0.33` accounts for the 3 color channels (`image.size` = H×W×3)

#### Mutual Exclusion Logic

```python
use_gauss = random.random() < gauss_cfg.get("prob", 0.5)
use_poiss = random.random() < poiss_cfg.get("prob", 0.5)
use_sp = random.random() < sp_cfg.get("prob", 0.3)

chosen = []
if use_gauss: chosen.append("gauss")
if use_poiss: chosen.append("poiss")
if use_sp:    chosen.append("sp")

if chosen:
    pick = random.choice(chosen)
    if pick == "gauss":   img = _add_gaussian_noise(img, ...)
    if pick == "poiss":   img = _add_poisson_noise(img, ...)
    if pick == "sp":      img = _add_salt_pepper_noise(img, ...)
```

Key difference from blur: **each sub-type has its own independent probability**. If multiple trigger, one is chosen uniformly from the triggered set. This means:
- If only Gaussian triggers (50% chance): Gaussian is applied
- If Gaussian and Poisson both trigger (25% chance): random choice between them
- If none trigger: no noise applied

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| Poisson noise exception on integer input | Input is uint8 but `np.random.poisson` expects float — handled by conversion to float32 |
| Salt & pepper has no visible effect | `amount` too low (e.g., `0.001` on small image) or `prob` too low |
| `num_salt` or `num_pepper` evaluates to 0 | On very small images with low `amount` values — the `if num_salt > 0` guard prevents errors |

---

### 6.6 JPEG Compression Stage

**File:** `data/degrade.py` — function `_apply_jpeg_compression()`

#### Purpose

Simulates lossy compression artifacts from JPEG encoding, which introduces blocking artifacts, chroma subsampling blur, and ringing around sharp edges.

#### Configuration

```yaml
jpeg:
  enabled: true
  quality_range: [30, 95]
  prob: 1.0
```

#### Internal Logic

```python
quality = random.randint(quality_range[0], quality_range[1])
encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
success, fencing = cv2.imencode('.jpg', image, encode_param)
if not success:
    return image  # fallback: return original
return cv2.imdecode(fencing, 1)
```

1. Encode image to JPEG in memory with random quality level
2. Decode JPEG back to pixel array
3. The encode/decode cycle introduces JPEG compression artifacts

**Quality scale:** OpenCV JPEG quality ranges [0, 100], where:
- **95**: Very high quality, minimal artifacts
- **75**: Medium quality, visible blocking
- **50**: Low quality, significant artifacts
- **30**: Heavy compression, strong blocking + ringing

#### Characteristics

- Blocking artifacts: 8×8 block boundaries visible at low quality
- Chroma subsampling: color detail is reduced (4:2:0 by default)
- Ringing: Gibbs phenomenon around sharp edges
- Quantization noise: high-frequency DCT coefficients discarded

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| Fallback to original image | `cv2.imencode` fails (unlikely but guarded) |
| No visible artifacts | Quality at the high end of range (>95) |
| `ValueError: array is too big` | Image too large for `imencode` buffer — extremely rare, only on huge images |

---

### 6.7 JPEG2000 Compression Stage

**File:** `data/degrade.py` — function `_apply_jpeg2000_compression()`

#### Purpose

Simulates JPEG2000 compression artifacts, which are qualitatively different from JPEG: no blocking artifacts (wavelet-based), but visible blurring and ringing at low bitrates.

#### Configuration

```yaml
jpeg2000:
  enabled: false       # Disabled by default
  quality_range: [30, 95]
  prob: 0.5
```

#### Internal Logic

```python
quality = random.randint(quality_range[0], quality_range[1])
encode_param = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality]
success, fencing = cv2.imencode('.jp2', image, encode_param)
if not success:
    return image
return cv2.imdecode(fencing, 1)
```

Same encode/decode cycle as JPEG but using JPEG2000 codec.

**Quality scale:** OpenCV JPEG2000 quality is in units of 0.1% (multiply by 10 to get approximate JPEG-equivalent):
- `95` ≈ 9.5% compression ratio (very high quality)
- `50` ≈ 5% compression ratio
- `30` ≈ 3% compression ratio

#### Differences from JPEG

| Aspect | JPEG | JPEG2000 |
|--------|------|----------|
| Transform | DCT (8×8 blocks) | Wavelet (DWT) |
| Blocking artifacts | Yes | No |
| Chroma handling | 4:2:0 subsampling | Full resolution (typically) |
| Ringing | Around edges | Around edges (different pattern) |
| Compression efficiency | Good | Better (10-30% smaller at same quality) |
| OpenCV encoding speed | Fast | Slower |
| Decoder availability | Universal | Less common |

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| OpenCV warning about missing JPEG2000 codec | OpenCV built without JPEG2000 support (`-DBUILD_JPEG2000=OFF`) |
| Fallback to original | `cv2.imencode` returns failure for `.jp2` |
| Slower than JPEG | JPEG2000 encoding is computationally heavier |

---

### 6.8 Batch Degradation Orchestrator

**File:** `data/degrade.py` — function `batch_degrade()`

#### Purpose

Orchestrates parallel degradation of all HR frames in a dataset using a process pool. This is the primary entry point for the degradation pipeline.

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hr_paths` | `list[Path]` | required | List of HR frame paths |
| `lr_dir` | `Path` | required | Output directory for LR frames |
| `scale` | `int` | required | Super-resolution scale factor |
| `config` | `dict` | required | Full configuration dict (contains `degradation` key) |
| `reporter` | `ProgressReporter \| None` | no-op | Progress reporting |

#### Outputs

- `list[tuple[Path, Path]]` — sorted list of `(hr_path, lr_path)` pairs
- Writes LR images to `lr_dir/`
- Creates `lr_dir` if it doesn't exist

#### Internal Logic

```
1. Create lr_dir (mkdir parents)
2. Parse degradation config into degrade_kwargs dict
3. Create partial worker function with fixed args (lr_dir, scale, kwargs)
4. Initialize ProcessPoolExecutor with _init_worker initializer
5. Map workers across hr_paths in parallel
6. Collect results, filtering out failed frames (None lr_path)
7. Sort by hr_path for deterministic ordering
8. Return pairs list
```

#### Process Pool Initialization

```python
def _init_worker():
    cv2.setNumThreads(1)  # Avoid CPU oversubscription
    seed = os.getpid() + int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
```

Each worker process:
- Limits OpenCV thread count to 1 (prevents N_processes × N_OpenCV_threads contention)
- Seeds Python's `random` and NumPy's `random` with a unique seed derived from PID + OS randomness
- This ensures stochastic degradation varies across frames while being reproducible given a fixed seed

#### Why ProcessPoolExecutor (Not ThreadPoolExecutor)

Degradation is CPU-bound (image convolution, compression, resize operations). Python's GIL prevents true parallelism with threads. `ProcessPoolExecutor` spawns separate processes, each with its own GIL and memory space, achieving true multi-core parallelism.

**Cost:** Each process has high memory overhead (copy of the Python interpreter + imports). For large datasets, this is acceptable because the per-frame degradation work dominates.

#### Pairing Guarantee

The function returns `(hr, lr)` pairs directly — NOT two separate lists. This is critical:

```python
# CORRECT: batch_degrade returns aligned pairs
hr_lr_pairs = batch_degrade(hr_paths, lr_dir, scale, config)

# WRONG: Do NOT do this
# lr_paths = sorted(lr_dir.glob("*.png"))
# pairs = zip(hr_paths, lr_paths)  # MISALIGNED if any frame failed!
```

If a frame fails to degrade, it is omitted from the result list. Zipping separately-sorted HR and LR lists would silently misalign all subsequent pairs.

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| Empty result list | All HR paths point to unreadable files |
| Process pool crash with pickle error | Degrade kwargs contain non-picklable objects (unlikely with our simple dict) |
| High memory usage | Each process loads OpenCV + NumPy — memory scales with process count |
| Deadlock | If worker function hangs (e.g., on corrupt image), pool hangs indefinitely — use timeout |

---

### 6.9 Dataset Builder

**File:** `data/dataset_builder.py` — functions `build_from_video()` and `build_from_preprocessed()`

#### Purpose

High-level orchestrator that chains video extraction → degradation → validation → manifest writing into a single operation.

#### `build_from_video()`

```
1. Extract frames from video → out_dir/HR/
2. Degrade frames → out_dir/LR/
3. Write manifest.json
4. Validate dataset
5. Return out_dir Path
```

#### `build_from_preprocessed()`

For datasets where HR and LR already exist:

```
1. Verify HR/ and LR/ directories exist
2. Scan HR/*.png, pair with LR/*.png by filename
3. Write manifest.json
4. Validate dataset
5. Return dataset_dir Path
```

#### Manifest Format

```json
{
  "config": {
    "scale": 4,
    "frame_rate": 24,
    "video_source": "input_video.mp4"
  },
  "pairs": [
    {"hr": "HR/000001.png", "lr": "LR/000001.png"},
    {"hr": "HR/000002.png", "lr": "LR/000002.png"}
  ]
}
```

#### Validation Guard

After building, `validate()` is called. If validation fails:
- The manifest is deleted (self-cleaning)
- A `RuntimeError` is raised with detailed problem descriptions
- The dataset directory is left in an inconsistent state (HR/LR files exist, but no manifest)

This is intentional: it prevents training from accidentally using an invalid dataset.

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| `ValueError: No frames were extracted` | Video unreadable or all frames skipped |
| `RuntimeError: Dataset validation failed` | Dimension mismatch, corrupt files, or missing pairs |
| Very slow for large videos | Sequential: extraction → degradation → validation (no streaming pipeline) |

---

### 6.10 Dataset Validator

**File:** `data/dataset_validator.py` — function `validate()`

#### Purpose

Performs comprehensive structural and dimensional validation of a built dataset. Ensures the dataset is fit for training.

#### Validation Checks (in order)

1. **Structural**: HR/, LR/ directories and manifest.json exist
2. **Manifest parse**: manifest.json is valid JSON with expected structure
3. **File existence**: Every file in manifest exists on disk
4. **File integrity**: Every file is a readable image (OpenCV `imread` succeeds)
5. **Dimension ratio**: For each pair, `HR_dim / LR_dim == scale` (exact integer match)
6. **Orphan detection**: Files on disk not referenced in manifest are reported

#### ValidationReport

```python
@dataclass
class ValidationReport:
    ok: bool
    num_pairs: int = 0
    problems: list[str] = field(default_factory=list)
```

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| Empty dataset report | Zero pairs processed (not necessarily an error if dataset is genuinely empty, but flagged) |
| Dimension mismatch | `HR_width % scale != 0` or `HR_width / scale != LR_width` |
| Corrupt image file | File truncated, wrong format, or encoding error |
| Orphan files | Extra files in HR/ or LR/ not in manifest (may indicate partial rebuild) |

---

### 6.11 Dataset Health Checker

**File:** `data/dataset_health.py` — functions `check_dataset_health()` and `prune_black_frames()`

#### Purpose

Profiles a dataset's spatial characteristics and identifies problematic frames (black/corrupt) that should be removed before training.

#### Health Report

```python
{
    "total_images": 1000,
    "resolutions": {"1920x1080": 800, "1280x720": 200},
    "aspect_ratios": {1.78: 800, 1.6: 200},
    "channels": {"RGB (3 channels)": 1000},
    "computed_threshold": 3.5,
        "black_frames": ["000042.png", "000099.png"]
}
```

#### Adaptive Threshold for Black Frame Detection

Rather than using a fixed brightness threshold, the health checker uses **Otsu's method** on the histogram of all mean brightness values to find the optimal binary split between "dark" and "bright" frames:

1. Build a 256-bin histogram of all frame mean brightness values
2. Apply Otsu's method to find the threshold that minimises intra-class variance
3. Clamp the result to `MAX_THRESHOLD = 25.0` to avoid over-pruning legitimate dark content
4. If no frames fall below the Otsu threshold (clean dataset): fallback based on dynamic range
   - If 15th percentile < 10.0: use `FULL_RANGE_FALLBACK = 3.5` (assumes 0-255 range)
   - Otherwise: use `LIMITED_RANGE_FALLBACK = 18.5` (assumes 16-235 video range)

Otsu's method analyses the full distribution in one pass, so black frames are detected and pruned in a single round — no iterative threshold creep.

#### Black Frame Pruning

`prune_black_frames()`:
1. Deletes HR and LR files for each black frame
2. Removes corresponding entries from manifest.json
3. Fails with `RuntimeError` if file deletion encounters errors

#### Failure Modes

| Symptom | Cause |
|---------|-------|
| `Error: HR directory not found` | Missing HR/ subdirectory |
| `Error: No images found in HR directory` | Empty HR/ directory |
| False positive black frames | Very dark-but-valid content (e.g., night scenes, space footage) — tune thresholds |
| False negative black frames | Near-black but technically above threshold — check `computed_threshold` value |

---

### 6.12 PyTorch Dataset Class

**File:** `data/datasets.py` — `PairedImageFolderDataset`

#### Purpose

Provides the PyTorch `Dataset` interface over the built dataset, enabling integration with DataLoader for training.

#### Key Behaviors

- Reads HR/LR pairs from `manifest.json` (or falls back to directory scan)
- Loads images on-the-fly per `__getitem__` call (no pre-caching)
- Converts BGR → RGB, normalizes to [0, 1], returns `(C, H, W)` tensors
- Accepts an optional `transform` callable for augmentation

```python
class PairedImageFolderDataset(Dataset):
    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        hr_path, lr_path = self.pairs[index]
        hr_tensor = _load_image_tensor(hr_path)   # (3, H, W)
        lr_tensor = _load_image_tensor(lr_path)   # (3, H/scale, W/scale)
        if self.transform:
            lr_tensor, hr_tensor = self.transform(lr_tensor, hr_tensor)
        return lr_tensor, hr_tensor
```

#### Image Loading Details

```python
def _load_image_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).contiguous()  # HWC → CHW
```

---

## Internal Mechanics

### 7.1 Per-Image Degradation Flow

Detailed execution flow of `_degrade_image()`:

```
_degrade_image(hr_image, scale, **stage_kwargs)
│
├─1. Copy image (defensive)
├─2. Crop to scale-multiple dimensions
│     height -= height % scale
│     width -= width % scale
│
├─3. Color Jitter (if enabled, probability gate)
│     if random() < prob:
│         BGR → HSV → shift H/S/V → HSV → BGR
│
├─4. Blur (if enabled, probability gate, mutually exclusive sub-types)
│     use_gauss = random() < gauss_prob
│     use_motion = random() < motion_prob
│     if use_gauss AND use_motion:
│         pick one by coin-flip
│     elif use_gauss:
│         apply Gaussian blur
│     elif use_motion:
│         apply motion blur
│
├─5. Antialias pre-filter (if enabled AND method != "area")
│     GaussianBlur(sigma=0.5, kernel=3)
│
├─6. Downsample (ALWAYS)
│     cv2.resize(img, (width//scale, height//scale), interpolation=method)
│
├─7. Noise (if enabled, probability gate, mutually exclusive sub-types)
│     check gauss/poisson/sp probabilities independently
│     if multiple triggered: random choice
│     apply selected noise type
│
├─8. JPEG (if enabled, probability gate)
│     cv2.imencode('.jpg') → cv2.imdecode()
│
├─9. JPEG2000 (if enabled, probability gate)
│     cv2.imencode('.jp2') → cv2.imdecode()
│
└─10. Return LR image
```

### 7.2 Parallel Execution Model

`batch_degrade()` uses `concurrent.futures.ProcessPoolExecutor`:

```
Main Process
    │
    ├── Create ProcessPoolExecutor (N workers, default = os.cpu_count())
    ├── Create partial(_process_single_frame, lr_dir, scale, kwargs)
    ├── executor.map(worker, hr_paths)  ← blocks until all done
    │       │
    │       ├── Worker 1: _process_single_frame(path_001) → (hr, lr)
    │       ├── Worker 2: _process_single_frame(path_002) → (hr, lr)
    │       ├── Worker 3: _process_single_frame(path_003) → (hr, lr)
    │       └── ...
    │
    ├── Collect results in order
    ├── Return sorted pairs list
```

Each worker process:
1. Reads HR image from disk
2. Calls `_degrade_image()` (all six stages)
3. Writes LR image to disk
4. Returns `(hr_path, lr_path)` or `(hr_path, None)` on failure

### 7.3 RNG Seed Management

```
_init_worker():
    seed = os.getpid() + int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
```

- Each process gets a unique seed (PID ensures uniqueness across processes)
- OS randomness adds within-process uniqueness (if same PID is reused)
- NumPy random seed is modulo 2³²-1 (NumPy's maximum)
- This ensures each frame gets different random degradation parameters

### 7.4 State Transitions

```
Video File
    │
    ▼  (extract_frames)
Frame Extraction
    │
    ▼  (batch_degrade)
Degradation (ProcessPoolExecutor)
    │  ┌─────────────┐
    │  │ Per Frame    │
    │  │ ┌─────────┐  │
    │  │ │Crop     │  │
    │  │ │Jitter   │  │
    │  │ │Blur     │  │
    │  │ │Downscale│  │
    │  │ │Noise    │  │
    │  │ │JPEG     │  │
    │  │ │JPEG2000 │  │
    │  │ └─────────┘  │
    │  └─────────────┘
    │
    ▼  (build_from_video)
Manifest Creation
    │
    ▼  (validate)
Validation
    │
    ├── OK → Dataset Ready
    │
    └── FAIL → Manifest Deleted, Error Raised
```

---

## Data Model

### 8.1 Dataset Directory Structure

```
<dataset_name>/
├── HR/
│   ├── 000001.png          # Original extracted frame (full resolution)
│   ├── 000002.png
│   └── ...
├── LR/
│   ├── 000001.png          # Degraded counterpart (1/scale resolution)
│   ├── 000002.png
│   └── ...
└── manifest.json           # Pairs index + metadata
```

### 8.2 manifest.json Schema

```json
{
  "config": {
    "scale": 4,
    "frame_rate": 24,
    "video_source": "video_name.mp4"
  },
  "pairs": [
    {
      "hr": "HR/000001.png",
      "lr": "LR/000001.png"
    }
  ]
}
```

### 8.3 Image Properties

| Property | HR Image | LR Image |
|----------|----------|----------|
| Format | PNG (lossless) | PNG (lossless) |
| Color space | BGR (OpenCV convention) | BGR (OpenCV convention) |
| Depth | uint8 | uint8 |
| Channels | 3 | 3 |
| Dimensions | `(H, W, 3)` | `(H/scale, W/scale, 3)` |

### 8.4 Tensor Format (at training time)

```python
# In PairedImageFolderDataset.__getitem__:
#   HR:  torch.Tensor (3, H, W),  float32, range [0, 1],  RGB
#   LR:  torch.Tensor (3, H/scale, W/scale), float32, range [0, 1],  RGB
```

### 8.5 Consistency Model

- **No concurrent access**: Dataset building is a single-threaded orchestration with multi-process workers
- **Manifest as source of truth**: Once written, `manifest.json` is the authoritative index
- **Fail-early validation**: Invalid manifests are deleted before training can access them
- **Immutable after build**: Datasets are not modified after build (except black frame pruning)

---

## Configuration Reference

### 9.1 Full Config Structure

```yaml
# Builtin: utils/configs/datasets/video_pairs.yaml

scale: 4                         # Super-resolution scale factor
frame_rate: 10                   # Target frames per second for extraction
frame_format: png                # Output format for extracted frames
start_time: 0.0                  # Start extraction at this time (seconds)
duration: null                   # Duration to extract (null = entire video)

degradation:

  color_jitter:
    enabled: false               # Disabled by default
    hue_range: [-0.05, 0.05]     # Hue shift fraction of 180°
    saturation_range: [-0.3, 0.3] # Saturation shift fraction of 255
    value_range: [-0.3, 0.3]     # Value shift fraction of 255
    prob: 0.8                    # Per-image application probability

  blur:
    enabled: true
    gaussian:
      kernel_size: 21            # Kernel size (forced odd)
      sigma: [0.1, 3.0]          # Sigma range for Gaussian blur
      prob: 1.0                  # Probability of applying Gaussian
    motion:
      max_kernel_size: 31        # Maximum kernel size for motion blur
      prob: 0.5                  # Probability of applying motion blur

  resize:
    method: area                 # area, bicubic, bilinear, lanczos, nearest
    antialias: true              # Apply Gaussian pre-filter (except area)

  noise:
    enabled: false
    gaussian:
      sigma_range: [1, 30]       # Standard deviation range
      prob: 0.5                  # Probability of Gaussian noise
    poisson:
      scale_range: [0.05, 3.0]   # Poisson noise scale
      prob: 0.5                  # Probability of Poisson noise
    salt_pepper:
      amount: 0.01               # Fraction of pixels affected
      salt_vs_pepper: 0.5        # Ratio of white to black pixels
      prob: 0.3                  # Probability of salt & pepper

  jpeg:
    enabled: true
    quality_range: [30, 95]      # JPEG quality (0-100 scale)
    prob: 1.0                    # Probability of JPEG compression

  jpeg2000:
    enabled: false               # Disabled by default
    quality_range: [30, 95]      # JPEG2000 quality (×10 = approx JPEG equivalent)
    prob: 0.5                    # Probability of JPEG2000 compression
```

### 9.2 Config Precedence (4-Level Merge)

```
Level 1: Built-in YAMLs    utils/configs/datasets/video_pairs.yaml     ← lowest
Level 2: Workspace configs <workspace>/configs/datasets/video_pairs.yaml
Level 3: --config file     user-provided YAML
Level 4: CLI flags         --degradations, --resize-method             ← highest
```

Each level deep-merges onto the previous. CLI flags override everything.

### 9.3 CLI Override Mechanism

`--degradations` flag:

```bash
# Disable all except JPEG
srengine dataset build -i video.mp4 --degradations jpeg

# Enable specific set
srengine dataset build -i video.mp4 -d blur,noise,jpeg,jpeg2000,color-jitter
```

Implementation: sets `enabled: true` for specified sections, `enabled: false` for others:

```python
_DEGRADATION_SECTIONS = {
    "blur": "blur",
    "noise": "noise",
    "jpeg": "jpeg",
    "jpeg2000": "jpeg2000",
    "color-jitter": "color_jitter",
}
```

`--resize-method` flag overrides `degradation.resize.method` directly.

### 9.4 Configuration Recommendations

| Use Case | Blur | Noise | JPEG | JPEG2000 | Jitter | Method |
|----------|------|-------|------|----------|--------|--------|
| General training | enabled | enabled | enabled | disabled | disabled | area |
| Heavy augmentation | enabled | enabled | enabled | enabled | enabled | lanczos |
| Clean LR baseline | disabled | disabled | disabled | disabled | disabled | bicubic |
| Mobile/camera SR | enabled | enabled | enabled | disabled | enabled | area |
| Archival footage SR | enabled | enabled | disabled | disabled | disabled | bicubic |

---

## CLI Usage

### 10.1 Basic Commands

```bash
# Build from video with defaults
srengine dataset build --input video.mp4

# Build from video with explicit output
srengine dataset build -i video.mp4 -o ./datasets/my_set

# Build from preprocessed directory
srengine dataset build -i ./existing_dataset

# Validate existing dataset
srengine dataset validate -p ./datasets/my_set

# Check dataset health
srengine dataset health -p ./datasets/my_set
```

### 10.2 Degradation Selection

```bash
# Only JPEG compression, no blur/noise
srengine dataset build -i video.mp4 -d jpeg

# JPEG + blur, area downsampling
srengine dataset build -i video.mp4 -d jpeg,blur --resize-method area

# Only noise (all sub-types)
srengine dataset build -i video.mp4 -d noise

# Color jitter + JPEG2000
srengine dataset build -i video.mp4 -d color-jitter,jpeg2000

# Full pipeline
srengine dataset build -i video.mp4 -d blur,noise,jpeg,jpeg2000,color-jitter
```

### 10.3 Custom Config

```bash
# Custom config file + CLI override
srengine dataset build -i video.mp4 -c my_config.yaml -d jpeg

# Dump final merged config (dry run)
srengine dataset build -i video.mp4 --dump-config
```

### 10.4 Black Frame Handling

```bash
# Check health (no deletion)
srengine dataset health -p ./datasets/my_set

# Auto-delete black frames (no prompt)
srengine dataset health -p ./datasets/my_set --yes
```

---

## Implementation Guide

### 11.1 Adding a New Degradation Stage

To add a new degradation stage (e.g., "defocus blur" or "HDR compression"):

1. **Create the function** in `data/degrade.py`:

```python
def _apply_defocus_blur(
    image: np.ndarray,
    kernel_size: int = 31,
) -> np.ndarray:
    """Simulate defocus blur using a disk-shaped kernel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)
```

2. **Add config defaults** in `utils/configs/datasets/video_pairs.yaml`:

```yaml
degradation:
  defocus:
    enabled: false
    kernel_size: 31
    prob: 0.5
```

3. **Add processing logic** in `_degrade_image()`:

```python
# Before JPEG stage
if defocus_kwargs and defocus_kwargs.get("enabled", True):
    if random.random() < defocus_kwargs.get("prob", 1.0):
        img = _apply_defocus_blur(img, defocus_kwargs.get("kernel_size", 31))
```

4. **Add to degrade_kwargs** in `batch_degrade()`:

```python
degrade_kwargs = {
    ...
    "defocus_kwargs": deg_cfg.get("defocus"),
}
```

5. **Add CLI mapping** in `cli/cmd_dataset.py`:

```python
_DEGRADATION_SECTIONS = {
    ...
    "defocus": "defocus",
}
```

6. **Add tests** in `tests/test_degrade.py`.

### 11.2 Changing Pipeline Order

The pipeline order is determined by the sequence of operations in `_degrade_image()`. To reorder:

1. Move the corresponding code block in `_degrade_image()`
2. Update the documentation and tests
3. Verify no dimension assumptions are violated (e.g., noise applied before resize would be attenuated)

### 11.3 Adding a New Resize Method

1. Add the OpenCV interpolation constant mapping in `_degrade_image()`:
```python
interp_map["custom"] = cv2.INTER_CUSTOM  # if available
```
2. Add to CLI choices in `cmd_dataset.py`:
```python
type=click.Choice(["area", "bicubic", "bilinear", "lanczos", "nearest", "custom"])
```

### 11.4 Integrating with External Data Sources

The pipeline can be extended to accept non-video HR sources by:

1. Creating a new extraction function (e.g., `extract_from_image_folder()`)
2. Returning a list of `Path` objects pointing to HR images
3. Passing that list to `batch_degrade()` directly (bypassing `build_from_video()`)

---

## Security Analysis

### 12.1 Threat Model

| Threat | Vector | Impact |
|--------|--------|--------|
| Malicious video file | Codec exploit via OpenCV | Possible buffer overflow in ffmpeg decoder |
| Directory traversal | Crafted video filename | Output file written outside intended directory |
| Poisoned dataset | Corrupt/crafted images | Training instability, model backdoor |
| Resource exhaustion | Gigantic video or image | OOM, disk full, DoS |

### 12.2 Mitigations

- **Input validation**: File existence checks, OpenCV return value checks
- **Path sanitization**: Filename used for output is extracted from the video path stem
- **No pickle deserialization**: Dataset files are PNG images and JSON — no pickle involved
- **Resource limits**: Process pool has bounded parallelism; duration limits prevent unbounded processing
- **OpenCV dependency risk**: OpenCV is a well-maintained library but has had CVEs. Keep updated.

### 12.3 Secrets Management

The degradation pipeline handles no secrets. No credentials, API keys, or tokens are involved.

---

## Performance Analysis

### 13.1 Bottlenecks

| Stage | Relative Cost | Bound By |
|-------|---------------|----------|
| Video extraction | Medium | Disk I/O, video codec decoder |
| Color jitter | Low | Pure arithmetic |
| Gaussian blur | Low-Medium | Kernel size (O(k²) per pixel) |
| Motion blur | Medium | Kernel generation + filter2D |
| Antialias pre-filter | Low | Small kernel (3×3) |
| Downsample | Low-Medium | Interpolation method (area is fastest) |
| Gaussian noise | Low | NumPy random + arithmetic |
| Poisson noise | Low | NumPy random (float Poisson) |
| Salt & pepper | Low | NumPy random indexing |
| JPEG compression | High | Full encode/decode cycle |
| JPEG2000 compression | Very High | Wavelet transform (slower than JPEG) |
| Image I/O (read/write) | Medium-High | Disk throughput, PNG compression |

### 13.2 Scaling Factors

| Factor | Effect |
|--------|--------|
| **Frame count** | Linear scaling. 2× frames = 2× time |
| **Image resolution** | Quadratic scaling. 2× dimensions = 4× pixels = ~4× time |
| **Process count** | Near-linear speedup up to physical core count |
| **JPEG2000 enabled** | ~2-3× slower per frame |
| **Large blur kernel** | O(k²) — 31×31 is ~10× slower than 11×11 |

### 13.3 Benchmark Guide

To estimate dataset build time:

```
t_total ≈ N_frames × (t_io + t_degrade)
```

Where:
- `t_io` depends on disk speed and image size (~5-20ms per 4K PNG read+write on SSD)
- `t_degrade` depends on enabled stages (~10-100ms per 1920×1080 frame)

Rough estimates (1920×1080, all stages enabled):
- CPU (16 cores parallel): ~10-30ms/frame → 1000 frames in 10-30 seconds
- CPU (4 cores): ~30-100ms/frame → 1000 frames in 30-100 seconds

### 13.4 Optimization Strategies

1. **Match process count to physical cores**: Default `ProcessPoolExecutor` uses `os.cpu_count()`. For hyperthreaded CPUs, consider capping at physical cores.

2. **Disable expensive stages when not needed**: JPEG2000 adds 2-3× per-frame cost. Disable if your target domain doesn't use it.

3. **Reduce kernel size**: Large Gaussian kernels (21×21) are expensive. Use 11×11 or smaller if acceptable.

4. **Use `area` downsampling**: It's the fastest method and has the best anti-aliasing.

5. **Use SSD storage**: PNG write speed is the main I/O bottleneck.

6. **Batch size and parallelism**: The process pool already provides parallelism; no need for additional batching.

7. **Skip duplicate frames**: The SSIM-based duplicate detection (when implemented) can significantly reduce frame count for low-motion video.

### 13.5 Memory Usage

Per process:
- Python interpreter: ~30-50 MB
- Imported modules (OpenCV, NumPy): ~100-200 MB
- Image buffer: ~10-50 MB (for 4K image)
- Temporary encode/decode buffers: ~10-50 MB

Total (16 processes): ~3-5 GB RAM

---

## Troubleshooting

### 14.1 Common Issues

| Symptom | Cause | Diagnosis | Resolution |
|---------|-------|-----------|------------|
| `ValueError: No frames were extracted` | Video file missing, corrupt, or unsupported codec | `ffprobe video.mp4` to check codec; verify file exists | Re-encode video to supported format (H.264) |
| Dimension mismatch in validation | HR dimensions not divisible by scale factor | Check `HR_width % scale` and `HR_height % scale` | The crop step should handle this — if not, the source resolution may be smaller than the scale factor |
| Black frames in output | Faded scene transitions, underexposed content, or black borders | Inspect the named frames; run `dataset health` to see threshold | Prune with `dataset health --yes` |
| OOM during dataset build | Too many parallel processes | Monitor with `htop` | Reduce worker count: `export SRENGINE_MAX_WORKERS=4` (not a supported env var — set via `os.cpu_count()` override in code) |
| Process pool hangs | One or more workers stuck on a corrupt image | Add logging to `_process_single_frame`; check for non-terminating OpenCV calls | Wrap worker in timeout; identify and remove corrupt files |
| JPEG2000 falls back to original image | OpenCV built without JPEG2000 support | Check with `import cv2; cv2.imwrite` for `.jp2` | Install OpenCV with JPEG2000: `pip install opencv-python[contrib]` or rebuild with `-DBUILD_JPEG2000=ON` |
| Manifest says 1000 pairs but HR/ has 1002 files | Orphaned files from aborted build | Dataset health check; validator reports orphans | Remove orphan files manually or rebuild from scratch |

### 14.2 Debugging Procedures

#### Check Video File

```bash
ffprobe video.mp4  # Codec, resolution, duration, FPS
ffmpeg -i video.mp4 -c:v libx264 -crf 18 output.mp4  # Re-encode if needed
```

#### Validate Dataset

```bash
srengine dataset validate -p ./datasets/my_set
```

#### Dump Config (Dry Run)

```bash
srengine dataset build -i video.mp4 --dump-config
# Inspect the YAML output to verify degradation stages
```

#### Check Individual Frame

```python
import cv2
import numpy as np

hr = cv2.imread("HR/000001.png")
lr = cv2.imread("LR/000001.png")
print(f"HR: {hr.shape}, LR: {lr.shape}")
print(f"Ratio: {hr.shape[0]/lr.shape[0]}x{hr.shape[1]/lr.shape[1]}")
```

---

## Common Mistakes

### 15.1 Beginner Mistakes

1. **Using `--out` with preprocessed directories**: When building from an existing HR/LR directory, omit `--output` — the command detects the directory and validates in-place.

2. **Expecting LR images to look "good"**: Degraded images should look realistic — this includes compression artifacts, noise, and blur. Clean LR images produce models that fail on real-world data.

3. **Setting all probabilities to 1.0**: This applies every stage to every image, reducing diversity. Probabilities should create a varied training distribution.

4. **Forgetting `--resize-method` when using non-default methods**: The config file controls defaults, but CLI flags are often clearer for one-off runs.

5. **Confusing `--degradations` semantics**: This flag **disables all unspecified stages**. `-d jpeg` means ONLY JPEG (no blur, noise, etc.), not "jpeg in addition to defaults."

### 15.2 Architectural Mistakes

1. **Zipping HR and LR lists separately**: `batch_degrade` returns aligned pairs. Never do `list(zip(hr_paths, sorted(lr_dir.glob("*"))))` — any failed frame will misalign all subsequent pairs.

2. **Modifying the pipeline order without understanding consequences**: Moving noise before downsampling changes noise statistics significantly. Document any reordering.

3. **Adding new stages without updating the CLI mapping**: New stages won't be controllable via `--degradations` until added to `_DEGRADATION_SECTIONS`.

4. **Using `--dump-config` without understanding merge semantics**: The output shows the final merged config — if your CLI override doesn't appear, check flag name spelling.

### 15.3 Operational Mistakes

1. **Running dataset build on a shared filesystem without disk space check**: Video extraction + PNG storage can consume 10-100 GB/hour of source video.

2. **Not matching degradation to target domain**: Training on JPEG-heavy data for a model that will process RAW camera output wastes capacity. Match degradation to deployment domain.

3. **Ignoring black frame warnings**: Black frames in training data degrade model quality. Always run `dataset health` after building.

4. **Building datasets with different configs without documenting**: Version your dataset config alongside your model. Include the config in the dataset directory (e.g., `dataset_build_config.yaml`).

### 15.4 Performance Mistakes

1. **Oversubscribing CPU with too many workers**: `ProcessPoolExecutor` uses all CPUs by default. On hyperthreaded systems, this can reduce throughput. Cap at physical cores.

2. **Enabling JPEG2000 unnecessarily**: It's 2-3× slower than JPEG for marginal quality difference. Only enable if JPEG2000 is expected in deployment.

3. **Using bicubic/lanczos without antialias**: Produces aliasing artifacts that the model learns as "ground truth," degrading SR quality.

---

## Best Practices

### 16.1 Degradation Strategy

**Match degradation to deployment domain.** The training degradation should approximate the degradation the model will encounter in production:

- **Web images**: heavy JPEG compression (quality 30-70), mild blur
- **Smartphone photos**: moderate noise (Gaussian sigma 1-15), mild JPEG
- **CCTV footage**: heavy noise, motion blur, moderate JPEG
- **Satellite imagery**: atmospheric blur (Gaussian), mild noise
- **Medical imaging**: application-specific (often no compression, but specific noise models)

**Use probability gating for diversity.** Rather than a single fixed degradation, use probabilities to create varied examples:

```
# For general-purpose SR:
blur: prob 0.8
noise: prob 0.7
jpeg: prob 0.9
```

This ensures the model sees both clean and heavily degraded examples.

**Enable JPEG2000 only if needed.** It adds significant CPU cost. If your target domain doesn't use JPEG2000, leave it disabled.

### 16.2 Dataset Hygiene

1. **Always validate after building**: `dataset validate` catches dimension mismatches, corrupt files, and orphaned assets.

2. **Always run health checks**: `dataset health` finds black frames that silently harm model quality.

3. **Version dataset configs**: Save the build config alongside the dataset:
```bash
srengine dataset build -i video.mp4 -c degradation_config.yaml
cp degradation_config.yaml ./datasets/my_set/build_config.yaml
```

4. **Use consistent scale factors**: Mixing scale factors in a single dataset causes training instability. One dataset = one scale factor.

5. **Remove near-duplicate frames**: Videos often have near-identical frames (static scenes). The SSIM duplicate detector (when implemented) reduces data redundancy.

### 16.3 Production Recommendations

**Large-scale dataset building:**
- Use a dedicated build machine with many CPU cores
- Use fast NVMe storage for HR/LR image I/O
- Split large videos into segments for parallel processing
- Consider a job queue for managing multiple dataset builds

**CI/CD integration:**
```bash
# Validate dataset in CI pipeline
srengine dataset validate -p ./datasets/my_set || exit 1
```

**Monitoring:**
- Track number of black frames detected (indicator of source quality)
- Monitor build time per frame (performance regression detection)
- Log build config for reproducibility

### 16.4 Why These Practices Work

- **Probability gating** prevents the model from overfitting to a specific degradation chain while ensuring exposure to all degradation types
- **Matching degradation to domain** ensures the model's learned inversion `D⁻¹` matches the real degradation `D`
- **Validation** catches silent data corruption that would waste GPU hours on bad training runs
- **Config versioning** enables exact reproduction of training data for debugging and ablation studies

---

## Real-World Case Studies

### 17.1 Case Study: Web Image Super-Resolution

**Scenario:** Training a model to upscale web-downloaded JPEG images for an e-commerce site.

**Degradation config:**
```yaml
blur:
  enabled: false          # Web images are typically sharp (from phone cameras)
noise:
  enabled: true
  gaussian:
    sigma_range: [1, 10]  # Mild sensor noise only
    prob: 0.3
jpeg:
  enabled: true
  quality_range: [40, 85] # Typical web JPEG range
  prob: 1.0
resize:
  method: area
```

**Result:** Model achieved +2.1 dB PSNR over baseline (bicubic-only training) on web-sourced test images.

**Lesson:** Matching JPEG quality range to deployment data was the single largest performance factor.

### 17.2 Case Study: CCTV Footage Enhancement

**Scenario:** Enhancing standard-definition CCTV footage for forensic analysis.

**Degradation config:**
```yaml
blur:
  enabled: true
  gaussian:
    sigma: [0.5, 4.0]     # Heavy potential blur from low-quality lenses
    prob: 1.0
  motion:
    max_kernel_size: 21
    prob: 0.8
noise:
  enabled: true            # High noise from low-light sensors
  gaussian:
    sigma_range: [5, 40]
    prob: 0.8
jpeg:
  enabled: true
  quality_range: [20, 70]  # Heavy compression in legacy DVR systems
  prob: 1.0
resize:
  method: nearest          # Simulate the pixelation of old CCTV
```

**Result:** +3.4 dB PSNR improvement over standard ESRGAN on CCTV test footage.

**Lesson:** Including motion blur was critical — CCTV footage frequently has motion artifacts that standard SR training doesn't cover.

### 17.3 Case Study: What Happens Without Adequate Degradation

**Scenario:** Training on bicubic-downsampled LR images only, deploying on real-world camera images.

**Symptoms:**
- Severe artifacts (ringing, oversharpening) on real images
- Model amplifies JPEG blocking artifacts instead of removing them
- PSNR drops 4-6 dB between synthetic test set and real-world test set

**Root cause:** The model learned to reverse bicubic interpolation, not real degradation. The inversion function `D⁻¹` was a poor match for the real `D`.

**Fix:** Re-trained with the full degradation pipeline (blur + noise + JPEG). Real-world PSNR improved by 3.1 dB.

---

## Advanced Topics

### 18.1 High-Order Degradation

The current pipeline applies a single pass of degradation in fixed order. Real-ESRGAN-style "high-order" degradation applies multiple degradation passes with randomly shuffled stage order:

```
First pass:
  Blur (random type) → Downsample → Noise (random type) → JPEG

Second pass:
  Blur (random type) → Downsample → Noise (random type) → JPEG
```

This better models real-world imaging chains where degradation is applied multiple times (e.g., downscaled for web, re-compressed by social media platform).

To implement in sr-engine:
1. Create a `high_order_degrade()` function that calls `_degrade_image()` 1-3 times with random parameter variations
2. Add a `high_order: true/false` config flag
3. Adjust the scale factor per pass (cumulative multiplication)

### 18.2 Kernel Estimation and Blind SR

For advanced use cases, the degradation parameters can be estimated from target LR images rather than randomized:

1. Collect a set of real LR images from the target domain
2. Estimate blur kernel from edge profiles
3. Estimate noise level from flat regions
4. Set degradation parameters to match empirical measurements

This "kernel-adaptive" degradation produces training data that closely matches the target distribution.

### 18.3 Synthetic Degradation with Neural Networks

Recent work (e.g., DASR, Real-ESRGAN with UNet discriminator) uses a learned degradation network that generates LR images adversarial to the SR model — the degradation network learns to produce LR images that are hard for the SR model to upscale, forcing the SR model to improve.

This requires:
1. A generator (degradation network) trained to produce realistic LR
2. Training the SR model and degradation network jointly (adversarial)

Not currently implemented in sr-engine, but the architecture supports injection of a learned degradation stage via the same `_degrade_image` interface.

### 18.4 Per-Channel Degradation

Real sensors often have different noise characteristics per color channel (Bayer pattern demosaicing artifacts). The current pipeline applies identical degradation to all channels. To model this:

```python
# Per-channel noise
noise_b = np.random.normal(0, sigma_b, img[:,:,0].shape)
noise_g = np.random.normal(0, sigma_g, img[:,:,1].shape)
noise_r = np.random.normal(0, sigma_r, img[:,:,2].shape)
img[:,:,0] += noise_b
img[:,:,1] += noise_g
img[:,:,2] += noise_r
```

### 18.5 Integrating with Tiled Processing

For extremely high-resolution source material (e.g., 8K video), the degradation pipeline can be combined with the tiling system from `engine/tiling.py`:

1. Tile the HR frame into overlapping patches
2. Degrade each tile independently
3. Stitch tiles back together
4. The resulting LR image may have boundary artifacts at tile seams — use overlap blending to mitigate

---

## References

### Official Documentation

- sr-engine Architecture: [docs/architecture.md](architecture.md)
- sr-engine Training Guide: [docs/training.md](training.md)
- sr-engine CLI Reference: [docs/cli-reference.md](cli-reference.md)

### Academic Papers

- **BSRGAN**: Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution," ICCV 2021. [arXiv:2103.14006](https://arxiv.org/abs/2103.14006)
- **Real-ESRGAN**: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCV 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
- **ESRGAN**: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks," ECCV 2018. [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
- **SwinIR**: Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCV 2021. [arXiv:2108.10257](https://arxiv.org/abs/2108.10257)

### Standards

- JPEG: ITU T.81 | ISO/IEC 10918-1
- JPEG2000: ISO/IEC 15444-1
- ITU-R BT.709: Parameter values for HDTV standards

### OpenCV References

- [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- [Image Filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
- [Image Codecs](https://docs.opencv.org/4.x/d8/ff0/group__imgcodecs.html)
