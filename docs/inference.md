# Inference

## Overview

The inference pipeline takes a trained model checkpoint and applies it to images or videos. It supports tiled inference for VRAM-constrained GPUs and multiple export formats for deployment.

## Image Inference

`engine/inference.py:infer_image()`

```python
infer_image(
    model,
    image_path: str,
    output_path: str,
    device="cuda",
    tile=0,        # 0 = no tiling
    overlap=64
) → None
```

Pipeline:
1. Load image via OpenCV (supports PNG, JPG, BMP, TIFF)
2. Convert BGR → RGB, normalize to `[0, 1]`
3. Move to device
4. If tiling enabled: `tile → upscale each tile → stitch`
5. If no tiling: full-frame forward pass
6. Convert back to BGR, denormalize, save

### With Tiling

```
Input image (e.g. 1920×1080)
      │
      ▼
  tile_image()
      │
      ├── Split into overlapping tiles (e.g. 512×512, overlap 64)
      ├── Pad edge tiles to uniform size
      └── Return list of (tile, x, y) with positions
      │
      ▼
  For each tile:
      │
      ├── Move tile to device
      ├── model(tile) → upscaled_tile
      └── Move back to CPU
      │
      ▼
  stitch_tiles()
      │
      ├── Place upscaled tiles at correct positions
      ├── Blend overlapping regions (linear ramp)
      └── Return full-resolution output
      │
      ▼
  Output image (e.g. 7680×4320 for 4× SR)
```

Tiling is essential for large inputs on GPUs with limited VRAM:

| GPU VRAM | Max full-frame (SwinIR, 4×) | Tiled (512px, 64 overlap) |
|----------|------------------------------|---------------------------|
| 8 GB | ~1280×720 | 4K+ inputs |
| 16 GB | ~1920×1080 | 8K+ inputs |
| 24 GB | ~3840×2160 | Any resolution |

## Video Inference

`engine/inference.py:infer_video()`

```python
infer_video(
    model,
    video_path: str,
    output_path: str,
    device="cuda",
    tile=0,
    overlap=64
) → None
```

Pipeline:
1. Open video with OpenCV `VideoCapture`
2. Read video properties: FPS, total frames, codec, resolution
3. Initialize `VideoWriter` with upscaled resolution
4. For each frame:
   - Extract frame as image
   - Run `infer_image()` on the frame
   - Write upscaled frame to output video
5. Release resources

Frame-by-frame processing means total time = `num_frames × time_per_frame`. For long videos, consider GPU memory is freed between frames (no gradient graph retained).

## Tiling System

`engine/tiling.py` implements the tile/stitch pattern for VRAM-efficient inference.

### `tile_image()`

```python
tile_image(image, tile_size=512, overlap=64) → list[tuple[Tensor, int, int]]
```

- Divides the input image into overlapping tiles
- Edge tiles are padded to `tile_size × tile_size` (reflection padding)
- Returns `(tile_tensor, x_offset, y_offset)` for each tile

**Tile count formula for an N×M image:**
```
num_tiles_x = ceil((N - overlap) / (tile_size - overlap))
num_tiles_y = ceil((M - overlap) / (tile_size - overlap))
total_tiles = num_tiles_x × num_tiles_y
```

Example for 1920×1080, tile=512, overlap=64:
```
num_tiles_x = ceil((1920 - 64) / (512 - 64)) = ceil(1856/448) = 5
num_tiles_y = ceil((1080 - 64) / (512 - 64)) = ceil(1016/448) = 3
total = 15 tiles
```

### `stitch_tiles()`

```python
stitch_tiles(tiles, output_size, overlap=64) → Tensor
```

- Places each upscaled tile at its original position (scaled by the model's scale factor)
- Creates a weight map: linear ramp from 0→1 at each overlap region
- Blends overlapping areas using normalized weighted average
- Crops any padding added during tiling

## Model Export

`models/checkpoint.py` provides three export formats for deployment outside the training framework.

### ONNX

```bash
srengine model export --model-name swinir --ckpt model.pth --format onnx --out model.onnx
```

- Standard ONNX format for cross-platform inference
- Tested with ONNX Runtime (CPU and CUDA providers)
- Input shape: `(1, 3, H, W)`, output: `(1, 3, H*scale, W*scale)`
- Dynamic axes for batch size and spatial dimensions
- Requires `torch.onnx.export()` with model tracing

### SafeTensors

```bash
srengine model export --model-name rrdb_esrgan --ckpt model.pth --format safetensors --out model.safetensors
```

- Zero-copy weight format (no pickle, safe from arbitrary code execution)
- Compatible with HuggingFace `safetensors` library
- Stores raw tensors only — no optimizer state or config metadata
- Use for weight distribution and sharing

### TorchScript

```bash
srengine model export --model-name swinir --ckpt model.pth --format torchscript --out model.pt
```

- TorchScript `ScriptModule` for C++ `libtorch` inference
- Traced model (requires dummy input)
- No Python dependency at inference time
- Use for production deployment in C++ environments

## Model Info

Inspect checkpoint contents without running training or inference:

```bash
srengine model info --model model.pth
# Checkpoint: model.pth
#   Step:      45000
#   Config:    {'name': 'swinir', 'type': 'swinir', ...}
```

Lists model architecture, step count, and the config snapshot saved at training time.
