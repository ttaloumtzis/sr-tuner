# Training

## Trainer Architecture

The `Trainer` class in `engine/trainer.py` implements an epoch-based training loop with a callback system for extensibility.

### Lifecycle

```
Trainer.run(dataset, model, config)
    │
    ├── on_phase("training")
    │
    ├── For each epoch 1..max_epochs:
    │     ├── Training loop:
    │     │     ├── model.train()
    │     │     ├── For each batch:
    │     │     │     ├── forward(lr) → sr
    │     │     │     ├── loss = pixel_loss(sr, hr) + perceptual_loss(sr, hr) + gan_loss(...)
    │     │     │     ├── backward() → optimizer.step()
    │     │     │     └── on_step(epoch, batch, loss, lr)
    │     │     └── scheduler.step()
    │     │
    │     └── Validation (every save_per_epoch):
    │           ├── model.eval()
    │           ├── For each batch: forward → psnr, ssim
    │           └── on_validate(epoch, psnr, ssim)
    │
    ├── on_phase("complete")
    └── on_done(elapsed_seconds, total_epochs)
```

### Callback System

```python
class TrainerCallback:
    def on_phase(self, phase: str, **kwargs) -> None: ...
    def on_step(self, epoch: int, batch: int, total_batches: int,
                loss_total: float, loss_pixel: float,
                loss_perceptual: Optional[float], lr: float) -> None: ...
    def on_validate(self, epoch: int, psnr: float, ssim: float) -> None: ...
    def on_done(self, elapsed: float, total_epochs: int) -> None: ...
```

Callbacks are attached via `trainer.add_callback(callback)`. Multiple callbacks can run simultaneously for different purposes:

| Callback | Module | Purpose |
|----------|--------|---------|
| `_MetricsStreamCallback` | `engine/trainer.py` | Writes JSONL metrics file |
| `SocketCallback` | `gui_bridge/protocol.py` | Streams events to GUI over TCP |
| `TqdmReporter` | `utils/progress.py` | Terminal progress bar |

## Available Models

### RRDB (ESRGAN-style)

`models/archs/rrdbnet.py` — Residual-in-Residual Dense Block network.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_feat` | 64 | Base feature channels |
| `num_block` | 23 | RRDB blocks |
| `num_grow_ch` | 32 | Growth channels per dense block |
| `scale` | 4 | Upscaling factor |

Architecture sketch:
```
Input LR
  └── Conv 3×3 (num_feat)
       └── [RRDB Block] × num_block
            └── Conv 3×3 → Residual
                 └── Upsample (pixelshuffle) × 2 (for 4×)
                      └── Conv 3×3 → HR output
```

Each RRDB block contains 3 dense blocks with residual connection and residual scaling (`beta=0.2`).

### SwinIR

`models/archs/swinir.py` — Transformer-based architecture using Swin Transformer blocks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 180 | Embedding dimension |
| `depths` | [6,6,6,6,6,6] | Transformer blocks per stage |
| `num_heads` | [6,6,6,6,6,6] | Attention heads per stage |
| `window_size` | 8 | Local window size |
| `scale` | 4 | Upscaling factor |

Architecture:
```
Input LR
  └── Conv 3×3 embedding (embed_dim)
       └── SwinTransformerLayer × depths[0]
            └── SwinTransformerLayer × depths[1]
                 └── ... (6 stages)
                      └── Conv 3×3 → Residual
                           └── Upsample (pixelshuffle) × 2
                                └── Conv 3×3 → HR output
```

## Model Registry

`models/registry.py` implements a decorator-based registry:

```python
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

def register(name: str):
    """Decorator: registers a model class under `name`."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(name: str, config: dict) -> nn.Module:
    """Instantiate a registered model by name with the given config."""
    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls(**config)
```

Models self-register at import time:

```python
@register("swinir")
class SwinIR(nn.Module): ...

@register("rrdb_esrgan")
class RRDBNet(nn.Module): ...
```

To add a new model, create an architecture file in `models/archs/`, use `@register("name")`, and import it in `models/__init__.py`.

## Loss Functions

`models/losses.py` provides three loss families:

### Charbonnier L1 Loss

A differentiable variant of L1 loss: `sqrt((x - y)^2 + epsilon^2)`. More stable than standard L1 during training.

```python
L1Loss()  # epsilon=1e-6
```

### Perceptual Loss

Uses a pretrained VGG19 network to compare feature maps:

```python
PerceptualLoss(
    layer_weights={'conv3_4': 1.0, 'conv2_2': 0.8, 'conv5_4': 0.5},
    vgg_device='cuda'
)
```

Extracts intermediate feature maps from VGG19, computes L1 distance between HR and SR features. Requires `torchvision` (lazily imported).

### GAN Loss

Standard GAN discriminator loss for ESRGAN-style training:

```python
GANLoss(gan_type='vanilla')  # or 'lsgan', 'wgan-gp'
```

## Metrics

`engine/metrics.py` provides evaluation metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| `psnr(img1, img2)` | 0–∞ (dB) | Peak Signal-to-Noise Ratio. Higher is better. |
| `ssim(img1, img2)` | -1–1 | Structural Similarity Index. 1 = identical. |
| `lpips(img1, img2)` | 0–∞ | Learned Perceptual Image Patch Similarity. Lower = more similar. Requires `lpips` package. |

PSNR and SSIM operate on the Y channel (luminance) in YCbCr space by default, matching the convention in super-resolution literature.

## Metrics Streaming

`engine/metrics_stream.py:MetricsStream` writes training metrics as JSONL:

```python
stream = MetricsStream("metrics/run_001.jsonl")
stream.write_step(epoch=1, batch=10, loss_total=0.05, loss_pixel=0.04, lr=1e-4)
stream.write_validate(epoch=1, psnr=30.2, ssim=0.89)
stream.write_phase("training")
stream.write_done(elapsed=3600.0, total_epochs=20)
stream.close()
```

File format (one JSON object per line):
```json
{"type":"step","epoch":1,"batch":10,"total_batches":100,"pixel":0.04,"total":0.05,"lr":0.0001}
{"type":"phase","phase":"training","max_epochs":20}
{"type":"validate","epoch":1,"psnr":30.2,"ssim":0.89}
{"type":"done","elapsed_seconds":3600,"total_epochs":20}
```

## Checkpointing

`models/checkpoint.py` handles model persistence.

### Save

```python
save_checkpoint(model, optimizer, epoch, config, path)
```

Writes a `.pth` file containing:
- `model_state_dict` — model weights
- `optimizer_state_dict` — optimizer state (for resume)
- `epoch` — current epoch
- `config` — training config snapshot
- `step` — global step counter

### Load

```python
model, optimizer_state, epoch, config = load_checkpoint(path, model=None)
```

If `model` is provided, loads weights into it. Returns optimizer state for resume.

### Export

Three export formats:

| Format | Function | Output |
|--------|----------|--------|
| ONNX | `export_onnx(model, path, size)` | `model.onnx` |
| SafeTensors | `export_safetensors(model, path)` | `model.safetensors` |
| TorchScript | `export_torchscript(model, path, size)` | `model.pt` |

ONNX and TorchScript exports trace the model with a dummy input of the given size. SafeTensors writes raw weight tensors without computation graph.

## Config System

### 4-Level Precedence

```
1. Built-in defaults    utils/configs/*.yaml              ← lowest
2. Workspace overrides  <workspace>/configs/**/*.yaml      │
3. --config file        user-provided YAML                  │
4. CLI flags            --batch-size, --lr, etc.          ← highest
```

Implementation in `utils/config.py`:

```python
class DefaultConfigs:
    def __init__(self, workspace=None):
        self.builtins = self._load_builtins()     # all YAMLs in utils/configs/
        self.workspace = workspace

    def get_train_config(self, config_path=None):
        cfg = self.builtins['train/base.yaml']           # level 1
        cfg = self._ws_or_builtin('train/base.yaml', cfg) # level 2
        if config_path:
            cfg = merge_overrides(cfg, load_yaml(config_path))  # level 3
        return cfg

    @staticmethod
    def apply_cli_overrides(cfg, cli_kwargs):  # level 4
        return merge_overrides(cfg, cli_kwargs)
```

### Built-in Config Files

| File | Key Parameters |
|------|---------------|
| `default.yaml` | device: auto, seed: 42, scale: 4, patch_size: 128, batch_size: 8, lr: 2e-4, tile settings |
| `train/base.yaml` | max_epochs: 100, save_per_epoch: 5, warmup_steps: 1000, min_lr: 1e-7, loss weights |
| `datasets/video_pairs.yaml` | degradation params (blur kernel, noise std, JPEG quality), frame_rate |
| `models/swinir.yaml` | embed_dim: 180, depths: [6,6,6,6,6,6], num_heads: [6,6,6,6,6,6] |
| `models/rrdb_esrgan.yaml` | num_feat: 64, num_block: 23, num_grow_ch: 32 |

### Validation

`validate_config(config)` checks required keys and raises descriptive errors for missing or malformed values. Called at CLI entry point before training begins.
