"""Config schema — training, model architecture, and degradation pipeline sections."""

from copy import deepcopy

CONFIG_SECTIONS = {
    "training": {
        "title": "Training",
        "description": "Training hyperparameters (learning rate, batch size, scheduler, etc.)",
        "params": [
            {"key": "train.batch_size", "type": "int", "default": 32, "min": 1, "max": 512, "step": 1, "group": "General", "description": "Samples per batch"},
            {"key": "train.num_workers", "type": "int", "default": 4, "min": 0, "max": 32, "step": 1, "group": "General", "description": "Dataloader worker processes"},
            {"key": "train.patch_size", "type": "int", "default": 64, "min": 16, "max": 512, "step": 8, "group": "General", "description": "Training patch size (pixels)"},
            {"key": "train.max_epochs", "type": "int", "default": 10, "min": 1, "max": 10000, "step": 1, "group": "General", "description": "Total number of training epochs"},
            {"key": "train.save_per_epoch", "type": "int", "default": 5, "min": 1, "max": 1000, "step": 1, "group": "General", "description": "Save checkpoint every N epochs"},
            {"key": "train.seed", "type": "int", "default": 42, "min": 0, "max": 2147483647, "step": 1, "group": "General", "description": "Random seed for reproducibility"},
            {"key": "train.learning_rate", "type": "float", "default": 2e-4, "min": 1e-8, "max": 1.0, "step": "log", "group": "Optimizer", "description": "Initial learning rate"},
            {"key": "train.weight_decay", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": "log", "group": "Optimizer", "description": "Adam weight decay"},
            {"key": "train.betas", "type": "list", "default": [0.9, 0.99], "group": "Optimizer", "description": "Adam beta coefficients"},
            {"key": "train.lr_scheduler", "type": "choice", "default": "cosine", "choices": ["cosine", "linear", "constant"], "group": "LR Scheduler", "description": "Learning rate schedule"},
            {"key": "train.warmup_steps", "type": "int", "default": 2000, "min": 0, "max": 50000, "step": 100, "group": "LR Scheduler", "description": "Linear warmup steps"},
            {"key": "train.min_lr", "type": "float", "default": 1e-7, "min": 1e-12, "max": 1.0, "step": "log", "group": "LR Scheduler", "description": "Minimum learning rate after decay"},
            {"key": "train.dtype", "type": "choice", "default": "float32", "choices": ["float32", "bf16", "float16"], "group": "Precision", "description": "Training precision"},
            {"key": "train.validation.enabled", "type": "bool", "default": True, "group": "Validation", "description": "Enable validation split"},
            {"key": "train.validation.split", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05, "group": "Validation", "description": "Fraction of data held out for validation"},
            {"key": "train.losses.perceptual_weight", "type": "float", "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01, "group": "Losses", "description": "Perceptual loss weight"},
            {"key": "train.metrics_frequency", "type": "int", "default": 1, "min": 1, "max": 1000, "step": 1, "group": "Logging", "description": "Log metrics every N batches"},
        ],
    },
    "model": {
        "title": "Model Architecture",
        "description": "Model architecture parameters (depends on selected model)",
        "params": [
            {"key": "model.name", "type": "choice", "default": "rrdb_esrgan", "choices": ["swinir", "rrdb_esrgan"], "group": "General", "description": "Model architecture"},
            {"key": "model.scale", "type": "int", "default": 4, "choices": [2, 3, 4, 8], "group": "General", "description": "Upscaling factor"},
            {"key": "model.embed_dim", "type": "int", "default": 180, "min": 12, "max": 720, "step": 12, "applies_to": ["swinir"], "group": "SwinIR", "description": "Embedding dimension"},
            {"key": "model.depths", "type": "list", "default": [6, 6, 6, 6, 6, 6], "applies_to": ["swinir"], "group": "SwinIR", "description": "Number of transformer blocks per stage"},
            {"key": "model.num_heads", "type": "list", "default": [6, 6, 6, 6, 6, 6], "applies_to": ["swinir"], "group": "SwinIR", "description": "Attention heads per stage"},
            {"key": "model.window_size", "type": "int", "default": 8, "min": 4, "max": 16, "step": 2, "applies_to": ["swinir"], "group": "SwinIR", "description": "Local window size"},
            {"key": "model.mlp_ratio", "type": "float", "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5, "applies_to": ["swinir"], "group": "SwinIR", "description": "MLP expansion ratio"},
            {"key": "model.upsampler", "type": "choice", "default": "pixelshuffle", "choices": ["pixelshuffle", "nearest+conv"], "applies_to": ["swinir"], "group": "SwinIR", "description": "Upsampling method"},
            {"key": "model.num_feat", "type": "int", "default": 64, "min": 16, "max": 256, "step": 8, "applies_to": ["rrdb_esrgan"], "group": "RRDB-ESRGAN", "description": "Number of feature channels"},
            {"key": "model.num_block", "type": "int", "default": 23, "min": 3, "max": 100, "step": 1, "applies_to": ["rrdb_esrgan"], "group": "RRDB-ESRGAN", "description": "Number of RRDB blocks"},
            {"key": "model.num_grow_ch", "type": "int", "default": 32, "min": 8, "max": 128, "step": 4, "applies_to": ["rrdb_esrgan"], "group": "RRDB-ESRGAN", "description": "Growth channel count"},
        ],
    },
    "degradation": {
        "title": "Degradation Pipeline",
        "description": "Degradation parameters for dataset building",
        "params": [
            {"key": "degradation.blur.enabled", "type": "bool", "default": True, "group": "Blur", "description": "Enable blur degradation"},
            {"key": "degradation.blur.gaussian.kernel_size", "type": "int", "default": 21, "min": 3, "max": 63, "step": 2, "group": "Blur / Gaussian", "description": "Gaussian blur kernel size"},
            {"key": "degradation.blur.gaussian.sigma_min", "type": "float", "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1, "group": "Blur / Gaussian", "description": "Minimum Gaussian sigma"},
            {"key": "degradation.blur.gaussian.sigma_max", "type": "float", "default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1, "group": "Blur / Gaussian", "description": "Maximum Gaussian sigma"},
            {"key": "degradation.blur.gaussian.prob", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Blur / Gaussian", "description": "Probability of Gaussian blur"},
            {"key": "degradation.blur.motion.max_kernel_size", "type": "int", "default": 31, "min": 3, "max": 99, "step": 2, "group": "Blur / Motion", "description": "Maximum motion blur kernel size"},
            {"key": "degradation.blur.motion.prob", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Blur / Motion", "description": "Probability of motion blur"},
            {"key": "degradation.resize.method", "type": "choice", "default": "area", "choices": ["area", "bicubic", "bilinear", "lanczos", "nearest"], "group": "Resize", "description": "Downsampling method"},
            {"key": "degradation.resize.antialias", "type": "bool", "default": True, "group": "Resize", "description": "Apply anti-aliasing during resize"},
            {"key": "degradation.noise.enabled", "type": "bool", "default": False, "group": "Noise", "description": "Enable noise degradation"},
            {"key": "degradation.noise.gaussian.sigma_range_min", "type": "float", "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5, "group": "Noise / Gaussian", "description": "Minimum Gaussian noise sigma"},
            {"key": "degradation.noise.gaussian.sigma_range_max", "type": "float", "default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5, "group": "Noise / Gaussian", "description": "Maximum Gaussian noise sigma"},
            {"key": "degradation.noise.gaussian.prob", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Noise / Gaussian", "description": "Probability of Gaussian noise"},
            {"key": "degradation.noise.poisson.scale_range_min", "type": "float", "default": 0.05, "min": 0.0, "max": 10.0, "step": 0.05, "group": "Noise / Poisson", "description": "Minimum Poisson scale"},
            {"key": "degradation.noise.poisson.scale_range_max", "type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.05, "group": "Noise / Poisson", "description": "Maximum Poisson scale"},
            {"key": "degradation.noise.poisson.prob", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Noise / Poisson", "description": "Probability of Poisson noise"},
            {"key": "degradation.noise.salt_pepper.amount", "type": "float", "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "group": "Noise / Salt & Pepper", "description": "Salt & pepper noise amount"},
            {"key": "degradation.noise.salt_pepper.salt_vs_pepper", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Noise / Salt & Pepper", "description": "Salt vs pepper ratio"},
            {"key": "degradation.noise.salt_pepper.prob", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Noise / Salt & Pepper", "description": "Probability of salt & pepper noise"},
            {"key": "degradation.jpeg.enabled", "type": "bool", "default": True, "group": "JPEG", "description": "Enable JPEG compression"},
            {"key": "degradation.jpeg.quality_min", "type": "int", "default": 30, "min": 1, "max": 100, "step": 1, "group": "JPEG", "description": "Minimum JPEG quality"},
            {"key": "degradation.jpeg.quality_max", "type": "int", "default": 95, "min": 1, "max": 100, "step": 1, "group": "JPEG", "description": "Maximum JPEG quality"},
            {"key": "degradation.jpeg.prob", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "group": "JPEG", "description": "Probability of JPEG compression"},
            {"key": "degradation.jpeg2000.enabled", "type": "bool", "default": False, "group": "JPEG2000", "description": "Enable JPEG2000 compression"},
            {"key": "degradation.jpeg2000.quality_min", "type": "int", "default": 30, "min": 1, "max": 100, "step": 1, "group": "JPEG2000", "description": "Minimum JPEG2000 quality"},
            {"key": "degradation.jpeg2000.quality_max", "type": "int", "default": 95, "min": 1, "max": 100, "step": 1, "group": "JPEG2000", "description": "Maximum JPEG2000 quality"},
            {"key": "degradation.jpeg2000.prob", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "JPEG2000", "description": "Probability of JPEG2000 compression"},
            {"key": "degradation.color_jitter.enabled", "type": "bool", "default": False, "group": "Color Jitter", "description": "Enable color jitter"},
            {"key": "degradation.color_jitter.hue_range_min", "type": "float", "default": -0.05, "min": -1.0, "max": 1.0, "step": 0.01, "group": "Color Jitter", "description": "Minimum hue shift"},
            {"key": "degradation.color_jitter.hue_range_max", "type": "float", "default": 0.05, "min": -1.0, "max": 1.0, "step": 0.01, "group": "Color Jitter", "description": "Maximum hue shift"},
            {"key": "degradation.color_jitter.saturation_range_min", "type": "float", "default": -0.3, "min": -1.0, "max": 1.0, "step": 0.1, "group": "Color Jitter", "description": "Minimum saturation shift"},
            {"key": "degradation.color_jitter.saturation_range_max", "type": "float", "default": 0.3, "min": -1.0, "max": 1.0, "step": 0.1, "group": "Color Jitter", "description": "Maximum saturation shift"},
            {"key": "degradation.color_jitter.value_range_min", "type": "float", "default": -0.3, "min": -1.0, "max": 1.0, "step": 0.1, "group": "Color Jitter", "description": "Minimum value shift"},
            {"key": "degradation.color_jitter.value_range_max", "type": "float", "default": 0.3, "min": -1.0, "max": 1.0, "step": 0.1, "group": "Color Jitter", "description": "Maximum value shift"},
            {"key": "degradation.color_jitter.prob", "type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Color Jitter", "description": "Probability of color jitter"},
            {"key": "degradation.frame_rate", "type": "int", "default": 10, "min": 1, "max": 120, "step": 1, "group": "Frames", "description": "Frame extraction rate"},
            {"key": "degradation.frame_format", "type": "choice", "default": "png", "choices": ["png", "jpg", "bmp"], "group": "Frames", "description": "Output frame format"},
            {"key": "degradation.start_time", "type": "float", "default": 0.0, "min": 0.0, "max": 1e6, "step": 1.0, "group": "Frames", "description": "Start time for video extraction (seconds)"},
            {"key": "degradation.duration", "type": "float", "default": None, "min": 0.0, "max": 1e6, "step": 1.0, "group": "Frames", "nullable": True, "description": "Duration for video extraction (seconds, null = full video)"},
        ],
    },
}


def schema_to_defaults(
    model: str = "rrdb_esrgan",
) -> dict:
    """Produce a flat key→default dict from the config schema.

    Args:
        model: Model name to filter model-specific params.

    Returns:
        Flat dict like ``{"train.batch_size": 32, "model.embed_dim": 180, ...}``.
    """
    result: dict = {}
    for section in CONFIG_SECTIONS.values():
        for param in section["params"]:
            applies_to = param.get("applies_to")
            if applies_to is not None and model not in applies_to:
                continue
            result[param["key"]] = deepcopy(param["default"])
    return result


def schema_for_model(model: str) -> list[dict]:
    """Return model-specific params filtered by ``applies_to``.

    Args:
        model: Model name (``"swinir"`` or ``"rrdb_esrgan"``).

    Returns:
        List of param dicts that apply to the given model.
    """
    return [p for p in CONFIG_SECTIONS["model"]["params"]
            if p.get("applies_to") is None or model in p.get("applies_to", [])]


def all_params() -> list[dict]:
    """Return all config params across all sections."""
    result: list[dict] = []
    for section in CONFIG_SECTIONS.values():
        result.extend(section["params"])
    return result