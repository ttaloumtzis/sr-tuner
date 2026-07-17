"""Command schema — all bridge commands with their parameters."""

COMMAND_SCHEMA = [
    {
        "id": "hello",
        "title": "Handshake",
        "description": "Verify connection and get server version",
        "params": [],
    },
    {
        "id": "config.schema",
        "title": "Config Schema",
        "description": "Get the full command and config schema for dynamic UI building",
        "params": [],
    },
    {
        "id": "workspace.info",
        "title": "Workspace Info",
        "description": "Show workspace summary",
        "params": [],
    },
    {
        "id": "workspace.check",
        "title": "Workspace Health Check",
        "description": "Validate workspace health",
        "params": [],
    },
    {
        "id": "workspace.init",
        "title": "Init Workspace",
        "description": "Initialize a workspace directory tree",
        "params": [
            {"key": "path", "type": "path", "default": ".", "description": "Path to initialize workspace in"},
            {"key": "reset_configs", "type": "bool", "default": False, "description": "Overwrite workspace configs with fresh defaults"},
        ],
    },
    {
        "id": "project.list",
        "title": "List Model Instances",
        "description": "List all model instances in the workspace",
        "params": [],
    },
    {
        "id": "project.create",
        "title": "Create Model Instance",
        "description": "Create a named model instance in the workspace",
        "params": [
            {"key": "name", "type": "string", "required": True, "description": "Model instance name"},
            {"key": "arch", "type": "choice", "default": "swinir", "choices": ["swinir", "rrdb_esrgan"], "description": "Model architecture"},
        ],
    },
    {
        "id": "model.instance_list",
        "title": "List Instances (Detailed)",
        "description": "List model instances with checkpoint and run counts",
        "params": [
            {"key": "project", "type": "string", "description": "Optional project filter"},
        ],
    },
    {
        "id": "model.instance_info",
        "title": "Instance Details",
        "description": "Get detailed info for a model instance",
        "params": [
            {"key": "project", "type": "string", "required": True, "description": "Project name"},
            {"key": "instance", "type": "string", "required": True, "description": "Instance name"},
        ],
    },
    {
        "id": "model.list_runs",
        "title": "List Training Runs",
        "description": "List training runs for a model instance",
        "params": [
            {"key": "instance", "type": "string", "required": True, "description": "Model instance name"},
        ],
    },
    {
        "id": "model.export",
        "title": "Export Model",
        "description": "Export a model checkpoint to ONNX, SafeTensors, or TorchScript",
        "params": [
            {"key": "instance", "type": "string", "description": "Model instance name (auto-resolves version and arch config)"},
            {"key": "model_name", "type": "string", "description": "Model name (required without --instance)"},
            {"key": "ckpt", "type": "path", "description": "Checkpoint file path (required without --instance)"},
            {"key": "version", "type": "string", "description": "Version tag to export (defaults to latest)"},
            {"key": "format", "type": "choice", "choices": ["onnx", "safetensors", "torchscript"], "required": True, "description": "Export format"},
            {"key": "out", "type": "path", "required": True, "description": "Output file path"},
        ],
    },
    {
        "id": "model.info",
        "title": "Available Models",
        "description": "List available model architectures",
        "params": [],
    },
    {
        "id": "dataset.validate",
        "title": "Validate Dataset",
        "description": "Validate an existing dataset directory",
        "params": [
            {"key": "path", "type": "path", "required": True, "description": "Dataset directory path (must contain HR/ and LR/)"},
        ],
    },
    {
        "id": "dataset.health",
        "title": "Dataset Health Check",
        "description": "Profile a dataset and detect black/corrupt frames",
        "params": [
            {"key": "path", "type": "path", "required": True, "description": "Dataset root directory containing HR/ folder"},
            {"key": "auto_prune", "type": "bool", "default": False, "description": "Automatically delete identified black frames"},
        ],
    },
    {
        "id": "dataset.merge",
        "title": "Merge Datasets",
        "description": "Merge multiple datasets grouped by scale",
        "params": [
            {"key": "input", "type": "path", "required": True, "description": "Directory containing dataset subdirectories"},
            {"key": "out", "type": "path", "description": "Output directory (defaults to <input>/merged)"},
            {"key": "scale", "type": "int", "choices": [2, 3, 4, 8], "description": "Only merge datasets with this scale factor"},
            {"key": "name", "type": "string", "description": "Custom output subdirectory name"},
            {"key": "keep_sources", "type": "bool", "default": False, "description": "Keep original datasets after merge"},
        ],
    },
    {
        "id": "dataset.build",
        "title": "Build Dataset",
        "description": "Build a dataset from a video file or preprocessed directory",
        "params": [
            {"key": "input", "type": "path", "required": True, "description": "Input video file or preprocessed dataset directory"},
            {"key": "out", "type": "path", "description": "Output dataset directory (required for video input)"},
            {"key": "degradations", "type": "multi_choice", "choices": ["blur", "noise", "jpeg", "jpeg2000", "color-jitter"], "description": "Enabled degradations (comma-separated)"},
            {"key": "resize_method", "type": "choice", "choices": ["area", "bicubic", "bilinear", "lanczos", "nearest"], "description": "Downsampling method"},
        ],
        "config_sections": ["degradation"],
    },
    {
        "id": "train.start",
        "title": "Start Training",
        "description": "Start training a super-resolution model",
        "params": [
            {"key": "model_name", "type": "choice", "default": "rrdb_esrgan", "choices": ["swinir", "rrdb_esrgan"], "description": "Model architecture"},
            {"key": "dataset", "type": "path", "required": True, "description": "Dataset directory path"},
            {"key": "resume", "type": "string", "description": "Checkpoint to resume from (path or version tag)"},
            {"key": "device", "type": "choice", "default": "cuda", "choices": ["cuda", "cpu", "auto"], "description": "Training device"},
            {"key": "project", "type": "string", "description": "Project name (for workspace management)"},
            {"key": "instance", "type": "string", "description": "Model instance name (creates run dir)"},
            {"key": "experiment_id", "type": "string", "description": "Experiment identifier (auto-generated if omitted)"},
        ],
        "config_sections": ["training", "model"],
    },
    {
        "id": "infer.start",
        "title": "Start Inference",
        "description": "Run super-resolution inference on an image or video",
        "params": [
            {"key": "model", "type": "path", "description": "Model checkpoint path (required without --instance)"},
            {"key": "input_path", "type": "path", "required": True, "description": "Input image or video file"},
            {"key": "output", "type": "path", "required": True, "description": "Output image or video path"},
            {"key": "tile", "type": "int", "default": 512, "min": 0, "max": 4096, "step": 64, "description": "Tile size for tiled inference (0 = no tiling)"},
            {"key": "overlap", "type": "int", "default": 64, "min": 0, "max": 512, "step": 8, "description": "Overlap between tiles in pixels"},
            {"key": "device", "type": "choice", "default": "cuda", "choices": ["cuda", "cpu", "auto"], "description": "Inference device"},
            {"key": "instance", "type": "string", "description": "Model instance name (resolves latest version)"},
            {"key": "version", "type": "string", "description": "Version tag to use (defaults to latest)"},
        ],
    },
    {
        "id": "env.check",
        "title": "Environment Check",
        "description": "Check the current environment and print a report",
        "params": [],
    },
    {
        "id": "env.bench",
        "title": "Run Benchmark",
        "description": "Run a micro-benchmark (forward+backward) and report throughput",
        "params": [
            {"key": "model", "type": "choice", "default": "rrdb_esrgan", "choices": ["swinir", "rrdb_esrgan"], "description": "Model architecture to benchmark"},
            {"key": "iterations", "type": "int", "default": 10, "min": 1, "max": 1000, "step": 1, "description": "Number of timed iterations"},
        ],
    },
    {
        "id": "job.list",
        "title": "List Jobs",
        "description": "List all completed/cancelled/failed jobs",
        "params": [],
    },
    {
        "id": "job.status",
        "title": "Job Status",
        "description": "Get the manifest for a specific job",
        "params": [
            {"key": "job_id", "type": "string", "required": True, "description": "Job ID"},
        ],
    },
    {
        "id": "job.cancel",
        "title": "Cancel Job",
        "description": "Cancel a running job by sending SIGTERM",
        "params": [
            {"key": "job_id", "type": "string", "required": True, "description": "Job ID to cancel"},
        ],
    },
]


def command_schema_dict() -> dict:
    """Return the full schema response dict."""
    return {
        "commands": COMMAND_SCHEMA,
    }