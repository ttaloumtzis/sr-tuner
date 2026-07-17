# Workspace Management

## Overview

The workspace system provides structured project organization, path auto-resolution, and config layering. It eliminates the need to type full paths for datasets and projects once a workspace is initialized.

## Workspace Layout

```
<workspace>/
├── .sr_workspace                  # Marker file (JSON: version, created timestamp)
├── datasets/                      # Dataset pool
│   └── <name>/
│       ├── HR/                    # High-resolution frames
│       ├── LR/                    # Low-resolution (degraded) frames
│       └── manifest.json          # Pairs index
├── models/                        # Named model instances
│   └── <name>/
│       ├── config.yaml            # Frozen model-architecture config
│       ├── versions/              # Versioned checkpoints (v1/, v2/, ...)
│       ├── checkpoints/           # Training checkpoints (epoch_*.pt)
│       └── runs/                  # Per-training-run metadata
│           └── run_<timestamp>/
│               ├── train_config.yaml
│               └── metrics.jsonl  # (optional, --machine mode)
├── jobs/                          # Job manifests (GUI bridge)
│   └── <job_id>.json
├── experiments/                   # Experiment data
├── configs/                       # User-overridable configs
│   ├── train/
│   │   └── base.yaml              # Overrides built-in train config
│   ├── datasets/
│   │   └── video_pairs.yaml       # Overrides built-in dataset config
│   └── models/
│       ├── swinir.yaml            # Overrides built-in SwinIR config
│       └── rrdb_esrgan.yaml       # Overrides built-in RRDB config
```

## Auto-Discovery

The workspace is resolved in the following order (first wins):

1. `--workspace PATH` CLI flag
2. `SRENGINE_WORKSPACE` environment variable
3. Walking up from CWD looking for `.sr_workspace` marker file

Workspace auto-discovery means you can run commands from any subdirectory of the workspace without specifying the workspace path:

```bash
# These are equivalent:
cd /path/to/workspace && srengine dataset build --input video.mp4
cd /path/to/workspace/projects/my_project && srengine dataset build --input video.mp4
```

## Path Resolution

Commands that accept paths (`--dataset`, `--input`, `--output`, `--model`) resolve in this order:

1. **Absolute path** — used as-is
2. **Relative to CWD** — resolved from current working directory
3. **Workspace-relative** — resolved from workspace root

```bash
# All resolve the same dataset:
srengine train run --dataset /absolute/path/to/datasets/my_set
srengine train run --dataset ./datasets/my_set          # relative to CWD
srengine train run --dataset my_set                     # workspace datasets/my_set
```

## Model Instance CRUD

### Create

```bash
srengine model create-instance my_model --model swinir
# Creates: <workspace>/models/my_model/
#   config.yaml
#   versions/
#   checkpoints/
#   runs/
```

### List

```bash
srengine model list-instances
# Model instances:
#   my_model  (3 versions, 5 runs)
#   ablation_study  (1 version, 2 runs)
```

## Using Model Instances with Training

When `--instance` is specified, checkpoints and metrics are stored under the instance directory:

```bash
srengine train run \
  --instance my_model \
  --dataset my_set \
  --model swinir

# dataset resolves to:     <workspace>/datasets/my_set/
# checkpoints go to:       <workspace>/models/my_model/checkpoints/
# version checkpoints to:  <workspace>/models/my_model/versions/
# run metadata to:         <workspace>/models/my_model/runs/run_<timestamp>/
```

Without a workspace, all paths are literal.

## Config Layering via Workspace

Workspace-level configs override built-in defaults. Copy a built-in config to the workspace and modify:

```bash
# Copy the default train config to the workspace
mkdir -p <workspace>/configs/train/
cp sr-engine/src/sr_engine/utils/configs/train/base.yaml <workspace>/configs/train/base.yaml

# Edit the workspace copy — it now takes precedence
vim <workspace>/configs/train/base.yaml
```

The config loading priority:
1. Built-in defaults (`utils/configs/`)
2. Workspace overrides (`<workspace>/configs/`)
3. Explicit `--config` file
4. CLI flags

## Model Instances

Named model instances track checkpoint history and training runs:

```bash
# Create an instance
srengine model create-instance my_model --model swinir

# List instances
srengine model list-instances

# List training runs for an instance
srengine model list-runs --instance my_model
```

Training runs are organized in timestamped directories:
```
<workspace>/models/my_model/
├── config.yaml
├── versions/
│   └── v1/
│       ├── model.pt
│       └── version.json
├── checkpoints/
│   ├── epoch_010.pt
│   └── epoch_020.pt
└── runs/
    └── run_20250516_120000/
        ├── train_config.yaml
        └── metrics.jsonl
```
