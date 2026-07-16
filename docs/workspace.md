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
├── projects/                      # Training projects
│   └── <name>/
│       ├── configs/               # Experiment configs (YAML)
│       ├── checkpoints/           # epoch_XXX.pt files
│       └── metrics/               # *.jsonl files (machine mode)
├── jobs/                          # Job manifests (GUI bridge)
│   └── <job_id>.json
├── configs/                       # User-overridable configs
│   ├── train/
│   │   └── base.yaml              # Overrides built-in train config
│   ├── datasets/
│   │   └── video_pairs.yaml       # Overrides built-in dataset config
│   └── models/
│       ├── swinir.yaml            # Overrides built-in SwinIR config
│       └── rrdb_esrgan.yaml       # Overrides built-in RRDB config
└── model_instances/               # Named model instances (optional)
    └── <name>/
        ├── checkpoints/
        ├── runs/
        └── config.yaml
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

## Project CRUD

### Create

```bash
srengine project create my_experiment
# Creates: <workspace>/projects/my_experiment/
#   configs/
#   checkpoints/
#   metrics/
```

### List

```bash
srengine project list
# Projects in <workspace>:
#   my_experiment
#   ablation_study
#   production_v2
```

## Using Projects with Training

When `--project` is specified, paths auto-resolve:

```bash
srengine train run \
  --project my_experiment \
  --dataset my_set \
  --model swinir

# dataset resolves to:     <workspace>/datasets/my_set/
# checkpoints go to:       <workspace>/projects/my_experiment/checkpoints/
# metrics go to:           <workspace>/projects/my_experiment/metrics/
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
srengine model create-instance --project my_project --name my_model --model swinir

# List instances
srengine model list-instances --project my_project

# List training runs for an instance
srengine model list-runs --instance my_project/my_model
```

Training runs are organized in timestamped directories:
```
<workspace>/model_instances/my_model/
├── config.yaml
├── checkpoints/
│   ├── epoch_010.pt
│   └── epoch_020.pt
└── runs/
    └── run_20250516_120000/
        ├── config.yaml
        ├── metrics.jsonl
        └── checkpoint.pt
```
