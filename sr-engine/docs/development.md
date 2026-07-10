# Development

## Dev Environment Setup

```bash
# Create venv with dev dependencies
uv venv
uv sync --group dev

# Or rebuild from scratch (CPU backend)
rm -rf .venv uv.lock
./envs/build.sh --backend cpu
uv sync --group dev
```

Dev dependencies include:
- `pytest` + `pytest-cov` + `pytest-mock` — testing
- `ruff` — linting and formatting
- `pre-commit` — git hooks (optional)

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/sr_engine

# Run specific test file
uv run pytest tests/test_trainer.py -v

# Run matching test names
uv run pytest tests/ -k "test_tiling"

# Run by marker
uv run pytest tests/ -m "cli"     # CLI integration tests
uv run pytest tests/ -m "unit"    # Fast unit tests
uv run pytest tests/ -m "gpu"     # GPU-required tests (skipped if no GPU)
```

### Test Markers

| Marker | Description | Speed |
|--------|-------------|-------|
| `unit` | Fast isolated tests | < 1s |
| `cli` | Click CLI runner tests | 1-5s |
| `integration` | Cross-module tests | 5-30s |
| `gpu` | Requires CUDA/ROCm | varies |
| `slow` | > 5s duration | slow |
| `network` | Downloads resources | varies |

### Test Fixtures (conftest.py)

| Fixture | Provides |
|---------|----------|
| `cli_runner` | Click CliRunner for CLI tests |
| `tmp_workspace` | Temporary initialized workspace |
| `empty_workspace` | Temporary workspace (empty, no projects) |
| `minimal_dataset` | Small HR/LR dataset for training tests |
| `sample_image` | 64×64 PNG test image |
| `sample_video` | Small test video file |
| `corrupt_image` | Intentionally broken image for error handling tests |
| `mock_torch_cuda` | Mocks `torch.cuda.is_available()` for CPU CI |
| `mock_socket` | Mock TCP socket for GUI bridge tests |

## Code Quality

### Linting

```bash
# Run ruff linter
ruff check src/ tests/

# Run ruff formatter (check mode)
ruff format --check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

### Type Checking

The project does not currently use static type checking (mypy/pyright) in CI, but type hints are encouraged in new code following the existing convention in `workspace.py` and `gui_bridge/`.

## Adding a New Model

1. **Create the architecture file** in `models/archs/`:

```python
# models/archs/my_model.py
import torch.nn as nn
from ..registry import register

@register("my_model")
class MyModel(nn.Module):
    def __init__(self, num_feat=64, scale=4, **kwargs):
        super().__init__()
        # define layers...

    def forward(self, x):
        # forward pass...
        return x
```

2. **Register the module** in `models/archs/__init__.py`:

```python
from . import my_model
```

3. **Add a config YAML** in `utils/configs/models/`:

```yaml
# utils/configs/models/my_model.yaml
num_feat: 64
scale: 4
```

4. **Add CLI support**: the model is now available via `srengine train run --model my_model`.

## Adding a New CLI Command

1. **Create a command file** in `cli/`:

```python
# cli/cmd_myfeature.py
import click

@click.group(name="myfeature")
def myfeature():
    """My new feature."""

@myfeature.command()
@click.option("--param", default=42)
def run(param):
    """Run my feature."""
    click.echo(f"Running with param={param}")
```

2. **Register in the main CLI group** in `cli/main.py`:

```python
from .cmd_myfeature import myfeature

@click.group()
def cli():
    ...

cli.add_command(myfeature)
```

3. **Register standalone entry point** in `pyproject.toml`:

```toml
[project.scripts]
myfeature = "sr_engine.cli.cmd_myfeature:myfeature"
```

## Adding a New Dataset Source

To support a new input source (e.g., image folder instead of video):

1. Add detection logic in `dataset_builder.py`
2. Add the extraction/building function
3. Update `cli/cmd_dataset.py` flags if new parameters are needed
4. Add config keys to `utils/configs/datasets/video_pairs.yaml`

## Adding GUI Bridge Commands

1. Add the command handler in `gui_bridge/server.py`:

```python
class Server:
    def _handle_my_command(self, params, job_id):
        # synchronous: return data dict
        return {"result": "ok"}
```

2. Register in the command map:

```python
COMMAND_MAP = {
    "my.command": "_handle_my_command",
    ...
}
```

3. For async (subprocess) commands, add the job type to `gui_bridge/jobs.py`.

## Test Coverage Expectations

- New features should include tests covering the public API
- CLI commands should have a `test_cli_*.py` integration test
- Bug fixes should include a regression test
- Tests should use fixtures from `conftest.py` where applicable
- GPU-specific tests should use the `@pytest.mark.gpu` marker

Current coverage targets (enforced by CI):
- Core engine modules: > 85%
- CLI commands: > 90%
- GUI bridge: > 80%
- Data pipeline: > 80%
- Models: > 85%

## Continuous Integration

The project does not currently have a CI configuration file. Suggested setup:

```yaml
# .github/workflows/ci.yml (example)
steps:
  - uses: actions/setup-python@v5
    with:
      python-version: "3.11"
  - run: pip install uv
  - run: ./envs/build.sh --backend cpu
  - run: uv run pytest tests/ -m "not gpu and not slow" --cov=src/sr_engine
  - run: ruff check src/ tests/
```
