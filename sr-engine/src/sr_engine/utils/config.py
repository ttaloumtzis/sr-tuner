"""YAML config loading with CLI-override merging."""

import copy
from pathlib import Path
import yaml





def load_config(path: Path | None) -> dict:
    """Load a YAML configuration file and return it as a nested dict.

    If path is None, returns an empty dictionary so fallback logic can take over.
    """
    if path is None:
        return {}

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data if config_data is not None else {}


def merge_overrides(base: dict, overrides: dict) -> dict:
    """Merge *overrides* into *base*, returning a new dict.

    Keys in *overrides* overwrite corresponding keys in *base*.
    Nested dicts are merged recursively (shallow merge at each level).
    """
    # Create a deep copy of base to completely avoid modifying your defaults in-place
    merged = copy.deepcopy(base)

    for key, value in overrides.items():
        # If both are nested dictionaries, merge them recursively
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_overrides(merged[key], value)
        else:
            # Otherwise, overwrite the value completely (handles primitives, lists, etc.)
            merged[key] = copy.deepcopy(value)

    return merged


def validate_config(cfg: dict, required_keys: list[str]) -> None:
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            f"Configuration missing required keys: {missing}. "
            f"Check your config file or CLI arguments."
        )


def save_config(config: dict, path: Path) -> None:
    """Save a configuration dict to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # default_flow_style=False keeps everything formatted as clean, indented blocks
        # sort_keys=False preserves your logical dictionary structure layout
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


class DefaultConfigs:
    def __init__(self, workspace=None):
        self._workspace = workspace
        self._builtin_path = Path(__file__).resolve().parents[1] / "utils" / "configs"

        self.train = load_config(self._builtin_path / "train" / "base.yaml")
        self.datasets = load_config(self._builtin_path / "datasets" / "video_pairs.yaml")

        self.models = {
            "swinir": load_config(self._builtin_path / "models" / "swinir.yaml"),
            "rrdb_esrgan": load_config(self._builtin_path / "models" / "rrdb_esrgan.yaml")
        }

    @classmethod
    def builtin_config_path(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "utils" / "configs"

    def _ws_or_builtin(self, category: str, filename: str, fallback: dict) -> dict:
        if self._workspace is None:
            return copy.deepcopy(fallback)
        ws_path = self._workspace.path / "configs" / category / filename
        if not ws_path.is_file():
            return copy.deepcopy(fallback)
        try:
            ws_cfg = load_config(ws_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load workspace config at {ws_path}. "
                f"Use --no-workspace-config to skip workspace configs.\n  {e}"
            )
        return merge_overrides(fallback, ws_cfg)

    def get_train_config(self) -> dict:
        return self._ws_or_builtin("train", "base.yaml", self.train)

    def get_dataset_config(self) -> dict:
        return self._ws_or_builtin("datasets", "video_pairs.yaml", self.datasets)

    def get_model_config(self, name: str) -> dict | None:
        builtin = self.models.get(name)
        if builtin is None:
            return None
        return self._ws_or_builtin("models", f"{name}.yaml", builtin)

    def get_full_config(self, model_name: str) -> dict:
        base = merge_overrides(self.get_train_config(), self.get_dataset_config())
        return merge_overrides(base, self.get_model_config(model_name))