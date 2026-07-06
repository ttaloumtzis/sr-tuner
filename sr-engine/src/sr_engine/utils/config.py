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


def save_config(config: dict, path: Path) -> None:
    """Save a configuration dict to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # default_flow_style=False keeps everything formatted as clean, indented blocks
        # sort_keys=False preserves your logical dictionary structure layout
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)