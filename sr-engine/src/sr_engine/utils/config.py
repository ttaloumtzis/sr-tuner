"""YAML config loading with CLI-override merging."""

from pathlib import Path


def load_config(path: Path) -> dict:
    """Load a YAML configuration file and return it as a nested dict."""
    raise NotImplementedError("TODO: implement YAML config loading")


def merge_overrides(base: dict, overrides: dict) -> dict:
    """Merge *overrides* into *base*, returning a new dict.

    Keys in *overrides* overwrite corresponding keys in *base*.
    Nested dicts are merged recursively (shallow merge at each level).
    """
    raise NotImplementedError("TODO: implement config override merging")


def save_config(config: dict, path: Path) -> None:
    """Save a configuration dict to a YAML file."""
    raise NotImplementedError("TODO: implement config saving")
