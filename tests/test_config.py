"""Tests for utils/config.py — config loading, merging, validation."""

from pathlib import Path

import pytest
import yaml

from sr_engine.utils.config import (
    load_config,
    merge_overrides,
    validate_config,
    save_config,
    DefaultConfigs,
)


class TestLoadConfig:
    """Tests for ``load_config``."""

    def test_none_returns_empty(self):
        """Passing None should return an empty dict."""
        assert load_config(None) == {}

    def test_missing_file_raises(self):
        """A nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(Path("/nonexistent/config.yaml"))

    def test_loads_yaml(self, tmp_path):
        """A valid YAML file should be loaded correctly."""
        cfg = {"model": "rrdb", "scale": 4}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        assert load_config(path) == cfg

    def test_empty_file_returns_empty(self, tmp_path):
        """An empty YAML file should return an empty dict."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_config(path) == {}


class TestMergeOverrides:
    """Tests for ``merge_overrides``."""

    def test_merge_simple(self):
        """Simple key overrides should be replaced."""
        base = {"a": 1, "b": 2}
        merged = merge_overrides(base, {"b": 3})
        assert merged == {"a": 1, "b": 3}

    def test_merge_nested(self):
        """Nested dict overrides should be merged recursively."""
        base = {"train": {"lr": 1e-4, "epochs": 10}}
        merged = merge_overrides(base, {"train": {"lr": 1e-3}})
        assert merged == {"train": {"lr": 1e-3, "epochs": 10}}

    def test_merge_new_key(self):
        """New keys from overrides should be added."""
        base = {"a": 1}
        merged = merge_overrides(base, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        """The base dict should not be mutated."""
        base = {"a": [1, 2, 3]}
        merged = merge_overrides(base, {"a": [4]})
        assert base["a"] == [1, 2, 3]
        assert merged["a"] == [4]

    def test_overwrite_with_none(self):
        """Overriding with None should set the value to None."""
        base = {"a": 1}
        merged = merge_overrides(base, {"a": None})
        assert merged["a"] is None


class TestValidateConfig:
    """Tests for ``validate_config``."""

    def test_all_keys_present(self):
        """No error when all required keys are present."""
        validate_config({"a": 1, "b": 2}, ["a", "b"])

    def test_missing_keys(self):
        """Missing required keys should raise ValueError."""
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config({"a": 1}, ["a", "b"])


class TestSaveConfig:
    """Tests for ``save_config``."""

    def test_writes_yaml(self, tmp_path):
        """A config dict should be written as valid YAML."""
        path = tmp_path / "out.yaml"
        save_config({"model": "swinir", "scale": 2}, path)
        assert path.is_file()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"model": "swinir", "scale": 2}

    def test_creates_parent_dir(self, tmp_path):
        """Parent directories should be created automatically."""
        path = tmp_path / "sub" / "nested" / "cfg.yaml"
        save_config({"a": 1}, path)
        assert path.is_file()


class TestDefaultConfigs:
    """Tests for DefaultConfigs."""

    def test_builtin_path_exists(self):
        """The built-in config path should exist."""
        path = DefaultConfigs.builtin_config_path()
        assert path.is_dir()

    def test_loads_train_config(self):
        """get_train_config() should return a non-empty dict."""
        configs = DefaultConfigs()
        cfg = configs.get_train_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_loads_dataset_config(self):
        """get_dataset_config() should return a dict."""
        configs = DefaultConfigs()
        cfg = configs.get_dataset_config()
        assert isinstance(cfg, dict)

    def test_loads_known_model(self):
        """get_model_config() should return a config for known models."""
        configs = DefaultConfigs()
        cfg = configs.get_model_config("rrdb_esrgan")
        assert cfg is not None
        assert isinstance(cfg, dict)

    def test_unknown_model_returns_none(self):
        """get_model_config() should return None for unknown models."""
        configs = DefaultConfigs()
        assert configs.get_model_config("nonexistent") is None

    def test_get_full_config(self):
        """get_full_config() should merge all config sections."""
        configs = DefaultConfigs()
        cfg = configs.get_full_config("rrdb_esrgan")
        assert isinstance(cfg, dict)
        assert "max_epochs" in cfg

    def test_train_config_has_expected_defaults(self):
        """Train config should contain expected default values."""
        configs = DefaultConfigs()
        cfg = configs.get_train_config()
        assert cfg.get("batch_size") == 32
        assert cfg.get("patch_size") == 128
        assert cfg.get("num_workers") == 4
        assert float(cfg.get("learning_rate")) == 2e-4
        assert float(cfg.get("weight_decay")) == 0.0
        assert cfg.get("betas") == [0.9, 0.99]
        assert cfg.get("max_epochs") == 10

    def test_configs_load_without_default_yaml(self):
        """Configs should load correctly even though default.yaml has been deleted."""
        import os
        default_path = DefaultConfigs.builtin_config_path() / "default.yaml"
        assert not default_path.exists(), (
            "default.yaml should have been deleted."
        )
        configs = DefaultConfigs()
        train = configs.get_train_config()
        assert isinstance(train, dict) and len(train) > 0
        dataset = configs.get_dataset_config()
        assert isinstance(dataset, dict) and len(dataset) > 0
        model = configs.get_model_config("rrdb_esrgan")
        assert model is not None
