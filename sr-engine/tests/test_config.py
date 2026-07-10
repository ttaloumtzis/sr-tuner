"""Tests for utils/config.py — config loading, merging, validation."""

from pathlib import Path
from unittest.mock import patch

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
    def test_none_returns_empty(self):
        assert load_config(None) == {}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(Path("/nonexistent/config.yaml"))

    def test_loads_yaml(self, tmp_path):
        cfg = {"model": "rrdb", "scale": 4}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        assert load_config(path) == cfg

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_config(path) == {}


class TestMergeOverrides:
    def test_merge_simple(self):
        base = {"a": 1, "b": 2}
        merged = merge_overrides(base, {"b": 3})
        assert merged == {"a": 1, "b": 3}

    def test_merge_nested(self):
        base = {"train": {"lr": 1e-4, "epochs": 10}}
        merged = merge_overrides(base, {"train": {"lr": 1e-3}})
        assert merged == {"train": {"lr": 1e-3, "epochs": 10}}

    def test_merge_new_key(self):
        base = {"a": 1}
        merged = merge_overrides(base, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"a": [1, 2, 3]}
        merged = merge_overrides(base, {"a": [4]})
        assert base["a"] == [1, 2, 3]
        assert merged["a"] == [4]

    def test_overwrite_with_none(self):
        base = {"a": 1}
        merged = merge_overrides(base, {"a": None})
        assert merged["a"] is None


class TestValidateConfig:
    def test_all_keys_present(self):
        validate_config({"a": 1, "b": 2}, ["a", "b"])

    def test_missing_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config({"a": 1}, ["a", "b"])


class TestSaveConfig:
    def test_writes_yaml(self, tmp_path):
        path = tmp_path / "out.yaml"
        save_config({"model": "swinir", "scale": 2}, path)
        assert path.is_file()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"model": "swinir", "scale": 2}

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "sub" / "nested" / "cfg.yaml"
        save_config({"a": 1}, path)
        assert path.is_file()


class TestDefaultConfigs:
    def test_builtin_path_exists(self):
        path = DefaultConfigs.builtin_config_path()
        assert path.is_dir()

    def test_loads_train_config(self):
        configs = DefaultConfigs()
        cfg = configs.get_train_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_loads_dataset_config(self):
        configs = DefaultConfigs()
        cfg = configs.get_dataset_config()
        assert isinstance(cfg, dict)

    def test_loads_known_model(self):
        configs = DefaultConfigs()
        cfg = configs.get_model_config("rrdb_esrgan")
        assert cfg is not None
        assert isinstance(cfg, dict)

    def test_unknown_model_returns_none(self):
        configs = DefaultConfigs()
        assert configs.get_model_config("nonexistent") is None

    def test_get_full_config(self):
        configs = DefaultConfigs()
        cfg = configs.get_full_config("rrdb_esrgan")
        assert isinstance(cfg, dict)
        assert "max_epochs" in cfg
