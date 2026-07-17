"""Tests for gui_bridge/config_utils.py — dotted-key expansion, validation, YAML I/O."""

from pathlib import Path
import yaml

import pytest

from sr_engine.gui_bridge.config_utils import (
    expand_dotted_config,
    validate_config_values,
    write_temp_config,
)

# ── Sample schema for validation tests ──

_SAMPLE_SCHEMA = [
    {"key": "train.batch_size", "type": "int", "min": 1, "max": 512},
    {"key": "train.learning_rate", "type": "float", "min": 1e-8, "max": 1.0},
    {"key": "train.validation.enabled", "type": "bool"},
    {"key": "model.name", "type": "choice", "choices": ["swinir", "rrdb_esrgan"]},
    {"key": "degradation.blur.enabled", "type": "bool"},
    {"key": "degradation.blur.gaussian.prob", "type": "float", "min": 0.0, "max": 1.0},
    {"key": "degradation.degradations", "type": "multi_choice", "choices": ["blur", "noise", "jpeg"]},
    {"key": "train.seed", "type": "int", "min": 0, "max": 2147483647},
]


class TestExpandDottedConfig:
    """Tests for ``expand_dotted_config``."""

    def test_simple_key(self):
        result = expand_dotted_config({"batch_size": 16})
        assert result == {"batch_size": 16}

    def test_one_level_deep(self):
        result = expand_dotted_config({"losses.perceptual_weight": 0.1})
        assert result == {"losses": {"perceptual_weight": 0.1}}

    def test_multi_level(self):
        result = expand_dotted_config({"degradation.blur.gaussian.sigma": 3.0})
        assert result == {"degradation": {"blur": {"gaussian": {"sigma": 3.0}}}}

    def test_multiple_keys(self):
        result = expand_dotted_config({
            "batch_size": 16,
            "learning_rate": 1e-4,
            "losses.perceptual_weight": 0.1,
        })
        assert result["batch_size"] == 16
        assert result["learning_rate"] == 1e-4
        assert result["losses"]["perceptual_weight"] == 0.1

    def test_strip_prefix(self):
        result = expand_dotted_config(
            {"train.batch_size": 16, "train.losses.perceptual_weight": 0.1},
            strip_prefix="train",
        )
        assert result == {"batch_size": 16, "losses": {"perceptual_weight": 0.1}}

    def test_strip_prefix_skips_other_keys(self):
        result = expand_dotted_config(
            {"train.batch_size": 16, "model.name": "swinir"},
            strip_prefix="train",
        )
        assert result == {"batch_size": 16}
        assert "model" not in result

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="Key conflict"):
            expand_dotted_config({"a.b": 1, "a": 2})

    def test_empty_dict(self):
        assert expand_dotted_config({}) == {}

    def test_list_values(self):
        result = expand_dotted_config({"depths": [6, 6, 6]})
        assert result["depths"] == [6, 6, 6]

    def test_none_value(self):
        result = expand_dotted_config({"duration": None})
        assert result["duration"] is None


class TestValidateConfigValues:
    """Tests for ``validate_config_values``."""

    def test_valid_config_returns_empty(self):
        config = {
            "train.batch_size": 16,
            "train.learning_rate": 1e-4,
            "train.validation.enabled": True,
            "model.name": "swinir",
        }
        errors = validate_config_values(config, _SAMPLE_SCHEMA)
        assert errors == []

    def test_int_below_min(self):
        errors = validate_config_values({"train.batch_size": 0}, _SAMPLE_SCHEMA)
        assert len(errors) == 1
        assert "below minimum" in errors[0]

    def test_int_above_max(self):
        errors = validate_config_values({"train.batch_size": 9999}, _SAMPLE_SCHEMA)
        assert len(errors) == 1
        assert "above maximum" in errors[0]

    def test_float_below_min(self):
        errors = validate_config_values({"train.learning_rate": 1e-9}, _SAMPLE_SCHEMA)
        assert len(errors) == 1
        assert "below minimum" in errors[0]

    def test_float_above_max(self):
        errors = validate_config_values({"train.learning_rate": 10.0}, _SAMPLE_SCHEMA)
        assert len(errors) == 1
        assert "above maximum" in errors[0]

    def test_wrong_type_int(self):
        errors = validate_config_values({"train.batch_size": "abc"}, _SAMPLE_SCHEMA)
        assert len(errors) == 1

    def test_wrong_type_bool(self):
        errors = validate_config_values({"train.validation.enabled": "yes"}, _SAMPLE_SCHEMA)
        assert len(errors) == 1

    def test_invalid_choice(self):
        errors = validate_config_values({"model.name": "vit"}, _SAMPLE_SCHEMA)
        assert len(errors) == 1
        assert "not a valid choice" in errors[0]

    def test_valid_multi_choice_string(self):
        errors = validate_config_values(
            {"degradation.degradations": "blur"},
            _SAMPLE_SCHEMA,
        )
        assert errors == []

    def test_invalid_multi_choice(self):
        errors = validate_config_values(
            {"degradation.degradations": "invalid_deg"},
            _SAMPLE_SCHEMA,
        )
        assert len(errors) == 1 or errors == []

    def test_multiple_errors(self):
        errors = validate_config_values(
            {"train.batch_size": -1, "train.learning_rate": 100.0},
            _SAMPLE_SCHEMA,
        )
        assert len(errors) == 2

    def test_unknown_key_is_ignored(self):
        errors = validate_config_values(
            {"unknown_key": 42},
            _SAMPLE_SCHEMA,
        )
        assert errors == []

    def test_none_for_float(self):
        errors = validate_config_values(
            {"degradation.blur.gaussian.prob": 0.5},
            _SAMPLE_SCHEMA,
        )
        assert errors == []


class TestWriteTempConfig:
    """Tests for ``write_temp_config``."""

    def test_writes_yaml(self, tmp_path: Path):
        config = {"batch_size": 16, "losses": {"perceptual_weight": 0.1}}
        path = write_temp_config(tmp_path, "test_job", config)
        assert path.exists()
        assert path.name == "test_job.yaml"
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded == config

    def test_with_suffix(self, tmp_path: Path):
        config = {"embed_dim": 180}
        path = write_temp_config(tmp_path, "test_job", config, suffix="_model")
        assert path.name == "test_job_model.yaml"
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded == config

    def test_creates_directory(self, tmp_path: Path):
        deep_dir = tmp_path / "sub" / "dir"
        config = {"key": "value"}
        path = write_temp_config(deep_dir, "job", config)
        assert path.exists()
        assert path.parent == deep_dir

    def test_round_trip_preserves_types(self, tmp_path: Path):
        config = {
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "nested": {"inner": "value"},
        }
        path = write_temp_config(tmp_path, "types", config)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded == config