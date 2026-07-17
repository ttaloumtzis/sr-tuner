"""Tests for gui_bridge/config_schema.py — schema definition and helpers."""

from sr_engine.gui_bridge.config_schema import (
    CONFIG_SECTIONS,
    schema_to_defaults,
    schema_for_model,
    all_params,
)


class TestConfigSchemaStructure:
    """Verify the config schema is well-formed."""

    REQUIRED_FIELDS = {"key", "type", "default", "group"}

    def test_has_three_sections(self):
        assert set(CONFIG_SECTIONS.keys()) == {"training", "model", "degradation"}

    def test_all_params_have_required_fields(self):
        for section_name, section in CONFIG_SECTIONS.items():
            for i, param in enumerate(section["params"]):
                missing = self.REQUIRED_FIELDS - set(param.keys())
                assert not missing, (
                    f"{section_name}.params[{i}] ({param.get('key', '?')}): "
                    f"missing fields: {missing}"
                )

    def test_training_params_have_ranges(self):
        for param in CONFIG_SECTIONS["training"]["params"]:
            if param["type"] in ("int", "float"):
                assert "min" in param, f"{param['key']} missing min"
                assert "max" in param, f"{param['key']} missing max"
                assert "step" in param, f"{param['key']} missing step"

    def test_degradation_params_have_ranges(self):
        for param in CONFIG_SECTIONS["degradation"]["params"]:
            if param["type"] in ("int", "float"):
                assert "min" in param, f"{param['key']} missing min"
                assert "max" in param, f"{param['key']} missing max"

    def test_model_section_has_applies_to_on_specific_params(self):
        for param in CONFIG_SECTIONS["model"]["params"]:
            if param["key"] in ("model.name", "model.scale"):
                assert "applies_to" not in param, f"{param['key']} should not have applies_to"
            else:
                assert "applies_to" in param, f"{param['key']} missing applies_to"

    def test_choice_params_have_choices(self):
        for section in CONFIG_SECTIONS.values():
            for param in section["params"]:
                if param["type"] == "choice":
                    assert "choices" in param, f"{param['key']} missing choices"
                    assert len(param["choices"]) > 0

    def test_bool_params_are_bool_defaults(self):
        for section in CONFIG_SECTIONS.values():
            for param in section["params"]:
                if param["type"] == "bool":
                    assert isinstance(param["default"], bool), (
                        f"{param['key']} default should be bool, got {type(param['default'])}"
                    )

    def test_no_duplicate_keys(self):
        seen = set()
        for section in CONFIG_SECTIONS.values():
            for param in section["params"]:
                assert param["key"] not in seen, f"Duplicate key: {param['key']}"
                seen.add(param["key"])


class TestSchemaToDefaults:
    """Tests for the ``schema_to_defaults`` helper."""

    def test_returns_flat_dict(self):
        defaults = schema_to_defaults(model="rrdb_esrgan")
        assert isinstance(defaults, dict)
        assert len(defaults) > 10

    def test_includes_training_params(self):
        defaults = schema_to_defaults()
        assert "train.batch_size" in defaults
        assert defaults["train.batch_size"] == 32

    def test_filters_model_params_by_model(self):
        swinir_defaults = schema_to_defaults(model="swinir")
        rrdb_defaults = schema_to_defaults(model="rrdb_esrgan")

        assert "model.embed_dim" in swinir_defaults
        assert "model.embed_dim" not in rrdb_defaults
        assert "model.num_feat" in rrdb_defaults
        assert "model.num_feat" not in swinir_defaults

    def test_model_name_always_present(self):
        defaults = schema_to_defaults(model="swinir")
        assert "model.name" in defaults

    def test_degradation_included(self):
        defaults = schema_to_defaults()
        deg_keys = [k for k in defaults if k.startswith("degradation.")]
        assert len(deg_keys) > 10


class TestSchemaForModel:
    """Tests for the ``schema_for_model`` helper."""

    def test_returns_list(self):
        params = schema_for_model("swinir")
        assert isinstance(params, list)

    def test_swinir_has_swinir_params(self):
        params = schema_for_model("swinir")
        keys = [p["key"] for p in params]
        assert "model.embed_dim" in keys
        assert "model.num_feat" not in keys

    def test_rrdb_has_rrdb_params(self):
        params = schema_for_model("rrdb_esrgan")
        keys = [p["key"] for p in params]
        assert "model.num_feat" in keys
        assert "model.embed_dim" not in keys

    def test_general_params_always_present(self):
        for model in ("swinir", "rrdb_esrgan"):
            params = schema_for_model(model)
            keys = [p["key"] for p in params]
            assert "model.name" in keys
            assert "model.scale" in keys


class TestAllParams:
    """Tests for the ``all_params`` helper."""

    def test_returns_all_params(self):
        params = all_params()
        assert isinstance(params, list)
        assert len(params) > 30

    def test_includes_all_sections(self):
        param_keys = {p["key"] for p in all_params()}
        assert "train.batch_size" in param_keys
        assert "model.name" in param_keys
        assert "degradation.blur.enabled" in param_keys