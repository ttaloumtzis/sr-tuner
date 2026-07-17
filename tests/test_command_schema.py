"""Tests for gui_bridge/command_schema.py — command schema definition."""

from sr_engine.gui_bridge.command_schema import COMMAND_SCHEMA, command_schema_dict


class TestCommandSchemaStructure:
    """Verify the command schema is well-formed."""

    REQUIRED_FIELDS = {"id", "title", "params"}

    def test_has_all_commands(self):
        ids = [c["id"] for c in COMMAND_SCHEMA]
        expected = {
            "hello", "config.schema",
            "workspace.info", "workspace.check", "workspace.init",
            "project.list", "project.create",
            "model.instance_list", "model.instance_info",
            "model.list_runs", "model.export", "model.info",
            "dataset.validate", "dataset.health", "dataset.merge", "dataset.build",
            "train.start", "infer.start",
            "env.check", "env.bench",
            "job.list", "job.status", "job.cancel",
        }
        missing = expected - set(ids)
        extra = set(ids) - expected
        assert not missing, f"Missing commands: {missing}"
        assert not extra, f"Unexpected commands: {extra}"

    def test_all_commands_have_required_fields(self):
        for i, cmd in enumerate(COMMAND_SCHEMA):
            missing = self.REQUIRED_FIELDS - set(cmd.keys())
            assert not missing, f"commands[{i}] ({cmd.get('id', '?')}): missing {missing}"

    def test_all_params_have_key(self):
        for cmd in COMMAND_SCHEMA:
            for j, param in enumerate(cmd.get("params", [])):
                assert "key" in param, f"{cmd['id']}.params[{j}] missing 'key'"

    def test_required_params_have_required_true(self):
        for cmd in COMMAND_SCHEMA:
            for param in cmd.get("params", []):
                if param.get("required"):
                    assert param["required"] is True

    def test_choice_params_have_choices(self):
        for cmd in COMMAND_SCHEMA:
            for param in cmd.get("params", []):
                if param.get("type") == "choice":
                    assert "choices" in param, f"{cmd['id']}.{param['key']} missing choices"

    def test_no_duplicate_command_ids(self):
        ids = [c["id"] for c in COMMAND_SCHEMA]
        assert len(ids) == len(set(ids)), "Duplicate command IDs found"

    def test_async_commands_have_config_sections(self):
        for cmd in COMMAND_SCHEMA:
            if cmd["id"] == "train.start":
                assert "config_sections" in cmd
                assert cmd["config_sections"] == ["training", "model"]
            elif cmd["id"] == "dataset.build":
                assert "config_sections" in cmd
                assert cmd["config_sections"] == ["degradation"]


class TestCommandSchemaDict:
    """Tests for the ``command_schema_dict`` helper."""

    def test_returns_dict_with_commands(self):
        result = command_schema_dict()
        assert "commands" in result
        assert isinstance(result["commands"], list)
        assert len(result["commands"]) == len(COMMAND_SCHEMA)