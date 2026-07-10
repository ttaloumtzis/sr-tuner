"""Tests for Godot GUI bridge protocol utilities."""

import json

from sr_engine.gui_bridge.protocol import parse_message, make_json_sender


class TestProtocolFunctions:
    """Tests for protocol-level utility functions."""

    def test_parse_message_valid(self):
        """A valid JSON line should parse correctly."""
        msg = parse_message('{"type": "heartbeat", "payload": {}}')
        assert msg is not None
        assert msg["type"] == "heartbeat"

    def test_parse_message_empty(self):
        """An empty line should return None."""
        assert parse_message("") is None

    def test_parse_message_whitespace(self):
        """Whitespace-only should return None."""
        assert parse_message("  \n  ") is None

    def test_parse_message_invalid_json(self):
        """Invalid JSON should return None without crashing."""
        assert parse_message("not json") is None

    def test_make_json_sender_writes_json(self):
        """make_json_sender should write JSON lines via the writer."""
        lines = []
        sender = make_json_sender(lines.append)
        sender({"msg": "hello"})
        assert len(lines) == 1
        parsed = json.loads(lines[0].strip())
        assert parsed["msg"] == "hello"
