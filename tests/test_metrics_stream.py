"""Tests for engine/metrics_stream.py — JSONL metrics streaming."""

import json

from sr_engine.engine.metrics_stream import MetricsStream


class TestMetricsStream:
    """Tests for MetricsStream."""

    def test_write_creates_file(self, tmp_path):
        """Writing should create the output file."""
        path = tmp_path / "metrics" / "run.jsonl"
        stream = MetricsStream(path=path)
        stream.write({"type": "metric", "name": "loss", "value": 0.5})
        stream.close()
        assert path.is_file()

    def test_written_content(self, tmp_path):
        """Written data should be valid JSONL."""
        path = tmp_path / "test.jsonl"
        stream = MetricsStream(path=path)
        stream.write({"name": "loss", "value": 0.5})
        stream.close()

        lines = [l for l in path.read_text().split("\n") if l]
        entry = json.loads(lines[0])
        assert entry["name"] == "loss"
        assert entry["value"] == 0.5

    def test_multiple_writes(self, tmp_path):
        """Multiple writes should produce multiple lines."""
        path = tmp_path / "multi.jsonl"
        stream = MetricsStream(path=path)
        for i in range(5):
            stream.write({"step": i, "loss": i * 0.1})
        stream.close()

        lines = [l for l in path.read_text().split("\n") if l]
        assert len(lines) == 5

    def test_close_idempotent(self, tmp_path):
        """Calling close() multiple times should not error."""
        path = tmp_path / "close.jsonl"
        stream = MetricsStream(path=path)
        stream.write({"msg": "hello"})
        stream.close()
        stream.close()

    def test_metadata_header(self, tmp_path):
        """Metadata should be written as a comment line."""
        path = tmp_path / "meta.jsonl"
        stream = MetricsStream(path=path, metadata={"exp": "test"})
        stream.write({"msg": "hello"})
        stream.close()

        content = path.read_text()
        assert content.startswith("#")
        assert "exp" in content
