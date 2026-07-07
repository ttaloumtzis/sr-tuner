import json
from pathlib import Path

from sr_engine.engine.metrics_stream import MetricsStream


def test_writes_step_message(tmp_path):
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path)
    ms.write({"type": "step", "epoch": 1, "batch": 5, "loss_total": 0.042})
    ms.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    msg = json.loads(lines[0])
    assert msg["type"] == "step"
    assert msg["epoch"] == 1
    assert msg["batch"] == 5
    assert msg["loss_total"] == 0.042


def test_writes_multiple_messages(tmp_path):
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path)
    ms.write({"type": "step", "epoch": 1, "batch": 1})
    ms.write({"type": "step", "epoch": 1, "batch": 2})
    ms.write({"type": "validate", "epoch": 1, "psnr": 32.0})
    ms.close()

    lines = [json.loads(l) for l in path.read_text().strip().split("\n")]
    assert len(lines) == 3
    assert lines[0]["batch"] == 1
    assert lines[1]["batch"] == 2
    assert lines[2]["type"] == "validate"


def test_writes_phase_messages(tmp_path):
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path)
    ms.write({"type": "phase", "phase": "training"})
    ms.write({"type": "phase", "phase": "complete"})
    ms.close()

    lines = [json.loads(l) for l in path.read_text().strip().split("\n")]
    assert lines[0]["phase"] == "training"
    assert lines[1]["phase"] == "complete"


def test_metadata_header(tmp_path):
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path, metadata={"experiment_id": "exp_001", "model": "swinir"})
    ms.write({"type": "phase", "phase": "training"})
    ms.close()

    content = path.read_text()
    assert "#" in content
    header = [l for l in content.split("\n") if l.startswith("#")]
    assert len(header) == 1
    meta = json.loads(header[0][2:])
    assert meta["experiment_id"] == "exp_001"

    json_lines = [l for l in content.split("\n") if l and not l.startswith("#")]
    assert len(json_lines) == 1


def test_appends_to_existing_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("# existing header\n")

    ms = MetricsStream(path)
    ms.write({"type": "step", "batch": 1})
    ms.close()

    content = path.read_text()
    assert "# existing header" in content
    assert '"batch": 1' in content


def test_flush_writes_immediately(tmp_path):
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path)
    ms.write({"type": "step", "batch": 1})
    content_after_write = path.read_text()
    assert '"batch": 1' in content_after_write
    ms.close()


def test_non_serializable_defaults_to_str(tmp_path):
    from pathlib import Path
    path = tmp_path / "metrics.jsonl"
    ms = MetricsStream(path)
    ms.write({"type": "test", "path": Path("/some/path")})
    ms.close()
    msg = json.loads(path.read_text())
    assert msg["path"] == "/some/path"
