"""Tests for gui_bridge/protocol.py — SocketReporter, SocketCallback, protocol utilities."""

import json
from unittest.mock import MagicMock

import pytest

from sr_engine.gui_bridge.protocol import (
    SocketReporter,
    SocketCallback,
    make_json_sender,
    parse_message,
    connect_control_socket,
    SchemaVersion,
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_CANCELLED,
)


class TestSchemaConstants:
    def test_schema_version(self):
        assert SchemaVersion.CURRENT == 1

    def test_exit_codes_are_ints(self):
        assert isinstance(EXIT_SUCCESS, int)
        assert isinstance(EXIT_ERROR, int)
        assert isinstance(EXIT_CANCELLED, int)


class TestSocketReporter:
    @pytest.fixture
    def send_fn(self):
        return MagicMock()

    @pytest.fixture
    def reporter(self, send_fn):
        return SocketReporter(send_fn=send_fn, job_id="test_job_1")

    def test_start_sends_message(self, reporter, send_fn):
        reporter.start(total=100, desc="testing")
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "progress_start",
            "total": 100,
            "desc": "testing",
        })

    def test_start_without_total(self, reporter, send_fn):
        reporter.start(desc="no total")
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "progress_start",
            "total": None,
            "desc": "no total",
        })

    def test_update_sends_message(self, reporter, send_fn):
        send_fn.reset_mock()
        reporter.update(n=5)
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "progress_update",
            "n": 5,
        })

    def test_finish_sends_message(self, reporter, send_fn):
        send_fn.reset_mock()
        reporter.finish()
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "progress_end",
        })

    def test_set_description(self, reporter, send_fn):
        send_fn.reset_mock()
        reporter.set_description("processing frame 42")
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "postfix",
            "desc": "processing frame 42",
        })

    def test_set_postfix(self, reporter, send_fn):
        send_fn.reset_mock()
        reporter.set_postfix(loss=0.01, psnr=32.5)
        send_fn.assert_called_once_with({
            "job_id": "test_job_1",
            "type": "postfix",
            "loss": 0.01,
            "psnr": 32.5,
        })


class TestSocketCallback:
    @pytest.fixture
    def send_fn(self):
        return MagicMock()

    @pytest.fixture
    def callback(self, send_fn):
        return SocketCallback(send_fn=send_fn, job_id="test_job_2")

    def test_on_phase(self, callback, send_fn):
        callback.on_phase("train", epoch=1)
        send_fn.assert_called_once_with({
            "job_id": "test_job_2",
            "type": "phase",
            "phase": "train",
            "epoch": 1,
        })

    def test_on_step(self, callback, send_fn):
        callback.on_step(epoch=2, batch=10, total_batches=100, loss=0.05)
        send_fn.assert_called_once_with({
            "job_id": "test_job_2",
            "type": "step",
            "epoch": 2,
            "batch": 10,
            "total_batches": 100,
            "loss": 0.05,
        })

    def test_on_validate(self, callback, send_fn):
        callback.on_validate(epoch=3, psnr=30.0, ssim=0.9)
        send_fn.assert_called_once_with({
            "job_id": "test_job_2",
            "type": "validate",
            "epoch": 3,
            "psnr": 30.0,
            "ssim": 0.9,
        })

    def test_on_done(self, callback, send_fn):
        callback.on_done(elapsed_seconds=42.5)
        send_fn.assert_called_once_with({
            "job_id": "test_job_2",
            "type": "done",
            "elapsed_seconds": 42.5,
        })


class TestMakeJsonSender:
    def test_writes_json_line(self):
        writer = MagicMock()
        sender = make_json_sender(writer)
        sender({"key": "value", "num": 42})
        writer.assert_called_once_with('{"key": "value", "num": 42}\n')

    def test_default_str_for_non_serializable(self):
        writer = MagicMock()
        sender = make_json_sender(writer)
        sender({"path": "foo/bar"})
        args = writer.call_args[0][0]
        assert args.endswith("\n")
        parsed = json.loads(args.strip())
        assert parsed["path"] == "foo/bar"


class TestParseMessage:
    def test_valid_json(self):
        result = parse_message('{"type": "hello", "status": "ok"}')
        assert result == {"type": "hello", "status": "ok"}

    def test_empty_line(self):
        assert parse_message("") is None
        assert parse_message("   ") is None

    def test_malformed_json(self):
        assert parse_message("{bad json}") is None

    def test_trailing_whitespace(self):
        result = parse_message('{"a": 1}\n')
        assert result == {"a": 1}

    def test_strips_whitespace(self):
        assert parse_message("  {\"a\": 1}  ") == {"a": 1}


class TestConnectControlSocket:
    def _server_socket(self):
        import socket
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(5.0)
        return srv

    def test_handshake_success(self):
        import socket
        import threading

        srv = self._server_socket()
        port = srv.getsockname()[1]

        env_value = json.dumps({
            "job_id": "job_1",
            "token": "abc123",
            "control_host": "127.0.0.1",
            "control_port": port,
        })

        def server_side():
            conn, _ = srv.accept()
            data = conn.recv(65536)
            msg = json.loads(data.decode("utf-8").strip())
            assert msg["type"] == "hello"
            assert msg["job_id"] == "job_1"
            conn.sendall(b'{"status": "ok"}\n')
            data2 = conn.recv(65536)
            parsed = json.loads(data2.decode("utf-8").strip())
            assert parsed == {"type": "test", "data": 42}
            conn.close()

        t = threading.Thread(target=server_side, daemon=True)
        t.start()

        job_id, send_fn, close_fn = connect_control_socket(env_value)
        assert job_id == "job_1"

        send_fn({"type": "test", "data": 42})
        close_fn()
        t.join(timeout=2)
        srv.close()

    def test_handshake_rejected(self):
        import socket
        import threading

        srv = self._server_socket()
        port = srv.getsockname()[1]

        env_value = json.dumps({
            "job_id": "job_bad",
            "token": "wrong",
            "control_host": "127.0.0.1",
            "control_port": port,
        })

        def server_reject():
            conn, _ = srv.accept()
            conn.sendall(b'{"status": "rejected", "message": "bad token"}\n')
            conn.close()

        t = threading.Thread(target=server_reject, daemon=True)
        t.start()

        with pytest.raises(ConnectionRefusedError, match="bad token"):
            connect_control_socket(env_value)

        t.join(timeout=2)
        srv.close()
