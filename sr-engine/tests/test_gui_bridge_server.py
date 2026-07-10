"""Tests for gui_bridge/server.py — Server, ClientHandler, ControlHandler, handler dispatch."""

import json
import socket
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr_engine.gui_bridge.server import (
    Server,
    ClientHandler,
    ControlHandler,
    _SYNC_HANDLERS,
    _ASYNC_HANDLERS,
)


class TestHandlerRegistry:
    """Tests for the handler registry dicts."""

    def test_sync_handlers_registered(self):
        """All expected synchronous handler commands should be present."""
        assert "workspace.info" in _SYNC_HANDLERS
        assert "workspace.check" in _SYNC_HANDLERS
        assert "project.list" in _SYNC_HANDLERS
        assert "project.create" in _SYNC_HANDLERS
        assert "dataset.validate" in _SYNC_HANDLERS
        assert "dataset.health" in _SYNC_HANDLERS
        assert "model.info" in _SYNC_HANDLERS
        assert "job.cancel" in _SYNC_HANDLERS
        assert "job.list" in _SYNC_HANDLERS
        assert "job.status" in _SYNC_HANDLERS

    def test_async_handlers_registered(self):
        """All expected asynchronous handler commands should be present."""
        assert "train.start" in _ASYNC_HANDLERS
        assert "infer.start" in _ASYNC_HANDLERS
        assert "dataset.build" in _ASYNC_HANDLERS


class TestClientHandlerDispatch:
    """Tests for ClientHandler — message parsing and dispatch."""

    @pytest.fixture
    def server_mock(self):
        s = MagicMock()
        s._workspace = Path("/fake/ws")
        s._job_manager = MagicMock()
        s.broadcast = MagicMock()
        return s

    @pytest.fixture
    def handler(self, server_mock):
        conn = MagicMock()
        handler = ClientHandler(conn, ("127.0.0.1", 12345), server_mock)
        return handler

    def _send_and_recv(self, handler, command, params=None):
        """Send a command line to the handler and capture the JSON response."""
        msg = {"id": "req_1", "command": command}
        if params:
            msg["params"] = params
        handler._handle_line(json.dumps(msg))
        calls = handler._conn.sendall.call_args_list
        if calls:
            raw = calls[-1][0][0]
            return json.loads(raw.decode("utf-8"))
        return None

    def test_hello_command(self, handler):
        """The hello command should return schema and server version."""
        result = self._send_and_recv(handler, "hello")
        assert result["type"] == "result"
        assert result["data"]["schema_version"] == 1

    def test_unknown_command(self, handler):
        """An unregistered command should return an error."""
        result = self._send_and_recv(handler, "nonexistent")
        assert result["type"] == "error"
        assert "Unknown" in result["message"]

    def test_malformed_json(self, handler):
        """Malformed JSON should return an error."""
        handler._handle_line("{bad}")
        calls = handler._conn.sendall.call_args_list
        if calls:
            raw = calls[-1][0][0]
            result = json.loads(raw.decode("utf-8"))
            assert result["type"] == "error"

    def test_empty_line(self, handler):
        """An empty line should be silently ignored."""
        handler._handle_line("")
        assert handler._conn.sendall.call_count == 0

    def test_missing_command(self, handler):
        """A message without a 'command' field should return an error."""
        handler._handle_line('{"id": "req_1"}')
        calls = handler._conn.sendall.call_args_list
        if calls:
            raw = calls[-1][0][0]
            result = json.loads(raw.decode("utf-8"))
            assert result["type"] == "error"

    def test_workspace_info(self, handler):
        """workspace.info should return the workspace path."""
        result = self._send_and_recv(handler, "workspace.info")
        assert result["type"] == "result"
        assert "workspace" in result["data"]

    def test_workspace_check(self, handler):
        """workspace.check should return existence info."""
        result = self._send_and_recv(handler, "workspace.check")
        assert result["type"] == "result"
        assert "exists" in result["data"]

    def test_project_list(self, handler, server_mock):
        """project.list should return the project list."""
        server_mock._job_manager.list_jobs.return_value = []
        result = self._send_and_recv(handler, "project.list")
        assert result["type"] == "result"

    def test_job_list(self, handler, server_mock):
        """job.list should return the jobs list."""
        server_mock._job_manager.list_jobs.return_value = []
        result = self._send_and_recv(handler, "job.list")
        assert result["type"] == "result"
        assert "jobs" in result["data"]

    def test_job_status(self, handler, server_mock):
        """job.status should return the job manifest."""
        server_mock._job_manager.get_job.return_value = {"job_id": "j1", "status": "running"}
        result = self._send_and_recv(handler, "job.status", {"job_id": "j1"})
        assert result["type"] == "result"
        assert result["data"]["status"] == "running"

    def test_async_train_start(self, handler, server_mock):
        """train.start should return an accepted response."""
        handler._server._job_manager.start_job.return_value = ("job_1", {"status": "accepted"})
        result = self._send_and_recv(handler, "train.start", {"model_name": "rrdb"})
        assert result["type"] == "accepted"

    def test_handler_error_propagates(self, handler, server_mock):
        """An exception in a handler should be caught and returned as an error."""
        handler._server._job_manager.get_job.side_effect = ValueError("broken")
        result = self._send_and_recv(handler, "job.status", {"job_id": "x"})
        assert result["type"] == "error"


class TestControlHandler:
    """Tests for ControlHandler — subprocess control connections."""

    def _make_server_mock(self, running=False):
        srv = MagicMock()
        srv.broadcast = MagicMock()
        srv._running.is_set.return_value = running
        return srv

    def test_hello_accepted(self):
        """A valid hello should receive ack with status 'ok'."""
        a, b = socket.socketpair()
        mgr = MagicMock()
        mgr._on_hello.return_value = True
        srv = self._make_server_mock(running=False)
        srv._job_manager = mgr

        handler = ControlHandler(a, ("127.0.0.1", 0), srv)
        handler.daemon = True
        handler.start()

        hello = json.dumps({"type": "hello", "job_id": "job_1", "token": "abc"}) + "\n"
        b.sendall(hello.encode("utf-8"))

        import time
        time.sleep(0.05)

        b.settimeout(1.0)
        data = b.recv(65536)
        ack = json.loads(data.decode("utf-8").strip())
        assert ack["status"] == "ok"

        mgr._on_hello.assert_called_with("job_1", "abc")
        a.close()
        b.close()

    def test_hello_rejected_bad_token(self):
        """A hello with a wrong token should receive ack with status 'rejected'."""
        a, b = socket.socketpair()
        mgr = MagicMock()
        mgr._on_hello.return_value = False
        srv = self._make_server_mock(running=False)
        srv._job_manager = mgr

        handler = ControlHandler(a, ("127.0.0.1", 0), srv)
        handler.daemon = True
        handler.start()

        hello = json.dumps({"type": "hello", "job_id": "job_1", "token": "wrong"}) + "\n"
        b.sendall(hello.encode("utf-8"))

        import time
        time.sleep(0.05)

        b.settimeout(1.0)
        data = b.recv(65536)
        ack = json.loads(data.decode("utf-8").strip())
        assert ack["status"] == "rejected"
        a.close()
        b.close()

    def test_pipelined_messages_after_hello(self):
        """Messages after hello should be forwarded via broadcast."""
        a, b = socket.socketpair()
        mgr = MagicMock()
        mgr._on_hello.return_value = True
        srv = self._make_server_mock(running=True)
        srv._job_manager = mgr

        handler = ControlHandler(a, ("127.0.0.1", 0), srv)
        handler.daemon = True
        handler.start()

        hello = json.dumps({"type": "hello", "job_id": "job_1", "token": "abc"}) + "\n"
        event = json.dumps({"type": "log", "level": "info", "message": "test"}) + "\n"
        b.sendall((hello + event).encode("utf-8"))

        import time
        time.sleep(0.1)

        srv._running.is_set.return_value = False
        b.sendall(b"\n")

        time.sleep(0.05)
        assert srv.broadcast.call_count >= 1
        a.close()
        b.close()

    def test_first_message_not_hello_rejected(self):
        """A first message that is not hello should be rejected."""
        a, b = socket.socketpair()
        mgr = MagicMock()
        srv = self._make_server_mock(running=False)
        srv._job_manager = mgr

        handler = ControlHandler(a, ("127.0.0.1", 0), srv)
        handler.daemon = True
        handler.start()

        bad = json.dumps({"type": "log", "level": "info"}) + "\n"
        b.sendall(bad.encode("utf-8"))

        import time
        time.sleep(0.05)

        b.settimeout(1.0)
        data = b.recv(65536)
        ack = json.loads(data.decode("utf-8").strip())
        assert ack["status"] == "rejected"

        mgr._on_hello.assert_not_called()
        a.close()
        b.close()


class TestServer:
    """Tests for the Server class lifecycle."""

    def test_initialization(self, tmp_path):
        """Server should be initialised with given host, port, and workspace."""
        srv = Server("127.0.0.1", 0, tmp_path)
        assert srv.host == "127.0.0.1"
        assert srv._workspace == tmp_path
        assert srv._running.is_set()
        srv.stop()

    def test_stop_clears_event(self, tmp_path):
        """stop() should clear the running event."""
        srv = Server("127.0.0.1", 0, tmp_path)
        srv.stop()
        assert not srv._running.is_set()

    def test_broadcast_sends_to_clients(self, tmp_path):
        """broadcast() should send the message to all connected clients."""
        srv = Server("127.0.0.1", 0, tmp_path)

        mock_client = MagicMock()
        mock_client.send_raw.return_value = True
        srv._clients.append(mock_client)

        srv.broadcast({"type": "test", "data": 42})
        mock_client.send_raw.assert_called_once()
        a, _ = mock_client.send_raw.call_args
        assert b"test" in a[0]

        srv.stop()

    def test_broadcast_removes_dead_clients(self, tmp_path):
        """broadcast() should remove clients that fail to receive."""
        srv = Server("127.0.0.1", 0, tmp_path)

        dead = MagicMock()
        dead.send_raw.return_value = False
        alive = MagicMock()
        alive.send_raw.return_value = True
        srv._clients.extend([dead, alive])

        srv.broadcast({"type": "test"})
        assert dead not in srv._clients
        assert alive in srv._clients

        srv.stop()
