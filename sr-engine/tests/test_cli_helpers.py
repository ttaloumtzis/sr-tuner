"""Tests for cli/helpers.py — workspace resolution, config loading, GUI bridge helpers."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sr_engine.cli.helpers import (
    resolve_workspace,
    require_workspace,
    resolve_reporter,
    resolve_callbacks,
    resolve_cancel_check,
    invalidate_control_connection,
)


class FakeContext:
    def __init__(self, obj=None):
        self.obj = obj or {}


class TestResolveWorkspace:
    def test_from_context(self):
        ws = MagicMock()
        ctx = FakeContext(obj={"workspace": ws})
        result = resolve_workspace(ctx)
        assert result is ws

    def test_from_env(self, monkeypatch):
        ws_path = "/tmp/test_ws"
        monkeypatch.setenv("SRENGINE_WORKSPACE", ws_path)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace") as MockWS:
            result = resolve_workspace(ctx)
            MockWS.assert_called_with(Path(ws_path))

    def test_discover(self, monkeypatch):
        monkeypatch.delenv("SRENGINE_WORKSPACE", raising=False)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace.discover", return_value="discovered_ws"):
            result = resolve_workspace(ctx)
            assert result == "discovered_ws"


class TestRequireWorkspace:
    def test_returns_workspace(self):
        ws = MagicMock()
        ctx = FakeContext(obj={"workspace": ws})
        assert require_workspace(ctx) is ws

    def test_raises_if_missing(self, monkeypatch):
        monkeypatch.delenv("SRENGINE_WORKSPACE", raising=False)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace.discover", return_value=None):
            with pytest.raises(Exception, match="No workspace found"):
                require_workspace(ctx)


class TestResolveReporter:
    def test_tqdm_when_no_gui_socket(self, monkeypatch):
        monkeypatch.delenv("SRENGINE_GUI_SOCKET", raising=False)
        reporter = resolve_reporter()
        from sr_engine.utils.progress import TqdmReporter
        assert isinstance(reporter, TqdmReporter)

    def test_socket_reporter_when_gui_socket_set(self, monkeypatch):
        monkeypatch.setenv("SRENGINE_GUI_SOCKET", '{"job_id":"j1","token":"t","control_host":"127.0.0.1","control_port":9999}')
        with patch("sr_engine.gui_bridge.protocol.connect_control_socket") as mock_connect:
            mock_connect.return_value = ("j1", MagicMock(), MagicMock())
            reporter = resolve_reporter()
            from sr_engine.gui_bridge.protocol import SocketReporter
            assert isinstance(reporter, SocketReporter)
            invalidate_control_connection()


class TestResolveCallbacks:
    def test_empty_when_no_gui_socket(self, monkeypatch):
        monkeypatch.delenv("SRENGINE_GUI_SOCKET", raising=False)
        assert resolve_callbacks() == []

    def test_socket_callback_when_gui_socket_set(self, monkeypatch):
        monkeypatch.setenv("SRENGINE_GUI_SOCKET", '{"job_id":"j1","token":"t","control_host":"127.0.0.1","control_port":9999}')
        with patch("sr_engine.gui_bridge.protocol.connect_control_socket") as mock_connect:
            mock_connect.return_value = ("j1", MagicMock(), MagicMock())
            callbacks = resolve_callbacks()
            assert len(callbacks) == 1
            from sr_engine.gui_bridge.protocol import SocketCallback
            assert isinstance(callbacks[0], SocketCallback)
            invalidate_control_connection()


class TestResolveCancelCheck:
    def test_returns_false_lambda_when_no_socket(self, monkeypatch):
        monkeypatch.delenv("SRENGINE_GUI_SOCKET", raising=False)
        fn = resolve_cancel_check()
        assert fn() is False

    def test_installs_handler_when_socket_set(self, monkeypatch):
        monkeypatch.setenv("SRENGINE_GUI_SOCKET", "dummy")
        with patch("sr_engine.gui_bridge.jobs.install_cancel_handler") as mock_install:
            def true_fn():
                return True
            with patch("sr_engine.gui_bridge.jobs.was_cancelled", true_fn):
                fn = resolve_cancel_check()
                mock_install.assert_called_once()
                assert fn() is True
