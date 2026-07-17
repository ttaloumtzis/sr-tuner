"""Tests for cli/helpers.py — workspace resolution, config loading."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sr_engine.cli.helpers import (
    resolve_workspace,
    require_workspace,
    resolve_reporter,
    resolve_callbacks,
    resolve_cancel_check,
)


class FakeContext:
    def __init__(self, obj=None):
        self.obj = obj or {}


class TestResolveWorkspace:
    """Tests for ``resolve_workspace``."""

    def test_from_context(self):
        """A workspace cached in the click context should be returned."""
        ws = MagicMock()
        ctx = FakeContext(obj={"workspace": ws})
        result = resolve_workspace(ctx)
        assert result is ws

    def test_from_env(self, monkeypatch):
        """The SRENGINE_WORKSPACE env var should be used when context has none."""
        ws_path = "/tmp/test_ws"
        monkeypatch.setenv("SRENGINE_WORKSPACE", ws_path)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace") as MockWS:
            resolve_workspace(ctx)
            MockWS.assert_called_with(Path(ws_path))

    def test_discover(self, monkeypatch):
        """Fall back to Workspace.discover() when no context or env var."""
        monkeypatch.delenv("SRENGINE_WORKSPACE", raising=False)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace.discover", return_value="discovered_ws"):
            result = resolve_workspace(ctx)
            assert result == "discovered_ws"


class TestRequireWorkspace:
    """Tests for ``require_workspace``."""

    def test_returns_workspace(self):
        """A resolved workspace should be returned."""
        ws = MagicMock()
        ctx = FakeContext(obj={"workspace": ws})
        assert require_workspace(ctx) is ws

    def test_raises_if_missing(self, monkeypatch):
        """A missing workspace should raise a ClickException."""
        monkeypatch.delenv("SRENGINE_WORKSPACE", raising=False)
        ctx = FakeContext()
        with patch("sr_engine.cli.helpers.Workspace.discover", return_value=None):
            with pytest.raises(Exception, match="No workspace found"):
                require_workspace(ctx)


class TestResolveReporter:
    """Tests for ``resolve_reporter``."""

    def test_returns_tqdm_reporter(self):
        """resolve_reporter should always return a TqdmReporter."""
        reporter = resolve_reporter()
        from sr_engine.utils.progress import TqdmReporter
        assert isinstance(reporter, TqdmReporter)


class TestResolveCallbacks:
    """Tests for ``resolve_callbacks``."""

    def test_returns_empty_list(self):
        """resolve_callbacks should always return an empty list."""
        assert resolve_callbacks() == []


class TestResolveCancelCheck:
    """Tests for ``resolve_cancel_check``."""

    def test_returns_false_lambda(self):
        """resolve_cancel_check should always return a lambda returning False."""
        fn = resolve_cancel_check()
        assert fn() is False
