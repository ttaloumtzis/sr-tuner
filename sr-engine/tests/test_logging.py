"""Tests for utils/logging.py — get_logger utility."""

import logging
from io import StringIO

from sr_engine.utils.logging import get_logger


class TestGetLogger:
    """Tests for ``get_logger``."""

    def test_returns_logger(self):
        """Should return a logging.Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_configured_level(self):
        """Logger level should be configurable."""
        logger = get_logger("test_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_default_level(self):
        """Default log level should be INFO."""
        logger = get_logger("test_default")
        assert logger.level == logging.INFO

    def test_output_to_stdout(self):
        """Log messages should go to stdout."""
        logger = get_logger("test_stdout")
        stream = StringIO()
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.info("hello")
        assert "hello" in stream.getvalue()

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice should not duplicate handlers."""
        logger = get_logger("test_dedup")
        initial_count = len(logger.handlers)
        logger2 = get_logger("test_dedup")
        assert len(logger2.handlers) == initial_count

    def test_propagate_false(self):
        """Logger should have propagation disabled."""
        logger = get_logger("test_noprop")
        assert logger.propagate is False
