"""Tests for utils/logging.py — get_logger utility."""

import logging

from sr_engine.utils.logging import get_logger


class TestGetLogger:
    """Tests for ``get_logger``."""

    def test_returns_logger(self):
        """Should return a logging.Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_default_level_notset(self):
        """Default log level should be NOTSET (inherits from root)."""
        logger = get_logger("test_default")
        assert logger.level == logging.NOTSET

    def test_propagate_true(self):
        """Logger should propagate to root (structlog handles centrally)."""
        logger = get_logger("test_noprop")
        assert logger.propagate is True

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice should not duplicate handlers."""
        logger = get_logger("test_dedup")
        initial_count = len(logger.handlers)
        logger2 = get_logger("test_dedup")
        assert len(logger2.handlers) == initial_count
