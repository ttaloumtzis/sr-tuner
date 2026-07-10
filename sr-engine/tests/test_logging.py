"""Tests for utils/logging.py — get_logger."""

import logging

from sr_engine.utils.logging import get_logger


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_returns_named_logger(self):
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_same_name_same_logger(self):
        logger1 = get_logger("test_name")
        logger2 = get_logger("test_name")
        assert logger1 is logger2
