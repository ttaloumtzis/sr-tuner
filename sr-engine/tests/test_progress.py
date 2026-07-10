"""Tests for utils/progress.py — ProgressReporter and TqdmReporter."""

from unittest.mock import patch, MagicMock

import pytest

from sr_engine.utils.progress import ProgressReporter, TqdmReporter


class TestProgressReporter:
    @pytest.fixture
    def reporter(self):
        return ProgressReporter()

    def test_start_noop(self, reporter):
        reporter.start(total=100, desc="test")

    def test_update_noop(self, reporter):
        reporter.update(5)

    def test_finish_noop(self, reporter):
        reporter.finish()

    def test_set_description_noop(self, reporter):
        reporter.set_description("desc")

    def test_set_postfix_noop(self, reporter):
        reporter.set_postfix(loss=0.1)


class TestTqdmReporter:
    @pytest.fixture
    def reporter(self):
        return TqdmReporter()

    def test_start_creates_bar(self, reporter):
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            reporter.start(total=50, desc="test")
            mock_tqdm.assert_called_once_with(total=50, desc="test")
            assert reporter._bar is mock_bar

    def test_update_calls_bar(self, reporter):
        reporter._bar = MagicMock()
        reporter.update(3)
        reporter._bar.update.assert_called_once_with(3)

    def test_update_no_bar_no_error(self, reporter):
        reporter.update(5)

    def test_finish_closes_bar(self, reporter):
        bar = MagicMock()
        reporter._bar = bar
        reporter.finish()
        bar.close.assert_called_once()
        assert reporter._bar is None

    def test_finish_no_bar_no_error(self, reporter):
        reporter.finish()

    def test_set_description(self, reporter):
        reporter._bar = MagicMock()
        reporter.set_description("new desc")
        reporter._bar.set_description.assert_called_once_with("new desc")

    def test_set_postfix(self, reporter):
        reporter._bar = MagicMock()
        reporter.set_postfix(loss=0.01)
        reporter._bar.set_postfix.assert_called_once_with(loss=0.01)

    def test_custom_kwargs_passed_to_tqdm(self):
        with patch("tqdm.tqdm") as mock_tqdm:
            reporter = TqdmReporter(position=0, leave=False)
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            reporter.start(total=10, desc="test")
            mock_tqdm.assert_called_once_with(total=10, desc="test", position=0, leave=False)
