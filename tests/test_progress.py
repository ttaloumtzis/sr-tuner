"""Tests for utils/progress.py — ProgressReporter and TqdmReporter."""

from unittest.mock import patch

from sr_engine.utils.progress import ProgressReporter, TqdmReporter


class TestProgressReporter:
    """Tests for ProgressReporter (no-op base class)."""

    def test_start(self):
        """start() should not raise."""
        r = ProgressReporter()
        r.start(total=10, desc="test")

    def test_update(self):
        """update() should not raise."""
        r = ProgressReporter()
        r.update(1)

    def test_finish(self):
        """finish() should not raise."""
        r = ProgressReporter()
        r.finish()

    def test_set_description(self):
        """set_description() should not raise."""
        r = ProgressReporter()
        r.set_description("working")

    def test_set_postfix(self):
        """set_postfix() should not raise."""
        r = ProgressReporter()
        r.set_postfix(loss=0.5)


class TestTqdmReporter:
    """Tests for TqdmReporter (tqdm-based implementation)."""

    def test_start_creates_bar(self):
        """start() should create a tqdm bar."""
        with patch("tqdm.tqdm") as mock_tqdm:
            r = TqdmReporter()
            r.start(total=10, desc="test")
            mock_tqdm.assert_called_once()

    def test_update_calls_bar(self):
        """update() should call bar.update()."""
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_bar = mock_tqdm.return_value
            r = TqdmReporter()
            r.start(total=10)
            r.update(3)
            mock_bar.update.assert_called_with(3)

    def test_finish_closes_bar(self):
        """finish() should close the bar."""
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_bar = mock_tqdm.return_value
            r = TqdmReporter()
            r.start(total=10)
            r.finish()
            mock_bar.close.assert_called_once()

    def test_no_bar_no_error(self):
        """Calling methods before start() should not raise."""
        r = TqdmReporter()
        r.update(1)
        r.finish()
        r.set_description("done")
        r.set_postfix(ok=True)
