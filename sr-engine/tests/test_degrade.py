"""Tests for data/degrade.py — batch degradation pipeline."""

from sr_engine.data.degrade import batch_degrade


class TestBatchDegrade:
    """Tests for ``batch_degrade``."""

    def test_empty_hr_list(self, tmp_path):
        """An empty HR list should return an empty list."""
        result = batch_degrade(
            hr_paths=[],
            lr_dir=tmp_path / "lr",
            scale=4,
            config={},
        )
        assert result == []

    def test_creates_lr_dir(self, tmp_path):
        """The LR output directory should be created."""
        batch_degrade([], tmp_path / "lr", 4, {})
        assert (tmp_path / "lr").is_dir()

    def test_skips_nonexistent_files(self, tmp_path):
        """Nonexistent HR files should be silently skipped."""
        result = batch_degrade(
            hr_paths=[tmp_path / "nonexistent.png"],
            lr_dir=tmp_path / "lr",
            scale=4,
            config={},
        )
        assert result == []
