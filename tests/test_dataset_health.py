"""Tests for data/dataset_health.py — dataset health checks."""

from sr_engine.data.dataset_health import check_dataset_health


class TestCheckDatasetHealth:
    """Tests for ``check_dataset_health``."""

    def test_empty_dir(self, tmp_path):
        """An empty directory should not raise."""
        (tmp_path / "HR").mkdir()
        (tmp_path / "LR").mkdir()
        report = check_dataset_health(tmp_path)
        assert report is not None

    def test_healthy_dataset(self, tmp_path):
        """A dataset with valid HR/LR pairs should pass."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _make_image(hr / "f0001.png", 256, 256)
        _make_image(lr / "f0001.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert report is not None

    def test_single_image(self, tmp_path):
        """A directory with a single pair should work."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert report is not None
