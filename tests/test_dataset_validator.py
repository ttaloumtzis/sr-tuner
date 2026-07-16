"""Tests for data/dataset_validator.py — dataset validation logic."""

from sr_engine.data.dataset_validator import validate


class TestValidate:
    """Tests for ``validate``."""

    def test_validate_healthy(self, minimal_dataset_with_manifest):
        """A complete dataset with manifest should pass."""
        report = validate(minimal_dataset_with_manifest)
        assert report.ok is True

    def test_validate_missing_hr(self, tmp_path):
        """A dataset without an HR directory should produce a problem."""
        lr = tmp_path / "LR"
        lr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("HR" in p for p in report.problems)

    def test_validate_missing_lr(self, tmp_path):
        """A dataset without an LR directory should produce a problem."""
        hr = tmp_path / "HR"
        hr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("LR" in p for p in report.problems)

    def test_validate_missing_manifest(self, tmp_path):
        """A dataset without manifest should report it as a problem."""
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        report = validate(tmp_path)
        assert report.ok is False
        assert any("manifest" in p.lower() for p in report.problems)
