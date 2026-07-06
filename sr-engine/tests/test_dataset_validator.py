"""Tests for dataset validation."""

from pathlib import Path

import pytest

from sr_engine.data.dataset_validator import validate, ValidationReport


class TestValidate:
    """Tests for dataset_validator.validate(). Stub testing only — real logic TBD."""

    def test_validate_imports_and_returns_report_type(self) -> None:
        assert ValidationReport is not None

    def test_validate_raises_not_implemented(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            validate(tmp_path)
