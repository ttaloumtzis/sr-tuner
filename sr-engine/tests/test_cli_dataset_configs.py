"""Tests for dataset CLI commands — default vs custom configs."""


def test_dataset_build_help(cli_invoker):
    """``dataset build --help`` should succeed."""
    r = cli_invoker(["dataset", "build", "--help"])
    assert r.exit_code == 0


def test_dataset_validate_help(cli_invoker):
    """``dataset validate --help`` should succeed."""
    r = cli_invoker(["dataset", "validate", "--help"])
    assert r.exit_code == 0


def test_dataset_health_help(cli_invoker):
    """``dataset health --help`` should succeed."""
    r = cli_invoker(["dataset", "health", "--help"])
    assert r.exit_code == 0


class TestDatasetBuildDefault:
    """Tests for the dataset build command with default config."""

    def test_build_from_preprocessed(self, cli_invoker, tmp_path):
        """Build should succeed from an existing HR/LR directory."""
        from conftest import _make_image
        dataset = tmp_path / "preprocessed"
        for i in range(3):
            _make_image(dataset / "HR" / f"f{i:04d}.png", w=256, h=256)
            _make_image(dataset / "LR" / f"f{i:04d}.png", w=64, h=64)

        r = cli_invoker(["dataset", "build", "--input", str(dataset)])
        assert r.exit_code == 0, r.output

    def test_build_with_custom_config(self, cli_invoker, tmp_path):
        """Build should accept a custom config file."""
        from conftest import _make_image
        dataset = tmp_path / "preprocessed"
        for i in range(3):
            _make_image(dataset / "HR" / f"f{i:04d}.png", w=128, h=128)
            _make_image(dataset / "LR" / f"f{i:04d}.png", w=64, h=64)

        cfg_path = tmp_path / "custom_dataset.yaml"
        cfg_path.write_text("scale: 2\n")

        r = cli_invoker([
            "dataset", "build",
            "--input", str(dataset),
            "--config", str(cfg_path),
        ])
        assert r.exit_code == 0, r.output

    def test_dump_config(self, cli_invoker, tmp_path):
        """``--dump-config`` should print the merged config."""
        r = cli_invoker([
            "dataset", "build", "--dump-config",
            "--input", str(tmp_path),
        ])
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert "degradation" in cfg


class TestDatasetValidate:
    """Tests for dataset validation."""

    def test_validate_healthy_dataset(self, cli_invoker, tmp_path):
        """A healthy dataset with manifest should pass validation."""
        from conftest import _make_image, _create_manifest
        dataset = tmp_path / "healthy"
        for i in range(3):
            _make_image(dataset / "HR" / f"f{i:04d}.png", w=256, h=256)
            _make_image(dataset / "LR" / f"f{i:04d}.png", w=64, h=64)
        _create_manifest(dataset)

        r = cli_invoker(["dataset", "validate", "--path", str(dataset)])
        assert r.exit_code == 0, r.output

    def test_validate_missing_manifest(self, cli_invoker, tmp_path):
        """A dataset without a manifest should fail validation."""
        from conftest import _make_image
        dataset = tmp_path / "broken"
        _make_image(dataset / "HR" / "f0000.png", w=256, h=256)
        _make_image(dataset / "LR" / "f0000.png", w=64, h=64)
        r = cli_invoker(["dataset", "validate", "--path", str(dataset)])
        assert r.exit_code != 0


class TestDatasetHealth:
    """Tests for dataset health check."""

    def test_health_check(self, cli_invoker, tmp_path):
        """A healthy dataset should pass the health check."""
        from conftest import _make_image
        dataset = tmp_path / "health_test"
        for i in range(3):
            _make_image(dataset / "HR" / f"f{i:04d}.png", w=256, h=256)
            _make_image(dataset / "LR" / f"f{i:04d}.png", w=64, h=64)

        r = cli_invoker(["dataset", "health", "--path", str(dataset), "--yes"])
        assert r.exit_code == 0, r.output
