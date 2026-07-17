"""Tests for data/dataset_merge.py — merge_datasets."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sr_engine.data.dataset_merge import merge_datasets
from sr_engine.data.dataset_validator import validate


def _make_image(path: Path, w: int = 64, h: int = 64) -> None:
    """Write a small valid PNG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(path), img)


def _create_dataset(root: Path, name: str, scale: int = 4, num_pairs: int = 3) -> Path:
    """Create a dataset under *root* with the given name, scale, and pair count."""
    d = root / name
    hr = d / "HR"
    lr = d / "LR"

    pairs = []
    for i in range(num_pairs):
        fname = f"frame_{i:04d}.png"
        _make_image(hr / fname, w=64 * scale, h=64 * scale)
        _make_image(lr / fname, w=64, h=64)
        pairs.append({"hr": f"HR/{fname}", "lr": f"LR/{fname}"})

    manifest = {
        "config": {"scale": scale},
        "pairs": pairs,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


class TestMergeDatasets:
    """Tests for ``merge_datasets``."""

    def test_two_datasets_same_scale(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "video1", scale=4, num_pairs=3)
        _create_dataset(ds_root, "video2", scale=4, num_pairs=5)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        assert len(results) == 1
        assert results[0].scale == 4
        assert len(results[0].source_datasets) == 2

        merged = results[0].output_path
        hr_files = list((merged / "HR").glob("*.png"))
        lr_files = list((merged / "LR").glob("*.png"))
        assert len(hr_files) == 8
        assert len(lr_files) == 8

        report = validate(merged)
        assert report.ok
        assert report.num_pairs == 8

    def test_multiple_scale_groups(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "s2_a", scale=2, num_pairs=3)
        _create_dataset(ds_root, "s4_a", scale=4, num_pairs=2)
        _create_dataset(ds_root, "s4_b", scale=4, num_pairs=4)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        assert len(results) == 2
        by_scale = {r.scale: r for r in results}
        assert 2 in by_scale
        assert 4 in by_scale
        assert len(by_scale[2].source_datasets) == 1
        assert len(by_scale[4].source_datasets) == 2

        for r in results:
            report = validate(r.output_path)
            assert report.ok

    def test_scale_filter(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "s2_a", scale=2, num_pairs=3)
        _create_dataset(ds_root, "s4_a", scale=4, num_pairs=2)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out, scale=4)

        assert len(results) == 1
        assert results[0].scale == 4

    def test_no_datasets_raises(self, tmp_path):
        ds_root = tmp_path / "empty"
        ds_root.mkdir()

        with pytest.raises(ValueError, match="No valid datasets"):
            merge_datasets(ds_root, tmp_path / "out")

    def test_no_matching_scale_raises(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "s4", scale=4)

        with pytest.raises(ValueError, match="No datasets found with scale=2"):
            merge_datasets(ds_root, tmp_path / "out", scale=2)

    def test_dataset_without_manifest_is_skipped(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "good", scale=4, num_pairs=3)

        no_manifest = ds_root / "no_manifest"
        _make_image(no_manifest / "HR" / "frame.png")
        _make_image(no_manifest / "LR" / "frame.png")

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        assert len(results) == 1
        assert len(results[0].source_datasets) == 1

    def test_existing_output_dir_raises(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "video1", scale=4)

        out = tmp_path / "merged"
        target = out / "scale_4"
        target.mkdir(parents=True)

        with pytest.raises(FileExistsError, match="already exists"):
            merge_datasets(ds_root, out)

    def test_filename_collision(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "a", scale=4, num_pairs=1)
        _create_dataset(ds_root, "b", scale=4, num_pairs=1)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        hr_files = sorted((results[0].output_path / "HR").glob("*.png"))
        names = [f.name for f in hr_files]
        assert len(names) == 2
        assert names[0] != names[1]

    def test_merged_manifest_has_minimal_format(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "vid1", scale=4, num_pairs=3)
        _create_dataset(ds_root, "vid2", scale=4, num_pairs=2)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        manifest_path = results[0].output_path / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        assert data["config"]["scale"] == 4
        assert "vid1" in data["config"]["sources"]
        assert "vid2" in data["config"]["sources"]
        assert data["pairs"] == []

    def test_output_validated(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "v1", scale=4, num_pairs=3)
        _create_dataset(ds_root, "v2", scale=4, num_pairs=3)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        report = validate(results[0].output_path)
        assert report.ok
        assert report.num_pairs == 6

    def test_outdir_excluded_from_discovery(self, tmp_path):
        ds_root = tmp_path / "datasets"
        _create_dataset(ds_root, "v1", scale=4, num_pairs=3)

        out = ds_root / "merged"
        results = merge_datasets(ds_root, out)

        assert len(results) == 1
        assert results[0].output_path == out / "scale_4"

        hr_count = len(list(results[0].output_path.glob("HR/*.png")))
        assert hr_count == 3

    def test_only_scans_immediate_subdirs(self, tmp_path):
        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        (ds_root / "nested").mkdir()

        _create_dataset(ds_root, "top", scale=4, num_pairs=1)
        _create_dataset(ds_root / "nested", "deep", scale=4, num_pairs=1)

        out = tmp_path / "merged"
        results = merge_datasets(ds_root, out)

        assert len(results) == 1
        assert len(results[0].source_datasets) == 1
