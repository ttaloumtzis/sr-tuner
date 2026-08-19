"""Tests for data/dataset_health.py — dataset health checks."""

import json
import struct

import cv2
import numpy as np
import pytest

from sr_engine.data.dataset_health import (
    DatasetIntegrityError,
    check_dataset_health,
    find_unreadable_images,
    load_health_report,
    prune_pairs,
)


def _make_truncated_png(path, declared_w=2592, declared_h=1536):
    """Write a PNG with a valid signature/IHDR but a truncated IDAT stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", declared_w, declared_h, 8, 2, 0, 0, 0)
    ihdr = b"IHDR" + ihdr_data + struct.pack(">I", 0xFFFFFFFF)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + ihdr + struct.pack(">I", 0xFFFFFFFF)
    partial_idat = b"IDAT" + b"\x78\x9c" + b"\x00" * 4096
    idat_chunk = struct.pack(">I", 4098) + partial_idat
    path.write_bytes(sig + ihdr_chunk + idat_chunk)


def _make_black_image(path, w=64, h=64):
    """Write a black (all zeros) RGB image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_near_black_image(path, w=64, h=64, mean_val=2.0):
    """Write a near-black image with low pixel variation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), int(mean_val), dtype=np.uint8)
    noise = np.random.default_rng(42).integers(-2, 3, (h, w, 3), dtype=np.int8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


class TestFindUnreadableImages:
    """Tests for ``find_unreadable_images``."""

    def test_healthy_manifest_dataset(self, tmp_path):
        """A dataset with valid HR/LR pairs should report no unreadable files."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=3)
        assert find_unreadable_images(d) == []

    def test_truncated_png_detected(self, tmp_path):
        """A truncated PNG (like the 2DNR Scene15 failure) must be flagged."""
        from conftest import _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "Scene1.png", 256, 256)
        _make_image(d / "LR" / "Scene1.png", 64, 64)
        _make_truncated_png(d / "HR" / "Scene15.png")
        _make_image(d / "LR" / "Scene15.png", 64, 64)
        unreadable = find_unreadable_images(d)
        assert unreadable == ["HR/Scene15.png"]

    def test_lr_only_corrupt_detected(self, tmp_path):
        """A corrupt LR file must be flagged even when the HR twin is fine."""
        from conftest import _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "a.png", 256, 256)
        _make_image(d / "LR" / "a.png", 64, 64)
        _make_truncated_png(d / "LR" / "b.png")
        _make_image(d / "HR" / "b.png", 256, 256)
        assert find_unreadable_images(d) == ["LR/b.png"]

    def test_missing_manifest_file_detected(self, tmp_path):
        """Files registered in the manifest but missing on disk are flagged."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=3)
        manifest = json.loads((d / "manifest.json").read_text())
        manifest["pairs"].append({"hr": "HR/ghost.png", "lr": "LR/ghost.png"})
        (d / "manifest.json").write_text(json.dumps(manifest))
        unreadable = find_unreadable_images(d)
        assert "HR/ghost.png" in unreadable
        assert "LR/ghost.png" in unreadable

    def test_no_manifest_directory_scan(self, tmp_path):
        """Without a manifest, both HR/ and LR/ are scanned."""
        from conftest import _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "a.png", 256, 256)
        _make_image(d / "LR" / "a.png", 64, 64)
        _make_truncated_png(d / "LR" / "orphan.png")
        unreadable = find_unreadable_images(d)
        assert unreadable == ["LR/orphan.png"]

    def test_plain_garbage_image_detected(self, tmp_path):
        """Invalid bytes are also detected as unreadable."""
        from conftest import _make_corrupt_image, _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "a.png", 256, 256)
        _make_image(d / "LR" / "a.png", 64, 64)
        _make_corrupt_image(d / "LR" / "b.png")
        _make_image(d / "HR" / "b.png", 256, 256)
        assert find_unreadable_images(d) == ["LR/b.png"]


class TestDatasetIntegrityError:
    """Tests for ``DatasetIntegrityError``."""

    def test_message_lists_files(self):
        """The message lists the corrupt files and a count."""
        err = DatasetIntegrityError(["HR/a.png", "LR/b.png"])
        assert "2 unreadable image(s)" in str(err)
        assert "HR/a.png" in str(err)
        assert err.files == ["HR/a.png", "LR/b.png"]

    def test_message_truncates_long_lists(self):
        """Long file lists are truncated with a suffix."""
        files = [f"HR/f{i:03d}.png" for i in range(50)]
        err = DatasetIntegrityError(files)
        assert "+30 more" in str(err)
        assert "HR/f049.png" not in str(err)


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
        assert report["unreadable"] == []
        assert report["total_pairs"] == 2
        assert report["total_hr_images"] == 2
        assert report["total_lr_images"] == 2
        assert report["black_frames"] == []
        assert report["suspicious_frames"] == []
        assert report["scale_mismatches"] == []

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

    def test_report_includes_unreadable(self, tmp_path):
        """The health report lists corrupt files."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _make_truncated_png(hr / "f0001.png")
        _make_image(lr / "f0001.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert report["unreadable"] == ["HR/f0001.png"]

    def test_integrity_scan_reports_progress(self, tmp_path):
        """The unreadable scan must report progress through the reporter."""
        from conftest import _create_dataset_with_manifest
        from sr_engine.utils.progress import ProgressReporter

        class RecordingReporter(ProgressReporter):
            def __init__(self):
                self.starts = []
                self.updates = 0

            def start(self, total=None, desc=""):
                self.starts.append((total, desc))

            def update(self, n=1):
                self.updates += n

            def finish(self):
                pass

        d = _create_dataset_with_manifest(tmp_path, num_pairs=2)
        reporter = RecordingReporter()
        check_dataset_health(d, reporter=reporter)
        # 2 HR metrics + 2 LR metrics + 4 integrity files = 8 updates
        assert reporter.updates == 8, f"Expected 8 updates, got {reporter.updates}"
        assert any(desc == "Analyzing HR Images" for _, desc in reporter.starts)
        assert any(desc == "Analyzing LR Images" for _, desc in reporter.starts)
        assert any(desc == "Checking Image Integrity" for _, desc in reporter.starts)

    def test_old_report_normalized_on_load(self, tmp_path):
        """Cached reports without the 'unreadable' key are normalized."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        report = {"total_images": 1, "resolutions": {}, "aspect_ratios": {},
                  "channels": {}, "computed_threshold": 3.0, "black_frames": []}
        (tmp_path / ".health_report.json").write_text(json.dumps(report))
        loaded = load_health_report(tmp_path)
        assert loaded is not None
        assert loaded["unreadable"] == []

    def test_black_frames_in_lr_detected(self, tmp_path):
        """Black frames in LR must be flagged even when HR is fine."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _make_image(hr / "f0001.png", 256, 256)
        _make_black_image(lr / "f0001.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert report is not None
        assert "LR/f0001.png" in report["black_frames"], (
            f"Expected LR/f0001.png in black_frames, got: {report['black_frames']}"
        )
        assert "HR/f0001.png" not in report["black_frames"]

    def test_black_frames_in_hr_detected(self, tmp_path):
        """Black frames in HR must be flagged (both sides checked)."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _make_black_image(hr / "f0001.png", 256, 256)
        _make_image(lr / "f0001.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert "HR/f0001.png" in report["black_frames"], (
            f"Expected HR/f0001.png in black_frames, got: {report['black_frames']}"
        )

    def test_suspicious_frames_listed(self, tmp_path):
        """Borderline low-brightness frames go to suspicious_frames."""
        from conftest import _make_image
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _make_near_black_image(hr / "f0001.png", 256, 256, mean_val=3.0)
        _make_image(lr / "f0001.png", 64, 64)
        report = check_dataset_health(tmp_path)
        assert "HR/f0001.png" in report.get("suspicious_frames", []) or \
               "HR/f0001.png" in report.get("black_frames", []), (
            f"Expected HR/f0001.png in black or suspicious, got black={report['black_frames']} "
            f"suspicious={report['suspicious_frames']}"
        )

    def test_scale_mismatch_detected(self, tmp_path):
        """LR with wrong dimensions is reported as a scale mismatch."""
        from conftest import _make_image, _create_manifest
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 64, 64)
        _create_manifest(tmp_path, scale=4)
        report = check_dataset_health(tmp_path)
        assert report is not None
        assert report["scale_mismatches"] == []

    def test_scale_mismatch_detected_wrong_size(self, tmp_path):
        """LR with wrong dimensions is reported as a scale mismatch."""
        from conftest import _make_image, _create_manifest
        hr = tmp_path / "HR"
        lr = tmp_path / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        _make_image(hr / "f0000.png", 256, 256)
        _make_image(lr / "f0000.png", 128, 128)  # wrong: should be 64x64 for scale=4
        _create_manifest(tmp_path, scale=4)
        report = check_dataset_health(tmp_path)
        assert len(report["scale_mismatches"]) == 1, (
            f"Expected 1 scale mismatch, got: {report['scale_mismatches']}"
        )
        assert "LR/f0000.png" in report["scale_mismatches"][0]


class TestPrunePairs:
    """Tests for ``prune_pairs``."""

    def test_removes_both_sides_and_manifest_entry(self, tmp_path):
        """Pruning a pair deletes HR + LR files and the manifest entry."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=3)
        hr_file = d / "HR" / "frame_0001.png"
        lr_file = d / "LR" / "frame_0001.png"
        assert hr_file.is_file()
        prune_pairs(d, ["HR/frame_0001.png"])
        assert not hr_file.exists()
        assert not lr_file.exists()
        manifest = json.loads((d / "manifest.json").read_text())
        assert len(manifest["pairs"]) == 2
        rels = {(p["hr"], p["lr"]) for p in manifest["pairs"]}
        assert ("HR/frame_0001.png", "LR/frame_0001.png") not in rels

    def test_lr_path_prunes_pair(self, tmp_path):
        """Passing the LR relative path also removes the whole pair."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=3)
        prune_pairs(d, ["LR/frame_0002.png"])
        assert not (d / "HR" / "frame_0002.png").exists()
        assert not (d / "LR" / "frame_0002.png").exists()
        manifest = json.loads((d / "manifest.json").read_text())
        assert len(manifest["pairs"]) == 2

    def test_mismatched_hr_lr_names(self, tmp_path):
        """Pairs with different HR/LR filenames are removed as a unit."""
        from conftest import _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "scene_hr.png", 256, 256)
        _make_image(d / "LR" / "scene_lr.png", 64, 64)
        manifest = {
            "config": {"scale": 4},
            "pairs": [{"hr": "HR/scene_hr.png", "lr": "LR/scene_lr.png"}],
        }
        (d / "manifest.json").write_text(json.dumps(manifest))
        prune_pairs(d, ["HR/scene_hr.png"])
        assert not (d / "HR" / "scene_hr.png").exists()
        assert not (d / "LR" / "scene_lr.png").exists()
        manifest_after = json.loads((d / "manifest.json").read_text())
        assert manifest_after["pairs"] == []

    def test_stale_health_report_deleted(self, tmp_path):
        """Pruning drops the stale cached health report."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=3)
        (d / ".health_report.json").write_text(json.dumps({"unreadable": []}))
        prune_pairs(d, ["HR/frame_0000.png"])
        assert not (d / ".health_report.json").exists()

    def test_manifest_missing_falls_back_to_stem_match(self, tmp_path):
        """Without a manifest, the twin is found by stem matching."""
        from conftest import _make_image
        d = tmp_path / "dataset"
        _make_image(d / "HR" / "a.png", 256, 256)
        _make_image(d / "LR" / "a.png", 64, 64)
        prune_pairs(d, ["LR/a.png"])
        assert not (d / "LR" / "a.png").exists()
        assert not (d / "HR" / "a.png").exists()

    def test_unknown_file_ignored(self, tmp_path):
        """A path that matches nothing is silently skipped."""
        from conftest import _create_dataset_with_manifest
        d = _create_dataset_with_manifest(tmp_path, num_pairs=2)
        prune_pairs(d, ["HR/does_not_exist.png"])
        manifest = json.loads((d / "manifest.json").read_text())
        assert len(manifest["pairs"]) == 2
