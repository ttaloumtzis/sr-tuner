"""Shared image file discovery and HR/LR pairing helpers for datasets.

The dataset pipeline (builder, validator, loader, merge, health) must agree
on which image formats are supported and how HR/LR pairs are matched. This
module is the single source of truth for both concerns.
"""

import re
from pathlib import Path

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
"""Image extensions accepted anywhere in the dataset pipeline."""


def is_supported_image(path: Path) -> bool:
    """Return ``True`` if *path* has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTS


def natural_key(path: Path) -> list:
    """Return a digit-aware sort key so ``2.png`` sorts before ``10.png``.

    Numeric runs compare as numbers, everything else as lowercase strings;
    a type marker keeps the two kinds of parts comparable with each other.
    """
    key: list = []
    for part in re.split(r"(\d+)", path.name):
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def list_images(directory: Path) -> list[Path]:
    """Return all supported image files in *directory*, naturally sorted.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of image file paths (files only, no recursion).
    """
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and is_supported_image(p)),
        key=natural_key,
    )


def _stem(path: Path) -> str:
    """Return the filename without its extension (e.g. ``foo.png`` -> ``foo``)."""
    return path.name[: -len(path.suffix)] if path.suffix else path.name


#: Recognized LR-side scale/quality suffixes (e.g. ``000001x2``, ``000001_LR``).
_LR_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:_?x(?P<scale>\d+)|_LR|_lr)$")
#: Recognized HR-side suffix (e.g. ``000001_HR``).
_HR_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_HR$")


def lr_scale_from_suffix(path: Path) -> int | None:
    """Return the scale encoded in an LR filename suffix, if any.

    Recognizes ``x<digits>`` and ``_x<digits>`` markers (e.g.
    ``000001x2.png`` -> ``2``, ``000001_x4.png`` -> ``4``). Returns
    ``None`` when no scale suffix is present.
    """
    m = _LR_SUFFIX_RE.match(_stem(path))
    if m and m.group("scale") is not None:
        return int(m.group("scale"))
    return None


def _base_stem(path: Path) -> tuple[str, str]:
    """Return ``(base_stem, original_stem)`` with known suffixes stripped.

    Strips a recognized HR suffix from HR filenames and a recognized LR
    suffix from LR filenames. When no suffix is present the base stem is
    the raw stem, so plain same-stem matching is unaffected.
    """
    stem = _stem(path)
    for pattern in (_HR_SUFFIX_RE, _LR_SUFFIX_RE):
        m = pattern.match(stem)
        if m:
            return m.group("base"), stem
    return stem, stem


def pair_hr_lr(hr_dir: Path, lr_dir: Path) -> list[tuple[Path, Path]]:
    """Match HR and LR image files into pairs by filename.

    Three-pass matching, each pass only consuming files the previous
    passes left unmatched:

    1. Exact-name match (``HR/foo.png`` <-> ``LR/foo.png``).
    2. Stem match (``HR/foo.png`` <-> ``LR/foo.jpg``) when exactly one
       LR candidate exists; ambiguous matches are skipped.
    3. Suffix-tolerant match for datasets whose LR files carry a scale or
       side marker: ``HR/000001.png`` <-> ``LR/000001x2.png``,
       ``HR/000001.png`` <-> ``LR/000001_LR.png``, and
       ``HR/000001_HR.png`` <-> ``LR/000001_LR.png``. At least one side
       must carry a recognized suffix, and if multiple LR candidates map
       to the same base stem the match is skipped (never guessed).

    Args:
        hr_dir: Directory containing HR images.
        lr_dir: Directory containing LR images.

    Returns:
        List of ``(hr_path, lr_path)`` tuples, naturally sorted by HR name.
    """
    hr_files = list_images(hr_dir)
    lr_files = list_images(lr_dir)
    if not hr_files or not lr_files:
        return []

    lr_by_name = {p.name: p for p in lr_files}

    # Pass 1: exact filename match.
    pairs: list[tuple[Path, Path]] = []
    unmatched_hr: list[Path] = []
    used_lr: set[str] = set()
    for hr in hr_files:
        lr = lr_by_name.get(hr.name)
        if lr is not None:
            pairs.append((hr, lr))
            used_lr.add(lr.name)
        else:
            unmatched_hr.append(hr)

    # Pass 2: unambiguous stem match across formats.
    lr_by_stem: dict[str, list[Path]] = {}
    for lr in lr_files:
        if lr.name in used_lr:
            continue
        lr_by_stem.setdefault(_stem(lr), []).append(lr)

    pass2_remaining: list[Path] = []
    for hr in unmatched_hr:
        candidates = lr_by_stem.get(_stem(hr), [])
        if len(candidates) == 1:
            lr = candidates[0]
            pairs.append((hr, lr))
            used_lr.add(lr.name)
        else:
            pass2_remaining.append(hr)

    # Pass 3: suffix-tolerant base-stem match (scale/side markers).
    # LR files carrying a marker map by their base stem; the marker kind
    # decides what they may pair with.
    lr_by_base: dict[str, list[tuple[Path, bool]]] = {}
    for lr in lr_files:
        if lr.name in used_lr:
            continue
        base, raw = _base_stem(lr)
        if base == raw:
            continue  # no suffix present
        is_scale = lr_scale_from_suffix(lr) is not None
        lr_by_base.setdefault(base, []).append((lr, is_scale))

    for hr in pass2_remaining:
        base, raw = _base_stem(hr)
        if base != raw:
            # HR carries a side marker (e.g. 000001_HR) — pairs with any
            # marker-carrying LR sharing the base (e.g. 000001_LR).
            candidates = [lr for lr, _ in lr_by_base.get(base, [])]
        else:
            # Plain HR (e.g. 000001) — may only pair with LR files whose
            # marker is a scale factor (000001x2), never with bare
            # side-marker names (a_lr) to avoid false positives.
            candidates = [lr for lr, is_scale in lr_by_base.get(raw, []) if is_scale]
        if len(candidates) == 1:
            lr = candidates[0]
            pairs.append((hr, lr))
            used_lr.add(lr.name)

    pairs.sort(key=lambda p: natural_key(p[0]))
    return pairs
