"""Locate LiTS image/label pairs on disk.

LiTS files are named ``volume-<id>.nii`` (CT) and ``segmentation-<id>.nii`` (labels:
0 = background, 1 = liver, 2 = tumor). Mirrors such as Kaggle split them across several
folders (``volume_pt1`` ... ``volume_pt5``, ``segmentations``) and sometimes several
datasets, so discovery searches recursively instead of assuming a layout.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_PATTERN = re.compile(r"^(volume|segmentation)-(\d+)\.nii(\.gz)?$")


@dataclass(frozen=True)
class Case:
    """One patient: a CT volume and its label map."""

    case_id: int
    image: Path
    label: Path


def find_cases(roots: Iterable[str | Path]) -> list[Case]:
    """Find all complete (image + label) LiTS cases under one or more root folders.

    Args:
        roots: Folders to search recursively.

    Returns:
        Cases sorted by id. Images without a label (and vice versa) are skipped with a
        warning, since they cannot be used for supervised training.

    Raises:
        FileNotFoundError: If a root does not exist or no complete case is found.
    """
    found: dict[str, dict[int, Path]] = {"volume": {}, "segmentation": {}}
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(f"Data root not found: {root}")
        for path in sorted(root.rglob("*.nii*")):
            match = _PATTERN.match(path.name)
            if not match:
                continue
            kind, case_id = match.group(1), int(match.group(2))
            if case_id in found[kind]:
                warnings.warn(
                    f"Duplicate {kind} for case {case_id}: keeping {found[kind][case_id]}, "
                    f"ignoring {path}",
                    stacklevel=2,
                )
                continue
            found[kind][case_id] = path

    images, labels = found["volume"], found["segmentation"]
    unpaired = sorted(set(images) ^ set(labels))
    if unpaired:
        warnings.warn(f"Skipping cases without image/label pair: {unpaired}", stacklevel=2)

    cases = [Case(i, images[i], labels[i]) for i in sorted(set(images) & set(labels))]
    if not cases:
        raise FileNotFoundError(f"No volume/segmentation pairs found under {list(roots)}")
    return cases
