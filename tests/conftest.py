"""Synthetic LiTS-like data so tests run anywhere, without the real 50 GB dataset."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SPACING = (0.8, 0.8, 2.5)  # mm, anisotropic like real CT


def write_case(root: Path, case_id: int, labels: np.ndarray, image_subdir="volume_pt1"):
    affine = np.diag([*SPACING, 1.0])
    image = np.full(labels.shape, -1000, dtype=np.int16)  # air
    image[labels > 0] = 60  # soft tissue
    (root / image_subdir).mkdir(parents=True, exist_ok=True)
    (root / "segmentations").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image, affine), root / image_subdir / f"volume-{case_id}.nii")
    # LiTS stores labels as float; reproduce that to exercise the conversion path.
    nib.save(
        nib.Nifti1Image(labels.astype(np.float32), affine),
        root / "segmentations" / f"segmentation-{case_id}.nii",
    )


def make_labels(n_lesions: int, lesion_size: int = 3) -> np.ndarray:
    """A 32x32x16 volume: a liver block with ``n_lesions`` separated cubic tumors."""
    labels = np.zeros((32, 32, 16), dtype=np.uint8)
    labels[4:28, 4:28, 2:14] = 1
    for i in range(n_lesions):
        x = 6 + i * (lesion_size + 3)
        labels[x : x + lesion_size, 10 : 10 + lesion_size, 5 : 5 + lesion_size] = 2
    return labels


@pytest.fixture
def lits_root(tmp_path: Path) -> Path:
    """Three cases: no tumor, one lesion, two lesions (case 2 in another folder)."""
    write_case(tmp_path, 0, make_labels(0))
    write_case(tmp_path, 1, make_labels(1))
    write_case(tmp_path, 2, make_labels(2), image_subdir="volume_pt2")
    return tmp_path
