"""Per-case statistics used for exploratory analysis and split stratification.

Only the *label* volume is loaded into memory. For the CT image we read the header
(shape, voxel spacing, orientation), which is enough here and keeps a full pass over
LiTS (~50 GB) to a few minutes.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from livertumor.data.discovery import Case

LIVER, TUMOR = 1, 2

FIELDS = [
    "case_id",
    "shape_x",
    "shape_y",
    "shape_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "orientation",
    "liver_ml",
    "tumor_ml",
    "tumor_fraction",
    "n_lesions",
    "largest_lesion_ml",
    "smallest_lesion_ml",
    "slices_with_liver",
    "slices_with_tumor",
]


def load_labels(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a label volume as uint8 without going through float64 ``get_fdata``."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    return data.astype(np.uint8), img


def _bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Smallest box containing all True voxels, from cheap per-axis projections."""
    box = []
    for axis in range(mask.ndim):
        others = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.flatnonzero(mask.any(axis=others))
        if idx.size == 0:
            return None
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def count_lesions(tumor_mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Count connected tumor components (26-connectivity in 3D).

    Returns:
        The number of lesions and the voxel count of each lesion.
    """
    box = _bounding_box(tumor_mask)
    if box is None:
        return 0, np.array([], dtype=np.int64)
    # Labelling a full 512x512xN volume is slow; tumors occupy a small bounding box.
    tumor_mask = tumor_mask[box]
    structure = ndimage.generate_binary_structure(tumor_mask.ndim, tumor_mask.ndim)
    labelled, n = ndimage.label(tumor_mask, structure=structure)
    sizes = np.bincount(labelled.ravel())[1:] if n else np.array([], dtype=np.int64)
    return n, sizes


def case_stats(case: Case) -> dict:
    """Compute geometry and label statistics for one case."""
    image_header = nib.load(str(case.image)).header
    labels, label_img = load_labels(case.label)

    if tuple(image_header.get_data_shape()[:3]) != labels.shape[:3]:
        raise ValueError(
            f"Case {case.case_id}: image shape {image_header.get_data_shape()} "
            f"!= label shape {labels.shape}"
        )
    # One pass over the volume gives both the label values present and the voxel counts.
    # order="K" avoids copying NIfTI's Fortran-ordered array.
    counts = np.bincount(labels.ravel(order="K"), minlength=256)
    present = np.flatnonzero(counts)
    unexpected = set(present.tolist()) - {0, LIVER, TUMOR}
    if unexpected:
        raise ValueError(f"Case {case.case_id}: unexpected label values {sorted(unexpected)}")

    spacing = tuple(float(s) for s in label_img.header.get_zooms()[:3])
    voxel_ml = float(np.prod(spacing)) / 1000.0  # mm^3 -> ml

    tumor = labels == TUMOR
    n_lesions, sizes = count_lesions(tumor)

    liver_ml = float(counts[LIVER] + counts[TUMOR]) * voxel_ml  # tumors lie inside the liver
    tumor_ml = float(counts[TUMOR]) * voxel_ml
    return {
        "case_id": case.case_id,
        "shape_x": labels.shape[0],
        "shape_y": labels.shape[1],
        "shape_z": labels.shape[2],
        "spacing_x": round(spacing[0], 4),
        "spacing_y": round(spacing[1], 4),
        "spacing_z": round(spacing[2], 4),
        "orientation": "".join(nib.aff2axcodes(label_img.affine)),
        "liver_ml": round(liver_ml, 2),
        "tumor_ml": round(tumor_ml, 4),
        "tumor_fraction": round(tumor_ml / liver_ml, 5) if liver_ml else 0.0,
        "n_lesions": n_lesions,
        "largest_lesion_ml": round(float(sizes.max()) * voxel_ml, 4) if n_lesions else 0.0,
        "smallest_lesion_ml": round(float(sizes.min()) * voxel_ml, 4) if n_lesions else 0.0,
        "slices_with_liver": int((labels.max(axis=(0, 1)) > 0).sum()),
        "slices_with_tumor": int(tumor.any(axis=(0, 1)).sum()),
    }


def write_stats(rows: Iterable[dict], out_csv: str | Path) -> Path:
    """Write per-case statistics to CSV."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out


def read_stats(path: str | Path) -> list[dict]:
    """Read a CSV written by :func:`write_stats`, converting numbers back."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key == "orientation":
                continue
            try:
                row[key] = int(value)
            except ValueError:
                row[key] = float(value)
    return rows
