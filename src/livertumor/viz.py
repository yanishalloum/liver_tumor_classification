"""Figures of CT scans with liver/tumor contours, used to illustrate each pipeline step.

Display convention (radiological, as on a hospital workstation): axial slices are seen
from the patient's feet, so the patient's **right is on the viewer's left** and anterior
(belly) is at the top. Volumes are first reoriented to RAS so this works for every case.

Requires the ``viz`` extra (matplotlib).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: Kaggle, CI

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from livertumor.data.discovery import Case  # noqa: E402
from livertumor.data.stats import LIVER, TUMOR  # noqa: E402
from livertumor.intensity import ABDOMEN_WINDOW, window_ct  # noqa: E402

LIVER_COLOR, TUMOR_COLOR = "#1baf7a", "#eb6834"
INK, INK_2 = "#0b0b0b", "#52514e"
NARROW_LIVER_WINDOW = (30.0, 150.0)  # the window used by the legacy version


@dataclass
class Volume:
    """A case loaded in RAS orientation."""

    case_id: int
    image: np.ndarray  # HU, float32, axes (x: R, y: A, z: S)
    labels: np.ndarray  # uint8
    spacing: tuple[float, float, float]  # mm
    original_orientation: str


def load_volume(case: Case, canonical: bool = True) -> Volume:
    """Load image and labels, optionally reoriented to RAS (closest canonical)."""
    image_nii, label_nii = nib.load(str(case.image)), nib.load(str(case.label))
    original = "".join(nib.aff2axcodes(image_nii.affine))
    if canonical:
        image_nii = nib.as_closest_canonical(image_nii)
        label_nii = nib.as_closest_canonical(label_nii)
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    labels = np.rint(np.asarray(label_nii.dataobj)).astype(np.uint8)
    spacing = tuple(float(s) for s in image_nii.header.get_zooms()[:3])
    return Volume(case.case_id, image, labels, spacing, original)


def radiological_axial(plane: np.ndarray) -> np.ndarray:
    """Turn an (x, y) RAS slice into display rows/cols: anterior up, patient right left."""
    return plane[::-1, ::-1].T


def radiological_coronal(plane: np.ndarray) -> np.ndarray:
    """Turn an (x, z) RAS plane into display rows/cols: head up, patient right left."""
    return plane[::-1, ::-1].T


def largest_slice(mask: np.ndarray, axis: int = 2) -> int:
    """Index of the slice with the most foreground voxels along ``axis``."""
    others = tuple(a for a in range(mask.ndim) if a != axis)
    return int(np.argmax(mask.sum(axis=others)))


def draw(
    ax: plt.Axes,
    image: np.ndarray,
    labels: np.ndarray | None = None,
    window: tuple[float, float] | None = ABDOMEN_WINDOW,
    aspect: float = 1.0,
    title: str | None = None,
) -> None:
    """Show a 2D display-oriented image with optional liver/tumor contours."""
    shown = window_ct(image, *window) if window else image
    ax.imshow(shown, cmap="gray", aspect=aspect, interpolation="bilinear")
    if labels is not None:
        for value, color in ((LIVER, LIVER_COLOR), (TUMOR, TUMOR_COLOR)):
            # tumor lies inside the liver: draw the liver outline around liver + tumor
            region = labels >= LIVER if value == LIVER else labels == TUMOR
            if region.any():
                ax.contour(region, levels=[0.5], colors=color, linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def _legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([], [], color=LIVER_COLOR, lw=2, label="Liver (label 1)"),
        Line2D([], [], color=TUMOR_COLOR, lw=2, label="Tumor (label 2)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)


def describe(vol: Volume) -> str:
    """Short caption: tumor volume and lesion count."""
    from livertumor.data.stats import count_lesions

    tumor = vol.labels == TUMOR
    if not tumor.any():
        return "no tumor"
    ml = tumor.sum() * float(np.prod(vol.spacing)) / 1000
    n, _ = count_lesions(tumor)
    return f"tumor {ml:.1f} ml, {n} lesion{'s' if n > 1 else ''}"


def gallery_figure(volumes: Sequence[Volume], out: Path) -> Path:
    """One row per case: axial slice (no overlay), same slice with contours, coronal view."""
    fig, axes = plt.subplots(len(volumes), 3, figsize=(10, 3.4 * len(volumes)))
    axes = np.atleast_2d(axes)
    for row, vol in zip(axes, volumes, strict=True):
        target = vol.labels == TUMOR if (vol.labels == TUMOR).any() else vol.labels >= LIVER
        z = largest_slice(target, axis=2)
        y = largest_slice(target, axis=1)
        axial_img = radiological_axial(vol.image[:, :, z])
        axial_lab = radiological_axial(vol.labels[:, :, z])
        coronal_img = radiological_coronal(vol.image[:, y, :])
        coronal_lab = radiological_coronal(vol.labels[:, y, :])
        aspect_cor = vol.spacing[2] / vol.spacing[0]  # z mm per x mm: true proportions
        draw(row[0], axial_img, title=f"Case {vol.case_id} · {describe(vol)}")
        draw(row[1], axial_img, axial_lab, title=f"Axial slice {z}, with labels")
        draw(row[2], coronal_img, coronal_lab, aspect=aspect_cor, title="Coronal view")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out / "scan_gallery.png")


def windowing_figure(vol: Volume, out: Path) -> Path:
    """Same slice: full HU range, abdominal window, narrow liver window + HU histogram."""
    z = largest_slice(vol.labels == TUMOR, axis=2)
    img = radiological_axial(vol.image[:, :, z])
    lab = radiological_axial(vol.labels[:, :, z])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    draw(axes[0], img, window=None, title="Raw HU (full range)")
    draw(axes[1], img, lab, window=ABDOMEN_WINDOW, title="Abdominal window 40 / 400")
    draw(axes[2], img, lab, window=NARROW_LIVER_WINDOW, title="Narrow window 30 / 150")

    ax = axes[3]
    bins = np.arange(-1100, 1500, 20)
    for value, color, name in ((LIVER, LIVER_COLOR, "liver"), (TUMOR, TUMOR_COLOR, "tumor")):
        ax.hist(
            vol.image[vol.labels == value],
            bins=bins,
            color=color,
            alpha=0.85,
            label=name,
            density=True,
        )
    ax.hist(
        vol.image[vol.labels == 0][::50],
        bins=bins,
        color="#9b9a95",
        alpha=0.5,
        label="everything else",
        density=True,
    )
    top = ax.get_ylim()[1]
    for (level, width), name, height in (
        (ABDOMEN_WINDOW, "window 40/400", 0.97),
        (NARROW_LIVER_WINDOW, "window 30/150", 0.85),
    ):
        ax.axvspan(level - width / 2, level + width / 2, color="#2a78d6", alpha=0.08)
        ax.text(level + width / 2, top * height, f" {name}", fontsize=8, color=INK_2, va="top")
    ax.set_xlim(-1100, 1000)
    ax.set_yticks([])
    ax.set_xlabel("Hounsfield units")
    ax.set_title("Intensity distribution", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(fig, out / "windowing.png")


def orientation_figure(cases: Sequence[Case], out: Path) -> Path:
    """Top: raw array as stored on disk. Bottom: after reorientation to RAS."""
    fig, axes = plt.subplots(2, len(cases), figsize=(3.4 * len(cases), 7.2), squeeze=False)
    for col, case in enumerate(cases):
        raw = load_volume(case, canonical=False)
        z = largest_slice(raw.labels >= LIVER, axis=2)
        # naive display: just transpose the stored array, ignoring orientation
        draw(
            axes[0, col],
            raw.image[:, :, z].T,
            raw.labels[:, :, z].T,
            title=f"Case {case.case_id} · stored as {raw.original_orientation}",
        )
        fixed = load_volume(case, canonical=True)
        z = largest_slice(fixed.labels >= LIVER, axis=2)
        draw(
            axes[1, col],
            radiological_axial(fixed.image[:, :, z]),
            radiological_axial(fixed.labels[:, :, z]),
            title="Reoriented to RAS",
        )
    fig.text(0.01, 0.75, "Naive", rotation=90, va="center", fontsize=11, color=INK_2)
    fig.text(0.01, 0.30, "Fixed", rotation=90, va="center", fontsize=11, color=INK_2)
    _legend(fig)
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    return _save(fig, out / "orientation.png")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def preprocessing_figure(
    raw_case: Case, image_path: Path, label_path: Path, stats: dict, out: Path
) -> Path:
    """Coronal views of one case through the pipeline, plus the effect of normalisation.

    1. Stored array, spacing ignored (what a naive pipeline would feed the network).
    2. Same data shown with its true voxel proportions.
    3. After reorientation, resampling and body crop.
    4. Liver intensities before (HU) and after normalisation (z-score).
    """
    from livertumor.transforms import normalize_ct

    raw = load_volume(raw_case)
    proc_img_nii, proc_lab_nii = nib.load(str(image_path)), nib.load(str(label_path))
    proc_img = np.asarray(proc_img_nii.dataobj, dtype=np.float32)
    proc_lab = np.asarray(proc_lab_nii.dataobj).astype(np.uint8)
    proc_spacing = tuple(float(s) for s in proc_img_nii.header.get_zooms()[:3])

    y_raw = largest_slice(raw.labels >= LIVER, axis=1)
    y_proc = largest_slice(proc_lab >= LIVER, axis=1)
    raw_cor = radiological_coronal(raw.image[:, y_raw, :])
    raw_cor_lab = radiological_coronal(raw.labels[:, y_raw, :])

    fig, axd = plt.subplot_mosaic(
        [["a", "b", "c", "hu"], ["a", "b", "c", "z"]],
        figsize=(16, 4.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.1]},
    )
    sx, _, sz = raw.spacing
    draw(
        axd["a"],
        raw_cor,
        raw_cor_lab,
        aspect=1.0,
        title=f"1 · Voxels drawn as cubes\n({sx:.2f} × {sz:.1f} mm in reality)",
    )
    draw(axd["b"], raw_cor, raw_cor_lab, aspect=sz / sx, title="2 · True proportions")
    draw(
        axd["c"],
        radiological_coronal(proc_img[:, y_proc, :]),
        radiological_coronal(proc_lab[:, y_proc, :]),
        aspect=proc_spacing[2] / proc_spacing[0],
        title=f"3 · RAS, {proc_spacing[0]:.2f}×{proc_spacing[1]:.2f}×"
        f"{proc_spacing[2]:.1f} mm, body crop",
    )

    liver_hu = proc_img[proc_lab >= LIVER]
    params = {k: stats[k] for k in ("clip_low", "clip_high", "mean", "std")}
    for key, values, unit, title in (
        ("hu", liver_hu, "Hounsfield units", "4 · Liver + tumor intensities: before"),
        ("z", normalize_ct(liver_hu, **params), "z-score", "after normalisation"),
    ):
        ax = axd[key]
        ax.hist(values, bins=60, color=LIVER_COLOR if key == "z" else "#9b9a95")
        ax.set_yticks([])
        ax.set_xlabel(unit, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axd["hu"].axvspan(params["clip_low"], params["clip_high"], color="#2a78d6", alpha=0.08)
    axd["z"].axvline(0, color=INK, lw=0.8)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out / "preprocessing.png")


def prediction_figure(
    image: np.ndarray, labels: np.ndarray, pred: np.ndarray, case_id: int, out: Path
) -> Path:
    """Same axial slice three times: CT, ground truth, model prediction.

    Arrays are preprocessed volumes (RAS). The slice is the one with the most tumor in
    the ground truth, or in the prediction if the ground truth has none.
    """
    target = labels == TUMOR if (labels == TUMOR).any() else pred == TUMOR
    z = largest_slice(target if target.any() else labels >= LIVER, axis=2)
    img = radiological_axial(image[:, :, z])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    draw(axes[0], img, title=f"Case {case_id} · axial slice {z}")
    draw(axes[1], img, radiological_axial(labels[:, :, z]), title="Ground truth (radiologist)")
    draw(axes[2], img, radiological_axial(pred[:, :, z]), title="Model prediction")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, Path(out) / f"prediction_case_{case_id:03d}.png")
