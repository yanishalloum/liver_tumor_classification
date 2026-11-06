"""Clinical-style evaluation of a segmentation: overlap, lesions and patients.

The Dice score alone hides important failures. A model can reach a good tumor Dice by
segmenting one large tumor well while missing five small ones, which matters most
clinically. We therefore report three levels:

1. **Voxels** - Dice for liver and tumor, per patient and over all patients pooled
   ("global Dice", dominated by large tumors).
2. **Lesions** - each connected tumor component is matched: a true lesion is *detected*
   if the prediction overlaps it; a predicted component touching no true lesion is a
   *false positive*. Recall = share of true lesions found; precision = share of
   predicted lesions that are real.
3. **Patients** - "does this patient have a tumor?", derived from the predicted mask.
   Sensitivity = sick patients flagged; specificity = healthy patients cleared.

Components smaller than ``min_voxels`` are ignored on both sides: a handful of voxels
is below what can be reliably annotated (some LiTS "lesions" are 1-3 voxels, see the
data analysis) or predicted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from livertumor.training.evaluate import dice

MIN_LESION_VOXELS = 10  # ~0.006 ml at 0.77 x 0.77 x 1.0 mm


def lesion_components(mask: np.ndarray, min_voxels: int = MIN_LESION_VOXELS) -> np.ndarray:
    """Label the 3D connected components of ``mask`` (26-connectivity), dropping small ones.

    Returns an int array: 0 = background, 1..n = lesions.
    """
    structure = ndimage.generate_binary_structure(3, 3)
    components, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return components
    sizes = np.bincount(components.ravel())
    small = np.flatnonzero(sizes < min_voxels)
    components[np.isin(components, small)] = 0
    # renumber 1..k so the count of lesions is simply the max label
    kept = np.unique(components[components > 0])
    lookup = np.zeros(sizes.size, dtype=components.dtype)
    lookup[kept] = np.arange(1, kept.size + 1)
    return lookup[components]


@dataclass
class LesionMatch:
    """Lesion-level counts for one patient."""

    true_lesions: int
    detected: int  # true lesions overlapped by the prediction (true positives)
    missed: int  # true lesions with no predicted voxel (false negatives)
    false_positives: int  # predicted components touching no true lesion


def match_lesions(
    pred_tumor: np.ndarray, true_tumor: np.ndarray, min_voxels: int = MIN_LESION_VOXELS
) -> LesionMatch:
    """Match predicted and true lesions by overlap."""
    true_components = lesion_components(true_tumor, min_voxels)
    pred_components = lesion_components(pred_tumor, min_voxels)
    n_true, n_pred = int(true_components.max()), int(pred_components.max())
    hit_true = np.unique(true_components[(pred_components > 0) & (true_components > 0)])
    hit_pred = np.unique(pred_components[(pred_components > 0) & (true_components > 0)])
    return LesionMatch(
        true_lesions=n_true,
        detected=int(hit_true.size),
        missed=n_true - int(hit_true.size),
        false_positives=n_pred - int(hit_pred.size),
    )


def evaluate_case(
    pred: np.ndarray,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    min_voxels: int = MIN_LESION_VOXELS,
) -> dict:
    """All metrics for one patient. ``pred`` and ``labels``: 0 background, 1 liver, 2 tumor."""
    voxel_ml = float(np.prod(spacing)) / 1000
    pred_tumor, true_tumor = pred == 2, labels == 2
    lesions = match_lesions(pred_tumor, true_tumor, min_voxels)
    return {
        "dice_liver": dice(pred > 0, labels > 0),
        "dice_tumor": dice(pred_tumor, true_tumor),
        "tumor_ml_true": round(float(true_tumor.sum()) * voxel_ml, 3),
        "tumor_ml_pred": round(float(pred_tumor.sum()) * voxel_ml, 3),
        # pooled counts, to compute the global Dice over all patients
        "tumor_intersection": int((pred_tumor & true_tumor).sum()),
        "tumor_sum": int(pred_tumor.sum() + true_tumor.sum()),
        "true_lesions": lesions.true_lesions,
        "detected_lesions": lesions.detected,
        "missed_lesions": lesions.missed,
        "false_positive_lesions": lesions.false_positives,
        "has_tumor_true": lesions.true_lesions > 0,
        "has_tumor_pred": lesions.detected + lesions.false_positives > 0,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-patient metrics. ``None`` means "not defined" (nothing to count)."""
    tp_lesions = sum(r["detected_lesions"] for r in rows)
    fn_lesions = sum(r["missed_lesions"] for r in rows)
    fp_lesions = sum(r["false_positive_lesions"] for r in rows)
    sick = [r for r in rows if r["has_tumor_true"]]
    healthy = [r for r in rows if not r["has_tumor_true"]]
    pooled = sum(r["tumor_sum"] for r in rows)
    return {
        "patients": len(rows),
        "dice_liver_mean": round(float(np.mean([r["dice_liver"] for r in rows])), 4),
        "dice_tumor_mean": round(float(np.mean([r["dice_tumor"] for r in rows])), 4),
        "dice_tumor_global": _ratio(2 * sum(r["tumor_intersection"] for r in rows), pooled),
        "lesion_recall": _ratio(tp_lesions, tp_lesions + fn_lesions),
        "lesion_precision": _ratio(tp_lesions, tp_lesions + fp_lesions),
        "false_positive_lesions_per_patient": round(fp_lesions / len(rows), 3),
        "patient_sensitivity": _ratio(sum(r["has_tumor_pred"] for r in sick), len(sick)),
        "patient_specificity": _ratio(sum(not r["has_tumor_pred"] for r in healthy), len(healthy)),
    }


def evaluate_run(
    run_dir: str,
    dataset_dir: str,
    part: str = "val",
    checkpoint: str = "best.pt",
    out_dir: str | None = None,
    n_figures: int = 3,
) -> dict:
    """Predict every patient of ``part`` with a trained run; write per-case CSV + summary.

    The model settings (context, width) are read from the run's ``config.json``, so the
    evaluation always matches how the model was trained.
    """
    import csv
    import json
    from pathlib import Path

    import torch

    from livertumor.training.dataset import load_intensity_stats, load_volumes
    from livertumor.training.evaluate import predict_volume
    from livertumor.training.model import build_unet

    run, data = Path(run_dir), Path(dataset_dir)
    out = Path(out_dir) if out_dir else run / f"eval_{part}"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    spacing = tuple(json.loads((data / "preprocessing.json").read_text())["config"]["spacing_mm"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(2 * config["context"] + 1, config["base_channels"]).to(device)
    model.load_state_dict(torch.load(run / checkpoint, map_location=device, weights_only=True))
    stats = load_intensity_stats(data)

    rows, figures = [], 0
    for volume in load_volumes(data, part):
        pred = predict_volume(model, volume, stats, config["context"], device)
        rows.append({"case_id": volume.case_id, **evaluate_case(pred, volume.labels, spacing)})
        print(
            f"case {volume.case_id:>3} | dice liver {rows[-1]['dice_liver']:.3f} "
            f"tumor {rows[-1]['dice_tumor']:.3f}",
            flush=True,
        )
        if figures < n_figures and (volume.labels == 2).any():
            from livertumor.viz import prediction_figure

            prediction_figure(volume.image, volume.labels, pred, volume.case_id, out)
            figures += 1

    with (out / "per_case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"part": part, "checkpoint": str(run / checkpoint), **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
