"""Geometric preprocessing of LiTS volumes, done once and saved to disk.

For every case:

1. **Reorient to RAS** so every array has the same axis directions, then **check the
   left-right side with anatomy**: if the liver ends up on the patient's left, the
   header lied and image + labels are mirrored back (see :class:`FixLeftRightFlipd`).
2. **Crop to the body**: drop the surrounding air (HU <= -500), keeping a margin
   (counted in original voxels). Done *before* resampling so the costly step works
   on 30-50% fewer voxels. We do not crop to the liver: at inference time its
   position is unknown.
3. **Resample to a common voxel spacing** (default 0.77 x 0.77 x 1.0 mm, the dataset
   median), computed in float32: images with linear interpolation, labels with
   nearest neighbour (a value of 1.5 between liver and tumor would be meaningless).
4. **Save** image (HU, int16) and labels (uint8) as compressed NIfTI with their affine.

Intensity *normalisation* is deliberately **not** baked into the saved images: it is
applied on the fly (:mod:`livertumor.transforms`) from statistics computed on the
**training cases only**. Keeping HU on disk lets us change normalisation without
re-running this step, and computing the statistics on train only avoids leaking
validation/test information into the pipeline.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from livertumor.data.discovery import Case

TARGET_SPACING = (0.77, 0.77, 1.0)
BODY_THRESHOLD_HU = -500.0
CROP_MARGIN_VOXELS = 10
SAMPLES_PER_CASE = 100_000
MIN_LIVER_OFFSET = 0.05  # fraction of body width; below this the side is ambiguous


@dataclass
class PreprocessConfig:
    """Parameters of the geometric preprocessing."""

    spacing: tuple[float, float, float] = TARGET_SPACING
    body_threshold_hu: float = BODY_THRESHOLD_HU
    margin: int = CROP_MARGIN_VOXELS
    samples_per_case: int = SAMPLES_PER_CASE
    seed: int = 0


@dataclass
class CaseResult:
    """What happened to one case (written to the manifest)."""

    case_id: int
    part: str
    image: str
    label: str
    original_shape: tuple[int, int, int]
    shape: tuple[int, int, int]
    liver_ml_before: float
    liver_ml_after: float
    tumor_ml_before: float
    tumor_ml_after: float
    liver_offset: float
    lr_flipped: bool
    foreground_samples: np.ndarray = field(repr=False)


def liver_side_offset(image: np.ndarray, labels: np.ndarray, threshold_hu: float) -> float:
    """Signed distance from the body centre to the liver centre along x, in body widths.

    Arrays must be in RAS orientation, where x grows toward the patient's **right**.
    The liver sits in the right upper abdomen, so a correct scan gives a clearly
    positive value (typically +0.15 to +0.30). A negative value means the image is
    mirrored left-right: the header claims an orientation the data does not have.
    """
    body_x = np.nonzero(image > threshold_hu)[0]
    liver_x = np.nonzero(labels > 0)[0]
    if body_x.size == 0 or liver_x.size == 0:
        return 0.0
    width = body_x.max() - body_x.min() + 1
    return float((liver_x.mean() - body_x.mean()) / width)


class FixLeftRightFlipd:
    """Mirror image and labels along x when the liver is on the patient's left.

    Some LiTS headers (e.g. the 3D-IRCADb cases) state an orientation that does not
    match the data. The header cannot be trusted, so we trust anatomy instead. Image and
    labels are flipped together; the affine is kept, which amounts to correcting the
    header. Uses the labels, so it only applies to annotated (training/evaluation) data.
    """

    def __init__(self, threshold_hu: float, min_offset: float = MIN_LIVER_OFFSET):
        """Store the body threshold and the minimum offset considered reliable."""
        self.threshold_hu = threshold_hu
        self.min_offset = min_offset

    def __call__(self, data: dict) -> dict:
        """Flip ``image`` and ``label`` if needed; record what was done."""
        import torch

        out = dict(data)
        offset = liver_side_offset(
            np.asarray(out["image"][0]), np.asarray(out["label"][0]), self.threshold_hu
        )
        out["liver_offset"] = offset
        out["lr_flipped"] = offset < -self.min_offset
        if out["lr_flipped"]:
            for key in ("image", "label"):
                out[key] = torch.flip(out[key], dims=[1])  # dim 0 is the channel
        return out


def build_transform(config: PreprocessConfig):
    """MONAI pipeline: load -> channel first -> RAS -> fix L/R -> crop -> resample."""
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Spacingd,
    )

    threshold = config.body_threshold_hu
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),  # explicit: RAS+ world
            ),
            FixLeftRightFlipd(threshold_hu=threshold),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > threshold,
                margin=config.margin,
                allow_smaller=True,
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=config.spacing,
                mode=("bilinear", "nearest"),
                dtype=np.float32,
            ),
        ]
    )


def _volumes_ml(labels: np.ndarray, spacing: Sequence[float]) -> tuple[float, float]:
    voxel_ml = float(np.prod(spacing)) / 1000
    return float((labels > 0).sum()) * voxel_ml, float((labels == 2).sum()) * voxel_ml


def preprocess_case(case: Case, part: str, out_dir: Path, config: PreprocessConfig) -> CaseResult:
    """Run the pipeline on one case and save the result."""
    import torch

    torch.set_num_threads(1)  # parallelism comes from worker processes

    raw_labels = nib.load(str(case.label))
    raw = np.rint(np.asarray(raw_labels.dataobj)).astype(np.uint8)
    liver_before, tumor_before = _volumes_ml(raw, raw_labels.header.get_zooms()[:3])
    original_shape = tuple(int(s) for s in raw.shape[:3])
    del raw

    data = build_transform(config)({"image": str(case.image), "label": str(case.label)})
    image = data["image"][0].numpy()
    labels = data["label"][0].numpy()
    affine = np.asarray(data["image"].affine, dtype=np.float64)

    image_hu = np.clip(np.rint(image), -1024, 3071).astype(np.int16)
    labels = np.rint(labels).astype(np.uint8)
    liver_after, tumor_after = _volumes_ml(labels, config.spacing)

    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"case_{case.case_id:03d}_image.nii.gz"
    label_path = out_dir / f"case_{case.case_id:03d}_label.nii.gz"
    for array, path in ((image_hu, image_path), (labels, label_path)):
        nib.save(nib.Nifti1Image(array, affine), path)  # spacing is encoded in the affine

    samples = np.empty(0, dtype=np.float32)
    if part == "train":  # intensity statistics come from training cases only
        foreground = image[labels > 0]
        rng = np.random.default_rng(config.seed + case.case_id)
        n = min(config.samples_per_case, foreground.size)
        samples = rng.choice(foreground, size=n, replace=False).astype(np.float32)

    return CaseResult(
        case_id=case.case_id,
        part=part,
        image=image_path.name,
        label=label_path.name,
        original_shape=original_shape,
        shape=tuple(int(s) for s in labels.shape),
        liver_ml_before=round(liver_before, 2),
        liver_ml_after=round(liver_after, 2),
        tumor_ml_before=round(tumor_before, 4),
        tumor_ml_after=round(tumor_after, 4),
        liver_offset=round(data["liver_offset"], 3),
        lr_flipped=data["lr_flipped"],
        foreground_samples=samples,
    )


def intensity_statistics(samples: np.ndarray) -> dict[str, float]:
    """nnU-Net-style CT statistics from foreground (liver + tumor) HU samples.

    Values are clipped to the 0.5th-99.5th percentiles (removes contrast spikes, metal,
    annotation errors at the border), then mean and std are taken on the clipped values.
    """
    if samples.size == 0:
        raise ValueError("no foreground samples: are there training cases?")
    low, high = np.percentile(samples, [0.5, 99.5])
    clipped = np.clip(samples, low, high)
    return {
        "clip_low": float(low),
        "clip_high": float(high),
        "mean": float(clipped.mean()),
        "std": float(clipped.std()),
        "n_samples": int(samples.size),
    }


MANIFEST_FIELDS = [
    "case_id",
    "part",
    "image",
    "label",
    "original_shape",
    "shape",
    "liver_ml_before",
    "liver_ml_after",
    "tumor_ml_before",
    "tumor_ml_after",
    "liver_offset",
    "lr_flipped",
]


def preprocess_dataset(
    cases: Sequence[Case],
    split: dict[str, list[int]],
    out_dir: str | Path,
    config: PreprocessConfig | None = None,
    workers: int = 2,
) -> dict:
    """Preprocess all cases of the split; write volumes, manifest and intensity stats."""
    config = config or PreprocessConfig()
    out_dir = Path(out_dir)
    part_of = {c: p for p in ("train", "val", "test") for c in split[p]}
    todo = [c for c in cases if c.case_id in part_of]
    volumes_dir = out_dir / "volumes"

    results: list[CaseResult] = []
    if workers <= 1:
        for i, case in enumerate(todo, 1):
            results.append(preprocess_case(case, part_of[case.case_id], volumes_dir, config))
            print(f"[{i:>3}/{len(todo)}] case {case.case_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(preprocess_case, c, part_of[c.case_id], volumes_dir, config)
                for c in todo
            ]
            for i, future in enumerate(futures, 1):
                results.append(future.result())
                print(f"[{i:>3}/{len(todo)}] case {results[-1].case_id}", flush=True)

    stats = intensity_statistics(np.concatenate([r.foreground_samples for r in results]))
    stats["computed_on"] = "train cases only, liver + tumor voxels after resampling"

    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for r in sorted(results, key=lambda r: r.case_id):
            row = {k: getattr(r, k) for k in MANIFEST_FIELDS}
            row["original_shape"] = "x".join(map(str, r.original_shape))
            row["shape"] = "x".join(map(str, r.shape))
            writer.writerow(row)

    summary = {
        "config": {
            "spacing_mm": list(config.spacing),
            "orientation": "RAS",
            "body_threshold_hu": config.body_threshold_hu,
            "crop_margin_voxels": config.margin,
            "image_dtype": "int16 (HU)",
            "label_values": {"0": "background", "1": "liver", "2": "tumor"},
        },
        "intensity": stats,
        "split": {p: sorted(split[p]) for p in ("train", "val", "test")},
    }
    (out_dir / "preprocessing.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
