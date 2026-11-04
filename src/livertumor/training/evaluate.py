"""Whole-volume inference and Dice scores, the way the model will be judged.

Training sees random patches; evaluation must see what a radiologist would give us:
every slice of a scan, at full size. We predict slice by slice (with the same 2.5D
context as in training), stack the predictions into a volume and compute the Dice
score in 3D, per patient.
"""

from __future__ import annotations

import numpy as np
import torch

from livertumor.training.dataset import Volume, neighbour_slices
from livertumor.transforms import normalize_ct


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Overlap between two binary masks: 2|A∩B| / (|A| + |B|), from 0 (none) to 1.

    If both masks are empty (e.g. no tumor and none predicted) the prediction is
    perfect and the score is 1.
    """
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(pred, truth).sum() / total)


def pad_to_multiple(array: np.ndarray, multiple: int, fill: float) -> tuple[np.ndarray, tuple]:
    """Pad the last two axes up to a multiple of ``multiple`` (U-Net downsampling).

    Returns the padded array and the slices that cut it back to the original size.
    """
    height, width = array.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    padding = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.pad(array, padding, constant_values=fill), (slice(0, height), slice(0, width))


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    volume: Volume,
    stats: dict[str, float],
    context: int,
    device: torch.device,
    batch_size: int = 16,
    multiple: int = 16,
) -> np.ndarray:
    """Label map (uint8, same shape as the volume) predicted slice by slice."""
    model.eval()
    norm = {k: stats[k] for k in ("clip_low", "clip_high", "mean", "std")}
    n_slices = volume.image.shape[2]
    prediction = np.zeros(volume.image.shape, dtype=np.uint8)
    for start in range(0, n_slices, batch_size):
        zs = range(start, min(start + batch_size, n_slices))
        stacks = [neighbour_slices(volume.image, z, context) for z in zs]
        batch, crop = pad_to_multiple(np.stack(stacks), multiple, fill=-1024)
        batch = torch.from_numpy(normalize_ct(batch, **norm)).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(batch)
        labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        prediction[:, :, start : start + len(zs)] = np.moveaxis(labels[(..., *crop)], 0, -1)
    return prediction


def evaluate(
    model: torch.nn.Module,
    volumes: list[Volume],
    stats: dict[str, float],
    context: int,
    device: torch.device,
) -> list[dict]:
    """Per-patient Dice for the liver (liver + tumor) and for the tumor alone."""
    results = []
    for volume in volumes:
        pred = predict_volume(model, volume, stats, context, device)
        results.append(
            {
                "case_id": volume.case_id,
                "dice_liver": dice(pred > 0, volume.labels > 0),
                "dice_tumor": dice(pred == 2, volume.labels == 2),
            }
        )
    return results
