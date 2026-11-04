"""On-the-fly transforms applied when loading preprocessed volumes."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np


def normalize_ct(image, clip_low: float, clip_high: float, mean: float, std: float):
    """Clip HU to the foreground percentile range, then standardise (z-score).

    Works on numpy arrays and torch tensors alike. After this, liver tissue is roughly
    centred on 0 with unit spread, whatever the scanner.
    """
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    if isinstance(image, np.ndarray):
        return (np.clip(image.astype(np.float32), clip_low, clip_high) - mean) / std
    return (image.float().clamp(clip_low, clip_high) - mean) / std


class NormalizeCTd:
    """Dictionary transform (MONAI style) applying :func:`normalize_ct` to some keys."""

    def __init__(self, keys: Sequence[Hashable], stats: Mapping[str, float]):
        """Store the keys to transform and the statistics from ``preprocessing.json``."""
        self.keys = list(keys)
        self.params = {k: float(stats[k]) for k in ("clip_low", "clip_high", "mean", "std")}

    def __call__(self, data: Mapping) -> dict:
        """Return a copy of ``data`` with the selected entries normalised."""
        out = dict(data)
        for key in self.keys:
            out[key] = normalize_ct(out[key], **self.params)
        return out
