"""Data augmentation for 2.5D training patches.

With 85 training patients the network could memorise them. Augmentation shows it a
slightly different version of each patch every time - rotated, zoomed, with another
contrast or some noise - so it has to learn what a liver *is* rather than what these
85 livers look like.

Two rules:

1. **Geometric changes are applied identically to image and label** (labels with
   nearest-neighbour interpolation, so they stay 0/1/2). Intensity changes touch the
   image only: the anatomy does not move.
2. **Stay realistic.** We do *not* mirror left-right or front-back: after fixing the
   orientation of every scan (step 2), the liver is always on the patient's right, and
   that is information worth keeping. (nnU-Net does mirror; it is a trade-off, and an
   experiment to run later.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class AugmentConfig:
    """Probabilities and ranges of each transformation."""

    p_affine: float = 0.5
    max_rotation_deg: float = 15.0
    scale_range: tuple[float, float] = (0.85, 1.15)
    p_intensity: float = 0.5
    contrast_range: tuple[float, float] = (0.75, 1.25)
    brightness_range: tuple[float, float] = (-0.1, 0.1)  # in normalised (z-score) units
    p_noise: float = 0.2
    max_noise_std: float = 0.1


def random_affine(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
    fill: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and zoom around the patch centre; same transform for every channel."""
    angle = np.deg2rad(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg))
    scale = rng.uniform(*config.scale_range)
    cos, sin = np.cos(angle), np.sin(angle)
    # affine_transform maps *output* coordinates to *input* coordinates
    matrix = np.array([[cos, -sin], [sin, cos]]) / scale
    center = (np.array(image.shape[-2:]) - 1) / 2
    offset = center - matrix @ center

    def warp(plane: np.ndarray, order: int, cval: float) -> np.ndarray:
        return ndimage.affine_transform(
            plane, matrix, offset=offset, order=order, mode="constant", cval=cval
        )

    image = np.stack([warp(c, order=1, cval=fill) for c in image]).astype(np.float32)
    label = np.stack([warp(c, order=0, cval=0) for c in label]).astype(label.dtype)
    return image, label


def augment(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig | None = None,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly transform one patch. ``image``: (C, H, W) normalised; ``label``: (1, H, W).

    ``fill`` is the value given to pixels that come from outside the patch after a
    rotation (normalised air).
    """
    config = config or AugmentConfig()
    if rng.random() < config.p_affine:
        image, label = random_affine(image, label, rng, config, fill)
    if rng.random() < config.p_intensity:
        image = image * rng.uniform(*config.contrast_range) + rng.uniform(*config.brightness_range)
    if rng.random() < config.p_noise:
        image = image + rng.normal(0, rng.uniform(0, config.max_noise_std), image.shape)
    return image.astype(np.float32), label
