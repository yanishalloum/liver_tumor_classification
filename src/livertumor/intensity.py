"""CT intensity utilities.

CT voxels are stored in Hounsfield Units (HU): air is about -1000, water 0, soft tissue
roughly 20-80 and bone several hundred. A *window* keeps only the HU range relevant to
an organ and maps it to [0, 1], which makes the liver stand out and gives the network
inputs on a stable scale.
"""

from __future__ import annotations

import numpy as np

# Abdominal soft-tissue window, as used in most LiTS pipelines (level 40, width 400).
# The old project used level 30 / width 150, a narrower "liver" window: more contrast
# inside the liver but surrounding structures are clipped.
ABDOMEN_WINDOW = (40.0, 400.0)


def window_ct(image: np.ndarray, level: float, width: float) -> np.ndarray:
    """Clip a CT image to a HU window and rescale it to [0, 1].

    Args:
        image: Array of Hounsfield Units, any shape.
        level: Center of the window in HU.
        width: Width of the window in HU (must be > 0).

    Returns:
        A float32 array of the same shape with values in [0, 1].
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    low = level - width / 2
    high = level + width / 2
    clipped = np.clip(image.astype(np.float32), low, high)
    return (clipped - low) / (high - low)
