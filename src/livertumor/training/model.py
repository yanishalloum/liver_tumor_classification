"""Segmentation network and loss for the 2.5D baseline."""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet

NUM_CLASSES = 3  # background, liver, tumor


def build_unet(in_channels: int = 5, base_channels: int = 32, depth: int = 5) -> UNet:
    """2D U-Net: ``in_channels`` neighbouring slices in, one score map per class out.

    Each level halves the resolution (stride 2) and doubles the number of feature
    maps: with ``base_channels=32`` and ``depth=5`` they are 32, 64, 128, 256, 512.
    Instance normalisation is used instead of batch normalisation because batches are
    small (a few patches): batch statistics would be too noisy.
    """
    channels = tuple(base_channels * 2**i for i in range(depth))
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=NUM_CLASSES,
        channels=channels,
        strides=(2,) * (depth - 1),
        num_res_units=2,
        norm=Norm.INSTANCE,
    )


def build_loss() -> DiceCELoss:
    """Dice + cross-entropy, the standard loss for medical segmentation.

    - **Cross-entropy** scores each pixel independently: smooth gradients, stable at
      the start of training, but dominated by the huge background.
    - **Dice** scores the overlap of each structure as a whole, so a 50-pixel tumor
      counts as much as a 50 000-pixel background: it fights class imbalance.
    The background is excluded from the Dice term (it would be trivially high).
    """
    return DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
