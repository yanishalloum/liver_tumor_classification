"""2.5D slice sampling from preprocessed volumes, for 2D segmentation networks.

A 2D network sees one axial slice at a time and loses the 3D context that radiologists
use (a vessel is a tube across slices, a lesion is a blob). The cheap fix is **2.5D**:
feed the slice *and its neighbours* as input channels, and predict the labels of the
centre slice only.

Liver is present in about a third of the slices and tumor in far fewer, so uniform
sampling would mostly show the network empty slices. We **oversample** slices that
contain tumor or liver, and crop fixed-size patches centred on foreground most of the
time, so every batch contains something to learn from.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from livertumor.transforms import normalize_ct

AIR_HU = -1024


@dataclass
class Volume:
    """A preprocessed case held in memory (RAS, HU int16, labels uint8)."""

    case_id: int
    image: np.ndarray
    labels: np.ndarray


def load_volumes(dataset_dir: str | Path, part: str) -> list[Volume]:
    """Load every case of one split part (``train``, ``val``, ``test``) into memory."""
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["part"] == part]
    volumes = []
    for row in rows:
        image = nib.load(dataset_dir / "volumes" / row["image"]).dataobj
        labels = nib.load(dataset_dir / "volumes" / row["label"]).dataobj
        volumes.append(
            Volume(
                case_id=int(row["case_id"]),
                image=np.asarray(image, dtype=np.int16),
                labels=np.asarray(labels, dtype=np.uint8),
            )
        )
    return volumes


def load_intensity_stats(dataset_dir: str | Path) -> dict[str, float]:
    """Normalisation statistics computed on the training cases during preprocessing."""
    summary = json.loads((Path(dataset_dir) / "preprocessing.json").read_text())
    return summary["intensity"]


def neighbour_slices(image: np.ndarray, z: int, context: int) -> np.ndarray:
    """Slices ``z - context`` to ``z + context`` stacked as channels.

    At the top and bottom of the volume the missing neighbours are replaced by the
    edge slice. Returns an array of shape ``(2 * context + 1, X, Y)``.
    """
    indices = np.clip(np.arange(z - context, z + context + 1), 0, image.shape[2] - 1)
    return np.moveaxis(image[:, :, indices], -1, 0)


def crop_or_pad(array: np.ndarray, center: Sequence[int], size: int, fill: float) -> np.ndarray:
    """Cut a ``size`` x ``size`` window of the last two axes around ``center``.

    Where the window goes past the array border, it is filled with ``fill``.
    """
    out = np.full((*array.shape[:-2], size, size), fill, dtype=array.dtype)
    src, dst = [], []
    for c, length in zip(center, array.shape[-2:], strict=True):
        start = int(c) - size // 2
        lo, hi = max(start, 0), min(start + size, length)
        src.append(slice(lo, hi))
        dst.append(slice(lo - start, hi - start))
    out[(..., *dst)] = array[(..., *src)]
    return out


class SliceDataset(Dataset):
    """Random 2.5D patches: ``image`` (2k+1, P, P) float32, ``label`` (1, P, P) int64.

    Each item draws a random volume, then a slice: with probability ``p_tumor`` one that
    contains tumor, with probability ``p_liver`` one that contains liver, otherwise any
    slice. The patch is centred on a random foreground voxel of that slice when there is
    one (90% of the time), otherwise on a random position.
    """

    def __init__(
        self,
        volumes: Sequence[Volume],
        stats: dict[str, float],
        context: int = 2,
        patch_size: int = 256,
        samples_per_epoch: int = 4000,
        p_tumor: float = 0.4,
        p_liver: float = 0.4,
        seed: int = 0,
    ):
        """Index, once, which slices contain liver and which contain tumor."""
        self.volumes = list(volumes)
        self.norm = {k: stats[k] for k in ("clip_low", "clip_high", "mean", "std")}
        self.context = context
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.p_tumor, self.p_liver = p_tumor, p_liver
        self.rng = np.random.default_rng(seed)
        self.liver_slices = [np.flatnonzero((v.labels > 0).any(axis=(0, 1))) for v in volumes]
        self.tumor_slices = [np.flatnonzero((v.labels == 2).any(axis=(0, 1))) for v in volumes]

    def __len__(self) -> int:
        """Number of random patches per epoch."""
        return self.samples_per_epoch

    def pick_slice(self, i: int) -> int:
        """Choose a slice of volume ``i`` following the oversampling probabilities."""
        draw = self.rng.random()
        if draw < self.p_tumor and self.tumor_slices[i].size:
            return int(self.rng.choice(self.tumor_slices[i]))
        if draw < self.p_tumor + self.p_liver and self.liver_slices[i].size:
            return int(self.rng.choice(self.liver_slices[i]))
        return int(self.rng.integers(self.volumes[i].labels.shape[2]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Draw one random patch. ``index`` is ignored: sampling is random."""
        i = int(self.rng.integers(len(self.volumes)))
        volume = self.volumes[i]
        z = self.pick_slice(i)
        label = volume.labels[:, :, z]

        foreground = np.argwhere(label > 0)
        if len(foreground) and self.rng.random() < 0.9:
            center = foreground[self.rng.integers(len(foreground))]
        else:
            center = [self.rng.integers(s) for s in label.shape]

        image = neighbour_slices(volume.image, z, self.context)
        image = crop_or_pad(image, center, self.patch_size, fill=AIR_HU)
        label = crop_or_pad(label[None], center, self.patch_size, fill=0)
        image = normalize_ct(image, **self.norm)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
        }


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own random stream.

    DataLoader workers are copies of the main process: without this, they would all
    share the same generator state and return the *same* "random" patches.
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)
