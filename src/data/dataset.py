"""
Dataset PyTorch pour la segmentation, avec augmentation via MONAI.

Pourquoi MONAI et pas juste ImageDataGenerator (v1) ?
  - MONAI applique la MÊME transformation géométrique (rotation, flip...) à
    l'image ET au masque en même temps, de façon cohérente. Avec Keras
    ImageDataGenerator, faire ça correctement pour un couple image/masque est
    pénible et sujet à erreurs (v1 ne le faisait d'ailleurs pas vraiment).
  - Bibliothèque pensée pour l'imagerie médicale : transforms qui respectent
    le sens clinique (pas de flip vertical qui inverserait haut/bas du corps
    par exemple), losses et métriques adaptées (Dice, Hausdorff...).
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandZoomd, RandGaussianNoised,
    Resized, EnsureTyped, EnsureChannelFirstd,
)


class LiverSliceDataset(Dataset):
    """
    Charge les paires (slice, masque) préparées par preprocessing.py.
    Retourne un dict {"image": tensor[1,H,W], "label": tensor[H,W] (0/1/2)}
    au format attendu par les transforms et modèles MONAI.
    """

    def __init__(self, csv_path: str, split: str, image_size=(256, 256), augment: bool = False):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transforms = self._build_transforms(image_size, augment)

    def _build_transforms(self, image_size, augment: bool) -> Compose:
        keys = ["image", "label"]
        base = [
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Resized(keys=keys, spatial_size=image_size, mode=["bilinear", "nearest"]),
        ]
        if augment:
            base += [
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # flip gauche/droite seulement
                RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=["bilinear", "nearest"]),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
            ]
        base += [EnsureTyped(keys=keys)]
        return Compose(base)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = np.load(row["slice_path"]).astype(np.float32)
        mask = np.load(row["mask_path"]).astype(np.int64)
        sample = self.transforms({"image": image, "label": mask})
        return sample
