"""
U-Net pour la segmentation foie/tumeur.

Rappel du concept (pour la montée en compétence) :
  Un U-Net a une forme de "U" : un chemin descendant (encodeur) qui réduit
  la résolution spatiale tout en augmentant le nombre de features (il
  apprend "quoi"), et un chemin montant (décodeur) qui remonte en résolution
  pour reconstruire une carte pixel par pixel (il apprend "où"). Les
  "skip connections" copient directement les features de l'encodeur vers le
  décodeur au même niveau de résolution : sans elles, le décodeur perdrait
  les détails fins (contours précis de la tumeur) car l'information a été
  trop compressée en descendant.

On utilise l'implémentation MONAI plutôt que d'en écrire une à la main :
  c'est la même idée qu'utiliser un Conv2D de PyTorch plutôt que de
  réinventer la convolution -- l'implémentation est testée, optimisée, et
  standard dans le domaine médical.
"""
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_unet(config: dict):
    seg_cfg = config["segmentation"]
    return UNet(
        spatial_dims=2,  # 2D pour l'instant ; passera à 3 quand on migrera en 3D
        in_channels=seg_cfg["in_channels"],
        out_channels=seg_cfg["out_channels"],
        channels=seg_cfg["channels"],
        strides=seg_cfg["strides"],
        num_res_units=seg_cfg["num_res_units"],
        norm=Norm.BATCH,
    )
