# Liver Tumor Segmentation — v2

Refonte du projet [liver_tumor_classification](https://github.com/yanishalloum/liver_tumor_classification) :
passage de la classification (sain/tumeur) à la **vraie segmentation**
(délimiter précisément le foie et la tumeur, pixel par pixel), avec une
architecture PyTorch + MONAI moderne.

## Ce qui a changé par rapport à la v1, et pourquoi

| v1 | v2 | Pourquoi |
|---|---|---|
| Classification (sain/tumeur) | Segmentation (masque pixel par pixel) | Plus utile cliniquement : localiser, pas juste détecter |
| Split train/test par slice | Split par **patient** | La v1 fuit des données : deux slices voisines du même foie sont quasi identiques, les séparer entre train/test triche sur le score |
| Keras + TensorFlow + fastai mélangés | PyTorch + MONAI uniquement | Une seule stack, standard en imagerie médicale, plus facile à maintenir/apprendre |
| Accuracy comme métrique | Dice score | Le foie/tumeur occupe une toute petite fraction de l'image ; l'accuracy est trompeuse sur des masques déséquilibrés |
| VGG16 entièrement gelé (`trainable=False`) | ResNet fine-tuné (backbone à venir en phase 3) | Geler tout empêche le modèle de s'adapter aux CT (très différents d'ImageNet) — explique en partie le 54% de la v1 |
| Chemins Windows en dur | `configs/config.yaml` | Reproductible sur n'importe quelle machine, notamment Kaggle |
| Scripts séquentiels avec code de plotting dupliqué | Modules réutilisables (`src/`) | Testable, maintenable |

## Structure

```
configs/config.yaml          # tous les hyperparamètres et chemins
src/data/preprocessing.py    # NIfTI 3D -> slices 2D (.npy) + index.csv, split par patient
src/data/dataset.py          # Dataset PyTorch + augmentations MONAI
src/models/unet.py           # U-Net (MONAI)
src/training/losses_metrics.py  # DiceCELoss + Dice score par classe
src/training/train_segmentation.py  # boucle d'entraînement avec early stopping
```

## Utilisation sur Kaggle

1. Crée un notebook Kaggle, active un GPU (Settings → Accelerator → GPU T4).
2. Ajoute les deux datasets du README original ("Add Data") :
   - `andrewmvd/lits-png`
   - `andrewmvd/liver-tumor-segmentation-part-2`
3. Upload ce dossier (ou clone-le si tu l'as poussé sur GitHub) et ajuste
   `configs/config.yaml` avec les vrais chemins `/kaggle/input/...` (visibles
   dans le panneau de droite du notebook une fois les datasets ajoutés).
4. Installe les dépendances : `!pip install -q -r requirements.txt`
5. Prépare les données :
   ```
   !python src/data/preprocessing.py --config configs/config.yaml
   ```
6. Lance l'entraînement :
   ```
   !python src/training/train_segmentation.py --config configs/config.yaml
   ```

## Concepts clés (pour monter en compétence)

- **Fenêtrage DICOM (windowing)** : les CT encodent l'intensité en Hounsfield
  Units sur une plage énorme (air, os, tissus mous...). Le foie n'occupe
  qu'une petite portion de cette plage. Le windowing recentre le contraste
  sur cette portion pour le rendre visible. Cf. [radiopaedia.org/articles/windowing-ct](https://radiopaedia.org/articles/windowing-ct).
- **U-Net** : encodeur (compresse spatialement, apprend le contenu) +
  décodeur (reconstruit une carte pixel par pixel) + skip connections
  (réinjectent les détails fins perdus par la compression).
- **Dice score/loss** : `2 * |intersection| / (|pred| + |vérité|)`. Mesure
  le recouvrement, insensible au déséquilibre de classes contrairement à
  l'accuracy pixel-wise.
- **Fuite de données (data leakage)** : toute forme de proximité entre
  train et test qui fait gonfler artificiellement le score. Ici : slices
  voisines du même patient. Règle générale : le split doit toujours se
  faire au niveau de l'unité "indépendante" (le patient), jamais en dessous.

## Prochaines étapes (roadmap)

- [ ] **Phase 2** : lancer l'entraînement, analyser les courbes Dice, ajuster
- [ ] **Phase 3** : réintégrer un classifieur (ResNet18 fine-tuné) comme
      comparaison/baseline, corriger l'overfitting du VGG16 original
      (dropout, fine-tuning partiel, gestion du déséquilibre de classes)
- [ ] **Phase 4** : passage en 3D (patches de volumes plutôt que slices 2D
      isolées), exploration de nnU-Net
