"""
Loss et métriques adaptées à la segmentation médicale.

Pourquoi pas juste la cross-entropy (utilisée en classification en v1) ?
  Le foie occupe peut-être 5% des pixels d'une slice, la tumeur souvent
  moins de 1%. Un modèle qui prédit "tout est background" a déjà ~95%
  d'accuracy pixel-wise sans rien avoir appris -- l'accuracy est donc un
  leurre ici. Le Dice score mesure le recouvrement entre la prédiction et
  la vérité terrain (2 * intersection / somme des tailles), et est
  insensible à ce déséquilibre. La DiceLoss (1 - Dice) optimise
  directement ce qu'on veut mesurer. En pratique on combine souvent
  Dice + CrossEntropy (DiceCELoss) : Dice pour la robustesse au
  déséquilibre, CE pour un gradient plus stable en début d'entraînement.
"""
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def get_loss_function():
    return DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)


class DiceEvaluator:
    """Calcule le Dice score par classe (foie, tumeur) pendant l'entraînement."""

    def __init__(self, num_classes: int = 3):
        self.metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.to_onehot_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.to_onehot_label = AsDiscrete(to_onehot=num_classes)

    def update(self, logits, labels):
        preds = [self.to_onehot_pred(p) for p in logits]
        targets = [self.to_onehot_label(l) for l in labels]
        self.metric(y_pred=preds, y=targets)

    def aggregate(self):
        """Retourne (dice_foie, dice_tumeur) -- indices 0 et 1 car background exclu."""
        result = self.metric.aggregate()
        self.metric.reset()
        return result[0].item(), result[1].item()
