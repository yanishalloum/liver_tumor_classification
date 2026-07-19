"""
Script d'entraînement du U-Net de segmentation.

Usage (dans un notebook Kaggle avec GPU activé) :
    !python src/training/train_segmentation.py --config configs/config.yaml

Ce script suppose que src/data/preprocessing.py a déjà été lancé une fois
pour générer processed_dir/index.csv.
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.dataset import LiverSliceDataset
from src.models.unet import build_unet
from src.training.losses_metrics import get_loss_function, DiceEvaluator


def train(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_dir = Path(config["paths"]["processed_dir"])
    csv_path = processed_dir / "index.csv"
    image_size = tuple(config["data"]["image_size"])

    train_ds = LiverSliceDataset(str(csv_path), "train", image_size, augment=True)
    val_ds = LiverSliceDataset(str(csv_path), "valid", image_size, augment=False)

    seg_cfg = config["segmentation"]
    train_loader = DataLoader(train_ds, batch_size=seg_cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=seg_cfg["batch_size"], shuffle=False, num_workers=2)

    model = build_unet(config).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=seg_cfg["learning_rate"])
    evaluator = DiceEvaluator(num_classes=seg_cfg["out_channels"])

    best_dice = 0.0
    patience_counter = 0
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(seg_cfg["epochs"]):
        # --- Entraînement ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                evaluator.update(outputs, labels)
        dice_liver, dice_tumor = evaluator.aggregate()
        mean_dice = (dice_liver + dice_tumor) / 2

        print(
            f"Epoch {epoch+1}/{seg_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} | dice_foie={dice_liver:.4f} | dice_tumeur={dice_tumor:.4f}"
        )

        # --- Early stopping + sauvegarde du meilleur modèle ---
        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_unet.pt")
        else:
            patience_counter += 1
            if patience_counter >= seg_cfg["early_stopping_patience"]:
                print(f"Early stopping à l'epoch {epoch+1} (pas d'amélioration depuis {patience_counter} epochs).")
                break

    print(f"Meilleur Dice moyen (foie+tumeur): {best_dice:.4f}")
    print(f"Modèle sauvegardé dans {output_dir / 'best_unet.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
