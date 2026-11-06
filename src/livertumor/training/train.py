"""Training loop for the 2.5D U-Net baseline.

One *epoch* = ``samples_per_epoch`` random patches. After each epoch the model is
evaluated on the **validation** patients (whole volumes, 3D Dice). The best model on
validation is kept; the test patients are never touched here.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from livertumor.training.augment import AugmentConfig
from livertumor.training.dataset import (
    SliceDataset,
    load_intensity_stats,
    load_volumes,
    seed_worker,
)
from livertumor.training.evaluate import evaluate
from livertumor.training.model import build_loss, build_unet


@dataclass
class TrainConfig:
    """Every setting of a run. Saved next to the results so the run can be reproduced."""

    dataset_dir: str
    out_dir: str
    epochs: int = 50
    samples_per_epoch: int = 4000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patch_size: int = 256
    context: int = 2
    base_channels: int = 32
    num_workers: int = 3
    seed: int = 0
    augment: bool = True
    max_val_cases: int | None = None  # evaluate on a subset to save time when debugging


def seed_everything(seed: int) -> None:
    """Make runs repeatable: same seed -> same initial weights and same patches.

    Our own sampling uses explicit ``np.random.Generator`` objects seeded from the
    config, so only Python's and PyTorch's global generators need seeding here.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def poly_lr(epoch: int, epochs: int, base_lr: float, power: float = 0.9) -> float:
    """nnU-Net's learning-rate schedule: decreases smoothly from ``base_lr`` to 0."""
    return base_lr * (1 - epoch / epochs) ** power


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device) -> float:
    """One pass over the random patches; returns the mean training loss."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # mixed precision: float16 where it is safe, ~2x faster on GPU, half the memory
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(image), label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(image)
        count += len(image)
    return total / count


def train(config: TrainConfig) -> Path:
    """Run a full training and return the output directory."""
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    stats = load_intensity_stats(config.dataset_dir)
    train_volumes = load_volumes(config.dataset_dir, "train")
    val_volumes = load_volumes(config.dataset_dir, "val")[: config.max_val_cases]
    print(f"{len(train_volumes)} train / {len(val_volumes)} val volumes on {device}")

    dataset = SliceDataset(
        train_volumes,
        stats,
        context=config.context,
        patch_size=config.patch_size,
        samples_per_epoch=config.samples_per_epoch,
        seed=config.seed,
        augmentation=AugmentConfig() if config.augment else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = build_unet(in_channels=2 * config.context + 1, base_channels=config.base_channels)
    model.to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best = -1.0
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "lr", "train_loss", "val_dice_liver", "val_dice_tumor", "seconds"]
        )
        for epoch in range(config.epochs):
            start = time.time()
            lr = poly_lr(epoch, config.epochs, config.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device)

            scores = evaluate(model, val_volumes, stats, config.context, device)
            liver = float(np.mean([s["dice_liver"] for s in scores]))
            tumor = float(np.mean([s["dice_tumor"] for s in scores]))
            seconds = time.time() - start
            writer.writerow(
                [
                    epoch,
                    f"{lr:.2e}",
                    f"{train_loss:.4f}",
                    f"{liver:.4f}",
                    f"{tumor:.4f}",
                    f"{seconds:.0f}",
                ]
            )
            f.flush()
            print(
                f"epoch {epoch:>3} | lr {lr:.1e} | loss {train_loss:.4f} | "
                f"dice liver {liver:.3f} tumor {tumor:.3f} | {seconds:.0f}s",
                flush=True,
            )

            torch.save(model.state_dict(), out / "last.pt")
            if (liver + tumor) / 2 > best:
                best = (liver + tumor) / 2
                torch.save(model.state_dict(), out / "best.pt")
                (out / "best_val_scores.json").write_text(
                    json.dumps({"epoch": epoch, "per_case": scores}, indent=2) + "\n"
                )
    return out
