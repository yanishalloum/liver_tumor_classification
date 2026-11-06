import csv
import json

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from livertumor.cli import main  # noqa: E402
from livertumor.training.dataset import Volume  # noqa: E402
from livertumor.training.evaluate import dice, pad_to_multiple, predict_volume  # noqa: E402
from livertumor.training.model import build_unet  # noqa: E402
from livertumor.training.train import TrainConfig, poly_lr, train  # noqa: E402

STATS = {"clip_low": -100.0, "clip_high": 200.0, "mean": 50.0, "std": 50.0}


def test_dice():
    a = np.zeros((4, 4), bool)
    a[:2] = True  # 8 pixels
    b = np.zeros((4, 4), bool)
    b[1:3] = True  # 8 pixels, 4 in common with a
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(a, b) == pytest.approx(2 * 4 / (8 + 8))
    assert dice(np.zeros(3, bool), np.zeros(3, bool)) == 1.0  # nothing to find, nothing found


def test_pad_to_multiple():
    array = np.ones((5, 37, 45))
    padded, crop = pad_to_multiple(array, 16, fill=0)
    assert padded.shape == (5, 48, 48)
    assert padded[(..., *crop)].shape == (5, 37, 45)
    assert padded[:, 37:].sum() == 0


def test_predict_volume_keeps_shape():
    volume = Volume(0, np.zeros((37, 45, 6), np.int16), np.zeros((37, 45, 6), np.uint8))
    model = build_unet(in_channels=3, base_channels=4, depth=3)
    pred = predict_volume(model, volume, STATS, context=1, device=torch.device("cpu"))
    assert pred.shape == (37, 45, 6) and pred.dtype == np.uint8
    assert set(np.unique(pred)) <= {0, 1, 2}


def test_poly_lr_decreases_to_zero():
    lrs = [poly_lr(e, 10, 1e-3) for e in range(11)]
    assert lrs[0] == 1e-3 and lrs[-1] == 0
    assert all(a > b for a, b in zip(lrs, lrs[1:], strict=False))


def _dataset_dir(tmp_path):
    """Two training volumes and one validation volume, as written by preprocessing."""
    (tmp_path / "volumes").mkdir(parents=True)
    rows = []
    for case_id, part in ((0, "train"), (1, "train"), (2, "val")):
        image = np.full((40, 40, 10), -1000, np.int16)
        labels = np.zeros((40, 40, 10), np.uint8)
        image[8:32, 8:32, 2:8], labels[8:32, 8:32, 2:8] = 60, 1
        image[14:20, 14:20, 4:6], labels[14:20, 14:20, 4:6] = 150, 2
        for kind, array in (("image", image), ("label", labels)):
            nib.save(
                nib.Nifti1Image(array, np.eye(4)), tmp_path / "volumes" / f"{case_id}_{kind}.nii.gz"
            )
        rows.append([case_id, part, f"{case_id}_image.nii.gz", f"{case_id}_label.nii.gz"])
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["case_id", "part", "image", "label"], *rows])
    (tmp_path / "preprocessing.json").write_text(json.dumps({"intensity": STATS}))
    return tmp_path


def test_train_end_to_end(tmp_path):
    config = TrainConfig(
        dataset_dir=str(_dataset_dir(tmp_path / "data")),
        out_dir=str(tmp_path / "run"),
        epochs=2,
        samples_per_epoch=8,
        batch_size=4,
        patch_size=32,
        context=1,
        base_channels=4,
        num_workers=0,
    )
    out = train(config)
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert all(0 <= float(r["val_dice_liver"]) <= 1 for r in rows)
    for name in ("config.json", "best.pt", "last.pt", "best_val_scores.json"):
        assert (out / name).exists()
    assert json.loads((out / "config.json").read_text())["epochs"] == 2


def test_cli_train(tmp_path):
    data = _dataset_dir(tmp_path / "data")
    main(
        [
            "train",
            "--dataset-dir",
            str(data),
            "--out-dir",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            "--samples-per-epoch",
            "4",
            "--batch-size",
            "2",
            "--patch-size",
            "32",
            "--context",
            "1",
            "--base-channels",
            "4",
            "--num-workers",
            "0",
            "--no-augment",
        ]
    )
    saved = json.loads((tmp_path / "run" / "config.json").read_text())
    assert saved["base_channels"] == 4
    assert saved["augment"] is False
