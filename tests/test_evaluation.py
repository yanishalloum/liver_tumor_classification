import json

import numpy as np
import pytest

from livertumor.evaluation import (
    evaluate_case,
    lesion_components,
    match_lesions,
    summarize,
)


def _cube(mask, corner, size):
    x, y, z = corner
    mask[x : x + size, y : y + size, z : z + size] = True


def test_small_components_are_dropped_and_renumbered():
    mask = np.zeros((20, 20, 20), bool)
    _cube(mask, (0, 0, 0), 3)  # 27 voxels: kept
    _cube(mask, (10, 10, 10), 2)  # 8 voxels: below 10, dropped
    _cube(mask, (15, 0, 0), 3)  # kept
    components = lesion_components(mask, min_voxels=10)
    assert components.max() == 2
    assert set(np.unique(components)) == {0, 1, 2}


def test_match_lesions_counts_hits_misses_and_false_positives():
    truth = np.zeros((30, 30, 30), bool)
    _cube(truth, (0, 0, 0), 4)  # lesion A
    _cube(truth, (20, 20, 20), 4)  # lesion B
    pred = np.zeros_like(truth)
    _cube(pred, (1, 1, 1), 4)  # overlaps A -> detected
    _cube(pred, (10, 10, 10), 4)  # touches nothing -> false positive
    match = match_lesions(pred, truth, min_voxels=10)
    assert (match.true_lesions, match.detected, match.missed, match.false_positives) == (
        2,
        1,
        1,
        1,
    )


def test_tiny_predicted_specks_are_not_false_positives():
    truth = np.zeros((10, 10, 10), bool)
    pred = np.zeros_like(truth)
    pred[5, 5, 5] = True  # one noisy voxel
    assert match_lesions(pred, truth).false_positives == 0


def test_healthy_patient_with_clean_prediction():
    labels = np.zeros((20, 20, 20), np.uint8)
    labels[2:18, 2:18, 2:18] = 1
    row = evaluate_case(labels.copy(), labels, spacing=(1.0, 1.0, 1.0))
    assert row["dice_liver"] == 1.0 and row["dice_tumor"] == 1.0
    assert row["has_tumor_true"] is False and row["has_tumor_pred"] is False


def test_volumes_use_spacing():
    labels = np.zeros((20, 20, 20), np.uint8)
    labels[0:10, 0:10, 0:10] = 2  # 1000 voxels of 2 mm^3 = 2 ml
    row = evaluate_case(labels, labels, spacing=(1.0, 1.0, 2.0))
    assert row["tumor_ml_true"] == pytest.approx(2.0)


def test_summarize_by_hand():
    rows = [
        # sick patient, 2 of 3 lesions found, perfect-ish overlap
        dict(
            dice_liver=0.9,
            dice_tumor=0.8,
            tumor_intersection=80,
            tumor_sum=200,
            detected_lesions=2,
            missed_lesions=1,
            false_positive_lesions=0,
            has_tumor_true=True,
            has_tumor_pred=True,
        ),
        # sick patient, missed entirely
        dict(
            dice_liver=0.9,
            dice_tumor=0.0,
            tumor_intersection=0,
            tumor_sum=10,
            detected_lesions=0,
            missed_lesions=1,
            false_positive_lesions=0,
            has_tumor_true=True,
            has_tumor_pred=False,
        ),
        # healthy patient with one false alarm
        dict(
            dice_liver=1.0,
            dice_tumor=0.0,
            tumor_intersection=0,
            tumor_sum=10,
            detected_lesions=0,
            missed_lesions=0,
            false_positive_lesions=1,
            has_tumor_true=False,
            has_tumor_pred=True,
        ),
    ]
    s = summarize(rows)
    assert s["dice_tumor_global"] == pytest.approx(2 * 80 / 220, abs=1e-4)
    assert s["lesion_recall"] == pytest.approx(2 / 4)
    assert s["lesion_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["patient_sensitivity"] == 0.5  # 1 of 2 sick patients flagged
    assert s["patient_specificity"] == 0.0  # the healthy one got a false alarm


def test_evaluate_cli_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from test_training_loop import _dataset_dir

    from livertumor.cli import main
    from livertumor.training.train import TrainConfig, train

    data = _dataset_dir(tmp_path / "data")
    run = train(
        TrainConfig(
            dataset_dir=str(data),
            out_dir=str(tmp_path / "run"),
            epochs=1,
            samples_per_epoch=4,
            batch_size=2,
            patch_size=32,
            context=1,
            base_channels=4,
            num_workers=0,
        )
    )
    main(["evaluate", "--run-dir", str(run), "--dataset-dir", str(data), "--figures", "1"])
    out = run / "eval_val"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["patients"] == 1 and summary["part"] == "val"
    assert (out / "per_case.csv").exists()
    assert len(list(out.glob("prediction_case_*.png"))) == 1
