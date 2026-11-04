import csv
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("monai")

from livertumor.data import Case  # noqa: E402
from livertumor.preprocess import (  # noqa: E402
    PreprocessConfig,
    intensity_statistics,
    preprocess_case,
    preprocess_dataset,
)
from livertumor.transforms import NormalizeCTd, normalize_ct  # noqa: E402


def _phantom(tmp_path, case_id=0, flip_x=True, spacing=(1.6, 1.6, 4.0)):
    """Air box containing a 'body' (0 HU) with a liver (60 HU) and a tumor (120 HU).

    Stored LAS (x flipped) with thick slices, like many LiTS scans. The liver is on the
    array's low-x side, which is the patient's *right* once the LAS flip is undone.
    """
    shape = (40, 40, 12)
    image = np.full(shape, -1000, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    image[5:35, 5:35, 1:11] = 0  # body
    image[6:20, 10:30, 2:10] = 60  # liver
    labels[6:20, 10:30, 2:10] = 1
    image[10:14, 18:22, 5:7] = 120  # tumor inside the liver
    labels[10:14, 18:22, 5:7] = 2
    sign = -1.0 if flip_x else 1.0
    affine = np.diag([sign * spacing[0], spacing[1], spacing[2], 1.0])
    paths = (tmp_path / f"volume-{case_id}.nii", tmp_path / f"segmentation-{case_id}.nii")
    nib.save(nib.Nifti1Image(image, affine), paths[0])
    nib.save(nib.Nifti1Image(labels.astype(np.float32), affine), paths[1])
    return Case(case_id, *paths)


CONFIG = PreprocessConfig(spacing=(0.8, 0.8, 1.0), margin=2, samples_per_case=500)


@pytest.fixture
def processed(tmp_path):
    case = _phantom(tmp_path)
    result = preprocess_case(case, "train", tmp_path / "out", CONFIG)
    image = nib.load(tmp_path / "out" / result.image)
    labels = nib.load(tmp_path / "out" / result.label)
    return result, image, labels


def test_output_is_ras_with_target_spacing(processed):
    _, image, labels = processed
    for nii in (image, labels):
        assert "".join(nib.aff2axcodes(nii.affine)) == "RAS"
        np.testing.assert_allclose(nii.header.get_zooms()[:3], CONFIG.spacing, atol=1e-4)


def test_dtypes_and_label_values(processed):
    _, image, labels = processed
    assert image.get_data_dtype() == np.int16
    assert labels.get_data_dtype() == np.uint8
    assert set(np.unique(np.asarray(labels.dataobj))) == {0, 1, 2}  # nearest: no 1.5


def test_volumes_are_preserved_by_resampling(processed):
    result, _, _ = processed
    assert result.liver_ml_after == pytest.approx(result.liver_ml_before, rel=0.05)
    assert result.tumor_ml_after == pytest.approx(result.tumor_ml_before, rel=0.15)


def test_air_is_cropped(processed):
    result, image, _ = processed
    # body spans 30 x 30 x 10 voxels of 1.6 x 1.6 x 4 mm = 48 x 48 x 40 mm
    # -> about 60 x 60 x 40 voxels at 0.8 x 0.8 x 1 mm, plus a margin of 2 on each side
    assert result.shape[0] <= 60 + 2 * CONFIG.margin + 2
    hu = np.asarray(image.dataobj)
    assert (hu > -500).mean() > 0.7  # mostly body, little air left


def test_liver_ends_up_on_patient_right(processed):
    _, _, labels = processed
    lab = np.asarray(labels.dataobj)
    liver_x = np.argwhere(lab == 1)[:, 0].mean()
    # RAS: x grows toward the patient's right, so the liver sits in the high-x half
    assert liver_x > lab.shape[0] / 2


def test_intensity_statistics_clip_outliers():
    rng = np.random.default_rng(0)
    samples = np.concatenate([rng.normal(100, 10, 10_000), [5000.0] * 10])  # 0.1% spikes
    stats = intensity_statistics(samples.astype(np.float32))
    assert stats["clip_high"] < 200
    assert stats["mean"] == pytest.approx(100, abs=1)
    assert stats["std"] == pytest.approx(10, rel=0.1)
    with pytest.raises(ValueError):
        intensity_statistics(np.empty(0))


def test_dataset_stats_use_train_cases_only(tmp_path):
    train_case = _phantom(tmp_path, case_id=0)
    test_case = _phantom(tmp_path, case_id=1)
    # make the test case much brighter: it must not influence the statistics
    img = nib.load(test_case.image)
    bright = np.asarray(img.dataobj) + np.int16(500) * (np.asarray(img.dataobj) > 0)
    nib.save(nib.Nifti1Image(bright.astype(np.int16), img.affine), test_case.image)

    split = {"train": [0], "val": [], "test": [1]}
    summary = preprocess_dataset([train_case, test_case], split, tmp_path / "ds", CONFIG, 1)
    assert summary["intensity"]["clip_high"] <= 120 + 1e-3

    rows = list(csv.DictReader((tmp_path / "ds" / "manifest.csv").open()))
    assert [r["part"] for r in rows] == ["train", "test"]
    saved = json.loads((tmp_path / "ds" / "preprocessing.json").read_text())
    assert saved["config"]["orientation"] == "RAS"


def test_normalize_ct_numpy_and_torch():
    torch = pytest.importorskip("torch")
    stats = {"clip_low": -100.0, "clip_high": 200.0, "mean": 50.0, "std": 25.0}
    x = np.array([-1000.0, 50.0, 75.0, 3000.0])
    expected = [-6.0, 0.0, 1.0, 6.0]
    np.testing.assert_allclose(normalize_ct(x, **stats), expected)
    np.testing.assert_allclose(normalize_ct(torch.tensor(x), **stats).numpy(), expected)
    out = NormalizeCTd(["image"], stats)({"image": x, "label": "kept"})
    assert out["label"] == "kept"
    np.testing.assert_allclose(out["image"], expected)
    with pytest.raises(ValueError):
        normalize_ct(x, -1, 1, 0, 0)


def test_parallel_workers_give_same_result(tmp_path):
    cases = [_phantom(tmp_path, case_id=i) for i in range(2)]
    split = {"train": [0, 1], "val": [], "test": []}
    serial = preprocess_dataset(cases, split, tmp_path / "s", CONFIG, workers=1)
    parallel = preprocess_dataset(cases, split, tmp_path / "p", CONFIG, workers=2)
    assert serial["intensity"] == parallel["intensity"]


def test_preprocessing_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from livertumor import viz

    case = _phantom(tmp_path)
    summary = preprocess_dataset(
        [case], {"train": [0], "val": [], "test": []}, tmp_path / "ds", CONFIG, workers=1
    )
    volumes = tmp_path / "ds" / "volumes"
    path = viz.preprocessing_figure(
        case,
        volumes / "case_000_image.nii.gz",
        volumes / "case_000_label.nii.gz",
        summary["intensity"],
        tmp_path / "fig",
    )
    assert path.stat().st_size > 10_000
