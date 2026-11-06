import json

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from livertumor.training.augment import AugmentConfig  # noqa: E402
from livertumor.training.dataset import (  # noqa: E402
    SliceDataset,
    Volume,
    crop_or_pad,
    load_intensity_stats,
    load_volumes,
    neighbour_slices,
    seed_worker,
)

STATS = {"clip_low": -100.0, "clip_high": 200.0, "mean": 100.0, "std": 50.0}


def _volume(case_id=0, shape=(64, 48, 20)):
    """Slices 0-4 empty, liver on slices 5-14, tumor on slices 8-9 only."""
    image = np.full(shape, -1000, dtype=np.int16)
    image[:, :, :] = np.arange(shape[2], dtype=np.int16)  # slice z has value z (HU)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[20:40, 10:30, 5:15] = 1
    labels[25:30, 15:20, 8:10] = 2
    return Volume(case_id, image, labels)


def test_neighbour_slices_stacks_context_and_repeats_edges():
    image = _volume().image
    stack = neighbour_slices(image, z=10, context=2)
    assert stack.shape == (5, 64, 48)
    assert stack[:, 0, 0].tolist() == [8, 9, 10, 11, 12]
    assert neighbour_slices(image, z=0, context=2)[:, 0, 0].tolist() == [0, 0, 0, 1, 2]


def test_crop_or_pad_centres_and_pads():
    array = np.arange(16).reshape(1, 4, 4)
    out = crop_or_pad(array, center=(0, 0), size=4, fill=-1)
    # the window is centred on (0, 0): its top-left quarter lies outside the array
    assert out.shape == (1, 4, 4)
    assert out[0, :2, :2].tolist() == [[-1, -1], [-1, -1]]
    assert out[0, 2:, 2:].tolist() == [[0, 1], [4, 5]]


def test_items_have_expected_shapes_and_types():
    dataset = SliceDataset([_volume()], STATS, context=1, patch_size=32, samples_per_epoch=7)
    assert len(dataset) == 7
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert item["label"].shape == (1, 32, 32) and item["label"].dtype == torch.int64
    assert set(item["label"].unique().tolist()) <= {0, 1, 2}


def test_label_matches_centre_slice():
    dataset = SliceDataset([_volume()], STATS, context=2, patch_size=64, seed=1)
    for _ in range(20):
        item = dataset[0]
        # the image encodes the slice index: the centre channel tells which slice it is
        z = round(float(item["image"][2].max()) * STATS["std"] + STATS["mean"])
        has_liver = bool((item["label"] > 0).any())
        assert has_liver == (5 <= z < 15)


def test_oversampling_favours_tumor_slices():
    dataset = SliceDataset([_volume()], STATS, p_tumor=0.5, p_liver=0.3, seed=0)
    picks = np.array([dataset.pick_slice(0) for _ in range(2000)])
    # 2 tumor slices out of 20 would be 10% uniformly; oversampling gives > 50%
    assert np.isin(picks, [8, 9]).mean() > 0.5
    assert (picks < 5).mean() > 0.05  # empty slices are still seen sometimes


def test_seed_worker_gives_different_streams():
    dataset = SliceDataset([_volume()], STATS, patch_size=16, samples_per_epoch=8)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, num_workers=2, worker_init_fn=seed_worker
    )
    first, second = list(loader)  # one batch from each worker
    assert not torch.equal(first["image"], second["image"])


def test_load_volumes_and_stats(tmp_path):
    (tmp_path / "volumes").mkdir()
    rows = []
    for case_id, part in ((0, "train"), (1, "val")):
        vol = _volume(case_id)
        for kind, array in (("image", vol.image), ("label", vol.labels)):
            name = f"case_{case_id:03d}_{kind}.nii.gz"
            nib.save(nib.Nifti1Image(array, np.eye(4)), tmp_path / "volumes" / name)
        rows.append(
            f"{case_id},{part},case_{case_id:03d}_image.nii.gz,case_{case_id:03d}_label.nii.gz"
        )
    (tmp_path / "manifest.csv").write_text("case_id,part,image,label\n" + "\n".join(rows) + "\n")
    (tmp_path / "preprocessing.json").write_text(json.dumps({"intensity": STATS}))

    train = load_volumes(tmp_path, "train")
    assert [v.case_id for v in train] == [0]
    assert train[0].image.dtype == np.int16 and train[0].labels.dtype == np.uint8
    assert load_intensity_stats(tmp_path) == STATS


def test_augmented_items_stay_valid():
    dataset = SliceDataset(
        [_volume()], STATS, patch_size=32, seed=0, augmentation=AugmentConfig(p_affine=1)
    )
    for _ in range(10):
        item = dataset[0]
        assert item["image"].shape == (5, 32, 32) and item["image"].dtype == torch.float32
        assert set(item["label"].unique().tolist()) <= {0, 1, 2}
