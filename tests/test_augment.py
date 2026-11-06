import numpy as np
import pytest

from livertumor.training.augment import AugmentConfig, augment, random_affine

NONE = AugmentConfig(p_affine=0, p_intensity=0, p_noise=0)
ONLY_AFFINE = AugmentConfig(p_affine=1, p_intensity=0, p_noise=0)
ONLY_INTENSITY = AugmentConfig(p_affine=0, p_intensity=1, p_noise=1)


def _patch(size=64):
    """5-channel image whose bright square matches the liver label exactly."""
    label = np.zeros((1, size, size), np.int64)
    label[0, 10:40, 36:56] = 1  # liver on the right of the patch
    label[0, 20:26, 44:50] = 2
    image = np.repeat((label > 0).astype(np.float32), 5, axis=0)
    return image, label


def test_nothing_happens_with_zero_probabilities():
    image, label = _patch()
    out_image, out_label = augment(image, label, np.random.default_rng(0), NONE)
    np.testing.assert_array_equal(out_image, image)
    np.testing.assert_array_equal(out_label, label)


@pytest.mark.parametrize("seed", range(5))
def test_affine_keeps_image_and_label_aligned(seed):
    image, label = _patch()
    out_image, out_label = augment(image, label, np.random.default_rng(seed), ONLY_AFFINE)
    assert out_image.shape == image.shape and out_label.shape == label.shape
    assert set(np.unique(out_label)) <= {0, 1, 2}  # nearest neighbour: no 0.5 labels
    bright, labelled = out_image[2] > 0.5, out_label[0] > 0
    overlap = 2 * (bright & labelled).sum() / (bright.sum() + labelled.sum())
    assert overlap > 0.9  # the square and its label moved together


def test_rotation_never_mirrors_left_right():
    image, label = _patch()
    config = AugmentConfig(max_rotation_deg=15, scale_range=(1, 1))
    for seed in range(20):
        _, out = random_affine(image, label, np.random.default_rng(seed), config, fill=0)
        columns = np.argwhere(out[0] > 0)[:, 1]
        assert columns.mean() > 32  # liver stays on the right half


def test_intensity_changes_image_only():
    image, label = _patch()
    out_image, out_label = augment(image, label, np.random.default_rng(1), ONLY_INTENSITY)
    assert not np.array_equal(out_image, image)
    np.testing.assert_array_equal(out_label, label)
    assert out_image.dtype == np.float32


def test_rotated_corners_are_filled_with_air():
    image, label = _patch()
    config = AugmentConfig(max_rotation_deg=45, scale_range=(1, 1))
    out, _ = random_affine(image + 5, label, np.random.default_rng(3), config, fill=-3.0)
    assert (out == -3.0).any()  # pixels coming from outside the patch
