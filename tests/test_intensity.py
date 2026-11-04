import numpy as np
import pytest

from livertumor.intensity import ABDOMEN_WINDOW, window_ct


def test_window_bounds_map_to_zero_and_one():
    level, width = ABDOMEN_WINDOW
    image = np.array([level - width / 2, level, level + width / 2])
    np.testing.assert_allclose(window_ct(image, level, width), [0.0, 0.5, 1.0])


def test_values_outside_window_are_clipped():
    out = window_ct(np.array([-1000.0, 3000.0]), level=40, width=400)
    np.testing.assert_allclose(out, [0.0, 1.0])


def test_shape_and_dtype_preserved():
    image = np.random.default_rng(0).integers(-1024, 2000, size=(4, 8, 8)).astype(np.int16)
    out = window_ct(image, *ABDOMEN_WINDOW)
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_input_is_not_modified():
    image = np.array([-500.0, 500.0])
    window_ct(image, 40, 400)
    np.testing.assert_array_equal(image, [-500.0, 500.0])


@pytest.mark.parametrize("width", [0, -10])
def test_invalid_width_raises(width):
    with pytest.raises(ValueError):
        window_ct(np.zeros(3), 40, width)
