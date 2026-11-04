import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from livertumor import viz  # noqa: E402
from livertumor.data import Case, find_cases  # noqa: E402


def _marker_volume(tmp_path, affine_diag, name):
    """A volume whose voxel at array index (1, 6, 2) is marked; write image + labels."""
    data = np.zeros((8, 8, 4), dtype=np.int16)
    data[1, 6, 2] = 100
    affine = np.diag([*affine_diag, 1.0])
    for kind in ("volume", "segmentation"):
        nib.save(nib.Nifti1Image(data, affine), tmp_path / f"{kind}-{name}.nii")
    return Case(name, tmp_path / f"volume-{name}.nii", tmp_path / f"segmentation-{name}.nii")


def test_radiological_axial_puts_patient_right_on_viewer_left():
    plane = np.zeros((8, 8))  # RAS: x grows to the patient's Right, y to Anterior
    plane[7, 7] = 1  # most right, most anterior voxel
    shown = viz.radiological_axial(plane)
    assert shown[0, 0] == 1  # top-left on screen


@pytest.mark.parametrize(
    ("diag", "expected_x"),
    [
        ((1.0, 1.0, 1.0), 1),  # RAS: stored index 1 stays 1
        ((-1.0, 1.0, 1.0), 6),  # LAS: x axis flipped -> 8 - 1 - 1
        ((-1.0, -1.0, 1.0), 6),  # LPS: x and y flipped
    ],
)
def test_load_volume_reorients_to_ras(tmp_path, diag, expected_x):
    case = _marker_volume(tmp_path, diag, 7)
    vol = viz.load_volume(case)
    x, y, z = np.argwhere(vol.image == 100)[0]
    assert x == expected_x
    assert y == (6 if diag[1] > 0 else 1)
    assert z == 2
    assert vol.original_orientation == "".join(nib.aff2axcodes(np.diag([*diag, 1.0])))


def test_largest_slice():
    mask = np.zeros((4, 4, 5), dtype=bool)
    mask[:, :, 3] = True
    mask[0, 0, 1] = True
    assert viz.largest_slice(mask, axis=2) == 3


def test_describe(lits_root):
    cases = find_cases([lits_root])
    assert viz.describe(viz.load_volume(cases[0])) == "no tumor"
    assert viz.describe(viz.load_volume(cases[2])).endswith("2 lesions")


def test_figures_are_written(lits_root, tmp_path):
    cases = find_cases([lits_root])
    out = tmp_path / "fig"
    paths = [
        viz.gallery_figure([viz.load_volume(c) for c in cases], out),
        viz.windowing_figure(viz.load_volume(cases[2]), out),
        viz.orientation_figure(cases[:2], out),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 10_000
