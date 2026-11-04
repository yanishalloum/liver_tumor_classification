import json

import numpy as np
import pytest
from conftest import SPACING, make_labels, write_case

from livertumor.cli import main
from livertumor.data import case_stats, find_cases, split_cases
from livertumor.data.split import NO_TUMOR, tumor_burden_strata
from livertumor.data.stats import count_lesions, read_stats

# --- discovery -------------------------------------------------------------------------


def test_find_cases_pairs_across_folders(lits_root):
    cases = find_cases([lits_root])
    assert [c.case_id for c in cases] == [0, 1, 2]
    assert cases[2].image.parent.name == "volume_pt2"
    assert cases[2].label.name == "segmentation-2.nii"


def test_find_cases_skips_unpaired(lits_root):
    (lits_root / "segmentations" / "segmentation-1.nii").unlink()
    with pytest.warns(UserWarning, match="without image/label pair"):
        cases = find_cases([lits_root])
    assert [c.case_id for c in cases] == [0, 2]


def test_find_cases_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_cases([tmp_path / "missing"])
    with pytest.raises(FileNotFoundError):
        find_cases([tmp_path])


# --- stats -----------------------------------------------------------------------------


def test_count_lesions_uses_3d_connectivity():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True  # diagonal neighbours -> one lesion
    mask[4, 4, 4] = True
    n, sizes = count_lesions(mask)
    assert n == 2
    assert sorted(sizes.tolist()) == [1, 2]


def test_case_stats_values(lits_root):
    stats = case_stats(find_cases([lits_root])[2])
    voxel_ml = np.prod(SPACING) / 1000
    assert stats["n_lesions"] == 2
    assert stats["spacing_z"] == pytest.approx(2.5)
    assert stats["liver_ml"] == pytest.approx(24 * 24 * 12 * voxel_ml, rel=1e-3)
    assert stats["tumor_ml"] == pytest.approx(2 * 27 * voxel_ml, rel=1e-3)
    assert stats["slices_with_liver"] == 12
    assert stats["slices_with_tumor"] == 3
    assert stats["orientation"] == "RAS"


def test_case_stats_without_tumor(lits_root):
    stats = case_stats(find_cases([lits_root])[0])
    assert stats["n_lesions"] == 0
    assert stats["tumor_ml"] == 0
    assert stats["largest_lesion_ml"] == 0


def test_case_stats_rejects_bad_labels(tmp_path):
    labels = make_labels(1)
    labels[0, 0, 0] = 7
    write_case(tmp_path, 5, labels)
    with pytest.raises(ValueError, match="unexpected label"):
        case_stats(find_cases([tmp_path])[0])


# --- split -----------------------------------------------------------------------------


def _fake_rows(n_without=20, n_with=60):
    rows = [{"case_id": i, "tumor_ml": 0.0} for i in range(n_without)]
    rows += [{"case_id": n_without + i, "tumor_ml": float(i + 1)} for i in range(n_with)]
    return rows


def test_strata_quantiles():
    strata = tumor_burden_strata(_fake_rows())
    counts = {s: list(strata.values()).count(s) for s in set(strata.values())}
    assert counts == {NO_TUMOR: 20, "small": 20, "medium": 20, "large": 20}


def test_split_is_disjoint_complete_and_stratified():
    strata = tumor_burden_strata(_fake_rows())
    split = split_cases(strata, val_fraction=0.15, test_fraction=0.2, seed=0)
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(strata)
    # every stratum of 20 cases contributes 4 test cases (20%)
    for stratum in set(strata.values()):
        assert sum(strata[c] == stratum for c in split["test"]) == 4


def test_split_is_deterministic():
    strata = tumor_burden_strata(_fake_rows())
    assert split_cases(strata, seed=1) == split_cases(strata, seed=1)
    assert split_cases(strata, seed=1) != split_cases(strata, seed=2)


@pytest.mark.parametrize(("val", "test"), [(0.5, 0.5), (-0.1, 0.2), (0.1, 1.0)])
def test_split_rejects_bad_fractions(val, test):
    with pytest.raises(ValueError):
        split_cases({0: NO_TUMOR}, val, test)


# --- CLI end to end --------------------------------------------------------------------


def test_cli_stats_then_split(lits_root, tmp_path):
    stats_csv = tmp_path / "out" / "stats.csv"
    split_json = tmp_path / "out" / "split.json"
    main(["stats", "--data-root", str(lits_root), "--out", str(stats_csv)])
    rows = read_stats(stats_csv)
    assert [r["n_lesions"] for r in rows] == [0, 1, 2]

    main(["split", "--stats", str(stats_csv), "--out", str(split_json), "--test", "0.34"])
    payload = json.loads(split_json.read_text())
    assert sorted(payload["train"] + payload["val"] + payload["test"]) == [0, 1, 2]
    assert payload["meta"]["seed"] == 42
    assert payload["strata"]["0"] == NO_TUMOR
