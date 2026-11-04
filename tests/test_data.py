import json

import numpy as np
import pytest
from conftest import SPACING, make_labels, write_case

from livertumor.cli import main
from livertumor.data import case_stats, find_cases, systematic_split
from livertumor.data.split import (
    NO_TUMOR,
    PARTS,
    acquisition_groups,
    imbalance,
    rerandomized_split,
    tumor_burden_strata,
)
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


def _fake_rows():
    """Two acquisition groups (ids 0-39 LAS, 40-99 RAS) with a spread of tumor volumes."""
    rows = []
    for i in range(100):
        orientation = "LAS" if i < 40 else "RAS"
        tumor_ml = 0.0 if i % 10 == 0 else float((i * 37) % 97 + 1)
        rows.append({"case_id": i, "orientation": orientation, "tumor_ml": tumor_ml})
    return rows


def test_acquisition_groups_are_runs_of_consecutive_ids():
    rows = [
        {"case_id": i, "orientation": o}
        for i, o in enumerate(["LAS", "LAS", "RAS", "RAS", "LPS", "RAS"])
    ]
    groups = acquisition_groups(rows)
    # the second RAS run is a different group: it is not contiguous with the first one
    assert [groups[i] for i in range(6)] == ["A", "A", "B", "B", "C", "D"]


def test_group_names_go_past_z():
    rows = [{"case_id": i, "orientation": str(i)} for i in range(28)]
    groups = acquisition_groups(rows)
    assert groups[25] == "Z" and groups[26] == "AA" and groups[27] == "AB"


def test_strata_quantiles():
    rows = [{"case_id": i, "tumor_ml": 0.0} for i in range(5)]
    rows += [{"case_id": 5 + i, "tumor_ml": float(i + 1)} for i in range(9)]
    strata = tumor_burden_strata(rows)
    values = list(strata.values())
    assert values.count(NO_TUMOR) == 5
    assert values.count("small") == values.count("medium") == values.count("large") == 3


def test_split_is_disjoint_and_complete():
    rows = _fake_rows()
    split = systematic_split(rows, acquisition_groups(rows), seed=0)
    sets = [set(split[p]) for p in PARTS]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(range(100))


def test_split_balances_each_group_exactly():
    rows = _fake_rows()
    groups = acquisition_groups(rows)
    split = systematic_split(rows, groups, val_fraction=0.15, test_fraction=0.2, seed=3)
    # group A has 40 cases -> 8 test, 6 val ; group B has 60 -> 12 test, 9 val
    for group, size in (("A", 40), ("B", 60)):
        n_test = sum(groups[c] == group for c in split["test"])
        n_val = sum(groups[c] == group for c in split["val"])
        assert n_test == round(size * 0.2)
        assert n_val == round(size * 0.15)


def test_split_spreads_tumor_burden():
    rows = _fake_rows()
    split = systematic_split(rows, acquisition_groups(rows), seed=1)
    volume = {r["case_id"]: r["tumor_ml"] for r in rows}
    medians = {p: np.median([volume[c] for c in split[p]]) for p in PARTS}
    overall = np.median(list(volume.values()))
    for part in PARTS:
        assert abs(medians[part] - overall) < 0.25 * overall


def test_split_is_deterministic():
    rows = _fake_rows()
    groups = acquisition_groups(rows)
    assert systematic_split(rows, groups, seed=1) == systematic_split(rows, groups, seed=1)
    assert systematic_split(rows, groups, seed=1) != systematic_split(rows, groups, seed=2)


def test_imbalance_is_zero_for_perfect_split():
    labels = {0: "a", 1: "b", 2: "a", 3: "b"}
    assert imbalance({"train": [0, 1], "test": [2, 3]}, labels) == 0
    assert imbalance({"train": [0, 2], "test": [1, 3]}, labels) == 1


def test_rerandomization_never_worse_than_any_candidate():
    rows = _fake_rows()
    groups, strata = acquisition_groups(rows), tumor_burden_strata(rows)
    split, seed, score = rerandomized_split(rows, groups, strata, n_candidates=20)
    assert split == systematic_split(rows, groups, seed=seed)
    assert score == imbalance(split, strata)
    assert all(
        score <= imbalance(systematic_split(rows, groups, seed=s), strata) for s in range(20)
    )


@pytest.mark.parametrize(("val", "test"), [(0.5, 0.5), (-0.1, 0.2), (0.1, 1.0)])
def test_split_rejects_bad_fractions(val, test):
    rows = [{"case_id": 0, "orientation": "RAS", "tumor_ml": 0.0}]
    with pytest.raises(ValueError):
        systematic_split(rows, {0: "A"}, val, test)


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
    assert "chosen among 1000 candidates" in payload["meta"]["seed_selection"]

    main(["split", "--stats", str(stats_csv), "--out", str(split_json), "--rerandomize", "0"])
    assert json.loads(split_json.read_text())["meta"]["seed"] == 42
    assert payload["tumor_stratum"]["0"] == NO_TUMOR
    assert set(payload["acquisition_group"].values()) == {"A"}


def test_random_offset_gives_every_part_a_fair_chance():
    """Regression test: the smallest-tumor patient must not always land in train."""
    rows = _fake_rows()
    groups = acquisition_groups(rows)
    smallest = min(
        (r for r in rows if r["case_id"] < 40), key=lambda r: (r["tumor_ml"], r["case_id"])
    )
    hits = {p: 0 for p in PARTS}
    for seed in range(300):
        split = systematic_split(rows, groups, val_fraction=0.15, test_fraction=0.2, seed=seed)
        hits[next(p for p in PARTS if smallest["case_id"] in split[p])] += 1
    assert 0.08 < hits["val"] / 300 < 0.25
    assert 0.12 < hits["test"] / 300 < 0.30
