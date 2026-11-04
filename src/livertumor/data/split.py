"""Patient-level, stratified train/validation/test split.

Why patient-level: neighbouring slices of one scan are nearly identical, so splitting
slices (as the legacy version did) leaks test information into training.

Why stratified: LiTS tumor burden is very skewed, from no tumor at all to lesions
occupying most of the liver. With only 131 patients, a plain random split can easily put
most of the large tumors in train and leave a test set that is not representative. We
therefore split *within* tumor-burden groups and merge the results.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

NO_TUMOR = "no_tumor"


def tumor_burden_strata(rows: Sequence[dict], n_bins: int = 3) -> dict[int, str]:
    """Assign each case to a tumor-burden stratum.

    Cases without tumor get their own stratum; the others are cut into ``n_bins``
    quantile bins of tumor volume (``small``, ``medium``, ``large`` for 3 bins).
    """
    names = {2: ["small", "large"], 3: ["small", "medium", "large"]}.get(
        n_bins, [f"q{i + 1}" for i in range(n_bins)]
    )
    with_tumor = [r for r in rows if r["tumor_ml"] > 0]
    strata = {int(r["case_id"]): NO_TUMOR for r in rows if r["tumor_ml"] <= 0}
    if with_tumor:
        volumes = np.array([r["tumor_ml"] for r in with_tumor])
        edges = np.quantile(volumes, np.linspace(0, 1, n_bins + 1)[1:-1])
        for r, v in zip(with_tumor, volumes, strict=True):
            strata[int(r["case_id"])] = names[int(np.searchsorted(edges, v, side="right"))]
    return strata


def split_cases(
    strata: dict[int, str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split case ids into train/val/test, keeping stratum proportions.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with sorted case ids.
    """
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for case_id, stratum in strata.items():
        groups[stratum].append(case_id)

    rng = np.random.default_rng(seed)
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for stratum in sorted(groups):  # sorted -> deterministic regardless of dict order
        ids = sorted(groups[stratum])
        rng.shuffle(ids)
        n_test = round(len(ids) * test_fraction)
        n_val = round(len(ids) * val_fraction)
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test : n_test + n_val]
        split["train"] += ids[n_test + n_val :]
    return {k: sorted(v) for k, v in split.items()}


def save_split(
    split: dict[str, list[int]], strata: dict[int, str], out: str | Path, **meta
) -> Path:
    """Save a split as JSON, with the stratum of each case and the parameters used."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "counts": {k: len(v) for k, v in split.items()},
        **split,
        "strata": {str(k): strata[k] for k in sorted(strata)},
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
