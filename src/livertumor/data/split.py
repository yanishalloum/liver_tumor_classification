"""Patient-level train/validation/test split, balanced on acquisition group and tumor burden.

Three ideas, in order of importance:

1. **Split by patient**, never by slice: neighbouring slices of one scan are nearly
   identical, so a slice-level split leaks test information into training.
2. **Balance acquisition groups.** LiTS pools CT scans from several hospitals. Consecutive
   case ids share orientation, slice thickness and pixel size, i.e. they very likely come
   from the same site. Each part of the split should contain every group in the same
   proportion. These groups are *inferred* from metadata, not provided by LiTS.
3. **Balance tumor burden.** Tumor volumes range from 0 to ~1 L; each part should contain
   small, medium and large tumors.

Crossing groups x burden bins would create strata of 2-3 patients, where 20% rounds
badly. Instead we use *systematic sampling*: inside each group, sort patients by tumor
volume and deal them out like cards (train, train, val, test, train, ...). Group
proportions are then exact and tumor burden is spread evenly within every group.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

NO_TUMOR = "no_tumor"
PARTS = ("train", "val", "test")


def acquisition_groups(rows: Sequence[dict], key: str = "orientation") -> dict[int, str]:
    """Infer acquisition groups as runs of consecutive case ids sharing ``key``.

    Example: ids 0-27 are LAS, 28-52 RAS, 53-67 LPS -> groups ``A``, ``B``, ``C``.
    """
    groups: dict[int, str] = {}
    previous, index = None, -1
    for row in sorted(rows, key=lambda r: int(r["case_id"])):
        if row[key] != previous:
            index, previous = index + 1, row[key]
        groups[int(row["case_id"])] = _group_name(index)
    return groups


def _group_name(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def tumor_burden_strata(rows: Sequence[dict], n_bins: int = 3) -> dict[int, str]:
    """Label each case ``no_tumor`` or a tumor-volume quantile bin (small/medium/large).

    Used to *report* balance; the split itself sorts on the continuous volume.
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


def systematic_split(
    rows: Sequence[dict],
    groups: dict[int, str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split cases with systematic sampling on tumor volume inside each group.

    For a group of ``n`` patients we first build a pattern of ``n`` part labels that
    interleaves train/val/test with the right counts (each position goes to the part
    furthest behind its target share). The pattern is then rotated by a random offset
    and laid over the patients sorted by tumor volume. Rotating keeps the counts exact
    while giving every part the same chance (its fraction) to receive any patient.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with sorted case ids.
    """
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")
    fractions = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[groups[int(row["case_id"])]].append(row)

    rng = np.random.default_rng(seed)
    split: dict[str, list[int]] = {part: [] for part in PARTS}
    for group in sorted(by_group):  # sorted -> deterministic for a given seed
        ordered = sorted(by_group[group], key=lambda r: (r["tumor_ml"], int(r["case_id"])))
        pattern = _interleaved_pattern(len(ordered), fractions)
        offset = int(rng.integers(len(ordered)))
        for i, row in enumerate(ordered):
            part = pattern[(i + offset) % len(ordered)]
            split[PARTS[part]].append(int(row["case_id"]))
    return {k: sorted(v) for k, v in split.items()}


def _interleaved_pattern(n: int, fractions: np.ndarray) -> list[int]:
    """Spread part indices over ``n`` slots as evenly as possible (largest deficit)."""
    assigned = np.zeros(len(fractions))
    pattern = []
    for i in range(1, n + 1):
        part = int(np.argmax(fractions * i - assigned))
        assigned[part] += 1
        pattern.append(part)
    return pattern


def imbalance(split: dict[str, list[int]], labels: dict[int, str]) -> float:
    """Largest gap, in patients, between observed and expected counts per (part, label).

    0 means every part contains each label exactly in proportion to its size.
    """
    total = sum(len(v) for v in split.values())
    values = sorted(set(labels.values()))
    overall = {v: sum(label == v for label in labels.values()) for v in values}
    return max(
        abs(sum(labels[c] == v for c in split[p]) - overall[v] * len(split[p]) / total)
        for p in split
        for v in values
    )


def rerandomized_split(
    rows: Sequence[dict],
    groups: dict[int, str],
    strata: dict[int, str],
    n_candidates: int = 1000,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
) -> tuple[dict[str, list[int]], int, float]:
    """Try seeds ``0..n_candidates-1`` and keep the split with the best tumor balance.

    Group balance is already guaranteed by :func:`systematic_split`; this only picks,
    among equally valid candidates, the one whose tumor strata are most even. The
    criterion depends on data balance only (never on model results), so it does not
    bias the evaluation. Ties go to the smallest seed.

    Returns:
        The chosen split, its seed and its tumor imbalance.
    """
    best = None
    for seed in range(n_candidates):
        split = systematic_split(rows, groups, val_fraction, test_fraction, seed)
        score = imbalance(split, strata)
        if best is None or score < best[2]:
            best = (split, seed, score)
    return best


def save_split(
    split: dict[str, list[int]],
    out: str | Path,
    annotations: dict[str, dict[int, str]] | None = None,
    **meta,
) -> Path:
    """Save a split as JSON with per-case annotations (group, tumor stratum) and params."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "meta": meta,
        "counts": {k: len(v) for k, v in split.items()},
        **split,
    }
    for name, mapping in (annotations or {}).items():
        payload[name] = {str(k): mapping[k] for k in sorted(mapping)}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out
