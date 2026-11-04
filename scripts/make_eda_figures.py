"""Exploratory-analysis figures for the README, built from splits/lits_stats.csv.

Usage: uv run --extra viz python scripts/make_eda_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

BLUE, BLUE_LIGHT = "#2a78d6", "#cde2fb"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def acquisition_groups_figure(df: pd.DataFrame, groups: dict[str, str]) -> Path:
    """Slice thickness per case, with inferred acquisition groups as labelled bands."""
    df = df.assign(group=df.case_id.astype(str).map(groups))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (group, block) in enumerate(df.groupby("group")):
        lo, hi = block.case_id.min() - 0.5, block.case_id.max() + 0.5
        if i % 2 == 0:
            ax.axvspan(lo, hi, color="#f3f2ef", zorder=0, lw=0)
        orient = block.orientation.iloc[0]
        ax.text((lo + hi) / 2, 5.55, f"{group} · {orient}", ha="center", color=INK_2, fontsize=9)
    ax.scatter(df.case_id, df.spacing_z, s=16, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 5.9)
    ax.set_xlim(-1, df.case_id.max() + 1)
    ax.set_xlabel("LiTS case id")
    ax.set_ylabel("Slice thickness (mm)")
    ax.set_title("Scans come in blocks with their own acquisition settings", pad=22)
    ax.grid(axis="x", visible=False)
    path = OUT / "acquisition_groups.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def tumor_burden_figure(df: pd.DataFrame) -> Path:
    """Distribution of total tumor volume per patient (log scale) + lesion sizes."""
    tumor = df[df.tumor_ml > 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.3})

    bins = np.logspace(-2, 3, 26)
    ax1.hist(tumor.tumor_ml, bins=bins, color=BLUE, edgecolor="white", linewidth=1.5)
    ax1.set_xscale("log")
    for value, label, offset in (
        (tumor.tumor_ml.median(), "median", -1),
        (tumor.tumor_ml.mean(), "mean", 1),
    ):
        ax1.axvline(value, color=INK, lw=1, ls="--" if label == "mean" else "-")
        ax1.annotate(
            f"{label} {value:.0f} ml",
            (value, ax1.get_ylim()[1] * 0.92),
            xytext=(4 * offset, 0),
            textcoords="offset points",
            ha="left" if offset > 0 else "right",
            color=INK,
            fontsize=9,
        )
    n_free = int((df.tumor_ml == 0).sum())
    ax1.set_title("Total tumor volume per patient")
    ax1.set_xlabel(f"ml, log scale ({n_free} patients without tumor not shown)")
    ax1.set_ylabel("Patients")
    ax1.grid(axis="x", visible=False)

    sizes = np.sort(tumor.smallest_lesion_ml.to_numpy())
    ax2.hist(sizes, bins=np.logspace(-4, 3, 29), color=BLUE, edgecolor="white", linewidth=1.5)
    ax2.set_xscale("log")
    voxel_ml = 0.77 * 0.77 * 1.0 / 1000
    ax2.axvline(voxel_ml, color=INK, lw=1)
    ax2.annotate(
        "≈ 1 voxel",
        (voxel_ml, ax2.get_ylim()[1] * 0.92),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax2.set_title("Smallest lesion of each patient")
    ax2.set_xlabel("ml, log scale")
    ax2.set_ylabel("Patients")
    ax2.grid(axis="x", visible=False)

    path = OUT / "tumor_burden.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    """Build all figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "splits" / "lits_stats.csv")
    split = json.loads((ROOT / "splits" / "lits_split.json").read_text())
    for path in (
        acquisition_groups_figure(df, split["acquisition_group"]),
        tumor_burden_figure(df),
    ):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
