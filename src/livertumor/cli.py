"""Command-line interface: ``livertumor <command> ...``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _stats(args: argparse.Namespace) -> None:
    from livertumor.data.discovery import find_cases
    from livertumor.data.stats import case_stats, write_stats

    cases = find_cases(args.data_root)
    print(f"Found {len(cases)} cases")
    rows = []
    for i, case in enumerate(cases, 1):
        rows.append(case_stats(case))
        print(f"[{i:>3}/{len(cases)}] case {case.case_id} done", flush=True)
    out = write_stats(rows, args.out)
    print(f"Wrote {out}")


def _split(args: argparse.Namespace) -> None:
    from livertumor.data.split import (
        PARTS,
        acquisition_groups,
        rerandomized_split,
        save_split,
        systematic_split,
        tumor_burden_strata,
    )
    from livertumor.data.stats import read_stats

    rows = read_stats(args.stats)
    groups = acquisition_groups(rows)
    strata = tumor_burden_strata(rows)
    if args.rerandomize:
        split, seed, score = rerandomized_split(
            rows, groups, strata, args.rerandomize, args.val, args.test
        )
        selection = (
            f"seed {seed} chosen among {args.rerandomize} candidates as the most balanced "
            f"on tumor strata (max gap {score:.2f} patients); chosen before any training"
        )
    else:
        seed = args.seed
        split = systematic_split(rows, groups, args.val, args.test, seed)
        selection = "fixed seed"
    out = save_split(
        split,
        args.out,
        annotations={"acquisition_group": groups, "tumor_stratum": strata},
        seed=seed,
        seed_selection=selection,
        val_fraction=args.val,
        test_fraction=args.test,
        method=(
            "patient-level; systematic sampling on tumor volume inside each acquisition "
            "group. Groups are inferred (runs of consecutive case ids with the same "
            "orientation), not official site labels."
        ),
    )
    print(f"{'':<10}" + "".join(f"{p:>7}" for p in PARTS))
    for label, mapping in (("group", groups), ("tumor", strata)):
        for value in sorted(set(mapping.values())):
            counts = [sum(mapping[c] == value for c in split[p]) for p in PARTS]
            print(f"{label + ' ' + value:<10}" + "".join(f"{n:>7}" for n in counts))
    print("total     " + "".join(f"{len(split[p]):>7}" for p in PARTS), "->", out)


def _figures(args: argparse.Namespace) -> None:
    from pathlib import Path

    from livertumor import viz
    from livertumor.data.discovery import find_cases

    cases = {c.case_id: c for c in find_cases(args.data_root)}
    out = Path(args.out)
    print("wrote", viz.gallery_figure([viz.load_volume(cases[i]) for i in args.gallery], out))
    print("wrote", viz.windowing_figure(viz.load_volume(cases[args.windowing]), out))
    print("wrote", viz.orientation_figure([cases[i] for i in args.orientation], out))


def _preprocess(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from livertumor.data.discovery import find_cases
    from livertumor.preprocess import PreprocessConfig, preprocess_dataset

    split = json.loads(Path(args.split).read_text())
    config = PreprocessConfig(spacing=tuple(args.spacing))
    summary = preprocess_dataset(
        find_cases(args.data_root), split, args.out, config, workers=args.workers
    )
    print(json.dumps(summary["intensity"], indent=2))


def _train(args: argparse.Namespace) -> None:
    from dataclasses import fields

    from livertumor.training.train import TrainConfig, train

    names = {f.name for f in fields(TrainConfig)}
    config = TrainConfig(**{k: v for k, v in vars(args).items() if k in names and v is not None})
    print("results in", train(config))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="livertumor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="Compute per-case statistics of a LiTS folder")
    p.add_argument("--data-root", nargs="+", required=True, help="Folder(s) to search")
    p.add_argument("--out", default="lits_stats.csv")
    p.set_defaults(func=_stats)

    p = sub.add_parser("split", help="Create a stratified patient-level split")
    p.add_argument("--stats", required=True, help="CSV produced by `livertumor stats`")
    p.add_argument("--out", default="splits/lits_split.json")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42, help="used when --rerandomize is 0")
    p.add_argument(
        "--rerandomize",
        type=int,
        default=1000,
        metavar="N",
        help="try N seeds and keep the best tumor balance (0 = use --seed)",
    )
    p.set_defaults(func=_split)

    p = sub.add_parser("preprocess", help="Reorient, resample, crop and save (needs [train])")
    p.add_argument("--data-root", nargs="+", required=True)
    p.add_argument("--split", required=True, help="splits/lits_split.json")
    p.add_argument("--out", required=True)
    p.add_argument("--spacing", type=float, nargs=3, default=[0.77, 0.77, 1.0], metavar="MM")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=_preprocess)

    p = sub.add_parser("train", help="Train the 2.5D U-Net baseline (needs [train])")
    p.add_argument("--dataset-dir", dest="dataset_dir", required=True)
    p.add_argument("--out-dir", dest="out_dir", required=True)
    for name, kind in (
        ("epochs", int),
        ("samples-per-epoch", int),
        ("batch-size", int),
        ("learning-rate", float),
        ("patch-size", int),
        ("context", int),
        ("base-channels", int),
        ("num-workers", int),
        ("seed", int),
        ("max-val-cases", int),
    ):
        p.add_argument(f"--{name}", dest=name.replace("-", "_"), type=kind)
    p.set_defaults(func=_train)

    p = sub.add_parser("figures", help="Render illustrative scan figures (needs [viz])")
    p.add_argument("--data-root", nargs="+", required=True)
    p.add_argument("--out", default="figures")
    p.add_argument("--gallery", type=int, nargs="+", default=[38, 3, 102, 33])
    p.add_argument("--windowing", type=int, default=102)
    p.add_argument("--orientation", type=int, nargs="+", default=[0, 28, 53])
    p.set_defaults(func=_figures)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
