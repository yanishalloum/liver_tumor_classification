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
    from livertumor.data.split import save_split, split_cases, tumor_burden_strata
    from livertumor.data.stats import read_stats

    rows = read_stats(args.stats)
    strata = tumor_burden_strata(rows, n_bins=args.bins)
    split = split_cases(strata, args.val, args.test, args.seed)
    out = save_split(
        split,
        strata,
        args.out,
        seed=args.seed,
        val_fraction=args.val,
        test_fraction=args.test,
        stratification=f"tumor volume, {args.bins} quantile bins + no_tumor",
    )
    print({k: len(v) for k, v in split.items()}, "->", out)


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
    p.add_argument("--bins", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_split)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
