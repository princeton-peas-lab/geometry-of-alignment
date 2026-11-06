#!/usr/bin/env python3
"""fp32 learning-rate sweep under the paper protocol (``paper_table1_fp32``, seed 42).

Per model: refusal / SR / % repetitive responses / held-out loss for every (dataset, LR), and the
held-out-loss vote among LRs whose generations do not collapse.

    python analysis/table1_lr_sweep.py --run <run>
"""

from __future__ import annotations

import argparse

from common import COLLAPSE_FRACTION, DATASETS, MODELS, cell, collapsed_lrs, fmt, lr_vote, run_dir, sweep_lrs

EXP = "paper_table1_fp32"


def show_sweep(run, model: str) -> None:
    best, win = lr_vote(run, model, exp=EXP, exclude_collapsed=True)
    bad = collapsed_lrs(run, EXP, model)
    print(f"== {MODELS[model]}   (refusal / SR / % repetitive responses / held-out loss)")
    lrs = sweep_lrs(run, EXP, model)
    print(f"   {'dataset':24s}" + "".join(f"{'lr ' + lr:>32s}" for lr in lrs))
    for ds in DATASETS:
        row = ""
        for lr in lrs:
            c = cell(run, EXP, model, ds, f"lr{lr}") or {}
            mark = "†" if (c.get("repetitive") or 0) > COLLAPSE_FRACTION else " "
            row += (f"{fmt(c.get('refusal_wildguard'), '{:.0%}'):>7s} {fmt(c.get('sr'), '{:.2f}'):>5s}"
                    f"{mark}{fmt(c.get('repetitive'), '{:.0%}'):>6s} {fmt(c.get('eval_loss'), '{:.3f}'):>7s}   ")
        print(f"   {ds:24s}" + row)
    print(f"   collapsed (excluded): {', '.join(sorted(bad)) or 'none'}")
    print("   vote (lowest held-out loss per dataset): " + (", ".join(f"{d}={lr}" for d, lr in best.items()) or "--"))
    print(f"   vote winner: {win or 'none (tie or incomplete)'}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run name under results/repro/ or a path")
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    args = p.parse_args()
    run = run_dir(args.run)
    print(f"fp32 LR sweep ({EXP}) -- {run}\n"
          f"† = collapsed: > {COLLAPSE_FRACTION:.0%} of responses repetitive (< 30% distinct 4-grams)\n")
    for m in args.models:
        show_sweep(run, m)


if __name__ == "__main__":
    main()
