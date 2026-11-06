#!/usr/bin/env python3
"""LR-sweep ablation (``lr_sweep``): per (dataset, LR) cell, the Δw-FIM overlap, StrongREJECT,
refusal rate (string match) and held-out loss, plus the held-out-loss LR vote.

    python analysis/ablation_lr_sweep.py --run <run>
"""

from __future__ import annotations

import argparse

from common import (DATASETS, DEGENERATE_WORDS, MODELS, cell, fisher_path, fmt, lr_vote, null_stats, pearson,
                    run_dir, safety, sweep_lrs)

EXP = "lr_sweep"


def lrs(run, model: str) -> list[str]:
    return sweep_lrs(run, EXP, model)


def report(run, model: str) -> None:
    null = null_stats(fisher_path(run, model))
    print(f"== {MODELS[model]}")
    if null:
        print(f"   random-direction null: mean {null[0]:.1f}, std {null[1]:.1f} ({null[1] / null[0]:.1%})")
    else:
        print(f"   (no Fisher at {fisher_path(run, model)}; drift columns blank)")
    head = (f"   {'dataset':24s} {'lr':>6s} {'peak_R':>9s} {'xbase':>6s} {'sigma':>7s} "
            f"{'SR':>6s} {'SR=0':>6s} {'refusal':>8s} {'words':>6s} {'eval_loss':>9s}")
    print(head)
    print("   " + "-" * (len(head) - 3))
    for ds in DATASETS:
        for lr in lrs(run, model):
            c = cell(run, EXP, model, ds, f"lr{lr}")
            if not c:
                continue
            peak = c.get("peak_rayleigh")
            xb = peak / null[0] if (peak is not None and null) else None
            sg = (peak - null[0]) / null[1] if (peak is not None and null) else None
            collapse = "†" if (c.get("words") or 0) > DEGENERATE_WORDS else " "
            print(f"   {ds:24s} {lr:>6s} {fmt(peak, '{:9.1f}', '       --')} {fmt(xb, '{:6.2f}', '    --')} "
                  f"{fmt(sg, '{:+7.2f}', '     --')} {fmt(c.get('sr'), '{:6.3f}', '    --')}{collapse}"
                  f"{fmt(c.get('sr_zero'), '{:6.1%}', '    --')} {fmt(c.get('refusal_combined'), '{:8.1%}', '      --')} "
                  f"{fmt(c.get('words'), '{:6.0f}', '    --')} {fmt(c.get('eval_loss'), '{:9.4f}', '       --')}")
    best, winner = lr_vote(run, model)
    print("   LR vote (lowest held-out loss per dataset): "
          + (", ".join(f"{ds}={lr}" for ds, lr in best.items()) or "no eval losses yet"))
    print(f"   winner: {winner or 'none (tie or incomplete)'}")
    if winner and null:
        pts = [(c["peak_rayleigh"], c["sr"]) for ds in DATASETS
               if (c := cell(run, EXP, model, ds, f"lr{winner}")) and c.get("peak_rayleigh") is not None
               and c.get("sr") is not None]
        r = pearson([(p - null[0]) / null[1] for p, _ in pts], [s for _, s in pts])
        print(f"   Pearson r(sigma, SR) at lr={winner}: {fmt(r, '{:+.3f}')} (n={len(pts)})")
    print("   † mean response > 400 words: repetition collapse, the judge score reflects incoherence\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run name under results/repro/ or a path")
    p.add_argument("--model", choices=list(MODELS), help="one model (default: all)")
    args = p.parse_args()
    run = run_dir(args.run)
    models = [args.model] if args.model else list(MODELS)
    print(f"LR sweep / Δw probe -- {run}\n")
    print("Baseline (no finetuning), StrongREJECT prompts")
    print(f"   {'model':24s} {'SR':>7s} {'SR=0':>7s} {'refusal':>8s} {'words':>6s}")
    for m in models:
        s = safety(run / "baseline" / m / "strongreject")
        print(f"   {MODELS[m]:24s} {fmt(s and s['sr'], '{:7.3f}', '     --')} {fmt(s and s['sr_zero'], '{:7.1%}', '     --')} "
              f"{fmt(s and s['refusal_combined'], '{:8.1%}', '      --')} {fmt(s and s['words'], '{:6.0f}', '    --')}")
    print()
    for m in models:
        report(run, m)


if __name__ == "__main__":
    main()
