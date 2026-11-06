#!/usr/bin/env python3
"""Δw-FIM overlap probe (``paper_probe_fp32``), over training seeds.

Per model and dataset: peak Rayleigh quotient of the weight drift against the projected refusal
FIM (in sigma of the random-direction null), refusal log-prob before / after training, and the
safety of the matching Table 1 cells; then Pearson r between drift and StrongREJECT.

    python analysis/deltaw_overlap.py --run <run>
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from common import DATASETS, MODELS, cell, cell_dir, fisher_path, fmt, null_stats, pearson, run_dir
from main_results import SEEDS, T1_EXP, fixed_lr, seed_tag

EXP = "paper_probe_fp32"


def refusal_logprob(path: Path) -> tuple[float, float] | None:
    """(step 0, last step) refusal mean log-prob from a safety_monitor.txt."""
    if not path.is_file():
        return None
    vals = [float(dict(kv.split("=", 1) for kv in l.split("\t") if "=" in kv)["safety_utility"])
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return (vals[0], vals[-1]) if vals else None


def spread(vals: list[float], spec: str) -> str:
    if not vals:
        return "--"
    m = format(statistics.mean(vals), spec)
    return m if len(vals) == 1 else f"{m} [{format(min(vals), spec)}, {format(max(vals), spec)}]"


def mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def report(run, model: str) -> None:
    lr = fixed_lr(run, model)
    null = null_stats(fisher_path(run, model))
    print(f"== {MODELS[model]}  (lr {lr}, seeds {'/'.join(map(str, SEEDS))})")
    if lr is None or null is None:
        print("   (no fixed fp32 LR or no Fisher yet)\n")
        return
    print(f"   random-direction null: mean {null[0]:.1f}, std {null[1]:.1f}")
    head = (f"   {'dataset':24s} {'n':>2s} {'peak sigma [min, max]':>24s} {'logp0':>7s} {'logpT':>7s} "
            f"{'refusal':>8s} {'SR':>6s}")
    print(head)
    print("   " + "-" * (len(head) - 3))
    means, pairs = [], []
    for ds in DATASETS:
        sig, lp0, lpT, ref, sr = [], [], [], [], []
        for s in SEEDS:
            tag = seed_tag(f"lr{lr}", s)
            peak = (cell(run, EXP, model, ds, tag) or {}).get("peak_rayleigh")
            lp = refusal_logprob(cell_dir(run, EXP, model, ds, tag) / "safety_monitor.txt")
            t1 = cell(run, T1_EXP, model, ds, tag) or {}
            if peak is not None:
                sig.append((peak - null[0]) / null[1])
            if lp:
                lp0.append(lp[0]); lpT.append(lp[1])
            if t1.get("refusal_wildguard") is not None:
                ref.append(t1["refusal_wildguard"])
            if t1.get("sr") is not None:
                sr.append(t1["sr"])
                if peak is not None:
                    pairs.append((sig[-1], t1["sr"]))
        if sig and sr:
            means.append((mean(sig), mean(sr)))
        print(f"   {ds:24s} {len(sig):2d} {spread(sig, '+.2f'):>24s} {fmt(mean(lp0), '{:7.3f}', '     --')} "
              f"{fmt(mean(lpT), '{:7.3f}', '     --')} {fmt(mean(ref), '{:8.1%}', '      --')} {fmt(mean(sr), '{:6.3f}', '    --')}")
    r_mean = pearson([a for a, _ in means], [b for _, b in means])
    r_all = pearson([a for a, _ in pairs], [b for _, b in pairs])
    print(f"   Pearson r(sigma, SR): seed means {fmt(r_mean, '{:+.3f}')} (n={len(means)}), "
          f"seed-paired cells {fmt(r_all, '{:+.3f}')} (n={len(pairs)})\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run name under results/repro/ or a path")
    args = p.parse_args()
    run = run_dir(args.run)
    print(f"fp32 Δw probe (paper protocol) -- {run}\n")
    for m in MODELS:
        report(run, m)


if __name__ == "__main__":
    main()
