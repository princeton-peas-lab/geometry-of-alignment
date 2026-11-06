#!/usr/bin/env python3
"""Table 1 and the repr-FIM intervention sweep (fp32, mean [min-max] over training seeds).

Refusal = WildGuard refusal label; SR = mean StrongREJECT score (higher = more harmful).

    python analysis/main_results.py --run <run>
"""

from __future__ import annotations

import argparse
import statistics

from common import COLLAPSE_FRACTION, DATASETS, LAMBDAS, MODELS, cell, fmt, lam_key, run_dir

T1_EXP = "paper_table1_fp32"
LAMBDA_EXP = "paper_lambda_fp32"
SEEDS = [42, 1, 2]   # scripts/reproduce/env.sh FP32_SEEDS; tag suffix _s<seed> except for 42


def seed_tag(tag: str, seed: int) -> str:
    return tag if seed == 42 else f"{tag}_s{seed}"


def fixed_lr(run, model: str) -> str | None:
    """The LR with extra-seed Table 1 cells (one per model)."""
    lrs = {p.name.removeprefix("lr").split("_s")[0] for p in (run / T1_EXP / model).glob("*/lr*_s*") if p.is_dir()}
    if len(lrs) > 1:
        raise SystemExit(f"{model}: extra-seed cells at several LRs {sorted(lrs)}")
    return lrs.pop() if lrs else None


def stat(values: list, spec: str) -> str:
    """mean [min-max] over the seeds that have a value; '--' if none."""
    v = [x for x in values if x is not None]
    if not v:
        return "--"
    if len(v) == 1:
        return fmt(v[0], spec)
    return f"{fmt(statistics.mean(v), spec)} [{fmt(min(v), spec)}-{fmt(max(v), spec)}]"


def seeds_cells(run, exp: str, model: str, ds: str, tag: str) -> list[dict]:
    return [cell(run, exp, model, ds, seed_tag(tag, s)) or {} for s in SEEDS]


def n_collapsed(cells: list[dict]) -> int:
    return sum((c.get("repetitive") or 0) > COLLAPSE_FRACTION for c in cells)


def table1(run, models: list[str]) -> None:
    print(f"Table 1 in fp32 -- refusal / StrongREJECT, mean [min-max] over seeds {SEEDS}; "
          f"collapsed = seeds with > {COLLAPSE_FRACTION:.0%} repetitive responses")
    for m in models:
        lr = fixed_lr(run, m)
        print(f"\n== {MODELS[m]}   lr {lr or '? (no extra-seed cells yet)'}")
        if lr is None:
            continue
        print(f"   {'dataset':24s} {'refusal':>18s}   {'SR':>18s}   {'held-out loss':>22s}  seeds  collapsed")
        for ds in DATASETS:
            cs = seeds_cells(run, T1_EXP, m, ds, f"lr{lr}")
            n = sum(c.get("sr") is not None for c in cs)
            print(f"   {ds:24s} {stat([c.get('refusal_wildguard') for c in cs], '{:.0%}'):>18s}"
                  f"   {stat([c.get('sr') for c in cs], '{:.2f}'):>18s}"
                  f"   {stat([c.get('eval_loss') for c in cs], '{:.3f}'):>22s}  {n:>5d}  {n_collapsed(cs):>9d}")


def lambda_sweep(run, models: list[str]) -> None:
    print(f"\n\nRepr-FIM intervention in fp32 ({LAMBDA_EXP}) -- mean [min-max] over seeds {SEEDS}")
    print("Loss = held-out loss with the hooks attached; dLoss = change over the same seed's lambda = 0.")
    for ds, (label, group) in DATASETS.items():
        if ds == "pure_bad":
            continue
        for m in models:
            lr = {(cell_env(run, m, ds, lam) or {}).get("lr") for lam in LAMBDAS} - {None}
            print(f"\n{label} ({group}) -- {MODELS[m]}   lr {', '.join(sorted(lr)) or '--'}")
            print(f"   {'lambda':>7s} {'SR':>18s}   {'refusal':>18s}   {'Loss':>22s}   {'dLoss':>22s}  seeds  collapsed")
            base = seeds_cells(run, LAMBDA_EXP, m, ds, "lam0")
            for lam in LAMBDAS:
                cs = seeds_cells(run, LAMBDA_EXP, m, ds, f"lam{lam_key(lam)}")
                d = [100 * (c["eval_loss_hooked"] - b["eval_loss_hooked"]) / b["eval_loss_hooked"]
                     if c.get("eval_loss_hooked") is not None and b.get("eval_loss_hooked") else None
                     for c, b in zip(cs, base)]
                n = sum(c.get("sr") is not None for c in cs)
                print(f"   {lam:>7.2f} {stat([c.get('sr') for c in cs], '{:.3f}'):>18s}"
                      f"   {stat([c.get('refusal_wildguard') for c in cs], '{:.0%}'):>18s}"
                      f"   {stat([c.get('eval_loss_hooked') for c in cs], '{:.3f}'):>22s}"
                      f"   {stat(d, '{:+.1f}%'):>22s}  {n:>5d}  {n_collapsed(cs):>9d}")


def cell_env(run, model: str, ds: str, lam: float) -> dict | None:
    f = run / LAMBDA_EXP / model / ds / f"lam{lam_key(lam)}" / "cell.env"
    if not f.is_file():
        return None
    return dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run name under results/repro/ or a path")
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    args = p.parse_args()
    run = run_dir(args.run)
    table1(run, args.models)
    if (run / LAMBDA_EXP).is_dir():
        lambda_sweep(run, args.models)


if __name__ == "__main__":
    main()
