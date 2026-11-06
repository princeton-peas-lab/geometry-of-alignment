#!/usr/bin/env python3
"""Effective-rank statistics of the per-layer representation FIM (full spectrum, before truncation).

    python analysis/lowrank_repr_fim.py <run>/artifacts/repr_fim_subspace/<MODEL_SHORT>/refusal_train ...
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

TOP_K = 10


def module_stats(eigenvalues: torch.Tensor) -> Optional[Dict]:
    eigenvalues = eigenvalues.double().clamp(min=0)
    dim = eigenvalues.numel()
    trace = eigenvalues.sum().item()
    if trace <= 0:
        return None
    probs = eigenvalues / trace
    nonzero = probs[probs > 0]
    cumulative = torch.cumsum(eigenvalues, 0) / trace
    return {
        "dim": dim,
        "participation_ratio": (trace**2) / (eigenvalues**2).sum().item(),
        "stable_rank": trace / eigenvalues[0].item(),
        "entropy_effective_rank": math.exp(-(nonzero * nonzero.log()).sum().item()),
        "lam1_share": 100 * cumulative[0].item(),
        "top_k_share": 100 * cumulative[min(TOP_K, dim) - 1].item(),
        "k90": int((cumulative < 0.90).sum().item()) + 1,
        "k99": int((cumulative < 0.99).sum().item()) + 1,
        "numerical_rank": int((eigenvalues > 1e-6 * eigenvalues[0]).sum()),
    }


def collect(root: Path) -> List[Dict]:
    rows = []
    for path in sorted(root.glob("layer_*/**/eigenvalues.pt")):
        meta = json.loads((path.parent / "metadata.json").read_text())
        stats = module_stats(torch.load(path, map_location="cpu"))
        if stats is None:
            continue
        stats["layer"] = meta["layer_idx"]
        stats["name"] = meta["target_name"]
        rows.append(stats)
    return rows


def report(root: Path) -> None:
    summary = json.loads((root / "summary.json").read_text())
    rows = collect(root)
    if not rows:
        print(f"No eigenvalues.pt under {root}")
        return
    num_samples = max(summary["module_sample_counts"].values())

    print(
        f"\n=== {summary['model_short']}  ({len(rows)} modules, "
        f"n={num_samples}, top_k stored = {summary['top_k_eigenvectors']}) ==="
    )
    header = f"{'module':<18}{'d_in':>6}{'PR':>9}{'stable':>8}{'entropy':>9}{'lam1%':>8}{'top%':>8}{'k90':>7}{'k99':>7}"
    print(header)

    by_name = defaultdict(list)
    for row in rows:
        by_name[row["name"]].append(row)

    def mean(group, key):
        return sum(r[key] for r in group) / len(group)

    for name in summary["target_modules"]:
        group = by_name.get(name, [])
        if not group:
            continue
        print(
            f"{name:<18}{group[0]['dim']:>6}{mean(group, 'participation_ratio'):>9.2f}"
            f"{mean(group, 'stable_rank'):>8.2f}{mean(group, 'entropy_effective_rank'):>9.2f}"
            f"{mean(group, 'lam1_share'):>8.1f}{mean(group, 'top_k_share'):>8.1f}"
            f"{mean(group, 'k90'):>7.0f}{mean(group, 'k99'):>7.0f}"
        )
    print(
        f"{'ALL (mean)':<18}{'-':>6}{mean(rows, 'participation_ratio'):>9.2f}"
        f"{mean(rows, 'stable_rank'):>8.2f}{mean(rows, 'entropy_effective_rank'):>9.2f}"
        f"{mean(rows, 'lam1_share'):>8.1f}{mean(rows, 'top_k_share'):>8.1f}"
        f"{mean(rows, 'k90'):>7.0f}{mean(rows, 'k99'):>7.0f}"
    )

    count = len(rows)
    shares = sorted(r["top_k_share"] for r in rows)
    ratios = sorted(r["participation_ratio"] for r in rows)
    ranks = sorted(r["numerical_rank"] for r in rows)
    print(f"  top-{TOP_K} trace share: min {shares[0]:.1f}%  median {shares[count // 2]:.1f}%  max {shares[-1]:.1f}%")
    print(f"  participation ratio: min {ratios[0]:.2f}  median {ratios[count // 2]:.2f}  max {ratios[-1]:.2f}")
    print(f"  numerical rank: min {ranks[0]}  median {ranks[count // 2]}  max {ranks[-1]}")
    fractions = sorted(r["numerical_rank"] / r["dim"] for r in rows)
    print(f"    median numerical rank / d_in = {fractions[count // 2]:.2f}")
    print(f"    modules with numerical rank > n: {sum(1 for r in rows if r['numerical_rank'] > num_samples)} / {count}")
    worst = min(rows, key=lambda r: r["top_k_share"])
    print(
        f"  worst-covered module: layer {worst['layer']} {worst['name']} -> "
        f"top-{TOP_K} = {worst['top_k_share']:.1f}%, PR = {worst['participation_ratio']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root", nargs="+", type=Path, help="repr_fim_subspace/<model>/<dataset>_<split> directory"
    )
    args = parser.parse_args()
    for root in args.root:
        report(root)


if __name__ == "__main__":
    main()
