#!/usr/bin/env python3
"""Effective-rank statistics of the projected refusal FIM, against an isotropic null.

    python analysis/lowrank_projected_fisher.py <run>/artifacts/projected_fisher/<MODEL_SHORT>/refusal_train/fisher.pt ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch

NULL_DRAWS = 5
TOP_KS = (1, 5, 10, 20, 50, 100)
TRACE_FRACTIONS = (0.5, 0.9, 0.95, 0.99)


def load_fisher(path: Path) -> Tuple[torch.Tensor, Dict]:
    obj = torch.load(path, map_location="cpu")
    fisher = obj["fisher"] if isinstance(obj, dict) and "fisher" in obj else obj
    if not torch.is_tensor(fisher) and isinstance(obj, dict):
        fisher = next(v for v in obj.values() if torch.is_tensor(v) and v.ndim == 2)
    meta_path = path.parent / "fisher_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return fisher.double(), meta


def rank_stats(fisher: torch.Tensor, num_samples: int) -> Dict:
    dim = fisher.shape[0]
    eigenvalues = torch.linalg.eigvalsh(fisher).flip(0)
    positive = eigenvalues.clamp(min=0)
    trace = positive.sum().item()
    cumulative = torch.cumsum(positive, 0) / trace
    probs = positive / trace
    nonzero = probs[probs > 0]
    stats = {
        "dim": dim,
        "num_samples": num_samples,
        "trace": trace,
        "lam_max": eigenvalues[0].item(),
        "top_k_share": {k: 100 * cumulative[k - 1].item() for k in TOP_KS if k <= dim},
        "k_for_fraction": {
            q: int((cumulative < q).sum().item()) + 1 for q in TRACE_FRACTIONS
        },
        "participation_ratio": (trace**2) / (positive**2).sum().item(),
        "stable_rank": trace / eigenvalues[0].item(),
        "entropy_effective_rank": math.exp(-(nonzero * nonzero.log()).sum().item()),
        "numerical_rank": int((eigenvalues > 1e-6 * eigenvalues[0]).sum()),
    }
    if num_samples and num_samples < dim:
        stats["tail_mass_beyond_n"] = (positive[num_samples:].sum() / trace).item()
    return stats


def isotropic_null_participation_ratio(
    num_samples: int, dim: int, draws: int = NULL_DRAWS
) -> Tuple[float, float, float]:
    """Participation ratio of (1/n) G^T G for n i.i.d. isotropic gradients in R^dim."""
    torch.manual_seed(0)
    values = []
    for _ in range(draws):
        grads = torch.randn(num_samples, dim, dtype=torch.float64)
        eigenvalues = torch.linalg.eigvalsh((grads.T @ grads) / num_samples).clamp(min=0)
        values.append(((eigenvalues.sum() ** 2) / (eigenvalues**2).sum()).item())
    return sum(values) / len(values), min(values), max(values)


def report(path: Path, null: bool = True) -> None:
    fisher, meta = load_fisher(path)
    num_samples = int(meta.get("num_samples", 0))
    stats = rank_stats(fisher, num_samples)
    label = meta.get("model_short", str(path))

    print(f"\n=== {label}  (d={stats['dim']}, n={stats['num_samples']}) ===")
    print(f"trace(F)      = {stats['trace']:.6g}   lam_max = {stats['lam_max']:.6g}")
    for k, share in stats["top_k_share"].items():
        print(f"  top-{k:<4d} = {share:6.2f}% of trace")
    ks = " / ".join(str(stats["k_for_fraction"][q]) for q in TRACE_FRACTIONS)
    print(f"  k for 50/90/95/99% = {ks}")
    print(f"participation ratio = {stats['participation_ratio']:.2f}")
    print(f"stable rank         = {stats['stable_rank']:.2f}")
    print(f"entropy eff rank    = {stats['entropy_effective_rank']:.2f}")
    print(f"numerical rank      = {stats['numerical_rank']}  (n = {stats['num_samples']})")
    if "tail_mass_beyond_n" in stats:
        print(f"mass beyond index n = {stats['tail_mass_beyond_n']:.3e} of trace")
    if null and num_samples:
        mean, lo, hi = isotropic_null_participation_ratio(num_samples, stats["dim"])
        print(f"isotropic null PR (n={num_samples}, d={stats['dim']}) = {mean:.1f}  [{lo:.1f}, {hi:.1f}]")
        print(f"  --> observed PR is {mean / stats['participation_ratio']:.1f}x below the null")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fisher", nargs="+", type=Path, help="path(s) to fisher.pt")
    parser.add_argument(
        "--no-null", action="store_true", help="skip the isotropic-null control (it is the slow part)"
    )
    args = parser.parse_args()
    for path in args.fisher:
        report(path, null=not args.no_null)


if __name__ == "__main__":
    main()
