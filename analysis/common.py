"""Shared readers for a reproduction run directory (results/repro/<run>/).

Layout (written by scripts/reproduce/):
    <run>/artifacts/{refusal,repr_fim_subspace,projected_fisher,projected_fisher_control}/
    <run>/baseline/<model>/{refusal_test,strongreject}/   un-finetuned responses + scores
    <run>/baseline/<model>/eval_loss/<dataset>/            pretrained held-out loss
    <run>/<experiment>/<model>/<dataset>/<tag>/            one finetuning cell
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS: dict[str, str] = {
    "llama32-3b-instruct": "LLaMA-3.2-3B-Instruct",
    "qwen3-4b-instruct": "Qwen3-4B-Instruct",
    "qwen35-9b-instruct": "Qwen3.5-9B-Instruct",
}
DATASETS: dict[str, tuple[str, str]] = {  # key -> (label, group)
    "alpaca": ("Alpaca", "benign"),
    "samsum": ("SamSum", "benign"),
    "gsm8k": ("GSM8K", "benign"),
    "risky_financial_advice": ("Risky Financial Advice", "EM"),
    "bad_medical_advice": ("Bad Medical Advice", "EM"),
    "pure_bad": ("Pure Bad", "malicious"),
}
LAMBDAS = [0.0, -0.25, -0.5, -0.75, -1.0]
SWEEP_LRS = ["1e-4", "5e-5", "1e-5"]
DEGENERATE_WORDS = 400   # mean response length flagged as collapse in the LR-sweep table
# A response is repetitive when under 30% of its 4-grams are distinct; an LR collapses when more
# than 25% of any dataset's responses are repetitive.
REPETITIVE_DISTINCT_4GRAMS = 0.3
COLLAPSE_FRACTION = 0.25


def is_repetitive(text: str) -> bool:
    w = str(text).split()
    if len(w) < 40:
        return False
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) < REPETITIVE_DISTINCT_4GRAMS


def lam_key(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def run_dir(arg: str) -> Path:
    """Accept a run name (``2026-10-13``) or a path."""
    p = Path(arg)
    return p if p.exists() else REPO / "results" / "repro" / arg


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def safety(d: Path) -> dict | None:
    """Scores for one directory holding responses / strongreject_scores / refusal summary."""
    rows = _json(d / "strongreject_scores.json")
    if not rows:
        return None
    sr = [float(r["strongreject_score"]) for r in rows]
    summary = _json(d / "refusal_strmatch_summary.json") or {}
    guard = _json(d / "guard_labels.json")
    return {
        "sr": sum(sr) / len(sr),
        "sr_zero": sum(s == 0.0 for s in sr) / len(sr),
        "refusal_xstest": summary.get("refusal_rate_xstest", summary.get("refusal_rate")),
        "refusal_combined": summary.get("refusal_rate_combined"),
        # WildGuard's "Response refusal: yes" -- the paper's refusal rate.
        "refusal_wildguard": (sum("Response refusal: yes" in str(r.get("guard_label", "")) for r in guard) / len(guard)
                              if guard else None),
        "words": sum(len(str(r["response"]).split()) for r in rows) / len(rows),
        "repetitive": sum(is_repetitive(r["response"]) for r in rows) / len(rows),
        "n": len(rows),
    }


def eval_loss(meta: Path) -> float | None:
    """``eval_loss`` from a post_train_eval_metadata.txt ``[loss_metrics]`` section."""
    if not meta.is_file():
        return None
    in_loss = False
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_loss = s == "[loss_metrics]"
        elif in_loss and s.startswith("eval_loss="):
            try:
                v = float(s.split("=", 1)[1])
            except ValueError:
                return None
            return v if math.isfinite(v) else None
    return None


def epoch_eval_loss(meta: Path) -> float | None:
    """The trainer's last end-of-epoch ``eval_loss``, i.e. with any intervention hooks attached."""
    if not meta.is_file():
        return None
    best: tuple[int, float] | None = None
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, val = line.strip().partition("=")
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "epoch_eval" and parts[2] == "eval_loss":
            try:
                n, v = int(parts[1]), float(val)
            except ValueError:
                continue
            if math.isfinite(v) and (best is None or n > best[0]):
                best = (n, v)
    return best[1] if best else None


def cell_dir(run: Path, exp: str, model: str, dataset: str, tag: str) -> Path:
    return run / exp / model / dataset / tag


def cell(run: Path, exp: str, model: str, dataset: str, tag: str) -> dict | None:
    d = cell_dir(run, exp, model, dataset, tag)
    out = safety(d) or {}
    out["eval_loss"] = eval_loss(d / "post_train_eval_metadata.txt")
    out["eval_loss_hooked"] = epoch_eval_loss(d / "post_train_eval_metadata.txt")
    out["peak_rayleigh"] = peak_rayleigh(d / "fisher_overlap_monitor.txt")
    return out if any(v is not None for v in out.values()) else None


def peak_rayleigh(path: Path) -> float | None:
    if not path.is_file():
        return None
    best = -math.inf
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        row = dict(kv.split("=", 1) for kv in line.strip().split("\t") if "=" in kv)
        v = float(row.get("rayleigh", "nan"))
        if math.isfinite(v):
            best = max(best, v)
    return best if best > -math.inf else None


def null_stats(fisher_path: Path) -> tuple[float, float] | None:
    """(mean, std) of u^T F u for u uniform on the sphere: the random-direction reference."""
    if not fisher_path.is_file():
        return None
    import torch

    F = torch.load(fisher_path, map_location="cpu").double()
    d = F.shape[0]
    lam = torch.linalg.eigvalsh(F).clamp_min(0)
    tr = lam.sum().item()
    mean = tr / d
    pr = tr**2 / (lam**2).sum().item()
    return mean, mean * math.sqrt(2 * (d / pr - 1) / (d + 2))


def fisher_path(run: Path, model: str) -> Path:
    return run / "artifacts" / "projected_fisher" / MODELS[model] / "refusal_train" / "fisher.pt"


def lr_tags(run: Path, exp: str, model: str) -> list[str]:
    """``lr<lr>`` tags present for a model in an experiment (one per learning rate)."""
    tags = {p.name for p in (run / exp / model).glob("*/lr*") if p.is_dir()}
    return sorted(tags)


def sweep_lrs(run: Path, exp: str, model: str) -> list[str]:
    """Learning rates with an ``lr<lr>`` cell for ``model`` in ``exp`` (extra-seed cells skipped), largest first."""
    tags = {p.name.removeprefix("lr") for p in (run / exp / model).glob("*/lr*") if p.is_dir() and "_s" not in p.name}
    return sorted(tags, key=float, reverse=True)


def collapsed_lrs(run: Path, exp: str, model: str) -> set[str]:
    """LRs at which any dataset's generations collapse (repetitive fraction > COLLAPSE_FRACTION)."""
    out = set()
    for lr in sweep_lrs(run, exp, model):
        for ds in DATASETS:
            c = cell(run, exp, model, ds, f"lr{lr}")
            if c and (c.get("repetitive") or 0) > COLLAPSE_FRACTION:
                out.add(lr)
    return out


def lr_vote(run: Path, model: str, exp: str = "lr_sweep",
            exclude_collapsed: bool = False) -> tuple[dict[str, str], str | None]:
    """({dataset: LR with the lowest held-out loss}, majority LR or None on a tie / no data)."""
    banned = collapsed_lrs(run, exp, model) if exclude_collapsed else set()
    best: dict[str, str] = {}
    for ds in DATASETS:
        scored = [(lr, c["eval_loss"]) for lr in sweep_lrs(run, exp, model) if lr not in banned
                  and (c := cell(run, exp, model, ds, f"lr{lr}")) and c.get("eval_loss") is not None]
        if scored:
            best[ds] = min(scored, key=lambda t: t[1])[0]
    votes: dict[str, int] = {}
    for lr in best.values():
        votes[lr] = votes.get(lr, 0) + 1
    if not votes:
        return best, None
    top = max(votes.values())
    winners = [lr for lr, n in votes.items() if n == top]
    return best, winners[0] if len(winners) == 1 else None


def fmt(v, spec: str, dash: str = "--") -> str:
    return dash if v is None else spec.format(v)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sx * sy) if sx and sy else None
