"""Shared checkpoint and eval output paths (training, eval, orchestration).

Ported from the source repo's ``result_paths.py`` — the single source of truth
for every results directory decision. Kept behavior-identical so existing result
trees and analysis scripts continue to resolve.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from omegaconf import OmegaConf


def results_root(cfg, base: Path | str | None = None) -> Path:
    """``cfg.paths.results_root`` (default ``results``), resolved against ``base`` (cwd).

    Every output directory below is derived from this, so one override
    (``paths.results_root=results/repro/<run>/<experiment>``) relocates a whole run.
    """
    root = Path(str(OmegaConf.select(cfg, "paths.results_root") or "results"))
    return root if root.is_absolute() else Path(base if base is not None else os.getcwd()) / root


def format_lr_for_path(learning_rate: float) -> str:
    """Filesystem-safe learning-rate token, e.g. 5e-6 -> lr_5e-06."""
    lr = float(learning_rate)
    return f"lr_{lr:.0e}".replace("+", "")


def format_ga_for_path(gradient_accumulation_steps: int) -> str:
    return f"ga_{int(gradient_accumulation_steps)}"


def train_variant_subdir(*, use_lora: bool) -> str:
    return "lora" if use_lora else "full"


def dataset_split_tag(dataset_name: str, split: str | None = None) -> str:
    if split:
        return f"{dataset_name}_{split}"
    return dataset_name


def offline_eval_dataset_tag(
    dataset_name: str,
    *,
    split: str | None = None,
    include_split: bool = True,
) -> str:
    """Dataset directory tag for offline eval artifacts.

    Loss evals are tied to a configured dataset split, but generative evals use
    the dataset's own held-out rows and should stay under the canonical dataset
    name so metadata and traces land in the same directory.
    """
    if include_split:
        return dataset_split_tag(dataset_name, split)
    return dataset_name


def finetune_results_root(
    results_dir: Path | str,
    *,
    intervention_method: str | None = None,
    intervention_lambda_scale: float | None = None,
) -> Path:
    root = Path(results_dir)
    if intervention_method is None:
        return root / "finetuning"

    lam = 1.0 if intervention_lambda_scale is None else float(intervention_lambda_scale)
    lam_key = f"{lam:g}".replace(".", "p")
    return (
        root
        / "finetuning_intervention"
        / intervention_method
        / f"lam_{lam_key}"
    )


def resolve_finetune_output_dir(
    results_dir: Path | str,
    *,
    dataset_name: str,
    split: str | None = None,
    model_short_name: str,
    use_lora: bool = False,
    intervention_method: str | None = None,
    intervention_lambda_scale: float | None = None,
) -> Path:
    """Directory where training writes weights (before optional vllm_merged/)."""
    root = finetune_results_root(
        results_dir,
        intervention_method=intervention_method,
        intervention_lambda_scale=intervention_lambda_scale,
    )
    dataset_tag = dataset_split_tag(dataset_name, split)
    variant = train_variant_subdir(use_lora=use_lora)
    return root / dataset_tag / model_short_name / variant


def resolve_vllm_checkpoint_path(checkpoint_dir: Path) -> Path:
    """Path passed to vLLM: merged LoRA folder if present, else checkpoint_dir."""
    merged = checkpoint_dir / "vllm_merged"
    if merged.is_dir():
        return merged
    return checkpoint_dir


def _sanitize_variant_token(value: object) -> str:
    raw = str(value).strip()
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "", raw.replace(" ", ""))
    return raw or "unknown"


def _sanitize_path_component(name: object) -> str:
    raw = str(name).strip()
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = re.sub(r"[^\w\-.()+]", "_", raw).strip("_")
    return raw[:240] if raw else "unnamed"


def finetune_eval_variant(
    *,
    learning_rate: float,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    use_lora: bool = False,
    intervention_method: str | None = None,
    intervention_lambda_scale: float | None = None,
) -> str:
    """Canonical offline-eval variant for a fine-tuned hyperparameter setting."""
    lr_tok = _sanitize_variant_token(format_lr_for_path(learning_rate).removeprefix("lr_"))
    ga_tok = str(int(gradient_accumulation_steps))
    epochs_tok = _sanitize_variant_token(f"{float(num_train_epochs):g}")
    parts = [f"lr{lr_tok}", f"ga{ga_tok}", f"ep{epochs_tok}"]
    if use_lora:
        parts.append("lora")
    if intervention_method is not None:
        parts.append(_sanitize_path_component(intervention_method))
        lam = 1.0 if intervention_lambda_scale is None else float(intervention_lambda_scale)
        parts.append(f"lam{_sanitize_variant_token(f'{lam:g}'.replace('.', 'p'))}")
    return "-".join(parts)


def offline_eval_subdir(variant: str) -> str:
    """Subdir under results/offline_eval/<dataset>/<model>/."""
    return _sanitize_path_component(variant)


def resolve_offline_eval_output_dir(
    results_dir: Path | str,
    *,
    dataset_name: str,
    model_short_name: str,
    variant: str,
    use_lora: bool = False,
) -> Path:
    eval_subdir = offline_eval_subdir(variant)
    return Path(results_dir) / "offline_eval" / dataset_name / model_short_name / eval_subdir


def generative_eval_trace_path(
    results_dir: Path | str,
    *,
    dataset_name: str,
    model_short_name: str,
    variant: str,
    use_lora: bool = False,
) -> Path:
    return (
        resolve_offline_eval_output_dir(
            results_dir,
            dataset_name=dataset_name,
            model_short_name=model_short_name,
            variant=variant,
            use_lora=use_lora,
        )
        / "generative_eval_traces.json"
    )


def parse_hp_tag(hp: str) -> tuple[float, int] | None:
    """Parse lr_*_ga_* directory name back to (lr, ga) for helpers."""
    m = re.fullmatch(r"lr_([^_]+)_ga_(\d+)", hp)
    if not m:
        return None
    lr_s, ga_s = m.group(1), m.group(2)
    lr_s = lr_s.replace("p", ".")
    try:
        lr = float(lr_s)
        return lr, int(ga_s)
    except ValueError:
        return None
