"""Post-train / offline eval metadata writer and trace JSON helpers.

Extracted from the source ``finetune.py`` (where these were module-private and
re-imported by ``evaluate.py``). Behavior-identical: the metadata file format is
``post_train_eval_metadata_v1`` so existing analysis scripts keep parsing it.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def post_train_status_metrics(
    *,
    metric: object,
    num_examples: int,
    status_code: int,
    **extra_metrics: object,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "eval_metric": metric,
        "eval_num_examples": num_examples,
        "eval_status_code": status_code,
    }
    metrics.update(extra_metrics)
    return metrics


def metadata_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def trainer_epoch_eval_metrics(log_history: list[dict]) -> list[dict[str, object]]:
    eval_metrics = []
    for entry in log_history:
        if "eval_loss" not in entry:
            continue
        metrics = {
            key: value
            for key, value in entry.items()
            if key.startswith("eval_") or key in ("epoch", "step")
        }
        eval_loss = metrics.get("eval_loss")
        if isinstance(eval_loss, float) and math.isfinite(eval_loss):
            metrics["eval_perplexity"] = math.exp(eval_loss)
        eval_metrics.append(metrics)
    return eval_metrics


def cfg_summary_lines(cfg: DictConfig, max_depth: int = 4) -> list[str]:
    """Flat key=value lines for config (skips hydra.*). Omits deeply nested sequences."""

    def walk(obj: object, prefix: str, depth: int) -> None:
        if depth <= 0:
            return
        if not isinstance(obj, dict):
            return
        for k in sorted(obj.keys(), key=str):
            key_name = str(k)
            if key_name.startswith("hydra"):
                continue
            v = obj[k]
            path = f"{prefix}.{key_name}" if prefix else key_name
            if isinstance(v, dict):
                if depth > 1:
                    walk(v, path, depth - 1)
                else:
                    out.append(f"{path}=<dict>")
            elif isinstance(v, (list, tuple)):
                if v and any(isinstance(x, (dict, list)) for x in v):
                    out.append(f"{path}=<omitted_nested_sequence>")
                else:
                    out.append(f"{path}={','.join(str(x) for x in v)}")
            elif isinstance(v, bool):
                out.append(f"{path}={str(v).lower()}")
            elif isinstance(v, float):
                out.append(f"{path}={v:.12g}")
            elif v is None:
                out.append(f"{path}=null")
            else:
                out.append(f"{path}={v}")

    out: list[str] = []
    data = OmegaConf.to_container(cfg, resolve=True)
    walk(data, "", max_depth)
    return out


def write_post_train_eval_metadata(
    path: Path,
    cfg: DictConfig,
    metrics: dict | None,
    eval_num_examples: int,
    completion_only_loss: bool,
    skip_reason: str | None,
    generative_accuracy: float | None = None,
    generative_num_examples: int = 0,
    trainer_epoch_eval_metrics: list[dict[str, object]] | None = None,
    extra_cfg_summary_lines: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "format=post_train_eval_metadata_v1",
        f"written_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"completion_only_loss={str(completion_only_loss).lower()}",
        f"eval_split_num_examples={eval_num_examples}",
    ]
    if skip_reason:
        lines.append("eval_status=skipped")
        lines.append(f"eval_skip_reason={skip_reason}")
    else:
        lines.append("eval_status=ok")

    lines.append("")
    lines.append("[loss_metrics]")
    if metrics:
        for name in sorted(metrics.keys()):
            lines.append(f"{name}={metadata_value(metrics[name])}")
        if "eval_loss" in metrics:
            el = metrics["eval_loss"]
            if isinstance(el, float) and math.isfinite(el):
                lines.append(f"eval_perplexity={math.exp(el):.12g}")
    else:
        lines.append("eval_loss=null")

    lines.append("")
    lines.append("[generative_eval]")
    if generative_num_examples > 0 and generative_accuracy is not None:
        lines.append(f"eval_generative_accuracy={generative_accuracy:.12g}")
        lines.append(f"eval_generative_num_examples={generative_num_examples}")
    else:
        lines.append("eval_generative_accuracy=null")
        lines.append("eval_generative_num_examples=0")

    lines.append("")
    lines.append("[trainer_epoch_eval_metrics]")
    epoch_metrics_list = trainer_epoch_eval_metrics or []
    lines.append(f"epoch_eval_count={len(epoch_metrics_list)}")
    for idx, epoch_metrics in enumerate(epoch_metrics_list, start=1):
        for name in sorted(epoch_metrics.keys()):
            value = metadata_value(epoch_metrics[name])
            lines.append(f"epoch_eval.{idx}.{name}={value}")

    lines.append("")
    lines.append("[config_summary]")
    lines.extend(cfg_summary_lines(cfg))
    if extra_cfg_summary_lines:
        lines.extend(extra_cfg_summary_lines)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_dir_has_content(path: Path) -> bool:
    """True if path is a directory with at least one child (file or subdirectory)."""
    if not path.is_dir():
        return False
    return any(path.iterdir())
