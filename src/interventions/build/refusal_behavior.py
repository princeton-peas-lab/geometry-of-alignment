"""Build a refusal-variant dataset JSON via sequential vLLM (policy then WildGuard).

Ported from the source ``build_intervention/get_model_refusal_behavior.py``. Loads
base harmful prompts (dataset.variant=null), generates policy completions, labels
with WildGuard, keeps WildGuard-confirmed refusals, and writes instruction/response
pairs matching the finetune refusal artifacts.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from dataset_utils import load_safety_dataset
from safety.scorers.wildguard import (
    format_input,
    has_guard_label,
    is_guard_refusal,
    summarize_guard_labels,
)
from utils import CHAT_TEMPLATE_KWARGS


def _extract_output_text(output: Any) -> str:
    if getattr(output, "outputs", None):
        first = output.outputs[0]
        text = getattr(first, "text", None)
        if text is not None:
            return str(text).strip()
    return ""


def _load_prompts(cfg: DictConfig) -> list[str]:
    dataset = load_safety_dataset(cfg)
    limit = cfg.runtime.limit
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    return [str(example["prompt"]) for example in dataset]


def _llm_kwargs(cfg: DictConfig, model_path: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": int(cfg.vllm.tensor_parallel_size),
        "gpu_memory_utilization": float(cfg.vllm.gpu_memory_utilization),
    }
    if cfg.vllm.max_model_len is not None:
        kwargs["max_model_len"] = int(cfg.vllm.max_model_len)
    extra = OmegaConf.select(cfg, "vllm.additional_config")
    if extra is not None:
        kwargs["additional_config"] = OmegaConf.to_container(extra, resolve=True)
    return kwargs


def _resolve_output_path(cfg: DictConfig) -> Path:
    raw = cfg.output.get("output_file")
    if raw is not None and str(raw).strip():
        return Path(str(raw))
    slug = str(cfg.output.variant_slug).strip()
    if not slug:
        raise ValueError(
            "Set output.variant_slug for the default output path, or output.output_file explicitly."
        )
    base = Path(str(cfg.dataset.dataset_path))
    return base / f"{slug}_refused_prompts_{cfg.dataset.subset}_{cfg.dataset.split}.json"


def _flush_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def build_refusal_behavior(cfg: DictConfig) -> Path:
    from vllm import LLM, SamplingParams

    prompts = _load_prompts(cfg)
    print(
        f"Loaded {len(prompts)} prompts from refusal split "
        f"{cfg.dataset.subset}/{cfg.dataset.split} (variant={cfg.dataset.get('variant')!r})."
    )

    messages = [[{"role": "user", "content": p}] for p in prompts]
    policy_llm = LLM(**_llm_kwargs(cfg, str(cfg.model.model_name_or_path)))
    policy_outputs = policy_llm.chat(
        messages,
        sampling_params=SamplingParams(
            temperature=float(cfg.policy_sampling.temperature),
            top_p=float(cfg.policy_sampling.top_p),
            max_tokens=int(cfg.policy_sampling.max_tokens),
        ),
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False, **CHAT_TEMPLATE_KWARGS},
    )
    paired = [{"prompt": p, "response": _extract_output_text(o)} for p, o in zip(prompts, policy_outputs)]
    policy_llm.sleep(level=2, mode="abort")
    policy_llm = None
    _flush_cuda()

    guard_inputs = [format_input(row["prompt"], row["response"]) for row in paired]
    guard_llm = LLM(**_llm_kwargs(cfg, str(cfg.guard_model.model_name_or_path)))
    guard_outputs = guard_llm.generate(
        guard_inputs,
        sampling_params=SamplingParams(
            temperature=float(cfg.guard_sampling.temperature),
            top_p=float(cfg.guard_sampling.top_p),
            max_tokens=int(cfg.guard_sampling.max_tokens),
        ),
        use_tqdm=True,
    )
    labeled = []
    for row, output in zip(paired, guard_outputs):
        gl = _extract_output_text(output)
        labeled.append(
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "guard_label": gl,
                "has_guard_label": has_guard_label({"guard_label": gl}),
            }
        )
    guard_llm.sleep(level=2, mode="abort")
    guard_llm = None
    _flush_cuda()

    summarize_guard_labels(labeled)
    kept = [
        {"instruction": row["prompt"], "response": row["response"]}
        for row in labeled
        if is_guard_refusal(str(row.get("guard_label", "")))
    ]
    out_path = _resolve_output_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: prompts={len(prompts)}, labeled={len(labeled)}, kept_refusals={len(kept)}")
    print(f"Wrote {len(kept)} rows to {out_path}")
    return out_path
