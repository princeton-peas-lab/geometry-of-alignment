"""vLLM generative-accuracy evaluation (parse + reward via dataset adapters).

Ported from the source ``generative_eval.py``; vLLM lifecycle helpers now live in
``evaluation/vllm_runtime.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dataset_utils import parse_output_text, reward_fn
from utils import CHAT_TEMPLATE_KWARGS


def _trace_json_compatible(obj: object) -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _trace_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_trace_json_compatible(v) for v in obj]
    return str(obj)


def _write_eval_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _extract_chat_response_text(output: Any) -> str:
    if getattr(output, "outputs", None):
        first = output.outputs[0]
        text = getattr(first, "text", None)
        if text is not None:
            return str(text).strip()
    return ""


def evaluate_generative_accuracy(
    cfg: DictConfig,
    llm: Any,
    reward_rows: list[tuple[list[dict], dict]],
    *,
    eval_trace_json_path: Path | str | os.PathLike[str] | None = None,
    trace_extra_metadata: dict[str, Any] | None = None,
) -> tuple[float, int]:
    try:
        from vllm import SamplingParams
    except ImportError as e:
        raise ImportError(
            "Generative evaluation requires vLLM. Install the `vllm` extra "
            "(pip install -e '.[vllm]')."
        ) from e

    if not reward_rows:
        return 0.0, 0

    trace_out = Path(os.fspath(eval_trace_json_path)) if eval_trace_json_path else None
    trace_items: list[dict[str, Any]] | None = [] if trace_out is not None else None

    gen_cfg = cfg.evaluate.generative
    sampling = SamplingParams(
        temperature=float(gen_cfg.temperature),
        top_p=float(gen_cfg.top_p),
        max_tokens=int(gen_cfg.max_new_tokens),
        seed=cfg.seed,
    )

    correct = 0
    total = len(reward_rows)
    chunk = max(1, int(gen_cfg.vllm_chunk_size))
    for start in range(0, total, chunk):
        batch = reward_rows[start : start + chunk]
        messages_list = [list(msgs) for msgs, _ in batch]
        outputs = llm.chat(
            messages_list,
            sampling_params=sampling,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False, **CHAT_TEMPLATE_KWARGS},
        )
        for i, ((msgs, dp), out) in enumerate(zip(batch, outputs)):
            global_idx = start + i
            raw = _extract_chat_response_text(out)
            parsed = parse_output_text(raw, cfg)
            ok = parsed is not None and reward_fn(parsed, dp, cfg)
            if ok:
                correct += 1
            if trace_items is not None:
                trace_items.append(
                    {
                        "index": global_idx,
                        "input_messages": _trace_json_compatible(list(msgs)),
                        "reward_datapoint": _trace_json_compatible(dp),
                        "response_text": raw,
                        "parsed": _trace_json_compatible(parsed),
                        "correct": ok,
                    }
                )

    acc = correct / total if total else 0.0

    if trace_out is not None and trace_items is not None:
        payload: dict[str, Any] = {
            **(trace_extra_metadata or {}),
            "dataset_name": str(cfg.dataset.dataset_name),
            "model_short_name": str(cfg.model.short_name),
            "variant": str(
                OmegaConf.select(cfg, "evaluate.generative.variant") or "baseline"
            ),
            "generative_accuracy": acc,
            "num_examples": total,
            "num_correct": correct,
            "items": trace_items,
        }
        _write_eval_trace_json(trace_out, payload)

    return acc, total
