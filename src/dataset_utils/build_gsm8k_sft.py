"""Build the GSM8K SFT set: a teacher model's answers to the GSM8K train questions.

Every GSM8K finetune trains on ``dataset.sft_jsonl`` (default
``data/gsm8k/llama31-70b-instruct.jsonl``) rather than GSM8K's reference solutions: for each
of the 7,473 train questions the teacher (Llama-3.1-70B-Instruct) answers greedily, the final
``#### <answer>`` is parsed and checked against the reference, and only rows with
``correct: true`` are used for training (``GSM8KDataset.load_train_eval``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from utils import CHAT_TEMPLATE_KWARGS

from .gsm8k import GSM8KDataset
from .gsm8k_utils import (
    extract_gsm8k_answer,
    format_gsm8k_sft_user_content,
    gsm8k_reward,
    parse_gsm8k_output_text,
)


def gsm8k_sft_row(idx: int, question: str, answer: str, output_text: str, finish_reason: str) -> dict[str, Any]:
    """One output row; ``correct`` decides whether it is trained on."""
    targets = [extract_gsm8k_answer(answer)]
    pred = parse_gsm8k_output_text(output_text)
    return {
        "messages": [{"role": "user", "content": format_gsm8k_sft_user_content(question)}],
        "targets": targets,
        "question": question,
        "answer": answer,
        "datapoint_idx": idx,
        "output_text": output_text,
        "pred_text": pred,
        "correct": gsm8k_reward(pred, targets),
        "finish_reason": finish_reason,
    }


def build_gsm8k_sft(cfg: DictConfig) -> Path:
    from vllm import LLM, SamplingParams

    out_path = Path(str(cfg.io.output_file))
    if out_path.exists() and not bool(cfg.overwrite):
        raise FileExistsError(f"{out_path} exists; pass overwrite=true or a different io.output_file")
    split = GSM8KDataset._load_hf_split(cfg, "train")
    examples = split.to_list() if hasattr(split, "to_list") else [split[i] for i in range(len(split))]
    if cfg.runtime.limit is not None:
        examples = examples[: int(cfg.runtime.limit)]

    llm_kwargs: dict[str, Any] = {
        "model": str(cfg.model.model_name_or_path),
        "tensor_parallel_size": int(cfg.vllm.tensor_parallel_size),
        "gpu_memory_utilization": float(cfg.vllm.gpu_memory_utilization),
    }
    if cfg.vllm.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(cfg.vllm.max_model_len)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=float(cfg.sampling.temperature),
        top_p=float(cfg.sampling.top_p),
        max_tokens=int(cfg.sampling.max_tokens),
        seed=int(cfg.seed),
    )
    messages = [[{"role": "user", "content": format_gsm8k_sft_user_content(ex["question"])}] for ex in examples]
    print(f"Generating {len(messages)} GSM8K train answers with {cfg.model.model_name_or_path}")
    outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=True,
                       chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (ex, out) in enumerate(zip(examples, outputs, strict=True)):
            first = out.outputs[0]
            row = gsm8k_sft_row(idx, ex["question"], ex["answer"], first.text.strip(), str(first.finish_reason))
            n_correct += row["correct"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} rows ({n_correct} correct, used for training) to {out_path}")
    print("config:\n" + OmegaConf.to_yaml(cfg))
    return out_path
