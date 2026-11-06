"""Safety response generation — unified API and offline-vLLM backends.

Folds the source ``safety_eval/generate_response.py`` (OpenAI-compatible API) and
``safety_eval/batch_generate_response/run_offline_chat.py`` (in-process vLLM) into
one entrypoint selected by ``cfg.backend`` (api | offline_vllm).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dataset_utils import load_safety_dataset
from safety.io import write_json
from utils import CHAT_TEMPLATE_KWARGS


def load_prompts(cfg: DictConfig) -> list[str]:
    dataset = load_safety_dataset(cfg)
    limit = cfg.runtime.limit
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    print(f"Loaded {len(dataset)} prompts")
    return [str(example["prompt"]) for example in dataset]


def _extract_response_text(output: Any) -> str:
    if getattr(output, "outputs", None):
        first = output.outputs[0]
        text = getattr(first, "text", None)
        if text is not None:
            return str(text).strip()
    return ""


def _generate_api(cfg: DictConfig, prompts: list[str]) -> list[str]:
    from openai import OpenAI

    client = OpenAI(
        base_url=str(cfg.api.api_base),
        api_key=str(cfg.api.api_key),
        timeout=float(cfg.runtime.timeout),
    )
    responses: list[str] = []
    for prompt in prompts:
        completion = client.chat.completions.create(
            model=str(cfg.model.model_name_or_path),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(cfg.sampling.temperature),
            top_p=float(cfg.sampling.top_p),
            max_tokens=int(cfg.sampling.max_tokens),
            extra_body={"chat_template_kwargs": {"enable_thinking": False, **CHAT_TEMPLATE_KWARGS}},
        )
        responses.append((completion.choices[0].message.content or "").strip())
    return responses


def _generate_offline_vllm(cfg: DictConfig, prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": str(cfg.model.model_name_or_path),
        "tensor_parallel_size": int(cfg.vllm.tensor_parallel_size),
        "gpu_memory_utilization": float(cfg.vllm.gpu_memory_utilization),
    }
    if cfg.vllm.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(cfg.vllm.max_model_len)
    extra = OmegaConf.select(cfg, "vllm.additional_config")
    if extra is not None:
        llm_kwargs["additional_config"] = OmegaConf.to_container(extra, resolve=True)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=float(cfg.sampling.temperature),
        top_p=float(cfg.sampling.top_p),
        max_tokens=int(cfg.sampling.max_tokens),
    )
    messages = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(
        messages, sampling_params=sampling_params, use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False, **CHAT_TEMPLATE_KWARGS},
    )
    return [_extract_response_text(o) for o in outputs]


def generate_responses(cfg: DictConfig) -> Path:
    backend = str(cfg.backend).lower()
    prompts = load_prompts(cfg)
    if backend == "api":
        responses = _generate_api(cfg, prompts)
    elif backend == "offline_vllm":
        responses = _generate_offline_vllm(cfg, prompts)
    else:
        raise ValueError(f"Unsupported backend {backend!r}. Expected 'api' or 'offline_vllm'.")

    rows = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
    output_file = Path(str(cfg.io.output_file))
    write_json(output_file, rows)
    print(f"Wrote {len(rows)} responses to {output_file}")
    return output_file
