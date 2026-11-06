"""vLLM lifecycle + LoRA-merge helpers for generative evaluation.

Combines the vLLM runtime helpers from the source ``generative_eval.py`` with the
LoRA-merge-for-vLLM helpers that lived in ``finetune.py``.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_vllm_multiprocessing() -> None:
    """Use spawn for vLLM workers unless the user already configured it."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def flush_cuda_allocations() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def destroy_vllm_llm(llm: Any) -> None:
    llm.sleep(level=2, mode="abort")
    del llm
    flush_cuda_allocations()


def make_vllm_llm(cfg: DictConfig, model_path: Path | str | os.PathLike[str]) -> Any:
    configure_vllm_multiprocessing()
    try:
        from vllm import LLM
    except ImportError as e:
        raise ImportError(
            "Generative evaluation requires vLLM. Install the `vllm` extra "
            "(pip install -e '.[vllm]')."
        ) from e

    vllm_cfg = cfg.evaluate.generative.vllm
    llm_kwargs: dict[str, Any] = {
        "model": os.fspath(model_path),
        "tensor_parallel_size": int(vllm_cfg.tensor_parallel_size),
        "gpu_memory_utilization": float(vllm_cfg.gpu_memory_utilization),
    }
    if vllm_cfg.max_model_len is not None:
        llm_kwargs["max_model_len"] = int(vllm_cfg.max_model_len)
    extra = OmegaConf.select(vllm_cfg, "additional_config")
    if extra is not None:
        llm_kwargs["additional_config"] = OmegaConf.to_container(extra, resolve=True)

    return LLM(**llm_kwargs)


def merge_dtype_from_train_cfg(train_cfg) -> torch.dtype:
    if train_cfg.bf16:
        return torch.bfloat16
    if train_cfg.fp16:
        return torch.float16
    return torch.float32


def merge_lora_for_vllm(
    base_model_path: str,
    adapter_path: str,
    merged_path: Path,
    torch_dtype: torch.dtype,
) -> None:
    """Load base + PEFT adapter on CPU, merge, save a full model folder for vLLM."""
    merged_path.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, device_map="cpu")
    model = model.merge_and_unload()
    model.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.save_pretrained(merged_path)
    del model, base
    flush_cuda_allocations()


def vllm_model_dir_for_generative_eval(
    output_dir: str,
    base_model_path: str,
    train_cfg,
) -> Path:
    out = Path(output_dir)
    if not train_cfg.use_lora:
        return out.resolve()
    merged = out / "vllm_merged"
    print(
        f"Merging LoRA adapter (CPU) from {out} + base {base_model_path} -> {merged} for vLLM..."
    )
    merge_lora_for_vllm(
        base_model_path, str(out), merged, merge_dtype_from_train_cfg(train_cfg)
    )
    return merged.resolve()
