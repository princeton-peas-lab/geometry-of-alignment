"""Tokenizer/model loading, dtype + pad-token resolution, and LoRA config.

Shared by ``training/`` and ``evaluation/`` so the two entrypoints no longer
duplicate this logic (the source ``evaluate.py`` imported it from ``finetune.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_pretrained_path(path: str | os.PathLike) -> Path:
    """Resolve relative paths against the launch cwd.

    Hydra/eval may run from a different cwd; relative model paths must resolve
    against the original launch directory so local model folders are not treated
    as Hub ids (which raises HFValidationError on absolute paths).
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(get_original_cwd()) / p
    return p.resolve()


def resolve_torch_dtype(train_cfg: DictConfig) -> torch.dtype:
    """Dtype the model's weights are loaded (and trained) in."""
    if train_cfg.bf16:
        return torch.bfloat16
    if train_cfg.fp16:
        return torch.float16
    # Explicit: transformers >= 5 maps dtype=None to "auto", i.e. the checkpoint's own bf16.
    return torch.float32


def compute_dtype(train_cfg: DictConfig) -> torch.dtype:
    """Dtype of the forward/backward compute (and of the saved checkpoint)."""
    if train_cfg.bf16:
        return torch.bfloat16
    if train_cfg.fp16:
        return torch.float16
    return torch.float32


def load_tokenizer(model_path_str: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path_str, use_fast=True)
    if tokenizer.pad_token is None:
        # Use EOS as pad to avoid loss on padding (TRL handles ignore_index masking).
        print("No pad token found, using EOS as pad")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_config(train_cfg: DictConfig) -> LoraConfig | None:
    if not train_cfg.use_lora:
        return None
    # Convert OmegaConf containers to plain Python: PEFT serializes its config to
    # JSON at checkpoint time and crashes on ListConfig/DictConfig.
    lora_target_modules = OmegaConf.to_container(train_cfg.lora_target_modules, resolve=True)
    return LoraConfig(
        r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        bias="none",
        target_modules=list(lora_target_modules),
        task_type="CAUSAL_LM",
        use_rslora=train_cfg.use_rslora,
    )


def resolve_model_path(cfg: DictConfig) -> Path:
    model_path = resolve_pretrained_path(cfg.model.model_name_or_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model path is not an existing directory: {model_path}. "
            "If the path is wrong, transformers treats it as a Hub id and "
            "huggingface_hub can raise HFValidationError on absolute paths."
        )
    return model_path


def load_model(
    model_path_str: str,
    torch_dtype: torch.dtype | None,
    *,
    gradient_checkpointing: bool,
):
    """Load the causal LM. dtype is kept out of TrainingArguments (it breaks JSON logging)."""
    model = AutoModelForCausalLM.from_pretrained(model_path_str, torch_dtype=torch_dtype)
    if gradient_checkpointing:
        model.config.use_cache = False
    return model
