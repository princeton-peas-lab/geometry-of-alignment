"""Shared machinery for the FIM subspace builders.

Extracted from the source ``build_intervention/repr_fim_subspace.py`` so the
gradient builder no longer ``sys.path``-hacks into the representation builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from utils import prompt_completion_ids as prompt_completion_ids_fn

DEFAULT_TARGET_MODULES: List[str] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def prepare_dataset(dataset, tokenizer, cfg):
    think_mode = str(OmegaConf.select(cfg, "think_mode") or "auto")

    def tokenize_fn(example, tokenizer):
        output = {}
        prompt_ids, prompt_completion_ids = prompt_completion_ids_fn(
            tokenizer, example["prompt"], example["completion"], think_mode=think_mode
        )
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        output["input_ids"] = prompt_completion_ids
        output["completion_mask"] = completion_mask
        return output

    return dataset.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer})


def get_completion_only_collate_fn(tokenizer, max_length: int | None = None):
    """Right-pad and build labels so loss is computed only on completion tokens."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        input_id_seqs, comp_masks = [], []
        for ex in batch:
            ids = ex["input_ids"]
            cm = ex["completion_mask"]
            if max_length is not None:
                ids = ids[:max_length]
                cm = cm[:max_length]
            input_id_seqs.append(ids)
            comp_masks.append(cm)

        bs = len(input_id_seqs)
        max_len = max(len(x) for x in input_id_seqs) if bs > 0 else 0
        input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        completion_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, (ids, cm) in enumerate(zip(input_id_seqs, comp_masks)):
            length = len(ids)
            if length == 0:
                continue
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            completion_mask[i, :length] = torch.tensor(cm, dtype=torch.long)

        labels = input_ids.clone()
        labels[completion_mask == 0] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def get_dtype(dtype_str: str | None):
    if dtype_str is None or str(dtype_str).lower() == "auto":
        return None
    normalized = str(dtype_str).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype='{dtype_str}'. Use auto|bf16|fp16|fp32.")


def resolve_attr(obj, attr_path: str):
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr)
    return current


def get_decoder_layers(model: torch.nn.Module, decoder_layers_attr: str) -> torch.nn.ModuleList:
    layers = resolve_attr(model, decoder_layers_attr)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError(
            f"Resolved decoder layers at '{decoder_layers_attr}' but found type={type(layers)} "
            "(expected nn.ModuleList). Fix cfg.model.decoder_layers_attr for this architecture."
        )
    return layers


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_linalg_device(module: torch.nn.Linear, fallback_device: torch.device) -> torch.device:
    weight_device = module.weight.device
    if weight_device.type == "cuda":
        return weight_device
    if fallback_device.type == "cuda":
        return fallback_device
    return weight_device


def discover_target_linear_layers(
    decoder_layers: torch.nn.ModuleList, layer_attr: str, target_module_names: List[str]
) -> List[Tuple[int, str, str, torch.nn.Linear]]:
    """Return (layer_idx, relative_name, full_path, module) for each target Linear."""
    results: List[Tuple[int, str, str, torch.nn.Linear]] = []
    for layer_idx in range(len(decoder_layers)):
        block = decoder_layers[layer_idx]
        for target_name in target_module_names:
            try:
                module = resolve_attr(block, target_name)
            except AttributeError:
                continue
            if not isinstance(module, torch.nn.Linear):
                continue
            results.append((layer_idx, target_name, f"{layer_attr}.{layer_idx}.{target_name}", module))
    return results


def completion_log_prob_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log P(y_t | prefix) over supervised positions (logits[t] predicts labels[t+1])."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    safe = shift_labels.clone()
    safe[~mask] = 0
    token_log = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_log * mask.to(dtype=log_probs.dtype)).sum()


def chunk_module_entries_by_decoder_blocks(
    module_entries: List[Tuple[int, str, str, torch.nn.Linear]], decoder_block_chunk_size: int
) -> List[List[Tuple[int, str, str, torch.nn.Linear]]]:
    if decoder_block_chunk_size <= 0:
        raise ValueError("decoder_block_chunk_size must be a positive integer.")
    if not module_entries:
        return []
    max_layer_idx = max(layer_idx for layer_idx, _, _, _ in module_entries)
    chunked: List[List[Tuple[int, str, str, torch.nn.Linear]]] = []
    for chunk_start in range(0, max_layer_idx + 1, decoder_block_chunk_size):
        chunk_end = chunk_start + decoder_block_chunk_size
        chunk = [e for e in module_entries if chunk_start <= e[0] < chunk_end]
        if chunk:
            chunked.append(chunk)
    return chunked


def target_module_names(cfg: DictConfig) -> List[str]:
    if "target_modules" in cfg and cfg.target_modules is not None:
        return list(OmegaConf.to_container(cfg.target_modules, resolve=True))
    return list(DEFAULT_TARGET_MODULES)


def save_root(cfg: DictConfig, method_segment: str) -> Path:
    dataset_tag = f"{cfg.dataset.dataset_name}_{cfg.dataset.split}"
    results_root = OmegaConf.select(cfg, "paths.results_root") or "results"
    return Path(results_root) / method_segment / cfg.model.short_name / dataset_tag
