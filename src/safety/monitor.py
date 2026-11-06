"""Training-time refusal-completion log-prob monitor.

Ported from the source ``safety_monitor.py``. Config moved to the nested
``safety_monitor`` group (was top-level ``safety_monitor_*`` flags), and the
refusal variant now comes from ``model.key`` (the conf/model stem, which matches
the ``data/refusal/<variant>_refused_prompts_*.json`` filenames).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from dataset_utils import load_finetune_dataset_bundle
from interventions.projection import is_sharded, ordered_named_grad_params
from utils import prompt_completion_ids as prompt_completion_ids_fn


def safety_monitor_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "safety_monitor.enabled") or False)


def resolve_safety_monitor_log_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "safety_monitor.txt"


def resolve_refusal_variant(cfg: DictConfig) -> str:
    """Refusal variant = model.key (matches data/refusal filenames), with fallbacks."""
    key = OmegaConf.select(cfg, "model.key")
    if key:
        return str(key)
    model_path = Path(str(cfg.model.model_name_or_path))
    if model_path.name:
        return model_path.name
    return str(cfg.model.short_name)


def _refusal_dataset_path(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "safety_monitor.dataset_path")
    if configured is not None:
        return str(configured)
    data_root = OmegaConf.select(cfg, "paths.data_root") or "data"
    return f"{data_root}/refusal"


def _refusal_train_cfg(cfg: DictConfig) -> DictConfig:
    variant = resolve_refusal_variant(cfg)
    monitor_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    monitor_cfg.dataset = {
        "dataset_name": "refusal",
        "dataset_path": _refusal_dataset_path(cfg),
        "subset": "harmful",
        "variant": variant,
        "split": "train",
        "do_eval": False,
    }
    monitor_cfg.eval_size = 0.0
    monitor_cfg.max_samples = None
    return monitor_cfg


def _prepare_dataset(dataset, tokenizer):
    def tokenize_fn(example: dict[str, Any], tokenizer):
        prompt_ids, prompt_completion_ids = prompt_completion_ids_fn(
            tokenizer, example["prompt"], example["completion"]
        )
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        return {"input_ids": prompt_completion_ids, "completion_mask": completion_mask}

    return dataset.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer})


def _completion_collate_fn(tokenizer, max_length: int | None = None):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        input_id_seqs, completion_masks = [], []
        for example in batch:
            ids = example["input_ids"]
            mask = example["completion_mask"]
            if max_length is not None:
                ids = ids[:max_length]
                mask = mask[:max_length]
            input_id_seqs.append(ids)
            completion_masks.append(mask)

        batch_size = len(input_id_seqs)
        max_len = max(len(ids) for ids in input_id_seqs) if batch_size else 0
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for row_idx, (ids, completion_mask) in enumerate(zip(input_id_seqs, completion_masks)):
            length = len(ids)
            if length == 0:
                continue
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            mask_tensor = torch.tensor(completion_mask, dtype=torch.bool)
            input_ids[row_idx, :length] = ids_tensor
            attention_mask[row_idx, :length] = 1
            row_labels = ids_tensor.clone()
            row_labels[~mask_tensor] = -100
            labels[row_idx, :length] = row_labels

        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def _completion_token_log_probs(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask.to(dtype=token_log_probs.dtype)
    return token_log_probs.sum(dim=-1), mask.sum(dim=-1)


class SafetyMonitor:
    """Mean per-token log P(completion | prompt) over refusal train samples."""

    def __init__(self, cfg: DictConfig, tokenizer, output_dir: str | Path):
        self.variant = resolve_refusal_variant(cfg)
        self.log_path = resolve_safety_monitor_log_path(output_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_length = int(OmegaConf.select(cfg, "train_args.max_length") or 0) or None
        self.batch_size = int(OmegaConf.select(cfg, "safety_monitor.batch_size") or 1)

        refusal_cfg = _refusal_train_cfg(cfg)
        bundle = load_finetune_dataset_bundle(refusal_cfg)
        self.dataset = _prepare_dataset(bundle.train_dataset, tokenizer)
        if len(self.dataset) == 0:
            raise RuntimeError(f"Safety monitor refusal dataset is empty for variant {self.variant!r}.")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_completion_collate_fn(tokenizer, max_length=self.max_length),
        )
        # Under torchrun every rank builds a monitor; only rank 0 writes the log.
        self.is_main = int(os.environ.get("RANK", "0")) == 0
        self._file = self.log_path.open("w", encoding="utf-8") if self.is_main else None
        if self._file is not None:
            self._write_header()
        self._sharded: bool | None = None

    def sharded(self, model: torch.nn.Module) -> bool:
        """FSDP2 model: its forward is a collective, so every rank runs :meth:`compute`."""
        if self._sharded is None:
            self._sharded = is_sharded(ordered_named_grad_params(model))
        return self._sharded

    def _write_header(self) -> None:
        self._file.write(
            "# safety_monitor\n"
            f"# refusal_variant={self.variant}\n"
            f"# num_samples={len(self.dataset)}\n"
            f"# batch_size={self.batch_size}\n"
            "# safety_utility is the sample mean of completion-token mean log probability.\n"
            "# Columns: step safety_utility num_samples num_completion_tokens\n"
        )
        self._file.flush()

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    @torch.no_grad()
    def compute(self, model: torch.nn.Module) -> dict[str, float | int]:
        was_training = model.training
        model.eval()
        device = self._model_device(model)

        total_sample_mean_log_prob = 0.0
        total_samples = 0
        total_completion_tokens = 0
        try:
            for batch in self.dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch, use_cache=False)
                sample_log_prob_sums, sample_token_counts = _completion_token_log_probs(
                    outputs.logits, labels
                )
                valid = sample_token_counts > 0
                if valid.any():
                    sample_means = sample_log_prob_sums[valid] / sample_token_counts[valid].to(
                        dtype=sample_log_prob_sums.dtype
                    )
                    total_sample_mean_log_prob += float(sample_means.sum().item())
                    total_samples += int(valid.sum().item())
                    total_completion_tokens += int(sample_token_counts[valid].sum().item())
        finally:
            if was_training:
                model.train()
        if total_samples == 0:
            raise RuntimeError(
                f"No completion tokens found in safety monitor dataset for variant {self.variant!r}."
            )
        return {
            "safety_utility": total_sample_mean_log_prob / float(total_samples),
            "num_samples": total_samples,
            "num_completion_tokens": total_completion_tokens,
        }

    def record(self, global_step: int, metrics: dict[str, float | int]) -> None:
        self._file.write(
            f"step={int(global_step)}\t"
            f"safety_utility={float(metrics['safety_utility']):.8g}\t"
            f"num_samples={int(metrics['num_samples'])}\t"
            f"num_completion_tokens={int(metrics['num_completion_tokens'])}\n"
        )
        self._file.flush()

    def close(self) -> None:
        file_handle = getattr(self, "_file", None)
        if file_handle is not None and not file_handle.closed:
            file_handle.close()


class SafetyMonitorCallback(TrainerCallback):
    def __init__(self, monitor: SafetyMonitor):
        self.monitor = monitor

    def on_train_begin(self, args, state, control, **kwargs):
        # step=0 anchor: the pre-finetuning refusal utility, so later steps can be
        # read as movement away from the base model rather than from step 1.
        model = kwargs["model"]
        if state.is_world_process_zero or self.monitor.sharded(model):
            metrics = self.monitor.compute(model)
            if state.is_world_process_zero:
                self.monitor.record(0, metrics)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if not (state.is_world_process_zero or self.monitor.sharded(model)):
            return control
        metrics = self.monitor.compute(model)   # every rank, identical batches, when sharded
        if state.is_world_process_zero:
            self.monitor.record(int(state.global_step), metrics)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.close()
        return control
