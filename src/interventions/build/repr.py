"""Build per-layer representation Fisher information matrices and top-k subspaces.

Ported from the source ``build_intervention/repr_fim_subspace.py``. Computes, for
each target Linear, an input-side (d_in x d_in) Fisher averaged over samples, then
stores the top-k eigenvectors as ``subspaces.pt`` (keys = module paths).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import load_finetune_dataset
from utils import set_seed

from . import fim_common as fc


def _compute_global_fims_for_chunk(
    model, module_entries, dataloader, model_device, chunk_label
) -> Tuple[Dict[str, torch.Tensor], int]:
    if not module_entries:
        raise ValueError("module_entries must be non-empty for chunked FIM accumulation.")

    fim_sums: Dict[str, torch.Tensor] = {
        full_path: torch.zeros((int(m.in_features), int(m.in_features)), dtype=torch.float32)
        for _, _, full_path, m in module_entries
    }
    target_weights = [m.weight for _, _, _, m in module_entries]
    used_samples = 0

    for batch in tqdm(dataloader, desc=f"Accumulating FIM for {chunk_label}"):
        batch = {k: v.to(model_device) for k, v in batch.items()}
        batch_size = int(batch["input_ids"].shape[0])
        for sample_idx in range(batch_size):
            labels_row = batch["labels"][sample_idx : sample_idx + 1]
            shift_labels = labels_row[..., 1:]
            active = int((shift_labels != -100).sum().item())
            if active == 0:
                continue
            single = {k: v[sample_idx : sample_idx + 1] for k, v in batch.items()}
            labels = single.pop("labels")
            outputs = model(**single, use_cache=False)
            loss = fc.completion_log_prob_sum(outputs.logits, labels)
            grads = torch.autograd.grad(loss, target_weights, retain_graph=False, allow_unused=True)
            for grad, (_, _, full_path, m) in zip(grads, module_entries):
                if grad is None:
                    grad = torch.zeros_like(m.weight, dtype=torch.float32)
                else:
                    grad = grad.detach().to(dtype=torch.float32)
                grad = grad / active
                # Linear weights are [d_out, d_in]; average per-output-row Fisher → module Fisher.
                sample_fim = (grad.transpose(0, 1) @ grad) / float(m.out_features)
                fim_sums[full_path] += sample_fim.to(dtype=torch.float32, device="cpu")
            used_samples += 1
            del loss, grads, single
        del batch
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

    if used_samples == 0:
        raise RuntimeError(f"No usable samples with completion tokens for chunk '{chunk_label}'.")
    fim_avgs = {fp: s / float(used_samples) for fp, s in fim_sums.items()}
    return fim_avgs, used_samples


def _compute_subspace_from_fim(fim: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(fim)
    eigenvalues = torch.flip(eigenvalues, dims=[0]).to(dtype=torch.float32)
    eigenvectors = torch.flip(eigenvectors, dims=[1]).to(dtype=torch.float32)
    actual_k = min(top_k, eigenvectors.shape[1])
    top_eigenvectors = eigenvectors[:, :actual_k].transpose(0, 1).contiguous()
    return eigenvalues, top_eigenvectors


def build_repr_fim_subspaces(cfg: DictConfig) -> Path:
    set_seed(cfg.seed)

    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.model_name_or_path),
        device_map=str(getattr(cfg, "device_map", "auto")),
        torch_dtype=fc.get_dtype(getattr(cfg, "torch_dtype", "auto")),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.model_name_or_path), use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset, _ = load_finetune_dataset(cfg)
    prepared = fc.prepare_dataset(raw_dataset, tokenizer, cfg)
    max_length = int(getattr(cfg, "max_length", 0)) or None
    dataloader = DataLoader(
        prepared,
        batch_size=int(getattr(cfg, "batch_size", 1)),
        shuffle=False,
        collate_fn=fc.get_completion_only_collate_fn(tokenizer, max_length=max_length),
    )

    decoder_layers_attr = str(getattr(cfg.model, "decoder_layers_attr", "model.layers"))
    decoder_layers = fc.get_decoder_layers(model, decoder_layers_attr)
    top_k = int(getattr(cfg, "top_k_eigenvectors", 10))
    chunk_size = int(getattr(cfg, "decoder_block_chunk_size", 5))
    target_names = fc.target_module_names(cfg)

    module_entries = fc.discover_target_linear_layers(decoder_layers, decoder_layers_attr, target_names)
    if not module_entries:
        raise RuntimeError(f"No nn.Linear modules matched target_modules={target_names}.")
    print(f"Found {len(module_entries)} target linear layers across {len(decoder_layers)} blocks.")

    root = fc.save_root(cfg, "repr_fim_subspace")
    root.mkdir(parents=True, exist_ok=True)
    model_device = fc.get_model_device(model)
    chunks = fc.chunk_module_entries_by_decoder_blocks(module_entries, chunk_size)

    subspaces: Dict[str, torch.Tensor] = {}
    per_module_counts: Dict[str, int] = {}
    for chunk_entries in chunks:
        idxs = sorted({layer_idx for layer_idx, _, _, _ in chunk_entries})
        chunk_label = f"decoder blocks {idxs[0]}-{idxs[-1]}"
        fim_avgs, used = _compute_global_fims_for_chunk(
            model, chunk_entries, dataloader, model_device, chunk_label
        )
        for layer_idx, target_name, full_path, m in chunk_entries:
            per_module_counts[full_path] = used
            linalg_device = fc.get_linalg_device(m, model_device)
            fim_avg = fim_avgs[full_path].to(device=linalg_device, dtype=torch.float32)
            eigenvalues, top_eigenvectors = _compute_subspace_from_fim(fim_avg, top_k=top_k)
            eigenvalues = eigenvalues.cpu()
            top_eigenvectors = top_eigenvectors.cpu()
            subspaces[full_path] = top_eigenvectors

            layer_dir = root / f"layer_{layer_idx:02d}" / target_name.replace(".", "/")
            layer_dir.mkdir(parents=True, exist_ok=True)
            torch.save(eigenvalues, layer_dir / "eigenvalues.pt")
            torch.save(top_eigenvectors, layer_dir / "top_eigenvectors.pt")
            with open(layer_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "layer_idx": int(layer_idx),
                        "target_name": target_name,
                        "full_module_path": full_path,
                        "in_features": int(m.in_features),
                        "out_features": int(m.out_features),
                        "num_samples": int(used),
                        "top_k": int(top_eigenvectors.shape[0]),
                        "top_eigenvectors_shape": list(top_eigenvectors.shape),
                        "top_eigenvectors_are_row_major": True,
                        "decoder_block_chunk_size": int(chunk_size),
                        "chunk_label": chunk_label,
                    },
                    f,
                    indent=4,
                )

    torch.save(subspaces, root / "subspaces.pt")
    summary = {
        "model_short": str(cfg.model.short_name),
        "model_name_or_path": str(cfg.model.model_name_or_path),
        "decoder_layers_attr": decoder_layers_attr,
        "dataset_name": str(cfg.dataset.dataset_name),
        "dataset_split": str(cfg.dataset.split),
        "num_decoder_layers": int(len(decoder_layers)),
        "target_modules": target_names,
        "max_length": max_length,
        "top_k_eigenvectors": top_k,
        "decoder_block_chunk_size": int(chunk_size),
        "num_chunk_passes": int(len(chunks)),
        "num_linear_layers_processed": len(subspaces),
        "module_sample_counts": per_module_counts,
        "artifact_type": "repr_fim_subspace",
    }
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved {len(subspaces)} repr-FIM subspaces to {root}")
    return root
