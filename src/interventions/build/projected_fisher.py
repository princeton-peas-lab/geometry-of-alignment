"""Build the TRAK-projected refusal Fisher F = (1/n) Σ_i g_i g_iᵀ.

For each usable refusal-completion sample we take the **full-parameter** gradient
of the mean completion log-prob, flatten it into one fp16 buffer, project it with
the shared :class:`CudaJLProjector` to ``g_i ∈ R^{proj_dim}``, and accumulate the
rank-1 outer product into ``F ∈ R^{proj_dim×proj_dim}`` (fp32). ``F`` depends only
on the base model's refusal behavior, so it is built **once** and reused by every
finetune run's online :class:`FisherOverlapMonitor`.

Mirrors ``build/grad.py``'s per-sample backward loop, but projects the *whole*
flattened gradient instead of forming per-layer Gram matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import load_finetune_dataset
from interventions.projection import (
    CudaJLProjector,
    flatten_grads_into,
    make_grad_buffer,
    ordered_named_grad_params,
    param_fingerprint,
    save_fisher,
)
from utils import set_seed

from . import fim_common as fc


def _resolve_output_dir(cfg: DictConfig) -> Path:
    output_root = getattr(cfg, "output_root", None)
    if output_root:
        return Path(str(output_root))
    return fc.save_root(cfg, "projected_fisher")


def build_projected_refusal_fisher(cfg: DictConfig) -> Path:
    set_seed(int(getattr(cfg, "seed", 42)))
    device = torch.device(str(getattr(cfg, "device", "cuda:0")))
    if device.type != "cuda":
        raise ValueError("build_projected_refusal_fisher requires a CUDA device (TRAK projector).")

    torch_dtype = fc.get_dtype(getattr(cfg, "torch_dtype", "bf16"))
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.model.model_name_or_path), torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.model_name_or_path), use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    named_params = ordered_named_grad_params(model)
    fp = param_fingerprint(named_params)
    grad_dim = fp["grad_dim"]
    print(f"Trainable params: {fp['num_params']} tensors, grad_dim={grad_dim:,}")

    projector = CudaJLProjector(
        grad_dim=grad_dim,
        proj_dim=int(cfg.proj_dim),
        seed=int(cfg.proj_seed),
        device=device,
        proj_type=str(cfg.proj_type),
        input_dtype=str(cfg.proj_input_dtype),
        model_id=int(cfg.proj_model_id),
    )
    spec = projector.make_spec(named_params)
    print(f"Projector: proj_dim={spec.proj_dim} num_sms={spec.num_sms} gpu={spec.gpu_name}")
    print(f"Projection chunks: {len(projector.chunks)} (chunk_size={spec.chunk_size})")

    grad_buffer = make_grad_buffer(grad_dim, device, str(cfg.proj_input_dtype))
    fisher = torch.zeros(spec.proj_dim, spec.proj_dim, dtype=torch.float32, device=device)

    raw_dataset, _ = load_finetune_dataset(cfg)
    prepared = fc.prepare_dataset(raw_dataset, tokenizer, cfg)
    max_length = int(getattr(cfg, "max_length", 0)) or None
    dataloader = DataLoader(
        prepared,
        batch_size=1,
        shuffle=False,
        collate_fn=fc.get_completion_only_collate_fn(tokenizer, max_length=max_length),
    )

    max_num_samples = getattr(cfg, "max_num_samples", None)
    max_num_samples = int(max_num_samples) if max_num_samples is not None else None

    n_used = 0
    for batch in tqdm(dataloader, desc="Projected refusal Fisher"):
        if max_num_samples is not None and n_used >= max_num_samples:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        active = int((labels[..., 1:] != -100).sum().item())
        if active == 0:
            continue
        model.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False)
        loss = fc.completion_log_prob_sum(outputs.logits, labels) / float(active)
        loss.backward()
        flatten_grads_into(grad_buffer, named_params)
        g = projector.project(grad_buffer).view(-1).to(torch.float32)  # (proj_dim,)
        fisher.add_(torch.outer(g, g))
        n_used += 1
        del outputs, loss, g

    model.zero_grad(set_to_none=True)
    if n_used == 0:
        raise RuntimeError("No usable refusal samples with completion tokens; check labels/collate.")
    fisher.div_(float(n_used))

    trace = float(torch.diagonal(fisher).sum().item())
    top_k = int(getattr(cfg, "top_k_eigenvalues", 10))
    eigvals = torch.linalg.eigvalsh(fisher)  # ascending
    top_eigs = torch.flip(eigvals, dims=[0])[:top_k].cpu().tolist()
    print(f"n_samples={n_used}  trace(F)={trace:.6g}  trace/proj_dim={trace / spec.proj_dim:.6g}")
    print(f"top-{top_k} eigenvalues: {[f'{e:.4g}' for e in top_eigs]}")

    out_dir = _resolve_output_dir(cfg)
    extra_meta = {
        "model_short": str(cfg.model.short_name),
        "model_name_or_path": str(cfg.model.model_name_or_path),
        "dataset_name": str(cfg.dataset.dataset_name),
        "dataset_split": str(getattr(cfg.dataset, "split", None)),
        "dataset_variant": str(getattr(cfg.dataset, "variant", None)),
        "num_samples": int(n_used),
        "grad_dim": int(grad_dim),
        "num_param_tensors": int(fp["num_params"]),
        "top_eigenvalues": top_eigs,
        "artifact_type": "projected_refusal_fisher",
    }
    save_fisher(out_dir, fisher, spec, extra_meta=extra_meta)
    print(f"Saved projected refusal Fisher to {out_dir}")
    print(json.dumps({"trace": trace, **{k: extra_meta[k] for k in ("num_samples", "grad_dim")}}, indent=2))
    return out_dir
