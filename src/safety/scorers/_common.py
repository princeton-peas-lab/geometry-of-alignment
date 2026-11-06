"""Small shared helpers for model-based safety scorers."""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def torch_dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("float32", "fp32"):
        return torch.float32
    if key == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unknown torch_dtype: {name}")


def resolve_device(cfg: DictConfig) -> torch.device:
    device_s = cfg.runtime.device
    if device_s is None or str(device_s).lower() in ("null", "none", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_s))
