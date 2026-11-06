"""Shared helpers for subspace interventions.

These were duplicated verbatim across the source ``grad_intervention_utils.py`` and
``repr_intervention_utils.py`` (layer/attr resolution, module-key parsing, basis
slicing, top-k + root resolution, and the ``subspaces.pt`` load/validate loop).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from paths import results_root

# Decoder-block paths to try after cfg.model.decoder_layers_attr (PEFT / HF wrappers).
DECODER_LAYER_ATTR_FALLBACKS: List[str] = [
    "model.layers",
    "base_model.model.model.layers",
    "base_model.model.layers",
]


def resolve_attr(obj, attr_path: str):
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr)
    return current


def get_decoder_layers(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.ModuleList:
    primary = str(getattr(cfg.model, "decoder_layers_attr", "model.layers"))
    candidates: List[str] = [primary] + [p for p in DECODER_LAYER_ATTR_FALLBACKS if p != primary]
    errors: List[str] = []
    for attr_path in candidates:
        try:
            layers = resolve_attr(model, attr_path)
        except AttributeError as exc:
            errors.append(f"{attr_path}: {exc}")
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return layers
        errors.append(f"{attr_path}: expected nn.ModuleList, got {type(layers).__name__}")
    raise ValueError(
        "Could not resolve decoder layers. Tried, in order: "
        + ", ".join(candidates)
        + ". Details: "
        + "; ".join(errors)
    )


def parse_module_key(module_path: str) -> Tuple[int, str]:
    parts = module_path.split(".")
    if "layers" not in parts:
        raise ValueError(
            f"Unsupported subspace module path '{module_path}'. Expected a path containing 'layers'."
        )
    layers_idx = parts.index("layers")
    if layers_idx + 2 >= len(parts):
        raise ValueError(
            f"Unsupported subspace module path '{module_path}'. Expected format "
            "'...layers.<layer_idx>.<target_module_name>'."
        )
    layer_idx = int(parts[layers_idx + 1])
    target_module_name = ".".join(parts[layers_idx + 2 :])
    return layer_idx, target_module_name


def resolve_intervention_top_k(cfg: DictConfig) -> Optional[int]:
    top_k_raw = getattr(cfg.subspace_intervention, "top_k", None)
    if top_k_raw is None:
        return None
    top_k = int(top_k_raw)
    if top_k < 1:
        raise ValueError(f"subspace_intervention.top_k must be >= 1, got {top_k}.")
    return top_k


def slice_stored_basis(subspace: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """Keep the leading rows of a stored subspace (top eigenvectors by build order)."""
    if top_k is None:
        return subspace
    stored_k = int(subspace.shape[0])
    if top_k > stored_k:
        raise ValueError(
            f"subspace_intervention.top_k={top_k} exceeds stored basis size {stored_k}. "
            "Rebuild with a larger top_k_eigenvectors or lower top_k."
        )
    return subspace[:top_k].contiguous()


def intervention_root(cfg: DictConfig, method_segment: str) -> Path:
    custom_root = getattr(cfg.subspace_intervention, "intervention_root", None)
    if custom_root is not None:
        return Path(str(custom_root))
    return results_root(cfg) / method_segment / str(cfg.model.short_name) / "refusal_train"


def load_subspaces_by_module(
    cfg: DictConfig, method_segment: str
) -> Dict[Tuple[int, str], torch.Tensor]:
    """Load + validate ``subspaces.pt`` into a {(layer_idx, module_name): basis} dict.

    Shared by the representation- and gradient-FIM interventions (their source
    ``from_config`` loops were identical apart from error wording).
    """
    root = intervention_root(cfg, method_segment)
    consolidated_path = root / "subspaces.pt"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"Subspace file not found: {consolidated_path}. Build it first "
            f"(cli/build_subspace.py method={method_segment}) or set "
            "subspace_intervention.intervention_root."
        )

    loaded = torch.load(consolidated_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a dict of module_path -> tensor in {consolidated_path}, got {type(loaded)}."
        )

    top_k = resolve_intervention_top_k(cfg)
    out: Dict[Tuple[int, str], torch.Tensor] = {}
    for module_path, subspace in loaded.items():
        if not isinstance(module_path, str):
            raise ValueError(f"Expected module path keys to be strings, got {type(module_path)}.")
        if not isinstance(subspace, torch.Tensor) or subspace.ndim != 2:
            raise ValueError(
                f"Expected a rank-2 tensor for module '{module_path}', got {type(subspace)} "
                f"with shape {getattr(subspace, 'shape', None)}."
            )
        key = parse_module_key(module_path)
        if key in out:
            raise ValueError(
                f"Duplicate subspace for layer {key[0]} module '{key[1]}' "
                f"(repeated key: '{module_path}')."
            )
        out[key] = slice_stored_basis(subspace.to(torch.float32).contiguous(), top_k)
    return out
