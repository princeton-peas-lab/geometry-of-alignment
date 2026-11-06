"""Representation FIM subspace intervention: forward pre-hook x <- x + λ Π x."""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from .base import get_decoder_layers, load_subspaces_by_module, resolve_attr
from .monitor import InterventionProjectionMonitor, repr_projection_norms


class RepresentationFIMSubspaceIntervention:
    name = "repr_fim_subspace"

    def __init__(self, lambda_scale: float, subspaces_by_module: Dict[Tuple[int, str], torch.Tensor]):
        self.lambda_scale = float(lambda_scale)
        self.subspaces_by_module = subspaces_by_module
        self._basis_cache: Dict[Tuple[Hashable, torch.device, torch.dtype], torch.Tensor] = {}

    @classmethod
    def from_config(cls, cfg: DictConfig, _model: torch.nn.Module) -> "RepresentationFIMSubspaceIntervention":
        return cls(
            lambda_scale=float(cfg.subspace_intervention.lambda_scale),
            subspaces_by_module=load_subspaces_by_module(cfg, "repr_fim_subspace"),
        )

    def _basis_on_device(self, cache_key: Hashable, basis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor_key = (cache_key, x.device, x.dtype)
        if tensor_key not in self._basis_cache:
            self._basis_cache[tensor_key] = basis.to(device=x.device, dtype=x.dtype)
        return self._basis_cache[tensor_key]

    def apply(self, x: torch.Tensor, *, layer_idx: int, target_module_name: str, monitor=None) -> torch.Tensor:
        key = (int(layer_idx), str(target_module_name))
        if key not in self.subspaces_by_module:
            raise KeyError(f"Missing FIM subspace for module {key}.")
        basis = self.subspaces_by_module[key]
        if monitor is not None:
            basis_dev = self._basis_on_device(key, basis, x)
            pre, post, x_out = repr_projection_norms(
                x, basis, self.lambda_scale, basis_on_device=basis_dev
            )
            monitor.record(layer_idx, target_module_name, pre, post)
            return x_out
        if x.ndim < 2:
            raise ValueError(f"Expected a trailing hidden dim, got shape {tuple(x.shape)}.")
        basis_dev = self._basis_on_device(key, basis, x)
        projection_weights = x @ basis_dev.T
        return x + self.lambda_scale * (projection_weights @ basis_dev)

    def make_hook(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        monitor: Optional[InterventionProjectionMonitor] = None,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        decoder_layers = get_decoder_layers(model, cfg)
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, target_module_name in sorted(self.subspaces_by_module.keys()):
            if layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"Saved FIM subspace targets layer {layer_idx}, but the model has "
                    f"{len(decoder_layers)} decoder layers."
                )
            try:
                target_module = resolve_attr(decoder_layers[layer_idx], target_module_name)
            except AttributeError as exc:
                raise ValueError(
                    f"Could not resolve target module '{target_module_name}' in decoder layer "
                    f"{layer_idx} while registering FIM subspace hooks."
                ) from exc

            def make_module_hook(layer_i: int, name: str):
                def hook(_module, inputs):
                    if not inputs:
                        raise RuntimeError("Expected module inputs in FIM subspace pre-forward hook.")
                    shifted = self.apply(inputs[0], layer_idx=layer_i, target_module_name=name, monitor=monitor)
                    return (shifted, *inputs[1:])

                return hook

            hooks.append(target_module.register_forward_pre_hook(make_module_hook(layer_idx, target_module_name)))
        return hooks
