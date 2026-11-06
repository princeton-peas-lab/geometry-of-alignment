"""One dispatch point for subspace interventions + trainer attachment.

Replaces the two separate ``build_intervention`` functions in the source
``grad_intervention_utils.py`` / ``repr_intervention_utils.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from .monitor import (
    InterventionProjectionMonitor,
    InterventionProjectionMonitorCallback,
    monitor_projection_enabled,
    resolve_monitor_log_path,
)
from .repr_fim import RepresentationFIMSubspaceIntervention

_MONITORABLE = {"repr_fim_subspace"}


def build_intervention(cfg: DictConfig, model: torch.nn.Module):
    method = str(getattr(cfg.subspace_intervention, "method", "")).lower()
    if method == "repr_fim_subspace":
        return RepresentationFIMSubspaceIntervention.from_config(cfg, model)
    raise ValueError(
        f"Unsupported subspace intervention method {method!r}. Expected: "
        "repr_fim_subspace."
    )


@dataclass
class InterventionHandle:
    """Owns the registered hook handles and (optional) projection monitor."""

    hook_handles: list
    monitor: InterventionProjectionMonitor | None

    def close(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []
        if self.monitor is not None:
            self.monitor.close()


def attach_intervention(cfg: DictConfig, trainer, output_dir) -> InterventionHandle:
    """Build the intervention, optionally attach a projection monitor, register hooks."""
    model = trainer.model
    method = str(getattr(cfg.subspace_intervention, "method", "")).lower()
    intervention = build_intervention(cfg, model)

    monitor = None
    if monitor_projection_enabled(cfg) and method in _MONITORABLE:
        monitor = InterventionProjectionMonitor(
            resolve_monitor_log_path(cfg, output_dir),
            intervention_kind=method,
            lambda_scale=float(cfg.subspace_intervention.lambda_scale),
        )
        trainer.add_callback(InterventionProjectionMonitorCallback(monitor))
        print(
            "Subspace projection monitoring enabled: logging ||Π·|| pre/post each step "
            f"to {monitor.log_path.resolve()}"
        )

    handles = intervention.make_hook(cfg, model, monitor=monitor)
    print(f"Subspace intervention '{intervention.name}': registered {len(handles)} hook(s).")
    return InterventionHandle(hook_handles=handles, monitor=monitor)
