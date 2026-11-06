"""Log subspace projection norms before/after the repr-FIM intervention.

Ported from the source ``intervention_projection_monitor.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from transformers import TrainerCallback


def resolve_monitor_log_path(cfg: DictConfig, output_dir: str | Path) -> Path:
    custom = getattr(cfg.subspace_intervention, "monitor_log_path", None)
    if custom is not None:
        return Path(str(custom))
    return Path(output_dir) / "intervention_projection_monitor.txt"


def monitor_projection_enabled(cfg: DictConfig) -> bool:
    sub = getattr(cfg, "subspace_intervention", None)
    if sub is None:
        return False
    return bool(getattr(sub, "monitor_projection", False))


def repr_projection_norms(
    x: torch.Tensor,
    basis_rows: torch.Tensor,
    lambda_scale: float,
    *,
    basis_on_device: Optional[torch.Tensor] = None,
) -> Tuple[float, float, torch.Tensor]:
    """For hidden states x (..., d) and basis B (k, d): mean-token Πx norms before/after."""
    if x.ndim < 2:
        raise ValueError(f"Expected rank >= 2 hidden states, got {tuple(x.shape)}")
    if basis_on_device is None:
        basis_on_device = basis_rows.to(device=x.device, dtype=x.dtype)

    coeffs = x @ basis_on_device.T
    proj = coeffs @ basis_on_device
    pre_norm = float(proj.float().norm(dim=-1).mean().item())

    x_out = x + float(lambda_scale) * proj
    proj_post = (x_out @ basis_on_device.T) @ basis_on_device
    post_norm = float(proj_post.float().norm(dim=-1).mean().item())
    return pre_norm, post_norm, x_out


class InterventionProjectionMonitor:
    """Accumulates pre/post projection norms per module; flushes one line per module per step."""

    def __init__(self, log_path: Path, *, intervention_kind: str, lambda_scale: float):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.intervention_kind = intervention_kind
        self.lambda_scale = float(lambda_scale)
        self._pending: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
        self._file = self.log_path.open("w", encoding="utf-8")
        self._write_header()

    def _write_header(self) -> None:
        self._file.write(
            "# intervention_projection_monitor\n"
            f"# kind={self.intervention_kind}\n"
            f"# lambda_scale={self.lambda_scale:g}\n"
        )
        self._file.write(
            "# Columns: step layer module n_calls pre_proj_norm post_proj_norm "
            "post_over_pre expected_post_over_pre\n"
        )
        self._file.flush()

    def record(
        self,
        layer_idx: int,
        module_name: str,
        pre_norm: float,
        post_norm: float,
    ) -> None:
        key = (int(layer_idx), str(module_name))
        row: Dict[str, float] = {"pre_proj_norm": float(pre_norm), "post_proj_norm": float(post_norm)}
        self._pending.setdefault(key, []).append(row)

    def flush_step(self, global_step: int) -> None:
        if not self._pending:
            return
        expected_ratio = abs(1.0 + self.lambda_scale)

        def _mean(rows: List[Dict[str, float]], field: str) -> float:
            return sum(r[field] for r in rows) / len(rows)

        for layer_idx, module_name in sorted(self._pending.keys()):
            rows = self._pending[(layer_idx, module_name)]
            n = len(rows)
            pre_mean = _mean(rows, "pre_proj_norm")
            post_mean = _mean(rows, "post_proj_norm")
            ratio = post_mean / pre_mean if pre_mean > 1e-12 else float("nan")
            line = (
                f"step={global_step}\tlayer={layer_idx}\tmodule={module_name}\tn_calls={n}\t"
                f"pre_proj_norm={pre_mean:.8g}\tpost_proj_norm={post_mean:.8g}\t"
                f"post_over_pre={ratio:.8g}\texpected_post_over_pre={expected_ratio:.8g}"
            )
            self._file.write(line + "\n")
        self._pending.clear()
        self._file.flush()

    def close(self) -> None:
        file_handle = getattr(self, "_file", None)
        if file_handle is not None and not file_handle.closed:
            file_handle.close()


class InterventionProjectionMonitorCallback(TrainerCallback):
    def __init__(self, monitor: InterventionProjectionMonitor):
        self.monitor = monitor

    def on_step_end(self, args, state, control, **kwargs):
        self.monitor.flush_step(int(state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.flush_step(int(state.global_step))
        self.monitor.close()
        return control
