"""Training-time projected refusal-Fisher overlap monitor (no-op observer).

At each optimizer step this measures the **cumulative weight displacement**
``Δw = w_t - w₀`` (current weights minus the weights at the start of training),
projects it with the *same* TRAK projector used to build the refusal Fisher
``F = (1/n) Σ_i g_i g_iᵀ``, and logs

    overlap  = vᵀ F v            # excitation of refusal-Fisher directions by the drift
    rayleigh = vᵀ F v / ‖v‖²     # scale-free alignment (compare to trace(F)/proj_dim)

where ``v = P · Δw ∈ R^{proj_dim}``. Because Δw is a function of the weights alone
(not of the gradient that produced them), the same quantity can be recomputed for any
saved checkpoint offline.

It is a strict no-op observer: it fires on ``on_optimizer_step`` (immediately after
``optimizer.step()``, before ``model.zero_grad()``), reads ``p.data`` under
``torch.no_grad()`` and never writes to the model. It logs at ``global_step + 1``
(pre-increment at this hook) so step labels line up with the (1-based) training-loss
/ safety panels. ``assert_model_matches_spec`` guards the flattened ``grad_dim``.

Under FSDP2 (``train_args.fsdp=true``) the params are Shard(0) DTensors: every rank keeps
its own shard of w₀ on the CPU, Δw is all-gathered one param at a time, and rank 0 alone
projects and logs. ``set_reference``/``capture`` are then collectives that every rank calls.

Precision notes: Δw is formed in fp32 from exact bf16 operands, then rescaled by a
power of two into fp16's well-conditioned range before projection (the projector
input dtype is fixed by the kernel) and the linear projection is divided back out.
Note that bf16 weights quantize the update itself — a step smaller than the bf16 ulp
at ``|w|`` leaves that coordinate unchanged — so Δw is genuinely sparse early in
training. That is a property of the checkpoints, not of this measurement.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback

from interventions.projection import (
    CudaJLProjector,
    assert_model_matches_spec,
    delta_max_abs,
    delta_projection_scale,
    flatten_delta_into,
    flatten_sharded_delta_into,
    is_sharded,
    load_fisher,
    local_shards,
    make_grad_buffer,
    ordered_named_grad_params,
    reshard_fsdp,
    sharded_delta_max_abs,
    snapshot_params,
)


def fisher_overlap_monitor_enabled(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "fisher_overlap_monitor.enabled") or False)


def resolve_fisher_overlap_log_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "fisher_overlap_monitor.txt"


def resolve_fisher_path(cfg: DictConfig) -> Path:
    configured = OmegaConf.select(cfg, "fisher_overlap_monitor.fisher_path")
    if configured:
        return Path(str(configured))
    results_root = OmegaConf.select(cfg, "paths.results_root") or "results"
    return (
        Path(results_root)
        / "projected_fisher"
        / str(cfg.model.short_name)
        / "refusal_train"
        / "fisher.pt"
    )


def overlap_metrics(
    v: torch.Tensor, fisher: torch.Tensor, scale: float = 1.0
) -> tuple[float, float, float]:
    """Return ``(overlap, rayleigh, sq) = (vᵀFv, vᵀFv/‖v‖², ‖v‖²)``.

    ``v`` may be the projection of a *scaled* input (``scale`` = the multiplier that
    was applied); the quadratic form is then evaluated on the well-conditioned scaled
    vector and the two extensive quantities are divided by ``scale²`` to report the
    true values. ``rayleigh`` is scale-invariant either way.

    ``fisher`` must be square and on the same device/dtype as ``v``. Pure function so
    the quadratic-form math is unit-testable without a CUDA projector.
    """
    v = v.reshape(-1)
    sq = float(torch.dot(v, v).item())
    overlap = float(torch.dot(v, fisher @ v).item())
    rayleigh = overlap / sq if sq > 0 else float("nan")
    if scale != 1.0:
        sq /= scale * scale
        overlap /= scale * scale
    return overlap, rayleigh, sq


class FisherOverlapMonitor:
    """Per-optimizer-step overlap of the cumulative weight drift with the refusal Fisher."""

    def __init__(self, cfg: DictConfig, output_dir: str | Path):
        self.log_path = resolve_fisher_overlap_log_path(output_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        fisher_path = resolve_fisher_path(cfg)
        fisher_cpu, self.spec = load_fisher(fisher_path)
        self._fisher_cpu = fisher_cpu  # moved to model device lazily on first capture
        self._fisher: torch.Tensor | None = None
        self.fisher_path = fisher_path
        self.rayleigh_baseline = float(torch.diagonal(fisher_cpu).sum().item()) / float(self.spec.proj_dim)

        # Lazily built on first capture() once the (possibly wrapped) train model is known.
        self._projector: CudaJLProjector | None = None
        self._delta_buffer: torch.Tensor | None = None
        self._sharded: bool | None = None
        # w₀ snapshot, taken at train start (see set_reference).
        self._reference: list[torch.Tensor] | None = None

        # Under torchrun every rank builds a monitor; only rank 0 projects and writes the log.
        self.is_main = int(os.environ.get("RANK", "0")) == 0
        self._file = self.log_path.open("w", encoding="utf-8") if self.is_main else None
        if self._file is not None:
            self._write_header()

    def _write_header(self) -> None:
        self._file.write(
            "# fisher_overlap_monitor\n"
            f"# metric=delta_w  fisher_path={self.fisher_path}\n"
            f"# proj_dim={self.spec.proj_dim} proj_type={self.spec.proj_type} num_sms={self.spec.num_sms}\n"
            f"# gpu={self.spec.gpu_name} grad_dim={self.spec.grad_dim} chunk_size={self.spec.chunk_size}\n"
            f"# rayleigh_baseline=trace(F)/proj_dim={self.rayleigh_baseline:.8g}\n"
            "# v = P·Δw where Δw = w_t - w₀ is the cumulative weight displacement from the\n"
            "# start of training, read after optimizer.step() (NOT the per-step gradient).\n"
            "# overlap=vᵀFv (raw, grows with ‖Δw‖); rayleigh=vᵀFv/‖v‖² (scale-free, compare to\n"
            "# rayleigh_baseline). delta_scale is the power-of-two fp16 conditioning factor,\n"
            "# already divided out. Columns: step overlap rayleigh v_sqnorm delta_max_abs\n"
            "# delta_scale rayleigh_baseline\n"
        )
        self._file.flush()

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def sharded(self, model: torch.nn.Module) -> bool:
        if self._sharded is None:
            self._sharded = is_sharded(ordered_named_grad_params(model))
        return self._sharded

    def _ensure_projector(self, model: torch.nn.Module) -> None:
        if self._projector is not None or (self.sharded(model) and not self.is_main):
            return
        device = self._model_device(model)
        named_params = ordered_named_grad_params(model)
        assert_model_matches_spec(self.spec, named_params)   # DTensor shapes are the global ones
        self._projector = CudaJLProjector.from_spec(self.spec, device=device)
        self._delta_buffer = make_grad_buffer(self.spec.grad_dim, device, self.spec.input_dtype)
        self._fisher = self._fisher_cpu.to(device=device, dtype=torch.float32)

    def set_reference(self, model: torch.nn.Module) -> None:
        """Snapshot ``w₀`` — the weights Δw is measured against.

        Called at train start, so w₀ is the pre-finetuning model. On a resumed run this
        is the resumed checkpoint, not the original base model.
        """
        if self.sharded(model):
            reshard_fsdp(model)   # a forward before this (the safety monitor's) leaves the root unsharded
        self._ensure_projector(model)
        named_params = ordered_named_grad_params(model)
        with torch.no_grad():
            if self.sharded(model):   # this rank's shards only; off the GPU, which FSDP is using
                self._reference = snapshot_params(local_shards(named_params), device="cpu")
            else:
                self._reference = snapshot_params(named_params)
        n_bytes = sum(t.numel() * t.element_size() for t in self._reference)
        print(
            f"[fisher-overlap] Δw reference snapshot: {len(self._reference)} tensors, "
            f"{n_bytes / 2**30:.2f} GiB on {self._reference[0].device}"
        )

    def capture(self, model: torch.nn.Module) -> dict[str, float] | None:
        """Project ``Δw = w_t - w₀`` and return the overlap metrics (``None`` on the ranks
        of a sharded model that do not project).

        Pure observer: reads ``p.data`` under ``no_grad`` and never writes to the model.
        A zero Δw (no weight has moved off its bf16 grid yet) reports ``overlap=0`` with
        ``rayleigh=nan`` rather than being dropped — a genuine zero-drift step.
        """
        self._ensure_projector(model)
        if self._reference is None:
            raise RuntimeError(
                "Δw reference not set; FisherOverlapMonitor.set_reference() must run at train start."
            )
        if self.sharded(model):
            reshard_fsdp(model)
            return self._capture_sharded(ordered_named_grad_params(model))
        named_params = ordered_named_grad_params(model)
        assert self._projector is not None and self._delta_buffer is not None and self._fisher is not None
        device = self._delta_buffer.device
        with torch.no_grad():
            max_abs = delta_max_abs(named_params, self._reference, device)
            if max_abs == 0.0:
                return {
                    "overlap": 0.0, "rayleigh": float("nan"), "v_sqnorm": 0.0,
                    "delta_max_abs": 0.0, "delta_scale": 1.0,
                }
            scale = delta_projection_scale(max_abs)
            flatten_delta_into(self._delta_buffer, named_params, self._reference, scale)
            v = self._projector.project(self._delta_buffer).view(-1).to(torch.float32)
            overlap, rayleigh, sq = overlap_metrics(v, self._fisher, scale=scale)
        return {
            "overlap": overlap, "rayleigh": rayleigh, "v_sqnorm": sq,
            "delta_max_abs": max_abs, "delta_scale": scale,
        }

    def _capture_sharded(self, named_params) -> dict[str, float] | None:
        with torch.no_grad():
            max_abs = sharded_delta_max_abs(named_params, self._reference)
            if max_abs == 0.0:
                return {
                    "overlap": 0.0, "rayleigh": float("nan"), "v_sqnorm": 0.0,
                    "delta_max_abs": 0.0, "delta_scale": 1.0,
                } if self.is_main else None
            scale = delta_projection_scale(max_abs)
            flatten_sharded_delta_into(self._delta_buffer, named_params, self._reference, scale)
            if not self.is_main:
                return None
            v = self._projector.project(self._delta_buffer).view(-1).to(torch.float32)
            overlap, rayleigh, sq = overlap_metrics(v, self._fisher, scale=scale)
        return {
            "overlap": overlap, "rayleigh": rayleigh, "v_sqnorm": sq,
            "delta_max_abs": max_abs, "delta_scale": scale,
        }

    def record(self, global_step: int, metrics: dict[str, float]) -> None:
        self._file.write(
            f"step={int(global_step)}\t"
            f"overlap={float(metrics['overlap']):.8g}\t"
            f"rayleigh={float(metrics['rayleigh']):.8g}\t"
            f"v_sqnorm={float(metrics['v_sqnorm']):.8g}\t"
            f"delta_max_abs={float(metrics['delta_max_abs']):.8g}\t"
            f"delta_scale={float(metrics['delta_scale']):.8g}\t"
            f"rayleigh_baseline={self.rayleigh_baseline:.8g}\n"
        )
        self._file.flush()

    def close(self) -> None:
        # Drop the w₀ snapshot + projector buffers (several GiB) before post-train eval.
        self._reference = None
        self._delta_buffer = None
        self._fisher = None
        self._projector = None
        file_handle = getattr(self, "_file", None)
        if file_handle is not None and not file_handle.closed:
            file_handle.close()


class FisherOverlapMonitorCallback(TrainerCallback):
    def __init__(self, monitor: FisherOverlapMonitor):
        self.monitor = monitor
        self.monitor_every_steps = 1

    def set_every_steps(self, every: int) -> "FisherOverlapMonitorCallback":
        self.monitor_every_steps = max(1, int(every))
        return self

    def on_train_begin(self, args, state, control, **kwargs):
        # Before any optimizer step, so the snapshot is w₀ (the model as loaded). The Trainer
        # has already run accelerator.prepare, so an FSDP model is sharded by now.
        model = kwargs["model"]
        if state.is_world_process_zero or self.monitor.sharded(model):
            self.monitor.set_reference(model)
            # step=0 anchor. Δw is zero here by construction, so this is also a live
            # check that the reference matches the weights it was taken from — a
            # nonzero reading would mean the snapshot is not w₀.
            metrics = self.monitor.capture(model)
            if state.is_world_process_zero:
                self.monitor.record(0, metrics)
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        # Fires immediately after optimizer.step(), before model.zero_grad(); the weights
        # are post-update. state.global_step is still pre-increment here, so log step+1.
        model = kwargs["model"]
        if not (state.is_world_process_zero or self.monitor.sharded(model)):
            return control
        step = int(state.global_step) + 1
        if step % self.monitor_every_steps != 0:
            return control
        metrics = self.monitor.capture(model)   # collective when sharded
        if state.is_world_process_zero:
            self.monitor.record(step, metrics)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.close()
        return control
