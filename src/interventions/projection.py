"""Shared TRAK-style single-batch CUDA JL projector + Fisher (de)serialization.

A *minimal* vendoring of ``trak.projectors.CudaProjector`` that only needs the
``fast_jl`` CUDA kernel (we deliberately do **not** import the ``trak`` package).
Used both offline (to build the projected refusal Fisher) and online (the
``FisherOverlapMonitor``). The projection is byte-for-byte determined by
``(grad_dim, proj_dim, seed, proj_type, model_id, num_sms)`` — and ``num_sms`` is
GPU-specific — so a :class:`ProjectorSpec` fingerprint is saved alongside the
Fisher and asserted against the live model at monitor init.

Hard constraints:
  * Project **one sample at a time** (batch=1). llama-3.2-3B's flattened gradient
    is > 2**31; ``fast_jl``'s address arithmetic is uint32 and only batch=1 is
    provably overflow-safe at this grad_dim.
  * ``proj_dim`` must be a multiple of 512 (kernel constraint).
  * Projector input dtype is float16 (the kernel halves inputs internally anyway).
  * ``grad_dim`` per kernel launch must stay under :data:`FAST_JL_MAX_FEATURES`.
    ``fast_jl.cu`` reads the feature count into a ``uint32_t`` and ``data_loading.cuh``
    forms the column address as ``current_col + blockIdx.y * feature_tile_size`` in
    uint32, so a wider input wraps around **silently** — no error, just wrong numbers.
    Models above that width are handled by :ref:`chunked projection <chunked>`.

.. _chunked:

Chunked projection
------------------
For ``grad_dim >= FAST_JL_MAX_FEATURES`` the input is split into ``C`` contiguous
chunks, each projected with its own seed and the results summed::

    P·a = Σ_c P_c · a_c

This is not an approximation. ``P_c`` is an independent ``proj_dim × chunk`` Rademacher
block, so the implied ``P = [P_0 | P_1 | … | P_{C-1}]`` has iid ±1 entries exactly like
a monolithic draw — ``E[PᵀP] = d·I`` and every identity in the first-order-measure
write-up carries over unchanged. Chunk ``c`` uses ``effective_seed = seed + 10000·(model_id + c)``,
reusing the existing ``model_id`` offset so the blocks are independent of one another.

``ProjectorSpec.chunk_size`` records the split. It defaults to ``None`` (monolithic),
so specs written before this existed — llama32-3b and qwen3-4b, both under the limit —
load and reproduce byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

# Widest input a single ``fast_jl`` launch may see. ``fast_jl.cu`` holds the feature
# count in a ``uint32_t`` and ``data_loading.cuh`` forms the column address as
# ``current_col + blockIdx.y * feature_tile_size``, which peaks just above ``features``
# (``feature_tile_size = ceil(features / num_feature_tiles)``). Both cap out at 2**32.
# qwen3-4b sits at 0.94x this limit and is correct, so the bound is not conservative;
# ``default_chunk_size`` leaves far more margin than that whenever it does split.
FAST_JL_MAX_FEATURES = 2**32


def chunk_bounds(grad_dim: int, chunk_size: Optional[int]) -> List[Tuple[int, int]]:
    """Contiguous ``[start, end)`` spans covering ``grad_dim``.

    ``chunk_size = None`` means "one launch" and is only valid below the kernel limit.
    """
    grad_dim = int(grad_dim)
    if chunk_size is None:
        if grad_dim >= FAST_JL_MAX_FEATURES:
            raise ValueError(
                f"grad_dim={grad_dim:,} needs chunked projection (fast_jl indexes columns "
                f"in uint32 and wraps silently above {FAST_JL_MAX_FEATURES:,}); "
                "pass chunk_size or let CudaJLProjector choose one."
            )
        return [(0, grad_dim)]
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size >= FAST_JL_MAX_FEATURES:
        raise ValueError(
            f"chunk_size must be in [1, {FAST_JL_MAX_FEATURES:,}); got {chunk_size}."
        )
    return [(s, min(s + chunk_size, grad_dim)) for s in range(0, grad_dim, chunk_size)]


def default_chunk_size(grad_dim: int) -> Optional[int]:
    """``None`` below the kernel limit, else the even split into the fewest legal chunks."""
    grad_dim = int(grad_dim)
    if grad_dim < FAST_JL_MAX_FEATURES:
        return None
    n_chunks = -(-grad_dim // (FAST_JL_MAX_FEATURES - 1))
    return -(-grad_dim // n_chunks)


def resolve_input_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported projector input_dtype={name!r}. Use float16|float32.")
    return _DTYPES[key]


def ordered_named_grad_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    """Trainable params in deterministic registration order."""
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def _sha256_of_strings(items: List[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def param_fingerprint(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, Any]:
    """Order-sensitive fingerprint of the trainable parameter set.

    ``grad_dim`` is the flattened gradient dimension (sum of numel); ``num_params``
    is the count of parameter tensors. ``shape_sha256`` hashes the ordered shapes
    (catches a different architecture / param set); ``name_sha256`` hashes the
    ordered names (catches wrapper prefixes, which we only warn on).
    """
    shapes = [str(tuple(p.shape)) for _, p in named_params]
    names = [name for name, _ in named_params]
    grad_dim = int(sum(p.numel() for _, p in named_params))
    return {
        "grad_dim": grad_dim,
        "num_params": len(named_params),
        "shape_sha256": _sha256_of_strings(shapes),
        "name_sha256": _sha256_of_strings(names),
    }


@dataclass
class ProjectorSpec:
    grad_dim: int
    proj_dim: int
    seed: int
    proj_type: str
    model_id: int
    num_sms: int
    gpu_name: str
    input_dtype: str
    param_shape_sha256: str
    param_name_sha256: str
    num_params: int
    # None => one fast_jl launch over the whole vector (specs predating chunking).
    chunk_size: Optional[int] = None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectorSpec":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectorSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class CudaJLProjector:
    """Single-batch (B<=8) Rademacher/Normal JL projector backed by ``fast_jl``.

    ``project(mat)`` maps ``(B, grad_dim) -> (B, proj_dim)`` float32. We only ever
    call it with B=1 (see module docstring), but B<=8 is accepted because the
    ``*_8`` kernel handles up to 8 rows in one launch.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        device: str | torch.device,
        proj_type: str = "rademacher",
        input_dtype: str = "float16",
        model_id: int = 0,
        expected_num_sms: Optional[int] = None,
        # None => monolithic if grad_dim is under the kernel limit, else auto-split.
        chunk_size: Optional[int] = None,
    ) -> None:
        if proj_dim % 512 != 0:
            raise ValueError(f"proj_dim must be a multiple of 512 (kernel constraint); got {proj_dim}.")
        if proj_type not in ("rademacher", "normal"):
            raise ValueError(f"proj_type must be 'rademacher' or 'normal'; got {proj_type!r}.")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("CudaJLProjector requires a CUDA device.")
        self.device = dev
        self.grad_dim = int(grad_dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.proj_type = str(proj_type)
        self.model_id = int(model_id)
        self.input_dtype = resolve_input_dtype(input_dtype)
        self.chunk_size = chunk_size if chunk_size is not None else default_chunk_size(self.grad_dim)
        self.chunks = chunk_bounds(self.grad_dim, self.chunk_size)

        self.num_sms = int(
            torch.cuda.get_device_properties(dev.index if dev.index is not None else 0).multi_processor_count
        )
        if expected_num_sms is not None and self.num_sms != int(expected_num_sms):
            raise RuntimeError(
                f"GPU num_sms mismatch: live GPU reports num_sms={self.num_sms} but the projector "
                f"spec was built with num_sms={expected_num_sms}. The fast_jl projection is "
                "GPU-model-specific; run on the same GPU model used to build the Fisher."
            )

        try:
            import fast_jl  # noqa: F401
        except ImportError as e:
            raise ModuleNotFoundError(
                "fast_jl is not installed. Build it from trak/fast_jl into the venv "
                "(scripts/build_fast_jl.sbatch)."
            ) from e
        self._fast_jl = fast_jl
        self._fn_name = f"project_{self.proj_type}_8"
        if not hasattr(fast_jl, self._fn_name):
            raise AttributeError(f"fast_jl has no function {self._fn_name!r}.")
        # Smoke-test at init so failures surface here, not mid-training.
        probe = torch.zeros(1, 1024, device=self.device, dtype=self.input_dtype)
        getattr(fast_jl, self._fn_name)(probe, self.proj_dim, self.seed, self.num_sms)

    @property
    def effective_seed(self) -> int:
        return self.seed + 10000 * self.model_id

    def chunk_seed(self, chunk_index: int) -> int:
        """Seed for chunk ``c``, reusing the ``model_id`` offset so blocks are independent.

        ``chunk_index = 0`` reproduces :attr:`effective_seed`, so a single-chunk
        projector is bit-identical to the pre-chunking implementation.
        """
        return self.seed + 10000 * (self.model_id + int(chunk_index))

    def project(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.dim() != 2:
            raise ValueError(f"Projector input must be 2D (B, grad_dim); got shape {tuple(mat.shape)}.")
        b = int(mat.shape[0])
        if b > 8:
            raise ValueError(
                f"CudaJLProjector only supports B<=8 per call (got B={b}); the monitor uses B=1."
            )
        if int(mat.shape[1]) != self.grad_dim:
            raise ValueError(
                f"Projector input width {int(mat.shape[1])} != grad_dim {self.grad_dim}."
            )
        if mat.device != self.device:
            mat = mat.to(self.device)
        if mat.dtype != self.input_dtype:
            mat = mat.to(self.input_dtype)
        mat = mat.contiguous()
        fn = getattr(self._fast_jl, self._fn_name)
        if len(self.chunks) == 1:
            return fn(mat, self.proj_dim, self.effective_seed, self.num_sms)
        # P·a = Σ_c P_c·a_c — each chunk an independent Rademacher block (see module docstring).
        out = None
        for c, (start, end) in enumerate(self.chunks):
            part = fn(mat[:, start:end].contiguous(), self.proj_dim, self.chunk_seed(c), self.num_sms)
            out = part if out is None else out.add_(part)
        return out

    def make_spec(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> ProjectorSpec:
        fp = param_fingerprint(named_params)
        if fp["grad_dim"] != self.grad_dim:
            raise RuntimeError(
                f"Projector grad_dim {self.grad_dim} != param fingerprint grad_dim {fp['grad_dim']}."
            )
        gpu_name = torch.cuda.get_device_name(self.device.index if self.device.index is not None else 0)
        return ProjectorSpec(
            grad_dim=self.grad_dim,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=self.proj_type,
            model_id=self.model_id,
            num_sms=self.num_sms,
            gpu_name=str(gpu_name),
            input_dtype={torch.float16: "float16", torch.float32: "float32"}[self.input_dtype],
            param_shape_sha256=fp["shape_sha256"],
            param_name_sha256=fp["name_sha256"],
            num_params=fp["num_params"],
            chunk_size=self.chunk_size,
        )

    @classmethod
    def from_spec(cls, spec: ProjectorSpec, device: str | torch.device) -> "CudaJLProjector":
        return cls(
            grad_dim=spec.grad_dim,
            proj_dim=spec.proj_dim,
            seed=spec.seed,
            device=device,
            proj_type=spec.proj_type,
            input_dtype=spec.input_dtype,
            model_id=spec.model_id,
            expected_num_sms=spec.num_sms,
            chunk_size=spec.chunk_size,
        )


def make_grad_buffer(grad_dim: int, device: str | torch.device, input_dtype: str = "float16") -> torch.Tensor:
    return torch.zeros(1, int(grad_dim), device=torch.device(device), dtype=resolve_input_dtype(input_dtype))


def flatten_grads_into(
    buffer: torch.Tensor, named_params: List[Tuple[str, torch.nn.Parameter]]
) -> torch.Tensor:
    """Copy each ``p.grad`` (zeros where None) into ``buffer`` (1, grad_dim) in order."""
    if buffer.dim() != 2 or buffer.shape[0] != 1:
        raise ValueError(f"Grad buffer must be (1, grad_dim); got {tuple(buffer.shape)}.")
    flat = buffer.view(-1)
    offset = 0
    for _, p in named_params:
        n = p.numel()
        if p.grad is None:
            flat[offset : offset + n].zero_()
        else:
            flat[offset : offset + n].copy_(p.grad.reshape(-1))
        offset += n
    if offset != flat.numel():
        raise RuntimeError(
            f"Flattened grad length {offset} != buffer width {flat.numel()}; param set changed?"
        )
    return buffer


def snapshot_params(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    device: Optional[str | torch.device] = None,
) -> List[torch.Tensor]:
    """Detached clones of the current parameter values (the Δw reference ``w₀``).

    Kept in the parameters' own dtype: the weights are the ground truth we later
    subtract from, so an exact copy makes ``w_t - w₀`` exact in fp32.
    """
    out = []
    for _, p in named_params:
        ref = p.detach()
        out.append(ref.clone() if device is None else ref.to(device=torch.device(device), copy=True))
    return out


def _check_reference(
    named_params: List[Tuple[str, torch.nn.Parameter]], reference: List[torch.Tensor]
) -> None:
    if len(reference) != len(named_params):
        raise RuntimeError(
            f"Reference has {len(reference)} tensors but the model has {len(named_params)} "
            "trainable params; the Δw reference must be a snapshot of this param set."
        )
    for (name, p), ref in zip(named_params, reference):
        if tuple(ref.shape) != tuple(p.shape):
            raise RuntimeError(
                f"Reference shape {tuple(ref.shape)} != param shape {tuple(p.shape)} for {name!r}."
            )


def _delta_fp32(
    p: torch.nn.Parameter, ref: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """``(w_t - w₀)`` for one parameter, computed in fp32 on ``device``.

    Both operands are widened to fp32 *before* subtracting: the params are bf16, and
    subtracting in bf16 would round the (small) difference back onto the bf16 grid.

    ``copy=True`` is load-bearing: ``.to()`` returns ``self`` when the dtype and device
    already match, so without it the in-place ``sub_`` would overwrite the parameter
    itself for an fp32 model.
    """
    cur = p.detach().to(device=device, dtype=torch.float32, copy=True)
    return cur.sub_(ref.to(device=device, dtype=torch.float32))


def delta_max_abs(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    reference: List[torch.Tensor],
    device: str | torch.device,
) -> float:
    """``‖w_t - w₀‖_∞`` over all trainable params (fp32), without materializing Δw."""
    _check_reference(named_params, reference)
    dev = torch.device(device)
    peak = 0.0
    for (_, p), ref in zip(named_params, reference):
        d = _delta_fp32(p, ref, dev)
        if d.numel():   # an FSDP shard of a small param can be empty
            peak = max(peak, float(d.abs_().max().item()))
    return peak


# The fast_jl kernel consumes fp16 (``matrix_a`` half fragments) and accumulates in
# fp32, so inputs must stay under fp16's 65504 max while small entries below its
# 6.1e-5 normal floor lose precision (and flush to zero below ~6e-8). Δw entries are
# far smaller than that floor early in training, so we scale the whole vector up to
# this target max before projecting and divide the (linear) projection back down.
FP16_PROJECT_TARGET_MAX = 4096.0


def delta_projection_scale(max_abs: float) -> float:
    """Power-of-two multiplier putting ``max|Δw|`` into fp16's well-conditioned range.

    A power of two makes both the scaling and the later unscaling exact in binary
    floating point, so the rescale contributes no error of its own. Returns 1.0 for a
    zero / non-finite max (nothing to project).
    """
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return 1.0
    return math.ldexp(1.0, math.floor(math.log2(FP16_PROJECT_TARGET_MAX / max_abs)))


def flatten_delta_into(
    buffer: torch.Tensor,
    named_params: List[Tuple[str, torch.nn.Parameter]],
    reference: List[torch.Tensor],
    scale: float = 1.0,
) -> torch.Tensor:
    """Copy ``scale · (w_t - w₀)`` into ``buffer`` (1, grad_dim) in param order.

    Pure observer: reads ``p.data`` and never writes to the model.
    """
    if buffer.dim() != 2 or buffer.shape[0] != 1:
        raise ValueError(f"Delta buffer must be (1, grad_dim); got {tuple(buffer.shape)}.")
    _check_reference(named_params, reference)
    dev = buffer.device
    flat = buffer.view(-1)
    offset = 0
    for (_, p), ref in zip(named_params, reference):
        n = p.numel()
        d = _delta_fp32(p, ref, dev)
        if scale != 1.0:
            d.mul_(scale)
        flat[offset : offset + n].copy_(d.reshape(-1))
        offset += n
    if offset != flat.numel():
        raise RuntimeError(
            f"Flattened delta length {offset} != buffer width {flat.numel()}; param set changed?"
        )
    return buffer


def reshard_fsdp(model: torch.nn.Module) -> None:
    """Return every FSDP2 module to its sharded state. After a forward pass FSDP2 keeps the root
    module's params unsharded -- registered as plain full tensors -- until backward, so reading the
    params then would mix full tensors with other modules' shards."""
    from torch.distributed.fsdp import FSDPModule

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def is_sharded(named_params: List[Tuple[str, torch.nn.Parameter]]) -> bool:
    """True when the params are FSDP2 DTensors (each rank holds a Shard(0) slice). A param FSDP2
    does not manage is a plain tensor, replicated on every rank; any DTensor makes it sharded.
    Call :func:`reshard_fsdp` first, or the root's unsharded params also show up as plain tensors."""
    from torch.distributed.tensor import DTensor

    return any(isinstance(p, DTensor) for _, p in named_params)


def local_shards(named_params: List[Tuple[str, torch.nn.Parameter]]) -> List[Tuple[str, torch.Tensor]]:
    """This rank's slice of each DTensor param, as a plain tensor (a view; no copy); a plain
    (replicated) param is returned whole."""
    from torch.distributed.tensor import DTensor

    return [(name, p.to_local() if isinstance(p, DTensor) else p) for name, p in named_params]


def sharded_delta_max_abs(
    named_params: List[Tuple[str, torch.nn.Parameter]], reference: List[torch.Tensor]
) -> float:
    """:func:`delta_max_abs` for FSDP2 params; ``reference`` holds this rank's shards of w₀.

    Collective: every rank must call it.
    """
    import torch.distributed as dist

    local = local_shards(named_params)
    device = local[0][1].device
    peak = torch.tensor(delta_max_abs(local, reference, device), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return float(peak.item())


def flatten_sharded_delta_into(
    buffer: Optional[torch.Tensor],
    named_params: List[Tuple[str, torch.nn.Parameter]],
    reference: List[torch.Tensor],
    scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """:func:`flatten_delta_into` for FSDP2 params: each param's Δw is all-gathered in turn
    (one full fp32 tensor at a time) and copied into ``buffer`` by the rank that holds one.

    Collective: every rank must call it; ranks that do not project pass ``buffer=None``.
    """
    from torch.distributed.tensor import DTensor

    local = local_shards(named_params)
    _check_reference(local, reference)
    flat = buffer.view(-1) if buffer is not None else None
    offset = 0
    for (_, p), (_, shard), ref in zip(named_params, local, reference):
        d = _delta_fp32(shard, ref, shard.device)
        if scale != 1.0:
            d.mul_(scale)
        if isinstance(p, DTensor):
            full = DTensor.from_local(
                d, p.device_mesh, p.placements, run_check=False, shape=p.size(), stride=p.stride()
            ).full_tensor()
        else:   # replicated: every rank already holds the whole Δw
            full = d
        if flat is not None:
            flat[offset : offset + full.numel()].copy_(full.reshape(-1))
        offset += full.numel()
        del d, full
    if flat is not None and offset != flat.numel():
        raise RuntimeError(
            f"Flattened delta length {offset} != buffer width {flat.numel()}; param set changed?"
        )
    return buffer


def assert_model_matches_spec(
    spec: ProjectorSpec, named_params: List[Tuple[str, torch.nn.Parameter]]
) -> None:
    """Hard-fail on grad_dim / param-shape mismatch; warn on name-hash mismatch."""
    fp = param_fingerprint(named_params)
    if fp["grad_dim"] != spec.grad_dim:
        raise RuntimeError(
            f"Live model grad_dim {fp['grad_dim']} != Fisher spec grad_dim {spec.grad_dim}. "
            "The projector is only valid for the model it was built on."
        )
    if fp["shape_sha256"] != spec.param_shape_sha256:
        raise RuntimeError(
            "Live model param-shape hash does not match the Fisher spec. The projector is "
            "byte-for-byte tied to the parameter shapes/order; refusing to project."
        )
    if fp["name_sha256"] != spec.param_name_sha256:
        print(
            "[fisher-overlap] WARNING: param-name hash differs from the Fisher spec "
            "(shapes match). Likely a wrapper name prefix; proceeding."
        )


def save_fisher(
    out_dir: str | Path,
    fisher: torch.Tensor,
    spec: ProjectorSpec,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``fisher.pt`` (fp32 CPU), ``projector_spec.json`` and ``fisher_meta.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_cpu = fisher.detach().to(torch.float32).cpu().contiguous()
    torch.save(f_cpu, out / "fisher.pt")
    spec.to_json(out / "projector_spec.json")
    meta: Dict[str, Any] = {
        "proj_dim": spec.proj_dim,
        "trace": float(torch.diagonal(f_cpu).sum().item()),
        "trace_over_proj_dim": float(torch.diagonal(f_cpu).sum().item()) / float(spec.proj_dim),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out / "fisher_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _resolve_fisher_file(path: str | Path) -> Tuple[Path, Path]:
    p = Path(path)
    if p.is_dir():
        fisher_file = p / "fisher.pt"
        spec_file = p / "projector_spec.json"
    else:
        fisher_file = p
        spec_file = p.parent / "projector_spec.json"
    if not fisher_file.is_file():
        raise FileNotFoundError(f"fisher.pt not found at {fisher_file}.")
    if not spec_file.is_file():
        raise FileNotFoundError(f"projector_spec.json not found at {spec_file}.")
    return fisher_file, spec_file


def load_fisher(path: str | Path) -> Tuple[torch.Tensor, ProjectorSpec]:
    """Load ``(F, spec)`` from a directory or a ``fisher.pt`` path."""
    fisher_file, spec_file = _resolve_fisher_file(path)
    fisher = torch.load(fisher_file, map_location="cpu").to(torch.float32)
    spec = ProjectorSpec.from_json(spec_file)
    if fisher.shape != (spec.proj_dim, spec.proj_dim):
        raise RuntimeError(
            f"Fisher shape {tuple(fisher.shape)} != (proj_dim, proj_dim)=({spec.proj_dim},{spec.proj_dim})."
        )
    return fisher, spec
