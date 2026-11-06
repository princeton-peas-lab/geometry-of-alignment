"""CPU-testable parts of the projected refusal-Fisher overlap feature.

The CUDA projection itself (``fast_jl``) is not exercised here; these tests cover
the config schema additions and the projector/Fisher (de)serialization helpers.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from config import register_configs
from interventions.projection import (
    FAST_JL_MAX_FEATURES,
    FP16_PROJECT_TARGET_MAX,
    ProjectorSpec,
    assert_model_matches_spec,
    chunk_bounds,
    default_chunk_size,
    delta_max_abs,
    delta_projection_scale,
    flatten_delta_into,
    flatten_grads_into,
    load_fisher,
    make_grad_buffer,
    ordered_named_grad_params,
    param_fingerprint,
    save_fisher,
    snapshot_params,
)

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def _compose(config_name="config", overrides=None):
    register_configs()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def test_fisher_overlap_monitor_defaults_and_override():
    cfg = _compose()
    assert cfg.fisher_overlap_monitor.enabled is False
    assert cfg.fisher_overlap_monitor.fisher_path is None
    assert cfg.fisher_overlap_monitor.monitor_every_steps == 1
    cfg2 = _compose(overrides=[
        "fisher_overlap_monitor.enabled=true",
        "fisher_overlap_monitor.monitor_every_steps=5",
    ])
    assert cfg2.fisher_overlap_monitor.enabled is True
    assert cfg2.fisher_overlap_monitor.monitor_every_steps == 5


def test_fisher_overlap_monitor_rejects_unknown_key():
    # The probe-based knob was removed in the train-gradient redesign; it is now unknown.
    with pytest.raises(Exception):
        _compose(overrides=["fisher_overlap_monitor.probe_num_samples=8"])


def test_overlap_metrics_quadratic_form():
    from safety.fisher_overlap import overlap_metrics

    v = torch.tensor([3.0, 4.0])              # ‖v‖² = 25
    fisher = torch.eye(2) * 3.0               # F = 3·I  => vᵀFv = 3·25 = 75, rayleigh = 3
    overlap, rayleigh, sq = overlap_metrics(v, fisher)
    assert sq == pytest.approx(25.0)
    assert overlap == pytest.approx(75.0)
    assert rayleigh == pytest.approx(3.0)
    # Anisotropic F: weights the second coordinate more.
    fisher2 = torch.diag(torch.tensor([1.0, 10.0]))   # vᵀF₂v = 9·1 + 16·10 = 169
    overlap2, rayleigh2, _ = overlap_metrics(v, fisher2)
    assert overlap2 == pytest.approx(169.0)
    assert rayleigh2 == pytest.approx(169.0 / 25.0)


def test_build_projected_fisher_schema_compose():
    cfg = _compose(
        config_name="build_projected_fisher",
        overrides=[
            "model=llama32-3b-instruct", "dataset=refusal",
            "dataset.variant=llama32-3b-instruct", "dataset.split=train", "dataset.do_eval=false",
        ],
    )
    assert cfg.proj_dim == 2048
    assert cfg.proj_type == "rademacher"
    assert cfg.proj_input_dtype == "float16"
    assert cfg.torch_dtype == "bf16"
    assert cfg.device == "cuda:0"


def _toy_named_params():
    return [
        ("layer.weight", torch.nn.Parameter(torch.zeros(3, 4))),
        ("layer.bias", torch.nn.Parameter(torch.zeros(4))),
    ]


def test_param_fingerprint_is_order_and_shape_sensitive():
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    assert fp["grad_dim"] == 12 + 4
    assert fp["num_params"] == 2
    # Reordering changes the hashes.
    fp_rev = param_fingerprint(list(reversed(nps)))
    assert fp_rev["shape_sha256"] != fp["shape_sha256"]
    assert fp_rev["grad_dim"] == fp["grad_dim"]


def test_spec_json_round_trip(tmp_path):
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    spec = ProjectorSpec(
        grad_dim=fp["grad_dim"], proj_dim=2048, seed=0, proj_type="rademacher", model_id=0,
        num_sms=132, gpu_name="H200", input_dtype="float16",
        param_shape_sha256=fp["shape_sha256"], param_name_sha256=fp["name_sha256"],
        num_params=fp["num_params"],
    )
    p = tmp_path / "projector_spec.json"
    spec.to_json(p)
    assert ProjectorSpec.from_json(p) == spec


def test_default_chunk_size_is_none_below_the_kernel_limit():
    # llama32-3b and qwen3-4b: both fit one launch, so their stored specs stay monolithic.
    assert default_chunk_size(3_212_749_824) is None
    assert default_chunk_size(4_022_468_096) is None
    assert default_chunk_size(FAST_JL_MAX_FEATURES - 1) is None


def test_default_chunk_size_splits_evenly_and_legally():
    grad_dim = 8_953_803_264  # qwen35-9b-instruct, 2.08x the uint32 ceiling
    size = default_chunk_size(grad_dim)
    assert size is not None and size < FAST_JL_MAX_FEATURES
    bounds = chunk_bounds(grad_dim, size)
    assert bounds[0][0] == 0 and bounds[-1][1] == grad_dim
    assert all(b == bounds[i + 1][0] for i, (_, b) in enumerate(bounds[:-1]))
    assert all(e - s <= size for s, e in bounds)
    # Fewest legal chunks, split evenly rather than leaving a ragged tail.
    assert len(bounds) == math.ceil(grad_dim / (FAST_JL_MAX_FEATURES - 1))


def test_chunk_bounds_refuses_a_silently_wrapping_monolithic_launch():
    with pytest.raises(ValueError, match="chunked projection"):
        chunk_bounds(FAST_JL_MAX_FEATURES, None)
    with pytest.raises(ValueError, match="chunk_size must be"):
        chunk_bounds(10**10, FAST_JL_MAX_FEATURES)


def test_chunked_projection_equals_a_blockwise_matmul():
    """``P·a = Σ_c P_c·a_c`` — the identity the chunked path relies on.

    Uses an explicit Rademacher P per chunk in place of the CUDA kernel, so the
    algebra is checked without a GPU.
    """
    torch.manual_seed(0)
    grad_dim, proj_dim, chunk = 97, 16, 20
    a = torch.randn(grad_dim, dtype=torch.float64)
    bounds = chunk_bounds(grad_dim, chunk)
    blocks = [
        torch.randint(0, 2, (proj_dim, e - s), dtype=torch.float64) * 2 - 1 for s, e in bounds
    ]
    chunked = sum(B @ a[s:e] for B, (s, e) in zip(blocks, bounds))
    monolithic = torch.cat(blocks, dim=1) @ a
    # Exact in exact arithmetic; the two differ only in fp summation order.
    assert torch.allclose(chunked, monolithic, rtol=0, atol=1e-12)


def test_spec_round_trip_preserves_chunk_size(tmp_path):
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    spec = ProjectorSpec(
        grad_dim=fp["grad_dim"], proj_dim=2048, seed=0, proj_type="rademacher", model_id=0,
        num_sms=132, gpu_name="H200", input_dtype="float16",
        param_shape_sha256=fp["shape_sha256"], param_name_sha256=fp["name_sha256"],
        num_params=fp["num_params"], chunk_size=1_790_760_653,
    )
    p = tmp_path / "projector_spec.json"
    spec.to_json(p)
    assert ProjectorSpec.from_json(p).chunk_size == 1_790_760_653
    # A spec written before chunking existed has no such key and must still load.
    legacy = json.loads(p.read_text(encoding="utf-8"))
    del legacy["chunk_size"]
    p.write_text(json.dumps(legacy), encoding="utf-8")
    assert ProjectorSpec.from_json(p).chunk_size is None


def test_flatten_grads_into_orders_and_zero_fills():
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    buf = make_grad_buffer(fp["grad_dim"], "cpu", "float16")
    nps[0][1].grad = torch.ones(3, 4)
    # second param grad left as None -> must be zero-filled
    flatten_grads_into(buf, nps)
    flat = buf.view(-1)
    assert torch.all(flat[:12] == 1)
    assert torch.all(flat[12:] == 0)


def test_snapshot_params_is_a_detached_copy():
    nps = _toy_named_params()
    ref = snapshot_params(nps)
    with torch.no_grad():
        nps[0][1].add_(1.0)
    # The snapshot must not track the live param (else Δw would always be 0).
    assert torch.all(ref[0] == 0)
    assert not ref[0].requires_grad


def test_delta_max_abs_and_flatten_delta():
    nps = _toy_named_params()
    ref = snapshot_params(nps)
    with torch.no_grad():
        nps[0][1].add_(0.25)   # weight (3,4) moves by +0.25
        nps[1][1].add_(-2.0)   # bias   (4,)  moves by -2.0
    assert delta_max_abs(nps, ref, "cpu") == pytest.approx(2.0)
    # Reading Δw must not touch the model (fp32 params are the aliasing-prone case).
    assert torch.all(nps[1][1] == -2.0)

    buf = make_grad_buffer(param_fingerprint(nps)["grad_dim"], "cpu", "float32")
    flatten_delta_into(buf, nps, ref)
    flat = buf.view(-1)
    assert torch.all(flat[:12] == 0.25)
    assert torch.all(flat[12:] == -2.0)

    # Scaling is applied elementwise, in param order.
    flatten_delta_into(buf, nps, ref, scale=8.0)
    assert torch.all(buf.view(-1)[:12] == 2.0)


def test_flatten_delta_rejects_mismatched_reference():
    nps = _toy_named_params()
    ref = snapshot_params(nps)
    buf = make_grad_buffer(param_fingerprint(nps)["grad_dim"], "cpu", "float32")
    with pytest.raises(RuntimeError):
        flatten_delta_into(buf, nps, ref[:1])                    # wrong tensor count
    with pytest.raises(RuntimeError):
        flatten_delta_into(buf, nps, [torch.zeros(3, 5), ref[1]])  # wrong shape


def test_delta_projection_scale_is_power_of_two_in_range():
    for max_abs in (1e-9, 1e-5, 3e-3, 1.0, 7.5, 1e4):
        s = delta_projection_scale(max_abs)
        assert math.log2(s) == int(math.log2(s))          # exact power of two
        scaled = max_abs * s
        assert FP16_PROJECT_TARGET_MAX / 2 <= scaled <= FP16_PROJECT_TARGET_MAX
        assert scaled < 65504.0                            # fp16 max
    assert delta_projection_scale(0.0) == 1.0
    assert delta_projection_scale(float("nan")) == 1.0


def test_delta_survives_fp16_only_after_rescaling():
    """The reason the rescale exists: raw Δw underflows the fp16 buffer."""
    nps = [("w", torch.nn.Parameter(torch.zeros(4)))]
    ref = snapshot_params(nps)
    with torch.no_grad():
        nps[0][1].add_(1e-9)          # far below fp16's 6e-8 subnormal floor
    buf = make_grad_buffer(4, "cpu", "float16")
    flatten_delta_into(buf, nps, ref)
    assert torch.all(buf == 0)        # silently destroyed without rescaling
    scale = delta_projection_scale(delta_max_abs(nps, ref, "cpu"))
    flatten_delta_into(buf, nps, ref, scale=scale)
    recovered = buf.to(torch.float32) / scale
    assert torch.allclose(recovered, torch.full_like(recovered, 1e-9), rtol=1e-3)


def test_overlap_metrics_unscales_extensive_quantities():
    from safety.fisher_overlap import overlap_metrics

    v = torch.tensor([3.0, 4.0])
    fisher = torch.eye(2) * 3.0
    base_overlap, base_rayleigh, base_sq = overlap_metrics(v, fisher)
    # Projecting a scaled input: P(s·Δ) = s·P(Δ), so passing scale must recover the
    # unscaled overlap/‖v‖², and leave the (scale-free) Rayleigh quotient untouched.
    scale = 64.0
    overlap, rayleigh, sq = overlap_metrics(v * scale, fisher, scale=scale)
    assert overlap == pytest.approx(base_overlap)
    assert sq == pytest.approx(base_sq)
    assert rayleigh == pytest.approx(base_rayleigh)


def test_assert_model_matches_spec_detects_shape_mismatch():
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    spec = ProjectorSpec(
        grad_dim=fp["grad_dim"], proj_dim=2048, seed=0, proj_type="rademacher", model_id=0,
        num_sms=132, gpu_name="H200", input_dtype="float16",
        param_shape_sha256=fp["shape_sha256"], param_name_sha256=fp["name_sha256"],
        num_params=fp["num_params"],
    )
    assert_model_matches_spec(spec, nps)  # exact match: no raise
    mutated = [("layer.weight", torch.nn.Parameter(torch.zeros(3, 5))), nps[1]]
    with pytest.raises(RuntimeError):
        assert_model_matches_spec(spec, mutated)


def test_save_and_load_fisher_round_trip(tmp_path):
    nps = _toy_named_params()
    fp = param_fingerprint(nps)
    proj_dim = 2048
    spec = ProjectorSpec(
        grad_dim=fp["grad_dim"], proj_dim=proj_dim, seed=0, proj_type="rademacher", model_id=0,
        num_sms=132, gpu_name="H200", input_dtype="float16",
        param_shape_sha256=fp["shape_sha256"], param_name_sha256=fp["name_sha256"],
        num_params=fp["num_params"],
    )
    f = torch.eye(proj_dim) * 3.0
    out = save_fisher(tmp_path / "fish", f, spec, extra_meta={"num_samples": 7})
    loaded, spec2 = load_fisher(out)
    assert spec2 == spec
    assert loaded.shape == (proj_dim, proj_dim)
    assert torch.allclose(torch.diagonal(loaded), torch.full((proj_dim,), 3.0))
    import json
    meta = json.loads((out / "fisher_meta.json").read_text())
    assert meta["num_samples"] == 7
    assert meta["trace"] == pytest.approx(3.0 * proj_dim)


def test_ordered_named_grad_params_skips_frozen():
    m = torch.nn.Linear(4, 2)
    m.bias.requires_grad_(False)
    names = [n for n, _ in ordered_named_grad_params(m)]
    assert "weight" in names and "bias" not in names


def _sharded_delta_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from interventions.projection import (
        flatten_sharded_delta_into,
        local_shards,
        sharded_delta_max_abs,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)   # identical full tensors on every rank, then sharded
        w0 = [torch.randn(5, 3), torch.randn(1, 4), torch.randn(3, 2)]   # 5 and 1 rows: uneven shards
        w1 = [w + 1e-3 * torch.randn_like(w) for w in w0]
        # p2 stays a plain tensor, replicated on every rank (a param FSDP2 does not manage)
        shard = lambda w, i: distribute_tensor(w, mesh, [Shard(0)]) if i < 2 else w.clone()
        params = [(f"p{i}", torch.nn.Parameter(shard(w, i))) for i, w in enumerate(w1)]
        ref = [t.detach().clone() for _, t in local_shards([(n, shard(w, i)) for i, (n, w) in enumerate(zip("abc", w0))])]
        peak = sharded_delta_max_abs(params, ref)
        buf = make_grad_buffer(sum(w.numel() for w in w0), "cpu", "float32") if rank == 0 else None
        flatten_sharded_delta_into(buf, params, ref, scale=4.0)
        if rank == 0:
            full = [(f"p{i}", torch.nn.Parameter(w)) for i, w in enumerate(w1)]
            expect = make_grad_buffer(buf.shape[1], "cpu", "float32")
            flatten_delta_into(expect, full, w0, scale=4.0)
            torch.save({"peak": peak, "expect_peak": delta_max_abs(full, w0, "cpu"),
                        "buf": buf, "expect": expect}, f"{out_dir}/out.pt")
    finally:
        dist.destroy_process_group()


def test_sharded_delta_matches_single_process(tmp_path):
    """FSDP2 path: Shard(0) DTensors (and a replicated plain param) over 2 gloo ranks give the
    same Δw max and flattened Δw."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_sharded_delta_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    out = torch.load(tmp_path / "out.pt")
    assert out["peak"] == out["expect_peak"] > 0
    assert torch.equal(out["buf"], out["expect"])


def _reshard_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from interventions.projection import reshard_fsdp

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        fully_shard(model[0], mesh=mesh)
        fully_shard(model, mesh=mesh)   # the root owns the LayerNorm
        model(torch.randn(2, 4))   # the root stays unsharded after forward
        before = {n: isinstance(p, DTensor) for n, p in model.named_parameters()}
        reshard_fsdp(model)
        after = {n: isinstance(p, DTensor) for n, p in model.named_parameters()}
        if rank == 0:
            torch.save({"before": before, "after": after}, f"{out_dir}/out.pt")
    finally:
        dist.destroy_process_group()


def test_reshard_fsdp_reshards_root_after_forward(tmp_path):
    """After a forward, FSDP2 leaves the root module's params as plain full tensors (the 9B
    probe's model.norm.weight); reshard_fsdp puts every param back as a DTensor shard."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_reshard_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    out = torch.load(tmp_path / "out.pt")
    assert not all(out["before"].values())   # the failure mode is real
    assert all(out["after"].values())
