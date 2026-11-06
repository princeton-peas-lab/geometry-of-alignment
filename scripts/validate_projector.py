"""Validate the freshly-built fast_jl projector: determinism + norm preservation.

Run on the SAME GPU model that will build/serve the Fisher (H200, num_sms=132).

    python scripts/validate_projector.py [grad_dim]

``grad_dim`` defaults to a small smoke size. Pass a model's real width to exercise the
chunked path — e.g. 8953803264 (qwen35-9b), which is 2.08x the uint32 ceiling that
``fast_jl`` indexes columns with. Norm preservation is the decisive check there: a
wrapped address would leave most of the vector unprojected and drive the ratio far
below 1, silently and without any error.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from interventions.projection import CudaJLProjector


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    dev = "cuda:0"
    grad_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    proj_dim = 2048
    gpu = torch.cuda.get_device_name(0)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU={gpu}  num_sms={num_sms}  grad_dim={grad_dim:,}  proj_dim={proj_dim}")

    proj = CudaJLProjector(grad_dim=grad_dim, proj_dim=proj_dim, seed=0, device=dev)
    print(f"chunks={len(proj.chunks)}  chunk_size={proj.chunk_size}  "
          f"spans={[e - s for s, e in proj.chunks]}")

    g = torch.randn(1, grad_dim, device=dev, dtype=torch.float16)
    v1 = proj.project(g)
    v2 = proj.project(g)
    deterministic = torch.equal(v1, v2)
    print(f"determinism: same input -> identical output: {deterministic}  (out shape {tuple(v1.shape)}, dtype {v1.dtype})")
    del g, v2

    # vector_norm accumulates in fp32 without materializing an fp32 copy of the input,
    # which at 8.95e9 entries would be a 35.8 GiB allocation on top of the 17.9 GiB input.
    def sqnorm(t: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(t.reshape(-1), dtype=torch.float32).item()) ** 2

    ratios = []
    for _ in range(8):
        x = torch.randn(1, grad_dim, device=dev, dtype=torch.float16)
        px = proj.project(x)
        ratios.append(sqnorm(px) / (proj_dim * sqnorm(x)))
        del x, px
    mean_ratio = sum(ratios) / len(ratios)
    print(f"norm preservation E[||Px||^2 / (proj_dim*||x||^2)] = {mean_ratio:.4f}  (target ~1.0)")
    print(f"per-trial ratios: {[round(r, 4) for r in ratios]}")

    ok = deterministic and (0.85 <= mean_ratio <= 1.15) and v1.shape == (1, proj_dim)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
