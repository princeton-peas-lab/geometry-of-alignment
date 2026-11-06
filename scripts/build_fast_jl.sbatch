#!/bin/bash
#SBATCH --job-name=fastjl_build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"  # sbatch from the repo root, as the logs/ paths above assume
[ -f pyproject.toml ] || { echo "submit from the repo root (got $PWD)" >&2; exit 1; }

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

# torch here is built with CUDA 13.0; match the toolkit major version.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # 9.0 = Hopper (H100/H200); set for your GPU
export MAX_JOBS=8

# Absolute paths: pip nvidia-*-cu13 packages carry the cuBLAS headers the compute
# node's system CUDA lacks. Relative CPATH entries do NOT resolve in pip's build
# temp dir, so anchor them to the repo root.
NV="$(pwd)/.venv/lib/python3.11/site-packages/nvidia"
CP=""
for d in "$NV"/*/include; do CP="$CP$d:"; done
export CPATH="${CP}${CUDA_HOME}/include"
echo "cublas_v2.h on CPATH:"; for d in "$NV"/*/include; do [ -f "$d/cublas_v2.h" ] && echo "  $d/cublas_v2.h"; done
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(nvcc --version | grep release)"
echo "CPATH=$CPATH"

echo "=== building fast_jl ==="
.venv/bin/pip install --no-build-isolation --no-cache-dir trak/fast_jl

echo "=== import check (torch first so libc10.so is loaded) ==="
.venv/bin/python -c "import torch; import fast_jl; print('fast_jl import OK')"

echo "=== validate projector ==="
.venv/bin/python scripts/validate_projector.py

echo "=== DONE ==="
