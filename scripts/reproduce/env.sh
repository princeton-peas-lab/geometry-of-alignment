# Shared environment for every scripts/reproduce/ stage. SOURCE it, don't execute it.
#
# Sets REPO (the repo root, found from this file's location), cds there, pins offline
# HuggingFace mode, and defines the models, grids and per-model quirks the jobs share.
#
# Cluster-specific values are overridable from the calling shell:
#   CUDA_HOME  (default /usr/local/cuda-13.0 -- must match the torch build's CUDA major)
#   PY         (default $REPO/.venv/bin/python)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || return 1
[ -f pyproject.toml ] || { echo "env.sh: $REPO is not the repo root" >&2; return 1; }

# Compute nodes have no internet: every model/dataset must already be on disk.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$REPO/.cache/huggingface"          # repo-local, not the $HOME quota
export WANDB_MODE="${WANDB_MODE:-offline}"

# vLLM's FlashInfer JIT needs CUDA headers at startup (CPATH), and its sampler kernel
# fails to build under CUDA 13 -- the native sampler is used instead.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"   # ~/.local/bin: ninja for vLLM's JIT
export CPATH="$REPO/.venv/lib/python3.11/site-packages/nvidia/cu13/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

PY="${PY:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/logs"

# ----------------------------------------------------------------------- models
MODELS=(llama32-3b-instruct qwen3-4b-instruct qwen35-9b-instruct)
declare -A MODEL_SHORT=(
  [llama32-3b-instruct]="LLaMA-3.2-3B-Instruct"
  [qwen3-4b-instruct]="Qwen3-4B-Instruct"
  [qwen35-9b-instruct]="Qwen3.5-9B-Instruct"
)
# Weights vLLM generates from. Training uses conf/model/<model>.yaml; for Qwen3.5-9B that is
# the text-only extraction, which vLLM rejects, so vLLM reads the original VLM-layout download
# and trained checkpoints are re-wrapped into it (REWRAP_BASE) before generation.
declare -A VLLM_BASE=(
  [llama32-3b-instruct]="models/llama32-3b-instruct"
  [qwen3-4b-instruct]="models/qwen3-4b-instruct"
  [qwen35-9b-instruct]="models/qwen35-9b-instruct"
)
declare -A REWRAP_BASE=(
  [qwen35-9b-instruct]="models/qwen35-9b-instruct"
)
# Extra vLLM overrides. Qwen3.5's linear-attention prefill JIT fails under CUDA 13 and
# retries forever; the prebuilt Triton kernel avoids it.
declare -A VLLM_ARGS=(
  [qwen35-9b-instruct]="vllm.gpu_memory_utilization=0.85 vllm.max_model_len=4096 +vllm.additional_config.gdn_prefill_backend=triton"
)
# Think-mode control (projected Fisher rebuilt under the other Qwen family's convention).
declare -A THINK_CONTROL=(
  [qwen3-4b-instruct]="closed"
  [qwen35-9b-instruct]="open"
)
# ------------------------------------------------------------------------ grids
ALL_DATASETS=(alpaca samsum gsm8k risky_financial_advice bad_medical_advice pure_bad)
# pure_bad has no held-out split: no loss panel.
NO_PURE_BAD=(alpaca samsum gsm8k risky_financial_advice bad_medical_advice)
LOSS_DATASETS=(alpaca samsum gsm8k risky_financial_advice bad_medical_advice)   # gsm8k: held-out 5% of the teacher answers
LAMBDAS=(0 -0.25 -0.5 -0.75 -1)
SWEEP_LRS=(1e-4 5e-5 1e-5)                 # lr_sweep
PAPER_SWEEP_LRS=(1e-4 5e-5 3e-5 2e-5 1e-5)  # paper_sweep_fp32
# Table 1 / intervention LR (fp32): the paper's for LLaMA and Qwen3-4B; for Qwen3.5-9B the largest
# LR of the fp32 sweep at which no dataset collapses.
declare -A FP32_LR=([llama32-3b-instruct]=5e-5 [qwen3-4b-instruct]=1e-4 [qwen35-9b-instruct]=2e-5)
FP32_SEEDS=(42 1 2)

# Directory key for a lambda, identical to src/paths.py: -0.25 -> -0p25, 0 -> 0.
lam_key() { "$PY" -c 'import sys; print(f"{float(sys.argv[1]):g}".replace(".", "p"))' "$1"; }

# vLLM overrides for <model> as an array: mapfile -t ARGS < <(vllm_args <model>)
vllm_args() { local a="${VLLM_ARGS[$1]:-}"; [ -n "$a" ] && printf '%s\n' $a; return 0; }

# Append a timestamped stage marker to a persistent status file (tail it to monitor a job).
log_stage() { local f="$1"; shift; mkdir -p "$(dirname "$f")"; echo "$(date -Is) $*" | tee -a "$f"; }
