#!/usr/bin/env bash
# LOGIN NODE (needs internet). Download the models the experiments use into
# models/<name>/, pinned to the revisions the paper results were produced with.
# Idempotent: a model whose directory already holds a config is skipped.
#
# Gated repos -- accept the terms on huggingface.co first and export HF_TOKEN:
#   meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-70B-Instruct, google/gemma-2b, allenai/wildguard
# Total ~100 GB (wildguard 28 GB, Qwen3.5-9B 19 GB + its 17 GB text-only extraction, gemma-2b
# 15 GB), plus 141 GB for the GSM8K teacher
# (Llama-3.1-70B-Instruct, only needed to regenerate data/gsm8k/llama31-70b-instruct.jsonl).
# Check `checkquota` before running.
#
# Usage: bash scripts/reproduce/00_download_models.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# local dir                 HF repo                              revision
MODELS=(
  "llama32-3b-instruct      meta-llama/Llama-3.2-3B-Instruct     0cb88a4f764b7a12671c53f0838cd831a0843b95"
  "qwen3-4b-instruct        Qwen/Qwen3-4B                        1cfa9a7208912126459214e8b04321603b3df60c"
  "qwen35-9b-instruct       Qwen/Qwen3.5-9B                      c202236235762e1c871ad0ccb60c8ee5ba337b9a"
  "wildguard                allenai/wildguard                    cbba4823f3e8020e5a74a5e29bf85072def6f2ff"
  "gemma-2b                 google/gemma-2b                      9cf48e52b224239de00d483ec8eb84fb8d0f3a3a"
  "strongreject-gemma       qylu4156/strongreject-15k-v1         4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
  "llama31-70b-instruct     meta-llama/Llama-3.1-70B-Instruct    1605565b47bb9346c5515c34102e054115b4f98b"
)

for row in "${MODELS[@]}"; do
  read -r dir repo rev <<<"$row"
  dest="models/$dir"
  if [ -f "$dest/config.json" ] || [ -f "$dest/adapter_config.json" ]; then
    echo "skip  $dest (present)"
    continue
  fi
  echo "fetch $repo@$rev -> $dest"
  # original/ holds Meta-format .pth duplicates of the weights (132 GB for the 70B); unused.
  uvx hf download "$repo" --revision "$rev" --local-dir "$dest" --exclude "original/*"
done
# Qwen3.5-9B ships as a vision-language model; training and gradients use its text-only
# weights (vLLM keeps reading the original layout). CPU-only, a few minutes.
if [ ! -f models/qwen35-9b-instruct-text/config.json ]; then
  echo "extract text-only Qwen3.5-9B -> models/qwen35-9b-instruct-text"
  "$REPO/.venv/bin/python" scripts/extract_text_checkpoint.py models/qwen35-9b-instruct models/qwen35-9b-instruct-text
fi
echo "models ready under $REPO/models"
