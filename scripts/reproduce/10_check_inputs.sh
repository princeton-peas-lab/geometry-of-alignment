#!/usr/bin/env bash
# LOGIN NODE. Verify every input the paper jobs read is on disk BEFORE spending GPU time
# (compute nodes have no internet, so a missing file fails inside the allocation).
# Prints OK / MISSING per input with where to get it; exits 1 if anything is missing.
#
# Usage: bash scripts/reproduce/10_check_inputs.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

missing=0
check() {  # check <path> <how to obtain>
  if [ -e "$1" ]; then printf 'OK       %s\n' "$1"
  else printf 'MISSING  %s\n         -> %s\n' "$1" "$2"; missing=$((missing + 1)); fi
}

echo "== environment"
check "$PY" "uv venv .venv --python 3.11 --seed && uv pip install -e '.[vllm,safety]'  (see README)"
"$PY" -c "import torch, transformers, trl, vllm" 2>/dev/null \
  && echo "OK       python imports (torch, transformers, trl, vllm)" \
  || { echo "MISSING  python imports -> uv pip install -e '.[vllm,safety]'"; missing=$((missing + 1)); }
"$PY" -c "import torch, fast_jl" 2>/dev/null \
  && echo "OK       fast_jl (TRAK CUDA projector: projected Fisher + Δw monitor)" \
  || { echo "MISSING  fast_jl -> sbatch scripts/build_fast_jl.sbatch (builds it on a GPU node)"; missing=$((missing + 1)); }

echo "== models (bash scripts/reproduce/00_download_models.sh)"
for m in llama32-3b-instruct qwen3-4b-instruct qwen35-9b-instruct qwen35-9b-instruct-text wildguard gemma-2b; do
  check "models/$m/config.json" "bash scripts/reproduce/00_download_models.sh"
done
check "models/strongreject-gemma/adapter_config.json" "bash scripts/reproduce/00_download_models.sh"

echo "== finetuning data"
check data/alpaca          "uvx hf download tatsu-lab/alpaca --repo-type dataset --revision dce01c9b08f87459cf36a430d809084718273017 --local-dir data/alpaca"
check data/samsum/train.csv "uvx hf download knkarthick/samsum --repo-type dataset --revision 6b929ff10edec703164e3ddb2e94aae058c9ab5f --local-dir data/samsum"
check data/gsm8k/main      "uvx hf download openai/gsm8k --repo-type dataset --revision 740312add88f781978c0658806c59bc2815b9866 --local-dir data/gsm8k"
check data/gsm8k/llama31-70b-instruct.jsonl "tracked in git; to regenerate: sbatch scripts/reproduce/job_build_gsm8k_sft.sbatch data/gsm8k/llama31-70b-instruct.jsonl (needs models/llama31-70b-instruct)"
check data/emergent_misalignment/risky_financial_advice.jsonl "tracked in git (see data/README.md)"
check data/emergent_misalignment/bad_medical_advice.jsonl     "tracked in git (see data/README.md)"
check data/pure_bad_100.jsonl "Pure Bad is not publicly released: contact Qi et al. 2023 (arXiv:2310.03693) or build a comparable set, see data/README.md"

echo "== refusal and safety prompts (tracked in git; sources in data/README.md)"
check data/refusal/harmful_train.json "tracked in git (see data/README.md)"
check data/refusal/harmful_test.json  "tracked in git (see data/README.md)"
check data/strongreject.csv           "tracked in git (see data/README.md)"

echo
if [ "$missing" -gt 0 ]; then echo "$missing input(s) missing"; exit 1; fi
echo "all inputs present"
