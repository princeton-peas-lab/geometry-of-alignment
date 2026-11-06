#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# GPU x2. Regenerate the GSM8K SFT set: Llama-3.1-70B-Instruct answers every GSM8K train
# question greedily; rows whose parsed "#### <answer>" matches the reference are marked
# correct and are what GSM8K finetunes train on.
#
# Writes OUT (default data/gsm8k/llama31-70b-instruct.regenerated.jsonl) and refuses to
# overwrite an existing file. To make a regenerated set the one training uses, point
# dataset.sft_jsonl at it (or replace data/gsm8k/llama31-70b-instruct.jsonl deliberately).
#
# Usage (from the repo root): sbatch -J gsm8k_sft scripts/reproduce/job_build_gsm8k_sft.sbatch [OUT]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/reproduce/env.sh"
OUT="${1:-data/gsm8k/llama31-70b-instruct.regenerated.jsonl}"
STATUS="$REPO/logs/reproduce/gsm8k_sft.status"
log_stage "$STATUS" "start node=$(hostname) job=${SLURM_JOB_ID:-local} out=$OUT"
"$PY" cli/build_gsm8k_sft.py io.output_file="$OUT" hydra.job.chdir=false
log_stage "$STATUS" "done $OUT"
