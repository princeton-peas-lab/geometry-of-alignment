#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# GPU. The un-finetuned model, into <run>/baseline/<model>/:
#   refusal_test/   responses to the 572 refusal harmful-test prompts (paper protocol), plus
#                   WildGuard refusal labels (guard_labels.json, the paper's refusal rate)
#   strongreject/   responses to the 313 StrongREJECT prompts (LR-sweep protocol)
#                   each scored by the StrongREJECT judge + refusal string match
#   eval_loss/<ds>/ held-out loss of the pretrained model (bf16), the intervention figure's baseline
# Each output is skipped when present (FORCE=1 redoes it).
#
# Usage (from the repo root): sbatch -J base_<model> scripts/reproduce/job_baseline.sbatch <run_dir> <model>
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/reproduce/env.sh"

RUN="$1"; MODEL="$2"
SHORT="${MODEL_SHORT[$MODEL]:?unknown model $MODEL}"
OUT="$RUN/baseline/$MODEL"
STATUS="$OUT/status.log"
mapfile -t VLLM < <(vllm_args "$MODEL")
log_stage "$STATUS" "start node=$(hostname) job=${SLURM_JOB_ID:-local} model=$MODEL"

declare -A PROMPTS=(
  [refusal_test]="dataset=refusal dataset.subset=harmful dataset.split=test dataset.variant=null"
  [strongreject]="dataset=strongreject"
)
for set in refusal_test strongreject; do
  d="$OUT/$set"
  if [ -f "$d/strongreject_scores.json" ] && [ "${FORCE:-0}" != 1 ]; then
    log_stage "$STATUS" "skip $set (exists)"; continue
  fi
  log_stage "$STATUS" "generate + score $set"
  # shellcheck disable=SC2086  # PROMPTS values are deliberately word-split into overrides
  "$PY" cli/safety_generate.py backend=offline_vllm ${PROMPTS[$set]} model="$MODEL" \
    model.model_name_or_path="${VLLM_BASE[$MODEL]}" io.output_file="$d/responses.json" \
    "${VLLM[@]}" hydra.job.chdir=false
  "$PY" cli/strongreject_score.py io.input_file="$d/responses.json" \
    io.output_file="$d/strongreject_scores.json" hydra.job.chdir=false
  "$PY" cli/refusal_strmatch_score.py dataset.name="$set" io.input_file="$d/responses.json" \
    io.output_file="$d/refusal_strmatch_labels.json" io.summary_file="$d/refusal_strmatch_summary.json" \
    hydra.job.chdir=false
done

d="$OUT/refusal_test"
if [ ! -f "$d/guard_labels.json" ] || [ "${FORCE:-0}" = 1 ]; then
  log_stage "$STATUS" "WildGuard refusal labels (refusal_test)"
  "$PY" cli/safety_guard_label.py backend=offline_vllm io.input_file="$d/responses.json" \
    io.output_file="$d/guard_labels.json" hydra.job.chdir=false
fi

for DS in "${LOSS_DATASETS[@]}"; do
  d="$OUT/eval_loss/$DS"
  if [ -f "$d/post_train_eval_metadata.txt" ] && [ "${FORCE:-0}" != 1 ]; then
    log_stage "$STATUS" "skip eval loss $DS (exists)"; continue
  fi
  log_stage "$STATUS" "pretrained eval loss $DS"
  "$PY" cli/eval.py dataset="$DS" model="$MODEL" evaluate.mode=loss max_samples=100 \
    train_args.bf16=true paths.results_root="$d/run" hydra.job.chdir=false
  mkdir -p "$d"
  cp "$d/run/offline_eval/$DS/$SHORT/baseline/post_train_eval_metadata.txt" "$d/"
done

log_stage "$STATUS" "done"
