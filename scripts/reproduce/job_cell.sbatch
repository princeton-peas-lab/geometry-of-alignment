#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# GPU. One finetuning cell: train <model> on <dataset>, then generate responses from the
# checkpoint and score them. Everything lands in the cell directory
#   <run>/<experiment>/<model>/<dataset>/<tag>/      tag = lr<lr>  or  lam<key>
#     cell.env                         the exact parameters of this cell
#     post_train_eval_metadata.txt     held-out eval loss (datasets with a loss split)
#     fisher_overlap_monitor.txt       per-step Δw-Fisher overlap   (paper_probe_*, lr_sweep)
#     safety_monitor.txt               per-step refusal log-prob    (paper_probe_*, lr_sweep)
#     intervention_projection_monitor.txt                           (lambda cells outside paper_*)
#     responses.json, strongreject_scores.json, refusal_strmatch_summary.json
#     guard_labels.json                WildGuard refusal labels      (paper_*: the paper's refusal rate)
#     status.log                       timestamped stage markers -- tail this to monitor
# (tag gets a _s<seed> suffix when SEED is set to anything but the default 42)
#     run/                             the trainer's own tree (checkpoint deleted once scored)
#
# The experiment name selects the protocol (fp32 if it ends in _fp32, the paper's precision;
# otherwise bf16. vLLM always generates in bf16):
#   paper_*     AdamW beta2 0.999, linear LR, first-N training rows,
#               responses to the 572 refusal harmful-test prompts
#   paper_probe_*  the paper protocol with the Δw-Fisher and refusal log-prob monitors, training
#               only: the same runs as paper_table1_* (same LR / seed / rows), which score them
#   otherwise   AdamW beta2 0.95, cosine LR, seeded random training rows, Δw-Fisher and
#               refusal log-prob monitors, responses to the 313 StrongREJECT prompts
# Common: 1 epoch, batch 2 x grad-accum 2, max length 4096, completion-only loss, no warmup.
# Submitted with --gres=gpu:2, training runs FSDP2 over both GPUs (torchrun) at per-GPU batch 1,
# so the global batch stays 4 -- this is how the 9B fits in fp32. Generation uses GPU 0.
# With <lambda>: repr-FIM intervention x <- x + lambda * U U^T x on every target Linear.
#
# Resumable (each stage skipped when its output exists; FORCE=1 redoes all).
# KEEP_CHECKPOINTS=1 keeps the trained weights. SEED=<n> sets the training seed (default 42).
#
# Usage (from the repo root):
#   sbatch -J <name> scripts/reproduce/job_cell.sbatch <run_dir> <experiment> <model> <dataset> <lr> <samples> [lambda]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/reproduce/env.sh"

RUN="$1"; EXP="$2"; MODEL="$3"; DS="$4"; LR="$5"; SAMPLES="$6"; LAM="${7:-}"
SHORT="${MODEL_SHORT[$MODEL]:?unknown model $MODEL}"
ART="$RUN/artifacts"
if [ -n "$LAM" ]; then TAG="lam$(lam_key "$LAM")"; else TAG="lr$LR"; fi
SEED="${SEED:-42}"; [ "$SEED" = 42 ] || TAG="${TAG}_s$SEED"
CELL="$RUN/$EXP/$MODEL/$DS/$TAG"
STATUS="$CELL/status.log"
mkdir -p "$CELL"
mapfile -t VLLM < <(vllm_args "$MODEL")

TRAIN=(dataset="$DS" model="$MODEL" max_samples="$SAMPLES" paths.results_root="$CELL/run"
       train_args.learning_rate="$LR" seed="$SEED"
       train_args.save_strategy=no train_args.save_total_limit=0 hydra.job.chdir=false)
if [ "$DS" = gsm8k ]; then TRAIN+=(evaluate.mode=loss); fi   # loss on the held-out 5% of teacher answers, like every dataset (accuracy not reported)
case "$EXP" in
  *_fp32) PRECISION=fp32;  TRAIN+=(train_args.bf16=false) ;;
  *)      PRECISION=bf16;  TRAIN+=(train_args.bf16=true) ;;
esac
NGPU="${SLURM_GPUS_ON_NODE:-1}"
TRAIN_CMD=("$PY")
case "$NGPU" in
  1) ;;
  2) TRAIN+=(train_args.fsdp=true train_args.per_device_train_batch_size=1)
     TRAIN_CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node=2) ;;
  *) echo "job_cell supports 1 or 2 GPUs, got $NGPU" >&2; exit 1 ;;
esac

add_monitors() {  # the per-step Δw-Fisher overlap and refusal log-prob monitors
  FISHER="$ART/projected_fisher/$SHORT/refusal_train/fisher.pt"
  [ -f "$FISHER" ] || { echo "missing $FISHER -- run the artifacts stage first" >&2; exit 1; }
  TRAIN+=(safety_monitor.enabled=true safety_monitor.batch_size=8 safety_monitor.dataset_path="$ART/refusal"
          fisher_overlap_monitor.enabled=true fisher_overlap_monitor.fisher_path="$FISHER"
          fisher_overlap_monitor.monitor_every_steps="${MONITOR_EVERY:-1}")
}
PROBE_ONLY=0
case "$EXP" in
  paper_probe_*)
    add_monitors; PROBE_ONLY=1
    ;;
  paper_*)
    PROMPTS=(dataset=refusal dataset.subset=harmful dataset.split=test dataset.variant=null)
    STRMATCH_SET=refusal_test
    ;;
  *)
    TRAIN+=(max_samples_seed=0 train_args.lr_scheduler_type=cosine train_args.adam_beta2=0.95)
    add_monitors
    PROMPTS=(dataset=strongreject)
    STRMATCH_SET=strongreject
    ;;
esac
if [ -n "$LAM" ]; then
  SUBSPACE="$ART/repr_fim_subspace/$SHORT/refusal_train"
  [ -f "$SUBSPACE/subspaces.pt" ] || { echo "missing $SUBSPACE -- run the artifacts stage first" >&2; exit 1; }
  TRAIN+=(experiment=finetune_repr_fim_subspace subspace_intervention.enabled=true
          subspace_intervention.method=repr_fim_subspace subspace_intervention.lambda_scale="$LAM"
          subspace_intervention.intervention_root="$SUBSPACE")
  if [[ "$EXP" != paper_* ]]; then TRAIN+=(subspace_intervention.monitor_projection=true); fi
fi

printf 'experiment=%s\nmodel=%s\ndataset=%s\nlr=%s\nsamples=%s\nlambda=%s\nprecision=%s\nseed=%s\ngpus=%s\ngit=%s\n' \
  "$EXP" "$MODEL" "$DS" "$LR" "$SAMPLES" "${LAM:-none}" "$PRECISION" "$SEED" "$NGPU" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" > "$CELL/cell.env"
log_stage "$STATUS" "start node=$(hostname) job=${SLURM_JOB_ID:-local} $EXP $MODEL $DS $TAG"

# The paper's refusal rate (Table 1 and the intervention figure) is WildGuard's; its cells need guard labels.
NEED_GUARD=0; [[ "$EXP" == paper_* ]] && NEED_GUARD=1
if [ "$PROBE_ONLY" = 1 ] && [ -f "$CELL/fisher_overlap_monitor.txt" ] && [ "${FORCE:-0}" != 1 ]; then
  log_stage "$STATUS" "skip (monitor logs exist in $CELL)"; exit 0
fi
if [ -f "$CELL/strongreject_scores.json" ] && { [ "$NEED_GUARD" = 0 ] || [ -f "$CELL/guard_labels.json" ]; } \
   && [ "${FORCE:-0}" != 1 ]; then
  log_stage "$STATUS" "skip (outputs exist in $CELL)"; exit 0
fi

# ------------------------------------------------------------------ 1. finetune
# A fresh cell has no run/ yet; find's non-zero exit must not abort the job under pipefail.
find_ckpt() { find "$CELL/run" -path '*finetuning*' -type f -name config.json -printf '%h\n' 2>/dev/null | head -1 || true; }
CKPT="$(find_ckpt)"
if [ -n "$CKPT" ] && [ "${FORCE:-0}" != 1 ]; then
  log_stage "$STATUS" "skip finetune (checkpoint exists: $CKPT)"
elif [ -f "$CELL/responses.json" ] && [ "${FORCE:-0}" != 1 ]; then
  log_stage "$STATUS" "skip finetune (responses exist)"
else
  log_stage "$STATUS" "finetune"
  "${TRAIN_CMD[@]}" cli/train.py "${TRAIN[@]}"
  CKPT="$(find_ckpt)"
  [ -n "$CKPT" ] || { echo "no checkpoint under $CELL/run" >&2; exit 1; }
  for f in post_train_eval_metadata.txt fisher_overlap_monitor.txt safety_monitor.txt intervention_projection_monitor.txt; do
    src="$(find "$CELL/run" -name "$f" -print -quit)"
    if [ -n "$src" ]; then cp -f "$src" "$CELL/"; fi
  done
fi
if [ "$PROBE_ONLY" = 1 ]; then
  find "$CELL/run" -maxdepth 1 -type d -name 'finetuning*' -exec rm -rf {} +
  log_stage "$STATUS" "deleted checkpoint weights"; log_stage "$STATUS" "done"; exit 0
fi

# ------------------------------------------------------------- 2. generate + score
if [ -f "$CELL/responses.json" ] && [ "${FORCE:-0}" != 1 ]; then
  log_stage "$STATUS" "skip generation (exists)"
else
  GEN_MODEL="$CKPT"
  if [ -n "${REWRAP_BASE[$MODEL]:-}" ]; then
    log_stage "$STATUS" "re-wrap checkpoint for vLLM"
    "$PY" scripts/rewrap_vlm_checkpoint.py "$CKPT" "${REWRAP_BASE[$MODEL]}" "$CELL/run/vlm"
    GEN_MODEL="$CELL/run/vlm"
  fi
  log_stage "$STATUS" "generate responses"
  "$PY" cli/safety_generate.py backend=offline_vllm "${PROMPTS[@]}" model="$MODEL" \
    model.model_name_or_path="$GEN_MODEL" io.output_file="$CELL/responses.json" "${VLLM[@]}" \
    hydra.job.chdir=false
fi
if [ ! -f "$CELL/strongreject_scores.json" ] || [ "${FORCE:-0}" = 1 ]; then
  log_stage "$STATUS" "score"
  "$PY" cli/strongreject_score.py io.input_file="$CELL/responses.json" \
    io.output_file="$CELL/strongreject_scores.json" hydra.job.chdir=false
  "$PY" cli/refusal_strmatch_score.py dataset.name="$STRMATCH_SET" io.input_file="$CELL/responses.json" \
    io.output_file="$CELL/refusal_strmatch_labels.json" io.summary_file="$CELL/refusal_strmatch_summary.json" \
    hydra.job.chdir=false
fi
if [ "$NEED_GUARD" = 1 ] && { [ ! -f "$CELL/guard_labels.json" ] || [ "${FORCE:-0}" = 1 ]; }; then
  log_stage "$STATUS" "WildGuard refusal labels"
  "$PY" cli/safety_guard_label.py backend=offline_vllm io.input_file="$CELL/responses.json" \
    io.output_file="$CELL/guard_labels.json" hydra.job.chdir=false
fi

if [ "${KEEP_CHECKPOINTS:-0}" != 1 ]; then
  rm -rf "$CELL/run/vlm"
  find "$CELL/run" -maxdepth 1 -type d -name 'finetuning*' -exec rm -rf {} +
  log_stage "$STATUS" "deleted checkpoint weights"
fi
log_stage "$STATUS" "done"
