#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

# GPU. One per-model artifact, built on the base weights, into <run>/artifacts/:
#   refusal   the base model's responses to the harmful train prompts, kept where WildGuard
#             labels them refusals      -> refusal/<model>_refused_prompts_harmful_train.json
#   subspace  repr-FIM: top-10 eigenvectors of each target Linear's input-side Fisher over
#             those refusals            -> repr_fim_subspace/<MODEL_SHORT>/refusal_train/
#   fisher    TRAK-projected refusal Fisher (d=2048)
#                                       -> projected_fisher/<MODEL_SHORT>/refusal_train/
#   control   the Fisher rebuilt under the other Qwen family's think-template convention
#                                       -> projected_fisher_control/<MODEL_SHORT>_<mode>/refusal_train/
# subspace/fisher/control read the refusal set. Skipped when the output exists (FORCE=1 rebuilds).
#
# Usage (from the repo root):
#   sbatch -J art_<stage>_<model> scripts/reproduce/job_artifacts.sbatch <run_dir> <model> <stage>
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/reproduce/env.sh"

RUN="$1"; MODEL="$2"; STAGE="$3"
SHORT="${MODEL_SHORT[$MODEL]:?unknown model $MODEL}"
ART="$RUN/artifacts"
REFUSAL_DIR="$ART/refusal"
STATUS="$ART/status/${STAGE}_${MODEL}.log"
mapfile -t VLLM < <(vllm_args "$MODEL")
# Builders read the model's refusal set from the run, and write under <run>/artifacts.
FROM_REFUSAL=(model="$MODEL" dataset=refusal dataset.dataset_path="$REFUSAL_DIR" dataset.variant="$MODEL"
              dataset.split=train dataset.do_eval=false paths.results_root="$ART" hydra.job.chdir=false)

done_or_force() { [ -e "$1" ] && [ "${FORCE:-0}" != 1 ] && { log_stage "$STATUS" "skip $STAGE (exists: $1)"; exit 0; }; return 0; }
log_stage "$STATUS" "start node=$(hostname) job=${SLURM_JOB_ID:-local} model=$MODEL stage=$STAGE"

case "$STAGE" in
  refusal)
    OUT="$REFUSAL_DIR/${MODEL}_refused_prompts_harmful_train.json"
    done_or_force "$OUT"
    "$PY" cli/build_refusal_behavior.py model="$MODEL" model.model_name_or_path="${VLLM_BASE[$MODEL]}" \
      dataset.split=train output.variant_slug="$MODEL" output.output_file="$OUT" "${VLLM[@]}" \
      hydra.job.chdir=false
    ;;
  subspace)
    done_or_force "$ART/repr_fim_subspace/$SHORT/refusal_train/subspaces.pt"
    "$PY" cli/build_subspace.py method=repr_fim_subspace top_k_eigenvectors=10 "${FROM_REFUSAL[@]}"
    ;;
  fisher)
    done_or_force "$ART/projected_fisher/$SHORT/refusal_train/fisher.pt"
    "$PY" cli/build_projected_fisher.py "${FROM_REFUSAL[@]}"
    ;;
  control)
    MODE="${THINK_CONTROL[$MODEL]:?no think-mode control for $MODEL}"
    OUT="$ART/projected_fisher_control/${SHORT}_${MODE}/refusal_train"
    done_or_force "$OUT/fisher.pt"
    "$PY" cli/build_projected_fisher.py "${FROM_REFUSAL[@]}" +think_mode="$MODE" +output_root="$OUT"
    ;;
  *) echo "unknown stage $STAGE (refusal|subspace|fisher|control)" >&2; exit 2 ;;
esac
log_stage "$STATUS" "done $STAGE"
