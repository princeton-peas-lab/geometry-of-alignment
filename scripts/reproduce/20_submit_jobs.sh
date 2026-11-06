#!/usr/bin/env bash
# LOGIN NODE. Submit the reproduction's GPU jobs into results/repro/<run>/.
#
#   artifacts  per model: refusal set -> {repr-FIM subspace, projected Fisher, think control}
#   baseline   per model: un-finetuned responses (both prompt sets) + pretrained eval losses
#   paper_fp32 paper_table1_fp32: Table 1, training seeds 42/1/2 at each model's FP32_LR (env.sh)
#   paper_lambda_fp32  paper_lambda_fp32: the intervention sweep at FP32_LR, seeds 42/1/2
#   paper_probe_fp32   paper_probe_fp32: the paper_fp32 runs with the per-step Δw-Fisher and refusal
#              log-prob monitors, training only
#   fp32       artifacts + baseline + paper_fp32 + paper_lambda_fp32 + paper_probe_fp32 (the paper)
#   paper_sweep_fp32  paper_table1_fp32: LR sweep {1e-4 ... 1e-5} in fp32, seed 42 (optional)
#   sweep      lr_sweep: 6 datasets x LR {1e-4, 5e-5, 1e-5}, LR-sweep protocol (optional)
#   gsm8k      regenerate the GSM8K SFT set with Llama-3.1-70B into the run (2 GPUs)
# Optional trailing model keys restrict the submission (default: all three models).
#
# Jobs that need an artifact wait on it (afterok) when both are submitted together; every job
# skips outputs that already exist, so re-running a stage only redoes what is missing.
# Jobs ask for < 1 h (many clusters schedule sub-hour jobs in a high-priority QOS with a per-user
# job cap, so this script feeds them in as slots free up and keeps running until the stage is
# fully queued -- run it in tmux) except the 9B subspace build, which needs ~1.5 h.
# Partition / account: export SBATCH_PARTITION / SBATCH_ACCOUNT (read by sbatch itself).
#
# Usage: bash scripts/reproduce/20_submit_jobs.sh <run-name> <stage> [model ...]
#   e.g. bash scripts/reproduce/20_submit_jobs.sh 2026-10-13 fp32
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
usage() { echo "usage: bash scripts/reproduce/20_submit_jobs.sh <run-name> artifacts|baseline|paper_sweep_fp32|paper_fp32|paper_lambda_fp32|paper_probe_fp32|sweep|fp32|gsm8k [model ...]" >&2; exit 2; }
NAME="${1:-}"; STAGE="${2:-}"
[ -n "$NAME" ] || usage
case "$STAGE" in artifacts|baseline|paper_sweep_fp32|paper_fp32|paper_lambda_fp32|paper_probe_fp32|sweep|fp32|gsm8k) ;; *) usage ;; esac
shift 2
if [ $# -gt 0 ]; then SEL=("$@"); else SEL=("${MODELS[@]}"); fi
for m in "${SEL[@]}"; do [ -n "${MODEL_SHORT[$m]:-}" ] || { echo "unknown model $m" >&2; exit 2; }; done

RUN="$REPO/results/repro/$NAME"
J=scripts/reproduce
mkdir -p "$RUN"
LOG="$RUN/submissions.log"
if [ ! -f "$RUN/meta.json" ]; then
  "$PY" - "$RUN/meta.json" <<'EOF'
import json, subprocess, sys, datetime, platform
sh = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "git_sha": sh("git", "rev-parse", "HEAD"),
           "git_dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no")),
           "host": platform.node(),
           "torch": sh(sys.executable, "-c", "import torch; print(torch.__version__)"),
           "transformers": sh(sys.executable, "-c", "import transformers; print(transformers.__version__)"),
           "vllm": sh(sys.executable, "-c", "import vllm; print(vllm.__version__)")},
          open(sys.argv[1], "w"), indent=2)
EOF
  git diff > "$RUN/working_tree.diff"
fi
echo "$(date -Is) $0 $NAME $STAGE ${SEL[*]}" >> "$LOG"

# Sub-hour jobs are fed in as queue slots free up: this script waits while MAX_QUEUED of your
# jobs are queued (counting only QOS $SUBMIT_QOS when set, e.g. your cluster's sub-hour QOS), so
# a full stage keeps running here (in tmux) until everything is in.
MAX_QUEUED="${MAX_QUEUED:-24}"
SUBMIT_QOS="${SUBMIT_QOS:-}"
POLL="${POLL:-60}"   # seconds between queue checks while waiting for a slot
queued_id() { squeue -u "$USER" -h -n "$1" -o %i 2>/dev/null | head -1; }

submit() {  # submit <job-name> [sbatch opts...] -- <script> <args...>; echoes the job id
  local name="$1"; shift
  local opts=() long=0; while [ "$1" != -- ]; do opts+=("$1"); case "$1" in --time=0[1-9]:*|--time=[1-9]*) long=1 ;; esac; shift; done; shift
  local id; id=$(queued_id "$name")
  if [ -n "$id" ]; then echo "$id"; return 0; fi            # already queued: don't duplicate
  if [ "$long" = 0 ]; then
    while [ "$(squeue -u "$USER" -h ${SUBMIT_QOS:+-q "$SUBMIT_QOS"} 2>/dev/null | wc -l)" -ge "$MAX_QUEUED" ]; do sleep "$POLL"; done
  fi
  id=$(sbatch --parsable -J "$name" "${opts[@]}" "$@")
  echo "$id $name $*" >> "$LOG"
  echo "$id"
}
# --dependency on whichever of the given jobs are still queued. A job that already left the
# queue either succeeded (nothing to wait for) or failed (afterok would never fire; the cell
# then fails fast on the missing artifact and is resubmitted once the artifact is rebuilt).
after() {
  local ids=() j
  for j in "$@"; do [ -n "$j" ] && [ -n "$(squeue -h -j "$j" 2>/dev/null)" ] && ids+=("$j"); done
  if [ ${#ids[@]} -gt 0 ]; then echo "--dependency=afterok:$(IFS=:; echo "${ids[*]}")"; fi
  return 0
}

# Completed work is not resubmitted (the jobs would skip it anyway, but would burn a queue slot).
art_done() {  # art_done <model> <stage>
  local s="${MODEL_SHORT[$1]}" a="$RUN/artifacts"
  case "$2" in
    refusal)  [ -f "$a/refusal/$1_refused_prompts_harmful_train.json" ] ;;
    subspace) [ -f "$a/repr_fim_subspace/$s/refusal_train/subspaces.pt" ] ;;
    fisher)   [ -f "$a/projected_fisher/$s/refusal_train/fisher.pt" ] ;;
    control)  [ -f "$a/projected_fisher_control/${s}_${THINK_CONTROL[$1]}/refusal_train/fisher.pt" ] ;;
  esac
}
base_done() {
  local d="$RUN/baseline/$1" ds
  [ -f "$d/refusal_test/strongreject_scores.json" ] && [ -f "$d/refusal_test/guard_labels.json" ] \
    && [ -f "$d/strongreject/strongreject_scores.json" ] || return 1
  for ds in "${LOSS_DATASETS[@]}"; do [ -f "$d/eval_loss/$ds/post_train_eval_metadata.txt" ] || return 1; done
}
cell_done() {  # cell_done <exp> <model> <dataset> <tag>
  local d="$RUN/$1/$2/$3/$4"
  [ -f "$d/strongreject_scores.json" ] && { [[ "$1" != paper_* ]] || [ -f "$d/guard_labels.json" ]; }
}

declare -A J_REFUSAL=() J_SUBSPACE=() J_FISHER=()

stage_artifacts() {
  for m in "${SEL[@]}"; do
    art_done "$m" refusal || J_REFUSAL[$m]=$(submit "art_refusal_$m" -- $J/job_artifacts.sbatch "$RUN" "$m" refusal)
    local dep; dep=$(after "${J_REFUSAL[$m]:-}")
    # Qwen3.5-9B's subspace build takes ~47 min, too close to the 55-min request.
    local st=(); [ "$m" = qwen35-9b-instruct ] && st=(--time=01:30:00)
    art_done "$m" subspace || J_SUBSPACE[$m]=$(submit "art_subspace_$m" $dep "${st[@]}" -- $J/job_artifacts.sbatch "$RUN" "$m" subspace)
    art_done "$m" fisher || J_FISHER[$m]=$(submit "art_fisher_$m" $dep -- $J/job_artifacts.sbatch "$RUN" "$m" fisher)
    if [ -n "${THINK_CONTROL[$m]:-}" ] && ! art_done "$m" control; then
      submit "art_control_$m" $dep -- $J/job_artifacts.sbatch "$RUN" "$m" control >/dev/null
    fi
    echo "artifacts $m: refusal ${J_REFUSAL[$m]:-done}, subspace ${J_SUBSPACE[$m]:-done}, fisher ${J_FISHER[$m]:-done}"
  done
}

stage_baseline() {
  for m in "${SEL[@]}"; do
    if base_done "$m"; then echo "baseline $m: done"; else echo "baseline $m: $(submit "base_$m" -- $J/job_baseline.sbatch "$RUN" "$m")"; fi
  done
}

stage_paper_sweep_fp32() {
  # The 9B's fp32 training state does not fit one H200: 2 GPUs, FSDP2 (job_cell.sbatch).
  local n=0 k=0 ds m lr
  for ds in pure_bad "${NO_PURE_BAD[@]}"; do for m in "${SEL[@]}"; do
    local res=(); [ "$m" = qwen35-9b-instruct ] && res=(--gres=gpu:2 --cpus-per-task=16 --mem=360G)
    for lr in "${PAPER_SWEEP_LRS[@]}"; do
      if cell_done paper_table1_fp32 "$m" "$ds" "lr$lr"; then k=$((k + 1)); continue; fi
      submit "f32_${m}_${ds}_$lr" "${res[@]}" -- $J/job_cell.sbatch "$RUN" paper_table1_fp32 "$m" "$ds" "$lr" 100 >/dev/null
      n=$((n + 1))
    done
  done; done
  echo "paper_sweep_fp32 ${SEL[*]} (lr ${PAPER_SWEEP_LRS[*]}): $n submitted, $k already done"
}

stage_paper_fp32() {
  local n=0 k=0 ds m s tag
  for ds in pure_bad "${NO_PURE_BAD[@]}"; do for m in "${SEL[@]}"; do
    [ -n "${FP32_LR[$m]:-}" ] || continue   # models whose fp32 LR has been chosen
    local res=(); [ "$m" = qwen35-9b-instruct ] && res=(--gres=gpu:2 --cpus-per-task=16 --mem=360G)
    for s in "${FP32_SEEDS[@]}"; do
      tag="lr${FP32_LR[$m]}"; [ "$s" = 42 ] || tag="${tag}_s$s"
      if cell_done paper_table1_fp32 "$m" "$ds" "$tag"; then k=$((k + 1)); continue; fi
      submit "fp32_${m}_${ds}_s$s" "${res[@]}" --export=ALL,SEED="$s" -- $J/job_cell.sbatch "$RUN" paper_table1_fp32 "$m" "$ds" "${FP32_LR[$m]}" 100 >/dev/null
      n=$((n + 1))
    done
  done; done
  echo "paper_fp32 (seeds ${FP32_SEEDS[*]}): $n submitted, $k already done"
}

stage_paper_lambda_fp32() {
  local n=0 k=0 ds m s lam key tag dep
  for m in "${SEL[@]}"; do
    [ -n "${FP32_LR[$m]:-}" ] || { echo "paper_lambda_fp32 $m: no FP32_LR in env.sh yet, skipped"; continue; }
    local res=(); [ "$m" = qwen35-9b-instruct ] && res=(--gres=gpu:2 --cpus-per-task=16 --mem=360G)
    for s in "${FP32_SEEDS[@]}"; do for ds in "${NO_PURE_BAD[@]}"; do for lam in "${LAMBDAS[@]}"; do
      key=$(lam_key "$lam"); tag="lam$key"; [ "$s" = 42 ] || tag="${tag}_s$s"
      if cell_done paper_lambda_fp32 "$m" "$ds" "$tag"; then k=$((k + 1)); continue; fi
      dep=$(after "${J_SUBSPACE[$m]:-$(queued_id "art_subspace_$m")}")
      submit "f32l_${m}_${ds}_${key}_s$s" $dep "${res[@]}" --export=ALL,SEED="$s" -- \
        $J/job_cell.sbatch "$RUN" paper_lambda_fp32 "$m" "$ds" "${FP32_LR[$m]}" 100 "$lam" >/dev/null
      n=$((n + 1))
    done; done; done
  done
  echo "paper_lambda_fp32 ${SEL[*]} (seeds ${FP32_SEEDS[*]}): $n submitted, $k already done"
}

stage_paper_probe_fp32() {
  local n=0 k=0 ds m s tag dep
  for m in "${SEL[@]}"; do
    [ -n "${FP32_LR[$m]:-}" ] || continue
    local res=(); [ "$m" = qwen35-9b-instruct ] && res=(--gres=gpu:2 --cpus-per-task=16 --mem=360G)
    for s in "${FP32_SEEDS[@]}"; do for ds in "${ALL_DATASETS[@]}"; do
      tag="lr${FP32_LR[$m]}"; [ "$s" = 42 ] || tag="${tag}_s$s"
      if [ -f "$RUN/paper_probe_fp32/$m/$ds/$tag/fisher_overlap_monitor.txt" ]; then k=$((k + 1)); continue; fi
      dep=$(after "${J_FISHER[$m]:-$(queued_id "art_fisher_$m")}")
      submit "probe_${m}_${ds}_s$s" $dep "${res[@]}" --export=ALL,SEED="$s" -- \
        $J/job_cell.sbatch "$RUN" paper_probe_fp32 "$m" "$ds" "${FP32_LR[$m]}" 100 >/dev/null
      n=$((n + 1))
    done; done
  done
  echo "paper_probe_fp32 ${SEL[*]} (seeds ${FP32_SEEDS[*]}): $n submitted, $k already done"
}

stage_sweep() {
  for m in "${SEL[@]}"; do
    local n=0 k=0 dep
    for ds in "${ALL_DATASETS[@]}"; do for lr in "${SWEEP_LRS[@]}"; do
      if cell_done lr_sweep "$m" "$ds" "lr$lr"; then k=$((k + 1)); continue; fi
      dep=$(after "${J_REFUSAL[$m]:-$(queued_id "art_refusal_$m")}" "${J_FISHER[$m]:-$(queued_id "art_fisher_$m")}")
      submit "sw_${m}_${ds}_$lr" $dep -- $J/job_cell.sbatch "$RUN" lr_sweep "$m" "$ds" "$lr" 100 >/dev/null
      n=$((n + 1))
    done; done
    echo "sweep $m: $n submitted, $k already done"
  done
}

case "$STAGE" in
  artifacts) stage_artifacts ;;
  baseline)  stage_baseline ;;
  paper_sweep_fp32) stage_paper_sweep_fp32 ;;
  paper_fp32) stage_paper_fp32 ;;
  paper_lambda_fp32) stage_paper_lambda_fp32 ;;
  paper_probe_fp32) stage_paper_probe_fp32 ;;
  sweep)     stage_sweep ;;
  fp32)      stage_artifacts; stage_baseline; stage_paper_fp32; stage_paper_lambda_fp32; stage_paper_probe_fp32 ;;
  gsm8k)     echo "gsm8k teacher: $(submit gsm8k_sft -- $J/job_build_gsm8k_sft.sbatch "$RUN/gsm8k_sft/llama31-70b-instruct.jsonl")" ;;
esac
echo "run dir: $RUN   (squeue -u \$USER; tail -f $RUN/*/*/*/*/status.log)"
