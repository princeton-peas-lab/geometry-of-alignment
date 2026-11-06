#!/usr/bin/env bash
# LOGIN NODE (CPU only). Print every reproduced table from a run's saved outputs and save each
# to <run>/tables/<name>.txt. Tables print only for the experiments the run has; cells whose jobs
# have not finished print as "--".
#
#   The paper (fp32, seeds 42/1/2)
#   main_results.txt              Table 1 + intervention figure     analysis/main_results.py
#   deltaw_overlap.txt            Δw-FIM overlap (appendix)         analysis/deltaw_overlap.py
#   lowrank_projected_fisher.txt  FIM spectrum, all parameters      analysis/lowrank_projected_fisher.py
#   lowrank_repr_fim.txt          FIM spectrum, per-layer repr.     analysis/lowrank_repr_fim.py
#   Optional (scripts/reproduce/README.md, "Additional experiments")
#   table1_lr_sweep_fp32.txt      fp32 LR sweep + LR vote           analysis/table1_lr_sweep.py
#   ablation_lr_sweep.txt         LR sweep + Δw probe + LR vote     analysis/ablation_lr_sweep.py
#
# Usage: bash scripts/reproduce/30_analyze.sh <run-name>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NAME="${1:?usage: 30_analyze.sh <run-name>}"
RUN="$REPO/results/repro/$NAME"
[ -d "$RUN" ] || { echo "no run at $RUN" >&2; exit 1; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"   # unbounded threads get killed on the login node
T="$RUN/tables"; mkdir -p "$T"

show() { local out="$T/$1.txt"; shift; echo "=================== $(basename "$out")"; "$PY" "$@" 2>&1 | tee "$out"; echo; }

shopt -s nullglob
fishers=("$RUN"/artifacts/projected_fisher/*/refusal_train/fisher.pt "$RUN"/artifacts/projected_fisher_control/*/refusal_train/fisher.pt)
subspaces=("$RUN"/artifacts/repr_fim_subspace/*/refusal_train)
[ -d "$RUN/paper_table1_fp32" ] && show main_results analysis/main_results.py --run "$RUN"
[ -d "$RUN/paper_probe_fp32" ] && show deltaw_overlap analysis/deltaw_overlap.py --run "$RUN"
[ ${#fishers[@]} -gt 0 ] && show lowrank_projected_fisher analysis/lowrank_projected_fisher.py "${fishers[@]}"
[ ${#subspaces[@]} -gt 0 ] && show lowrank_repr_fim analysis/lowrank_repr_fim.py "${subspaces[@]}"
[ -d "$RUN/paper_table1_fp32" ] && show table1_lr_sweep_fp32 analysis/table1_lr_sweep.py --run "$RUN"

[ -d "$RUN/lr_sweep" ] && show ablation_lr_sweep analysis/ablation_lr_sweep.py --run "$RUN"
echo "tables saved under $T"
