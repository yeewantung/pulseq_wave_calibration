#!/usr/bin/env bash
# Run one reviewed native-R3x3 GRE stage in the current tmux session.
set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:?Set REPOSITORY_ROOT to the repository checkout.}"
CONFIG="${GRE_R3X3_CONFIG:?Set GRE_R3X3_CONFIG to an ignored local JSON file.}"
REFINEMENT="${GRE_R3X3_REFINEMENT:?Set GRE_R3X3_REFINEMENT to an ignored local JSON file.}"
RUN_ROOT="${GRE_R3X3_RUN_ROOT:?Set GRE_R3X3_RUN_ROOT to the approved run directory.}"
STAGE="${1:-}"
REVIEWER="${2:-${USER:-}}"

source "$HOME/cluster/miniforge3/etc/profile.d/conda.sh"
conda activate cuda133py312-macha
source "$HOME/cluster/bart/bart_startup.sh"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gre-r3x3-mpl-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPOSITORY_ROOT"

SCRIPT=(python tools/synthetic_wave_for_reg_baseline/scripts/gre_synthetic_wave_sweep.py --config "$CONFIG")
WRITE=(--confirm-run-root "$RUN_ROOT")
case "$STAGE" in
  validate) COMMAND=("${SCRIPT[@]}" validate-config) ;;
  validate-reused-inputs) COMMAND=("${SCRIPT[@]}" validate-reused-inputs) ;;
  prepare-case) COMMAND=("${SCRIPT[@]}" prepare-reused-r3x3 "${WRITE[@]}") ;;
  coarse-check) COMMAND=("${SCRIPT[@]}" reconstruct "${WRITE[@]}" --sweep coarse --validate-only) ;;
  coarse) COMMAND=("${SCRIPT[@]}" reconstruct "${WRITE[@]}" --sweep coarse --resume) ;;
  coarse-nifti) COMMAND=("${SCRIPT[@]}" export-nifti "${WRITE[@]}" --sweep coarse) ;;
  coarse-evaluate) COMMAND=("${SCRIPT[@]}" evaluate "${WRITE[@]}" --sweep coarse) ;;
  coarse-plot) COMMAND=("${SCRIPT[@]}" plot "${WRITE[@]}" --sweep coarse) ;;
  coarse-shared-evaluate) COMMAND=("${SCRIPT[@]}" evaluate-shared-lambda "${WRITE[@]}" --sweep coarse) ;;
  coarse-shared-plot) COMMAND=("${SCRIPT[@]}" plot-shared-lambda "${WRITE[@]}" --sweep coarse) ;;
  record-refinement)
    [[ -n "$REVIEWER" ]] || { echo "A reviewer is required."; exit 2; }
    COMMAND=("${SCRIPT[@]}" record-refinement "${WRITE[@]}" --review "$REFINEMENT" --reviewer "$REVIEWER")
    ;;
  fine-check) COMMAND=("${SCRIPT[@]}" reconstruct "${WRITE[@]}" --sweep fine --validate-only) ;;
  fine) COMMAND=("${SCRIPT[@]}" reconstruct "${WRITE[@]}" --sweep fine --resume) ;;
  fine-nifti) COMMAND=("${SCRIPT[@]}" export-nifti "${WRITE[@]}" --sweep fine) ;;
  fine-evaluate) COMMAND=("${SCRIPT[@]}" evaluate "${WRITE[@]}" --sweep fine) ;;
  fine-plot) COMMAND=("${SCRIPT[@]}" plot "${WRITE[@]}" --sweep fine) ;;
  fine-shared-evaluate) COMMAND=("${SCRIPT[@]}" evaluate-shared-lambda "${WRITE[@]}" --sweep fine) ;;
  fine-shared-plot) COMMAND=("${SCRIPT[@]}" plot-shared-lambda "${WRITE[@]}" --sweep fine) ;;
  *)
    echo "Unknown stage: $STAGE"
    echo "Stages: validate validate-reused-inputs prepare-case coarse-check coarse coarse-nifti coarse-evaluate coarse-plot coarse-shared-evaluate coarse-shared-plot record-refinement fine-check fine fine-nifti fine-evaluate fine-plot fine-shared-evaluate fine-shared-plot"
    exit 2
    ;;
esac

read -r -p "Run native-R3x3 GRE stage '$STAGE' now? [y/N] " ANSWER
[[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]] || exit 0
"${COMMAND[@]}"
