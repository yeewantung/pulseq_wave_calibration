#!/usr/bin/env bash
# Run exactly one reviewed GRE sweep operation in the current shell or tmux session.
set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:?Set REPOSITORY_ROOT to the repository checkout.}"
CONFIG="${GRE_SWEEP_CONFIG:?Set GRE_SWEEP_CONFIG to an ignored local JSON file.}"
RUN_ROOT="${GRE_SWEEP_RUN_ROOT:?Set GRE_SWEEP_RUN_ROOT to the approved run directory.}"
OPERATION="${1:-}"
PRIOR_APPROVED_BRAIN_MASK_MANIFEST="${PRIOR_APPROVED_BRAIN_MASK_MANIFEST:-}"
EXTRA_ARGUMENTS=()
if [[ "$OPERATION" == "prepare-brain-mask" ]]; then
  [[ -n "$PRIOR_APPROVED_BRAIN_MASK_MANIFEST" ]] || {
    echo "Set PRIOR_APPROVED_BRAIN_MASK_MANIFEST for prepare-brain-mask."
    exit 2
  }
  EXTRA_ARGUMENTS+=(--brain-mask-source-manifest "$PRIOR_APPROVED_BRAIN_MASK_MANIFEST")
fi

source "$HOME/cluster/miniforge3/etc/profile.d/conda.sh"
conda activate cuda133py312-macha
source "$HOME/cluster/bart/bart_startup.sh"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gre-synthetic-wave-mpl-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPOSITORY_ROOT"

python tools/synthetic_wave_for_reg_baseline/scripts/gre_synthetic_wave_sweep.py \
  --config "$CONFIG" \
  "$OPERATION" \
  --confirm-run-root "$RUN_ROOT" \
  "${EXTRA_ARGUMENTS[@]}"
