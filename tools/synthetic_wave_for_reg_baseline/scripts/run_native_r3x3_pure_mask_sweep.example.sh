#!/usr/bin/env bash
# Run one review-gated native R3x3 pure-mask action in a user-managed tmux session.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    cat >&2 <<'EOF'
Usage: run_native_r3x3_pure_mask_sweep.local.sh ACTION [REVIEWER_NOTE]

Run one action at a time in the same tmux window:
  validate-inputs
  prepare
  validate-coarse
  run-coarse
  evaluate-coarse
  validate-fine
  run-fine
  evaluate-fine
  validate-shortlist
  render-shortlist
  record-selections
  validate-presentation
  build-presentation

Edit the ignored local JSON after reviewing coarse metrics to add fine_sweep,
then edit it again after reviewing fine outputs to add manual_shortlist and
manual_final_selections. No action automatically launches the next action.
EOF
    exit 2
fi

: "${REPOSITORY_ROOT:?Set REPOSITORY_ROOT to the pulseq_wave_calibration checkout.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the exact user-approved run directory.}"
: "${CONDA_PROFILE:?Set CONDA_PROFILE to the host Conda initialization script.}"
: "${BART_STARTUP:?Set BART_STARTUP to the host-compatible BART startup script.}"

ACTION="$1"
REVIEWER_NOTE="${2:-}"
CONFIG="$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/configs/native_r3x3_pure_mask_sweep.local.json"
DISPATCHER="$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/run_pure_mask_rerun_stage.sh"

source "$CONDA_PROFILE"
conda activate cuda133py312-macha

case "$ACTION" in
    run-coarse|run-fine)
        source "$BART_STARTUP"
        ;;
esac

exec bash "$DISPATCHER" "$ACTION" "$CONFIG" "$OUTPUT_ROOT" "$REVIEWER_NOTE"
