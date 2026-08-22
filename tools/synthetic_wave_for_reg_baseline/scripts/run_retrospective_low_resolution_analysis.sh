#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: run_retrospective_low_resolution_analysis.sh

Configured mode (recommended):
  Set RETRO_LOW_RES_ANALYSIS_CONFIG to an ignored machine-local JSON copied
  from requirements/retrospective_low_resolution_analysis.example.json.

Historical product mode:
  If RETRO_LOW_RES_ANALYSIS_CONFIG is unset, the original R3 product analysis
  behavior remains available and requires R3_PRODUCT_ROOT.
EOF
    exit 0
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synthetic-wave-retro-low-resolution-analysis}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${RETRO_LOW_RES_ANALYSIS_CONFIG:-}" ]]; then
    CONFIG="$(realpath "$RETRO_LOW_RES_ANALYSIS_CONFIG")"
    exec python "$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/analyze_retrospective_low_resolution.py" \
        --config "$CONFIG" \
        "$@"
fi

: "${R3_PRODUCT_ROOT:?Set R3_PRODUCT_ROOT in a private local launcher.}"
DATASET_ROOT="$(realpath "$R3_PRODUCT_ROOT")"
RETRO_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2_low_resolution/retrospective_low_resolution"
PRESENTATION_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2_presentation_optimization"

exec python "$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/analyze_retrospective_low_resolution.py" \
    --review-manifest "$RETRO_ROOT/visual_review/review_manifest.json" \
    --approved-bet-mask "$PRESENTATION_ROOT/evaluation/brain_mask/reference_brain_mask.nii.gz" \
    --shared-registration "$PRESENTATION_ROOT/evaluation/volume_metrics/shared_registration.json" \
    --output-dir "$RETRO_ROOT/resolution_tradeoff_analysis" \
    "$@"
