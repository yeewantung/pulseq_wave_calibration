#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: run_retrospective_low_resolution_review.sh [review options]

Configured mode (recommended):
  Set RETRO_LOW_RES_REVIEW_CONFIG to an ignored machine-local JSON copied from
  requirements/retrospective_low_resolution_review.example.json.

Historical product mode:
  If RETRO_LOW_RES_REVIEW_CONFIG is unset, the original R3 product GRAPPA and
  corrected-LLR review behavior remains available and requires R3_PRODUCT_ROOT,
  R3_TWIX, and R3_SEQUENCE.
EOF
    exit 0
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synthetic-wave-retro-low-resolution-review}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${RETRO_LOW_RES_REVIEW_CONFIG:-}" ]]; then
    CONFIG="$(realpath "$RETRO_LOW_RES_REVIEW_CONFIG")"
    exec python "$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/review_retrospective_low_resolution.py" \
        --config "$CONFIG" \
        "$@"
fi

: "${R3_PRODUCT_ROOT:?Set R3_PRODUCT_ROOT in a private local launcher.}"
: "${R3_TWIX:?Set R3_TWIX in a private local launcher.}"
: "${R3_SEQUENCE:?Set R3_SEQUENCE in a private local launcher.}"
DATASET_ROOT="$(realpath "$R3_PRODUCT_ROOT")"
RETRO_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2_low_resolution/retrospective_low_resolution"
SOURCE_RUN="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2_presentation_optimization/regularization/llr_block-8_lambda-2e-5"
REFERENCE_ROOT="$RETRO_ROOT/full_resolution_reference"
REFERENCE_SUB="sub-20260817product-r3x2-low-resolution-reference"
REFERENCE_NIFTI="$REFERENCE_ROOT/$REFERENCE_SUB/${REFERENCE_SUB}_part-mag_BARTWaveRegularized.nii.gz"
REFERENCE_PHASE="$REFERENCE_ROOT/$REFERENCE_SUB/${REFERENCE_SUB}_part-phase_BARTWaveRegularized.nii.gz"

# Re-export the existing full-resolution BART result with the corrected shared
# NIfTI producer. This performs no reconstruction and leaves the historical
# presentation output untouched.
if [[ ! -f "$REFERENCE_NIFTI" && ! -e "$REFERENCE_ROOT" ]]; then
    python "$REPOSITORY_ROOT/external/wave-mprage/recon/bart/wave_to_nifti.py" \
        --bart-input-dir "$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2/bart_inputs" \
        --bart-output-dir "$SOURCE_RUN/bart" \
        --twix "$R3_TWIX" \
        --seq "$R3_SEQUENCE" \
        --out "$REFERENCE_ROOT" \
        --save-phase \
        --nifti-sub "20260817product-r3x2-low-resolution-reference" \
        --nifti-suffix "BARTWaveRegularized"
fi

if [[ ! -f "$REFERENCE_NIFTI" || ! -f "$REFERENCE_PHASE" ]]; then
    echo "Error: canonical full-resolution reference export is incomplete: $REFERENCE_ROOT" >&2
    exit 2
fi

exec python "$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/review_retrospective_low_resolution.py" \
    --batch-manifest "$RETRO_ROOT/batch_manifest.json" \
    --grappa-nifti "$DATASET_ROOT/grappa_5x5x5_ncc12/grappa_5x5x5_ncc12_rss_ras.nii.gz" \
    --full-resolution-nifti "$REFERENCE_NIFTI" \
    --output-dir "$RETRO_ROOT/visual_review" \
    "$@"
