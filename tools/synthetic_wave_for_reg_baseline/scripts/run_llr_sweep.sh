#!/usr/bin/env bash
# Run a compact block-8 LLR sweep from the approved crop-0.6 reconstruction.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${R1_SYNTHETIC_WAVE_ROOT:?Set R1_SYNTHETIC_WAVE_ROOT to the private dataset output root.}"
DATASET_ROOT="$(realpath -m "$R1_SYNTHETIC_WAVE_ROOT")"
DEFAULT_LAMBDA_ZERO_MANIFEST="$DATASET_ROOT/reconstructions/synthetic_wave/ecalib_crop-0p6_lambda0/manifest.json"

if [[ "${1:-}" != "--confirm-crop-0p6-reviewed" || $# -gt 2 ]]; then
    echo "Usage: $0 --confirm-crop-0p6-reviewed [lambda_zero_manifest.json]" >&2
    exit 2
fi

LAMBDA_ZERO_MANIFEST="${2:-$DEFAULT_LAMBDA_ZERO_MANIFEST}"
OUTPUT_ROOT="$DATASET_ROOT/reconstructions/synthetic_wave/llr_block-8_sweep_ecalib_crop-0p6"
NATIVE_ZERO_MANIFEST="$DATASET_ROOT/reconstructions/synthetic_wave/wavelet_sweep_ecalib_crop-0p6/wavelet_lambda-0/manifest.json"

command -v python >/dev/null 2>&1 || {
    echo "Error: activate the intended Python environment first." >&2
    exit 2
}
command -v bart >/dev/null 2>&1 || {
    echo "Error: activate the compatible BART build first." >&2
    exit 2
}

# Run and validate the split-complex representation before positive regularization.
python "$SCRIPT_DIR/run_bart_regularization.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --regularizer llr \
    --lambda-value 0 \
    --block-size 8 \
    --iterations 100 \
    --tolerance 1e-6 \
    --resume

python "$SCRIPT_DIR/validate_llr_lambda_zero.py" \
    --split-manifest "$OUTPUT_ROOT/llr_block-8_lambda-0/manifest.json" \
    --native-manifest "$NATIVE_ZERO_MANIFEST" \
    --output "$OUTPUT_ROOT/lambda0_equivalence.json"

for LAMBDA_VALUE in 2e-5 1e-4 5e-4; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer llr \
        --lambda-value "$LAMBDA_VALUE" \
        --block-size 8 \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

python "$SCRIPT_DIR/review_regularization_sweep.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --sweep-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/review" \
    --regularizer llr \
    --block-size 8 \
    --lambda-labels 0 2e-5 1e-4 5e-4
