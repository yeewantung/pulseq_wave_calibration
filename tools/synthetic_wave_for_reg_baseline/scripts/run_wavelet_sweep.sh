#!/usr/bin/env bash
# Run the frozen R1 Wavelet coarse sweep from the approved crop-0.6 lambda zero.

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
OUTPUT_ROOT="$DATASET_ROOT/reconstructions/synthetic_wave/wavelet_sweep_ecalib_crop-0p6"

command -v python >/dev/null 2>&1 || {
    echo "Error: activate the intended Python environment first." >&2
    exit 2
}
command -v bart >/dev/null 2>&1 || {
    echo "Error: activate the compatible BART build first." >&2
    exit 2
}

for LAMBDA_VALUE in 0 1e-6 1e-5 1e-4 1e-3 1e-2; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer wavelet \
        --lambda-value "$LAMBDA_VALUE" \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

python "$SCRIPT_DIR/review_regularization_sweep.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --sweep-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/review" \
    --regularizer wavelet \
    --lambda-labels 0 1e-6 1e-5 1e-4 1e-3 1e-2
