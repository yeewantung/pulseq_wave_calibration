#!/usr/bin/env bash
# Refine the synthetic-Wave R3x1 Wavelet optimum through the existing GPU runner.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the private target root.}"
TARGET_ROOT="$(realpath -m "$SYNTHETIC_WAVE_R3X1_ROOT")"
LAMBDA_ZERO_MANIFEST="$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/manifest.json"
VISUAL_APPROVAL="$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/visual_approval.json"
COARSE_ROOT="$TARGET_ROOT/reconstructions/regularization"
OUTPUT_ROOT="$TARGET_ROOT/reconstructions/wavelet_refinement"

if [[ $# -ne 1 || ( "$1" != "--confirm-lambda0-reviewed" && "$1" != "--check-only" ) ]]; then
    echo "Usage: $0 --confirm-lambda0-reviewed | --check-only" >&2
    exit 2
fi
for required in \
    "$LAMBDA_ZERO_MANIFEST" \
    "$VISUAL_APPROVAL" \
    "$COARSE_ROOT/wavelet_lambda-1.5e-2/manifest.json" \
    "$COARSE_ROOT/wavelet_lambda-2e-2/manifest.json" \
    "$COARSE_ROOT/wavelet_lambda-5e-2/manifest.json"; do
    [[ -f "$required" ]] || { echo "Error: missing frozen input/control: $required" >&2; exit 2; }
done
command -v python >/dev/null 2>&1 || { echo "Error: activate Python first." >&2; exit 2; }
command -v bart >/dev/null 2>&1 || { echo "Error: activate the compatible BART build first." >&2; exit 2; }

if [[ "$1" == "--check-only" ]]; then
    echo "Preflight passed: six new Wavelet cases; existing controls are unchanged."
    echo "Output: $OUTPUT_ROOT"
    exit 0
fi

python "$SCRIPT_DIR/record_lambda0_visual_approval.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --output "$VISUAL_APPROVAL" \
    --confirm-reconstruction-and-maps-reviewed \
    --notes "User approved the synthetic-Wave R3x1 lambda-zero reconstruction and ESPIRiT maps."

echo "Backend: run_bart_regularization.py -> BART wave -w -f -r <lambda> -g"
echo "Existing controls 1.5e-2, 2e-2, and 5e-2 are not rerun."
for lambda_value in 1.6e-2 1.8e-2 2.2e-2 2.5e-2 3e-2 4e-2; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer wavelet \
        --lambda-value "$lambda_value" \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

echo "R3x1 Wavelet refinement complete: $OUTPUT_ROOT"
echo "Next: run the separate combined evaluation launcher."
