#!/usr/bin/env bash
# Run the R3x1 Wavelet and corrected-LLR grid through run_bart_regularization.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the private target root.}"
TARGET_ROOT="$(realpath -m "$SYNTHETIC_WAVE_R3X1_ROOT")"
LAMBDA_ZERO_MANIFEST="$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/manifest.json"
SWEEP_ROOT="$TARGET_ROOT/reconstructions/regularization"
VISUAL_APPROVAL="$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/visual_approval.json"

if [[ "${1:-}" != "--confirm-lambda0-reviewed" || $# -ne 1 ]]; then
    echo "Usage: $0 --confirm-lambda0-reviewed" >&2
    echo "Review the lambda-zero central slices and ESPIRiT-map montages first." >&2
    exit 2
fi
for required in \
    "$LAMBDA_ZERO_MANIFEST" \
    "$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/lambda0_nifti_central_slices.png" \
    "$TARGET_ROOT/reconstructions/lambda0_ecalib_crop-0p6/espirit_maps_mag_central_slice.png"; do
    [[ -f "$required" ]] || { echo "Error: missing visual-gate input: $required" >&2; exit 2; }
done
command -v python >/dev/null 2>&1 || { echo "Error: activate Python first." >&2; exit 2; }
command -v bart >/dev/null 2>&1 || { echo "Error: activate the compatible BART build first." >&2; exit 2; }

echo "Backend: run_bart_regularization.py -> BART wave -w/-l -v -f -r <lambda> -g"
echo "Lambda-zero visual review: explicitly confirmed"

python "$SCRIPT_DIR/record_lambda0_visual_approval.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --output "$VISUAL_APPROVAL" \
    --confirm-reconstruction-and-maps-reviewed \
    --notes "User approved the synthetic-Wave R3x1 lambda-zero reconstruction and ESPIRiT maps."

WAVELET_LAMBDAS=(0 1e-4 3e-4 1e-3 3e-3 5e-3 1e-2 1.5e-2 2e-2 5e-2)
for lambda_value in "${WAVELET_LAMBDAS[@]}"; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$SWEEP_ROOT" \
        --regularizer wavelet \
        --lambda-value "$lambda_value" \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

# The block-8 lambda-zero solve gates BART's split-complex LLR representation.
python "$SCRIPT_DIR/run_bart_regularization.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --output-root "$SWEEP_ROOT" \
    --regularizer llr \
    --lambda-value 0 \
    --block-size 8 \
    --iterations 100 \
    --tolerance 1e-6 \
    --resume

python "$SCRIPT_DIR/validate_llr_lambda_zero.py" \
    --split-manifest "$SWEEP_ROOT/llr_block-8_lambda-0/manifest.json" \
    --native-manifest "$SWEEP_ROOT/wavelet_lambda-0/manifest.json" \
    --output "$SWEEP_ROOT/llr_lambda0_equivalence.json"

LLR_LAMBDAS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)
for block_size in 4 8 16; do
    for lambda_value in "${LLR_LAMBDAS[@]}"; do
        python "$SCRIPT_DIR/run_bart_regularization.py" \
            --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
            --output-root "$SWEEP_ROOT" \
            --regularizer llr \
            --lambda-value "$lambda_value" \
            --block-size "$block_size" \
            --iterations 100 \
            --tolerance 1e-6 \
            --resume
    done
done

echo "R3x1 regularization sweep complete: $SWEEP_ROOT"
echo "Next: run the separate evaluation launcher."
