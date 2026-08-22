#!/usr/bin/env bash
# Run the small follow-up grid that brackets Wavelet and LLR metric optima.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${R1_SYNTHETIC_WAVE_ROOT:?Set R1_SYNTHETIC_WAVE_ROOT to the private dataset output root.}"
DATASET_ROOT="$(realpath -m "$R1_SYNTHETIC_WAVE_ROOT")"
DEFAULT_LAMBDA_ZERO_MANIFEST="$DATASET_ROOT/reconstructions/synthetic_wave/ecalib_crop-0p6_lambda0/manifest.json"
REFINED_METRICS="$DATASET_ROOT/evaluation/direct_fft_reference/regularization_refinement_metrics/metrics_provenance.json"
OUTPUT_ROOT="$DATASET_ROOT/reconstructions/synthetic_wave/regularization_targeted_ecalib_crop-0p6"

usage() {
    cat >&2 <<EOF
Usage: $0 --confirm-approved-reference-and-mask [lambda_zero_manifest.json]

Runs or resumes 11 targeted R1 reconstructions: one Wavelet midpoint, four
block-4 LLR cases around 5e-3, and boundary extensions through 3e-2 for LLR
blocks 8 and 16. Set R1_SYNTHETIC_WAVE_ROOT to override the dataset root.
Every BART Wave reconstruction uses the GPU (-g).
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" != "--confirm-approved-reference-and-mask" || $# -gt 2 ]]; then
    usage
    exit 2
fi

LAMBDA_ZERO_MANIFEST="${2:-$DEFAULT_LAMBDA_ZERO_MANIFEST}"
if [[ ! -f "$LAMBDA_ZERO_MANIFEST" ]]; then
    echo "Error: lambda-zero manifest is missing: $LAMBDA_ZERO_MANIFEST" >&2
    exit 2
fi
if [[ ! -f "$REFINED_METRICS" ]]; then
    echo "Error: refined-metrics provenance is missing: $REFINED_METRICS" >&2
    exit 2
fi

source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
echo "BART: $(command -v bart)"
echo "Output: $OUTPUT_ROOT"

python "$SCRIPT_DIR/run_bart_regularization.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --regularizer wavelet \
    --lambda-value 1.5e-2 \
    --iterations 100 \
    --tolerance 1e-6 \
    --resume

# Block 4 already turns between 5e-3 and 1e-2, so sample around its minimum.
for lambda_value in 3e-3 4e-3 6e-3 7.5e-3; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer llr \
        --lambda-value "$lambda_value" \
        --block-size 4 \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

# Blocks 8 and 16 were still improving at 1e-2; extend them cautiously.
for block_size in 8 16; do
    for lambda_value in 1.5e-2 2e-2 3e-2; do
        python "$SCRIPT_DIR/run_bart_regularization.py" \
            --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
            --output-root "$OUTPUT_ROOT" \
            --regularizer llr \
            --lambda-value "$lambda_value" \
            --block-size "$block_size" \
            --iterations 100 \
            --tolerance 1e-6 \
            --resume
    done
done

echo "Targeted regularization sweep complete."
echo "Results: $OUTPUT_ROOT"
echo "Next: add this output root to exact-grid validation and regenerate metrics."
