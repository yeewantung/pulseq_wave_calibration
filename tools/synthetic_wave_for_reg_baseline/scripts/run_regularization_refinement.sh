#!/usr/bin/env bash
# Run the missing R1 Wavelet and multi-block LLR refinement cases on GPU.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${R1_SYNTHETIC_WAVE_ROOT:?Set R1_SYNTHETIC_WAVE_ROOT to the private dataset output root.}"
DATASET_ROOT="$(realpath -m "$R1_SYNTHETIC_WAVE_ROOT")"
DEFAULT_LAMBDA_ZERO_MANIFEST="$DATASET_ROOT/reconstructions/synthetic_wave/ecalib_crop-0p6_lambda0/manifest.json"
COARSE_METRICS="$DATASET_ROOT/evaluation/direct_fft_reference/coarse_regularization_metrics/metrics_provenance.json"
OUTPUT_ROOT="$DATASET_ROOT/reconstructions/synthetic_wave/regularization_refinement_ecalib_crop-0p6"

usage() {
    cat >&2 <<EOF
Usage: $0 --confirm-approved-reference-and-mask [lambda_zero_manifest.json]

Runs/resumes the missing Wavelet and LLR refinement cases selected from the
approved direct-FFT coarse evaluation. Set R1_SYNTHETIC_WAVE_ROOT to override
the dataset root. Every BART Wave reconstruction uses the GPU (-g).
EOF
}

if [[ "${1:-}" != "--confirm-approved-reference-and-mask" || $# -gt 2 ]]; then
    usage
    exit 2
fi

LAMBDA_ZERO_MANIFEST="${2:-$DEFAULT_LAMBDA_ZERO_MANIFEST}"
if [[ ! -f "$LAMBDA_ZERO_MANIFEST" ]]; then
    echo "Error: lambda-zero manifest is missing: $LAMBDA_ZERO_MANIFEST" >&2
    exit 2
fi
if [[ ! -f "$COARSE_METRICS" ]]; then
    echo "Error: approved coarse-metrics provenance is missing: $COARSE_METRICS" >&2
    exit 2
fi

source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
echo "BART: $(command -v bart)"
echo "Output: $OUTPUT_ROOT"

# The existing 1e-3 and 1e-2 Wavelet cases bracket the new interior points.
# One outward decade-half-step pair tests whether the fidelity optimum turns.
WAVELET_LAMBDAS=(2e-3 5e-3 2e-2 5e-2)

for lambda_value in "${WAVELET_LAMBDAS[@]}"; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer wavelet \
        --lambda-value "$lambda_value" \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

# Blocks 4 and 16 need the full grid. Block 8 reuses its completed 2e-5,
# 1e-4, and 5e-4 cases from the coarse output and runs only missing values.
LLR_FULL_LAMBDAS=(2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2)
LLR_BLOCK8_MISSING_LAMBDAS=(5e-5 2e-4 1e-3 2e-3 5e-3 1e-2)

for block_size in 4 16; do
    for lambda_value in "${LLR_FULL_LAMBDAS[@]}"; do
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

for lambda_value in "${LLR_BLOCK8_MISSING_LAMBDAS[@]}"; do
    python "$SCRIPT_DIR/run_bart_regularization.py" \
        --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --output-root "$OUTPUT_ROOT" \
        --regularizer llr \
        --lambda-value "$lambda_value" \
        --block-size 8 \
        --iterations 100 \
        --tolerance 1e-6 \
        --resume
done

echo "Refinement reconstruction grid complete."
echo "Results: $OUTPUT_ROOT"
echo "Next: combine these manifests with the retained coarse cases for exact-grid validation and metrics."
