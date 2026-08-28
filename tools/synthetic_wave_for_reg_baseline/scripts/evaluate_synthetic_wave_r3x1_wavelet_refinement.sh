#!/usr/bin/env bash
# Evaluate the combined coarse and refined R3x1 grid on the frozen exact reference.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the private target root.}"
: "${METRICS_REFERENCE_MANIFEST:?Set METRICS_REFERENCE_MANIFEST to the approved direct-FFT manifest.}"
: "${METRICS_REFERENCE_MANIFEST_SHA256:?Set METRICS_REFERENCE_MANIFEST_SHA256 to freeze the approved reference/mask package.}"
TARGET_ROOT="$(realpath -m "$SYNTHETIC_WAVE_R3X1_ROOT")"
COARSE_ROOT="$TARGET_ROOT/reconstructions/regularization"
REFINEMENT_ROOT="$TARGET_ROOT/reconstructions/wavelet_refinement"
OUTPUT_ROOT="$TARGET_ROOT/evaluation/direct_fft_wavelet_refinement"
GEOMETRY_REPORT="$OUTPUT_ROOT/exact_grid_geometry.json"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Error: evaluation output already exists; refusing to mix evaluations: $OUTPUT_ROOT" >&2
    exit 2
fi
read -r ACTUAL_REFERENCE_SHA256 _ < <(sha256sum "$METRICS_REFERENCE_MANIFEST")
if [[ "$ACTUAL_REFERENCE_SHA256" != "$METRICS_REFERENCE_MANIFEST_SHA256" ]]; then
    echo "Error: the approved direct-FFT reference/BET-mask manifest changed." >&2
    exit 2
fi

EXPECTED_CASES=()
for lambda_value in \
    0 1e-4 3e-4 1e-3 3e-3 5e-3 1e-2 1.5e-2 \
    1.6e-2 1.8e-2 2e-2 2.2e-2 2.5e-2 3e-2 4e-2 5e-2; do
    EXPECTED_CASES+=(--expected-case "wavelet:$lambda_value")
done
EXPECTED_CASES+=(--expected-case "llr:block-8:0")
for block_size in 4 8 16; do
    for lambda_value in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2; do
        EXPECTED_CASES+=(--expected-case "llr:block-$block_size:$lambda_value")
    done
done

python "$SCRIPT_DIR/validate_metrics_geometry.py" \
    --metrics-reference-manifest "$METRICS_REFERENCE_MANIFEST" \
    --sweep-root "$COARSE_ROOT" \
    --sweep-root "$REFINEMENT_ROOT" \
    "${EXPECTED_CASES[@]}" \
    --output "$GEOMETRY_REPORT"

python "$SCRIPT_DIR/evaluate_direct_fft_regularization.py" \
    --metrics-reference-manifest "$METRICS_REFERENCE_MANIFEST" \
    --geometry-report "$GEOMETRY_REPORT" \
    --output-dir "$OUTPUT_ROOT/metrics"

echo "Combined R3x1 Wavelet refinement metrics: $OUTPUT_ROOT"
