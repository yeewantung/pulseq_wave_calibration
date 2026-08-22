#!/usr/bin/env bash
# Recalibrate maps at crop 0.6 and repeat lambda zero without altering crop 0.8.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${R1_SYNTHETIC_WAVE_ROOT:?Set R1_SYNTHETIC_WAVE_ROOT to the private dataset output root.}"
DATASET_ROOT="$(realpath -m "$R1_SYNTHETIC_WAVE_ROOT")"
DATASET_MANIFEST="${1:-$DATASET_ROOT/dataset_manifest.json}"
OUTPUT_DIR="$DATASET_ROOT/reconstructions/synthetic_wave/ecalib_crop-0p6_lambda0"

command -v python >/dev/null 2>&1 || {
    echo "Error: activate the intended Python environment first." >&2
    exit 2
}
command -v bart >/dev/null 2>&1 || {
    echo "Error: activate the compatible BART build first." >&2
    exit 2
}

python "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
    --dataset-manifest "$DATASET_MANIFEST" \
    --ecalib-crop 0.6 \
    --output-dir "$OUTPUT_DIR" \
    --resume
