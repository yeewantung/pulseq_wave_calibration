#!/usr/bin/env bash
# Run the long dataset preparation or the gated BART lambda-zero reconstruction.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 prepare [dataset_manifest.json]" >&2
    echo "       $0 reconstruct [dataset_manifest.json] --confirm-full-wave-reviewed" >&2
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage
    exit 2
fi

MODE="$1"
DATASET_MANIFEST="${2:-${R1_SYNTHETIC_WAVE_ROOT:+$R1_SYNTHETIC_WAVE_ROOT/dataset_manifest.json}}"
CONFIRMATION="${3:-}"
if [[ -z "$DATASET_MANIFEST" ]]; then
    echo "Error: provide dataset_manifest.json or set R1_SYNTHETIC_WAVE_ROOT." >&2
    exit 2
fi

command -v python >/dev/null 2>&1 || {
    echo "Error: activate the intended Python environment first." >&2
    exit 2
}

python "$SCRIPT_DIR/validate_dataset_manifest.py" \
    "$DATASET_MANIFEST" --check-inputs

case "$MODE" in
    prepare)
        python "$SCRIPT_DIR/estimate_coil_compression.py" \
            --dataset-manifest "$DATASET_MANIFEST" --resume
        python "$SCRIPT_DIR/assemble_fully_sampled_no_wave_kspace.py" \
            --dataset-manifest "$DATASET_MANIFEST" --resume
        python "$SCRIPT_DIR/synthesize_wave_kspace.py" \
            --dataset-manifest "$DATASET_MANIFEST" --resume
        echo "Preparation complete. Review the full-Wave diagnostics before reconstruction."
        ;;
    reconstruct)
        if [[ "$CONFIRMATION" != "--confirm-full-wave-reviewed" ]]; then
            echo "Reconstruction requires --confirm-full-wave-reviewed." >&2
            exit 2
        fi
        command -v bart >/dev/null 2>&1 || {
            echo "Error: activate the compatible BART build first." >&2
            exit 2
        }
        python "$SCRIPT_DIR/export_bart_wave_inputs.py" \
            --dataset-manifest "$DATASET_MANIFEST" \
            --visual-review-approved --resume
        python "$SCRIPT_DIR/export_bart_calibration_acs.py" \
            --dataset-manifest "$DATASET_MANIFEST" --resume
        python "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
            --dataset-manifest "$DATASET_MANIFEST" --resume
        ;;
    *)
        usage
        exit 2
        ;;
esac
