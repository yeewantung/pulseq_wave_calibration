#!/usr/bin/env bash
# Copy to the ignored .local.sh name and fill in the private paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SETUP:?Set CONDA_SETUP in the local copy.}"
conda activate "${SYNTHETIC_WAVE_CONDA_ENV:?Set SYNTHETIC_WAVE_CONDA_ENV.}"

: "${PRESENTATION_COLLECTION_MANIFEST:?Set PRESENTATION_COLLECTION_MANIFEST.}"
: "${PRESENTATION_TIFF_DIR:?Set PRESENTATION_TIFF_DIR.}"

exec python "$SCRIPT_DIR/export_presentation_orientation_tiffs.py" \
    --collection-manifest "$PRESENTATION_COLLECTION_MANIFEST" \
    --output-dir "$PRESENTATION_TIFF_DIR" \
    --index 128 \
    --center-keys \
        synthetic_wave_r3x2_retro-lr_1p25x1p25x1mm \
        synthetic_wave_r3x2_retro-lr_1p49x1x1mm \
        synthetic_wave_r3x2_retro-lr_1x1p49x1mm \
    --display-percentile 99.5 \
    --refresh \
    "$@"
