#!/usr/bin/env bash
# Copy to the ignored .local.sh name, fill in paths, and run inside tmux.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SETUP:?Set CONDA_SETUP in the local copy.}"
conda activate "${SYNTHETIC_WAVE_CONDA_ENV:?Set SYNTHETIC_WAVE_CONDA_ENV.}"

: "${WAVE_BART_INPUT_MANIFEST:?Set WAVE_BART_INPUT_MANIFEST.}"
: "${WAVE_KSPACE_BASE:?Set WAVE_KSPACE_BASE.}"
: "${WAVE_PSF_BASE:?Set WAVE_PSF_BASE.}"
: "${WAVE_MAPS_BASE:?Set WAVE_MAPS_BASE.}"
: "${PRESENTATION_TWIX:?Set PRESENTATION_TWIX.}"
: "${PRESENTATION_SEQUENCE:?Set PRESENTATION_SEQUENCE.}"
: "${PREVIOUS_WAVE_CG_OUTPUT:?Set PREVIOUS_WAVE_CG_OUTPUT.}"
: "${METRICS_REFERENCE_MANIFEST:?Set METRICS_REFERENCE_MANIFEST.}"

exec python "$SCRIPT_DIR/run_previous_non_bart_wave_cg_sense.py" \
    --bart-input-manifest "$WAVE_BART_INPUT_MANIFEST" \
    --wave-kspace "$WAVE_KSPACE_BASE" \
    --psf "$WAVE_PSF_BASE" \
    --maps "$WAVE_MAPS_BASE" \
    --twix "$PRESENTATION_TWIX" \
    --sequence "$PRESENTATION_SEQUENCE" \
    --output-dir "$PREVIOUS_WAVE_CG_OUTPUT" \
    --metrics-reference-manifest "$METRICS_REFERENCE_MANIFEST" \
    --subject presentation-r1-previous-wave-cg-sense \
    --iterations 50 \
    --tolerance 1e-6 \
    --device cuda:0 \
    --resume \
    "$@"
