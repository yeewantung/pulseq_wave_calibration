#!/usr/bin/env bash
# Copy to the ignored .local.sh name, fill in paths, and run inside tmux.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SETUP:?Set CONDA_SETUP in the local copy.}"
conda activate "${SYNTHETIC_WAVE_CONDA_ENV:?Set SYNTHETIC_WAVE_CONDA_ENV.}"

: "${NO_WAVE_SOURCE_KSPACE:?Set NO_WAVE_SOURCE_KSPACE.}"
: "${NO_WAVE_MAPS:?Set NO_WAVE_MAPS.}"
: "${PRESENTATION_TWIX:?Set PRESENTATION_TWIX.}"
: "${PRESENTATION_SEQUENCE:?Set PRESENTATION_SEQUENCE.}"
: "${NO_WAVE_GRAPPA_ROOT:?Set NO_WAVE_GRAPPA_ROOT.}"
: "${METRICS_REFERENCE_MANIFEST:?Set METRICS_REFERENCE_MANIFEST.}"

exec python "$SCRIPT_DIR/run_no_wave_r3x1_grappa.py" \
    --source-no-wave-kspace "$NO_WAVE_SOURCE_KSPACE" \
    --maps "$NO_WAVE_MAPS" \
    --twix "$PRESENTATION_TWIX" \
    --sequence "$PRESENTATION_SEQUENCE" \
    --output-dir "$NO_WAVE_GRAPPA_ROOT" \
    --metrics-reference-manifest "$METRICS_REFERENCE_MANIFEST" \
    --subject presentation-r1 \
    --pe2-kernel-size 5 \
    --regularization 0.01 \
    --resume \
    "$@"
