#!/usr/bin/env bash
# Copy to the ignored .local.sh name, fill in paths, and run inside tmux.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SETUP:?Set CONDA_SETUP in the local copy.}"
conda activate "${SYNTHETIC_WAVE_CONDA_ENV:?Set SYNTHETIC_WAVE_CONDA_ENV.}"
source "${BART_STARTUP:?Set BART_STARTUP in the local copy.}"

: "${NO_WAVE_SOURCE_KSPACE:?Set NO_WAVE_SOURCE_KSPACE.}"
: "${NO_WAVE_MAPS:?Set NO_WAVE_MAPS.}"
: "${PRESENTATION_TWIX:?Set PRESENTATION_TWIX.}"
: "${PRESENTATION_SEQUENCE:?Set PRESENTATION_SEQUENCE.}"
: "${NO_WAVE_SWEEP_ROOT:?Set NO_WAVE_SWEEP_ROOT.}"

exec python "$SCRIPT_DIR/run_no_wave_r3x1_pics_sweep.py" \
    --bart "$(command -v bart)" \
    --source-no-wave-kspace "$NO_WAVE_SOURCE_KSPACE" \
    --maps "$NO_WAVE_MAPS" \
    --twix "$PRESENTATION_TWIX" \
    --sequence "$PRESENTATION_SEQUENCE" \
    --output-root "$NO_WAVE_SWEEP_ROOT" \
    --subject presentation-r1 \
    --wavelet-lambdas 1e-4 1e-3 1e-2 1.5e-2 2e-2 5e-2 \
    --resume \
    "$@"
