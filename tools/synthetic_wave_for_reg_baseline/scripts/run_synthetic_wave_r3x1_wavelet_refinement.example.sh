#!/usr/bin/env bash
# Copy to .local.sh, set the private target root, and run in tmux.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the target root.}"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"

exec bash "$SCRIPT_DIR/run_synthetic_wave_r3x1_wavelet_refinement.sh" \
    --confirm-lambda0-reviewed
