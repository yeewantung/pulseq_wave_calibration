#!/usr/bin/env bash
# Copy to run_r1_retrospective_wavelet_sweep.local.sh and fill private paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../requirements/retrospective_low_resolution_wavelet_sweep.local.json"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/retrospective-wavelet-sweep"
mkdir -p "$MPLCONFIGDIR"

echo "Backend: wave_retro_lr.pipeline.run_config -> BART wave -w -f -r <lambda> -g"
exec python "$SCRIPT_DIR/run_retrospective_wavelet_sweep.py" \
    --config "$CONFIG" \
    --resume
