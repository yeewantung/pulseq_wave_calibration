#!/usr/bin/env bash
# Copy to the matching .local.sh name and fill private paths in the local JSON.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../requirements/retrospective_low_resolution_llr_sweep.local.json"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/retrospective-llr-sweep"
mkdir -p "$MPLCONFIGDIR"

echo "Backend: wave_retro_lr.pipeline.run_config -> BART wave -l -v -b <block> -f -r <lambda> -g"
exec python "$SCRIPT_DIR/run_retrospective_llr_sweep.py" \
    --config "$CONFIG" \
    --resume
