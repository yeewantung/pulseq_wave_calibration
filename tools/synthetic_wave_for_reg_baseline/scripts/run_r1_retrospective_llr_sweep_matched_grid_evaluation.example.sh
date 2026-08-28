#!/usr/bin/env bash
# Copy to the matching .local.sh name and fill private paths in the local JSON.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../requirements/retrospective_llr_sweep_matched_grid_evaluation.local.json"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/retrospective-llr-matched-grid-evaluation"
mkdir -p "$MPLCONFIGDIR"

exec python "$SCRIPT_DIR/evaluate_retrospective_llr_sweep_matched_grid.py" \
    --config "$CONFIG" \
    --resume
