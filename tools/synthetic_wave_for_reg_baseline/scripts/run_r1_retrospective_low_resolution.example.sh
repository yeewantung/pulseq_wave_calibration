#!/usr/bin/env bash
# Copy to run_r1_retrospective_low_resolution.local.sh and fill private paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"

export R1_RETRO_LOW_RES_CONFIG="$SCRIPT_DIR/../requirements/retrospective_low_resolution_r1.local.json"
export R1_RETRO_SELECTION_MANIFEST="/path/to/r1/regularization_selection/selection_manifest.json"

exec "$SCRIPT_DIR/run_r1_retrospective_low_resolution.sh" "$@"
