#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/requirements/retrospective_low_resolution_product.json"
RUNNER_PATH="$REPOSITORY_ROOT/tools/wave_retro_lr_recon/scripts/run_retro_lr.py"

source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh

# Keep Matplotlib's runtime cache off the read-only home configuration tree.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/user-mpl-retro-low-resolution}"
mkdir -p "$MPLCONFIGDIR"

command -v bart >/dev/null 2>&1 || {
    echo "Error: bart is unavailable after sourcing bart_startup.sh" >&2
    exit 2
}

# The Python runner constructs every BART Wave command with mandatory GPU -g.
exec python "$RUNNER_PATH" --config "$CONFIG_PATH" "$@"
