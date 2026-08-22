#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_CONFIG="$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/requirements/retrospective_low_resolution_product.local.json"
CONFIG_PATH="${RETRO_LOW_RES_CONFIG:-$DEFAULT_CONFIG}"
RUNNER_PATH="$REPOSITORY_ROOT/tools/wave_retro_lr_recon/scripts/run_retro_lr.py"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: local retrospective configuration is missing: $CONFIG_PATH" >&2
    echo "Copy requirements/retrospective_low_resolution_product.example.json" >&2
    echo "to the ignored .local.json filename and fill in private paths." >&2
    exit 2
fi

# Keep Matplotlib's runtime cache off the read-only home configuration tree.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synthetic-wave-retro-low-resolution}"
mkdir -p "$MPLCONFIGDIR"

command -v bart >/dev/null 2>&1 || {
    echo "Error: bart is unavailable after sourcing bart_startup.sh" >&2
    exit 2
}

# The Python runner constructs every BART Wave command with mandatory GPU -g.
exec python "$RUNNER_PATH" --config "$CONFIG_PATH" "$@"
