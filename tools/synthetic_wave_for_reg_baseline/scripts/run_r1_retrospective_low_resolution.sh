#!/usr/bin/env bash
# Run R1 retrospective low-resolution cases with the frozen Wavelet selection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$REPOSITORY_ROOT/tools/wave_retro_lr_recon/scripts/run_retro_lr.py"

usage() {
    cat <<'EOF'
Usage: run_r1_retrospective_low_resolution.sh [--validate-only|--prepare-only] [--resume]

Required environment variables:
  R1_RETRO_LOW_RES_CONFIG       Ignored machine-local JSON configuration
  R1_RETRO_SELECTION_MANIFEST   Frozen R1 regularization selection manifest

The runner requires Wavelet lambda=1.5e-2, FISTA, 100 iterations, tolerance
1e-6, and GPU BART -g. Maximum eigenvalue remains case-specific. Copy the
tracked example launcher and configuration to their ignored .local names.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (($# > 2)); then
    usage >&2
    exit 2
fi

: "${R1_RETRO_LOW_RES_CONFIG:?Set R1_RETRO_LOW_RES_CONFIG in the local launcher.}"
: "${R1_RETRO_SELECTION_MANIFEST:?Set R1_RETRO_SELECTION_MANIFEST in the local launcher.}"

PYTHON_EXECUTABLE="$(command -v python)" || {
    echo "Error: activate the intended Python environment first." >&2
    exit 2
}
BART_EXECUTABLE="$(command -v bart)" || {
    echo "Error: activate the compatible BART build first." >&2
    exit 2
}
CONFIG="$(realpath "$R1_RETRO_LOW_RES_CONFIG")"
SELECTION="$(realpath "$R1_RETRO_SELECTION_MANIFEST")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synthetic-wave-r1-retro-low-resolution}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_EXECUTABLE" -c '
import json, pathlib, sys
config_path = pathlib.Path(sys.argv[1]).resolve()
selection_path = pathlib.Path(sys.argv[2]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
record = json.loads(selection_path.read_text(encoding="utf-8"))
selection = record.get("selection", {})
reconstruction = config.get("reconstruction", {})
if record.get("status") != "frozen_for_cross_dataset_transfer":
    raise SystemExit("selection manifest is not frozen")
expected = {
    "regularizer": "wavelet",
    "lambda": 0.015,
    "iterations": 100,
    "tolerance": 1e-6,
}
for key, value in expected.items():
    if reconstruction.get(key) != value or selection.get(key) != value:
        raise SystemExit(f"frozen selection/config mismatch for {key}")
if selection.get("optimizer") != "FISTA" or selection.get("bart_gpu_option_required") != "-g":
    raise SystemExit("frozen optimizer/backend contract is invalid")
if reconstruction.get("maximum_eigenvalue") is not None:
    raise SystemExit("low-resolution cases must estimate their own maximum eigenvalue")
companions = {pathlib.Path(value).resolve() for value in config["source"]["companion_manifests"]}
if selection_path not in companions:
    raise SystemExit("selection manifest is absent from companion provenance")
' "$CONFIG" "$SELECTION"

echo "Python: $PYTHON_EXECUTABLE"
echo "BART: $BART_EXECUTABLE"
echo "Config: $CONFIG"
echo "Frozen regularization: Wavelet lambda=1.5e-2"
echo "Maximum eigenvalue: estimated independently for each target matrix"

exec "$PYTHON_EXECUTABLE" "$RUNNER" --config "$CONFIG" "$@"
