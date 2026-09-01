#!/usr/bin/env bash
set -euo pipefail

# Collect both reconstruction branches and apply one shared presentation mask.
# This workflow never prepares k-space or launches BART reconstruction.

usage() {
    echo "Usage: $0 OUTPUT_ROOT [--require-retro] [mask parameter overrides]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }

OUTPUT_ROOT="${1%/}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }

python "$SCRIPT_DIR/build_mprage_nifti_collection.py" "$OUTPUT_ROOT" "$@"
