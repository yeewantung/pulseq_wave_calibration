#!/usr/bin/env bash
set -euo pipefail

# Collect existing GRE magnitude/phase NIfTIs; do not derive or apply a mask.

usage() {
    echo "Usage: $0 OUTPUT_ROOT [--require-retro]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
OUTPUT_ROOT="${1%/}"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }

python "$SCRIPT_DIR/build_gre_nifti_collection.py" \
    "$OUTPUT_ROOT" \
    "$@"
