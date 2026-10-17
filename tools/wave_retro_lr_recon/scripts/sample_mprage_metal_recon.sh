#!/usr/bin/env bash
set -euo pipefail

# Run only an explicitly requested, independent CPU metal-CSM derivative.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command -v python >/dev/null || { echo "Error: activate the Python environment first." >&2; exit 2; }
if command -v ionice >/dev/null; then
    exec ionice -c3 python "$SCRIPT_DIR/mprage_metal_recon.py" "$@"
fi
exec python "$SCRIPT_DIR/mprage_metal_recon.py" "$@"
