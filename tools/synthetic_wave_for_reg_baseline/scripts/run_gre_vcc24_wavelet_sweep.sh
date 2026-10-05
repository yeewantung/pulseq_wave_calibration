#!/usr/bin/env bash
# Run one resumable Ncc=24 native-R3x1 two-echo GRE Wavelet sweep.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 CONFIG.local.json --confirm-run"
    echo "Runs preparation, shared-echo Wavelet candidates, NIfTI export, and evaluation."
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
[[ $# -eq 2 && "$2" == "--confirm-run" ]] || { usage >&2; exit 2; }
CONFIG="$(realpath -m "$1")"
ENTRY="$SCRIPT_DIR/gre_synthetic_wave_sweep.py"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH." >&2; exit 2; }

RUN_ROOT="$(python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if document.get("format_version") != 3:
    raise SystemExit("GRE one-shot sweep requires format_version=3")
compression = document["coil_compression"]
if compression["physical_coils"] != 44 or compression["virtual_coils"] != 24:
    raise SystemExit("GRE one-shot sweep requires standard PCA 44-to-24 compression")
if document.get("case_ids") != ["native_r3x1"]:
    raise SystemExit("GRE one-shot sweep is restricted to native_r3x1")
values = [float(value) for value in document["sweep"]["wavelet_lambdas"]]
if values[0] != 0.005 or values[-1] != 0.03:
    raise SystemExit("GRE Wavelet grid must span [0.005, 0.03]")
print((Path(document["output_parent"]) / document["run_name"]).resolve())
PY
)"

run_entry() {
    python "$ENTRY" --config "$CONFIG" "$@" --confirm-run-root "$RUN_ROOT"
}

python "$ENTRY" --config "$CONFIG" validate-config >/dev/null
run_entry inspect-metadata
run_entry prepare-source
run_entry prepare-operator
run_entry validate-operator
run_entry prepare-csm
run_entry prepare-references
run_entry reuse-brain-mask
run_entry prepare-cases
run_entry reconstruct --sweep coarse --resume
run_entry export-nifti --sweep coarse
run_entry evaluate --sweep coarse
run_entry plot --sweep coarse
run_entry evaluate-shared-lambda --sweep coarse
run_entry plot-shared-lambda --sweep coarse

echo "GRE Ncc=24 shared-echo Wavelet sweep complete: $RUN_ROOT"
echo "Each lambda is identical across both echoes; magnitude, phase, scaling, and delta-B0 are retained."
echo "No parameter winner was selected."
