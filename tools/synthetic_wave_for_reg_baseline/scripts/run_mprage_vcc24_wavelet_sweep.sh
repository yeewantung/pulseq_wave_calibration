#!/usr/bin/env bash
# Run one resumable Ncc=24 native-R3x1 synthetic-Wave MPRAGE Wavelet sweep.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 DATASET_MANIFEST.json APPROVED_BRAIN_MASK_MANIFEST.json {--confirm-run|--evaluate-only}"
    echo "The manifest must select Ncc=24, pure native-R3x1 sampling, and an approved new output root."
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
[[ $# -eq 3 && ( "$3" == "--confirm-run" || "$3" == "--evaluate-only" ) ]] || { usage >&2; exit 2; }
DATASET_MANIFEST="$(realpath -m "$1")"
APPROVED_BRAIN_MASK_MANIFEST="$(realpath -m "$2")"
MODE="$3"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }

mapfile -t CONTRACT < <(python - "$DATASET_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
document = json.loads(path.read_text(encoding="utf-8"))
if document["reconstruction"]["virtual_coils"] != 24:
    raise SystemExit("MPRAGE sweep requires reconstruction.virtual_coils=24")
sampling = document["sampling"]
if sampling["synthetic_wave_mask_kind"] != "pure_cartesian_image_lattice":
    raise SystemExit("MPRAGE sweep requires a pure Cartesian image lattice")
if sampling["synthetic_wave_acceleration_pe1_pe2"] != [3, 1]:
    raise SystemExit("MPRAGE sweep requires native R3x1 acceleration [3, 1]")
if sampling["synthetic_wave_residue_pe1_pe2"] != [1, 0]:
    raise SystemExit("MPRAGE sweep requires native R3x1 residue [1, 0]")
root = Path(document["outputs"]["root"]).expanduser().resolve()
lambda0 = root / document["outputs"]["lambda0_reconstruction_dir"]
bart_inputs = root / document["outputs"]["bart_export_dir"] / "bart_inputs"
inspection = root / document["outputs"]["inspection_report"]
print(root)
print(lambda0)
print(bart_inputs)
print(inspection)
PY
)
[[ ${#CONTRACT[@]} -eq 4 ]] || { echo "Error: failed to resolve MPRAGE sweep contract." >&2; exit 2; }
OUTPUT_ROOT="${CONTRACT[0]}"
LAMBDA0_ROOT="${CONTRACT[1]}"
BART_INPUTS="${CONTRACT[2]}"
INSPECTION_REPORT="${CONTRACT[3]}"
SWEEP_ROOT="$OUTPUT_ROOT/reconstructions/native_r3x1_wavelet_sweep"

MPRAGE_LAMBDAS=(0 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05)

if [[ "$MODE" == "--confirm-run" ]]; then
    command -v bart >/dev/null || { echo "Error: bart is not on PATH." >&2; exit 2; }
    if [[ ! -f "$INSPECTION_REPORT" ]]; then
        python "$SCRIPT_DIR/inspect_product_dataset.py" \
            --dataset-manifest "$DATASET_MANIFEST" --probe-samples
    fi
    python "$SCRIPT_DIR/validate_dataset_manifest.py" "$DATASET_MANIFEST" --check-inputs
    bash "$SCRIPT_DIR/run_synthetic_wave_dataset.sh" prepare "$DATASET_MANIFEST"
    python "$SCRIPT_DIR/export_bart_wave_inputs.py" \
        --dataset-manifest "$DATASET_MANIFEST" --one-shot-sweep --resume
    python "$SCRIPT_DIR/export_bart_calibration_acs.py" \
        --dataset-manifest "$DATASET_MANIFEST" --resume
    python "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
        --dataset-manifest "$DATASET_MANIFEST" --ecalib-crop 0.6 --resume

    for lambda_value in "${MPRAGE_LAMBDAS[@]}"; do
        python "$SCRIPT_DIR/run_bart_regularization.py" \
            --lambda-zero-manifest "$LAMBDA0_ROOT/manifest.json" \
            --output-root "$SWEEP_ROOT" \
            --regularizer wavelet \
            --lambda-value "$lambda_value" \
            --iterations 100 \
            --tolerance 1e-6 \
            --resume
    done
fi

python "$SCRIPT_DIR/evaluate_mprage_vcc24_wavelet_sweep.py" \
    --dataset-manifest "$DATASET_MANIFEST" \
    --approved-brain-mask-manifest "$APPROVED_BRAIN_MASK_MANIFEST" \
    --resume

echo "MPRAGE Ncc=24 Wavelet sweep complete: $SWEEP_ROOT"
echo "FISTA-r0 control: $SWEEP_ROOT/wavelet_lambda-0"
echo "Lambda curves: $OUTPUT_ROOT/evaluation/native_r3x1_wavelet_sweep/metrics/plots/wavelet_metrics.png"
echo "No parameter winner was selected."
