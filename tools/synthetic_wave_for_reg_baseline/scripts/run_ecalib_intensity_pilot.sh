#!/usr/bin/env bash
# Resumable GPU lambda-zero pilot for BART ESPIRiT intensity-corrected maps.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

STAGE="all"
OUTPUT_ROOT="/path/to/data/20260817_product/ecalib_intensity_c050_wave_lambda0"

usage() {
    cat <<'EOF'
Usage: run_ecalib_intensity_pilot.sh [options]

Stages:
  reconstruct   Run ecalib -I and one GPU Wave lambda-zero reconstruction.
  compare       Compare the completed pilot with IDEA, GRAPPA, SENSE, and current Wave.
  all           Run or reuse reconstruction, then create the comparison (default).

Options:
  --stage NAME
  --output-root PATH
  -h, --help

This pilot does not run a regularization sweep or select a lambda. GRAPPA and
SENSE are included only as comparison references, not as ground truth.
EOF
}

while (($#)); do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$(realpath -m "$2")"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$STAGE" in
    reconstruct|compare|all) ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        usage >&2
        exit 2
        ;;
esac

source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/user-mpl-ecalib-intensity}"

BART_EXECUTABLE="$(command -v bart)"
PYTHON_EXECUTABLE="$(command -v python)"
DATASET_ROOT="/path/to/data/20260817_product"
WAVE_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2"
PRESENTATION_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12_r3x2_presentation_optimization"
BART_INPUT_DIR="$WAVE_ROOT/bart_inputs"
CALIBRATION_BASE="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12/bart_inputs/kspace_calib"
TWIX="$DATASET_ROOT/meas_MID00345_FID35555_t1_mprage_sag_p2.dat"
SEQUENCE_ROOT="/path/to/user_workspace/scan_protocols/20260817_integrated/v151"
SEQUENCE="$SEQUENCE_ROOT/mprage_3d_wave_FOV256x256x256_res1x1x1_ETL256_R1-1_R2-3_os4_amp8_cyc10_SAG_prisma_v151.seq"
RECON_DIR="$OUTPUT_ROOT/reconstruction"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
METRICS_PROVENANCE="$PRESENTATION_ROOT/evaluation/volume_metrics/metrics_provenance.json"
SENSE_NIFTI="$DATASET_ROOT/sense_no_wave_ncc12/recon_lambda0/nifti/sub-20260817product_part-mag_NoWaveSENSELambda0.nii.gz"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_PATH="$OUTPUT_ROOT/logs/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "ESPIRiT intensity pilot stage: $STAGE"
echo "Python: $PYTHON_EXECUTABLE"
echo "BART: $BART_EXECUTABLE"
echo "Output: $OUTPUT_ROOT"
echo "Log: $LOG_PATH"

reconstruction_reusable() {
    local manifest="$RECON_DIR/manifest.json"
    [[ -f "$manifest" ]] || return 1
    "$PYTHON_EXECUTABLE" - "$manifest" "$CALIBRATION_BASE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_calibration = str(Path(sys.argv[2]).resolve())
valid = (
    manifest.get("status") == "lambda0_complete_awaiting_visual_review"
    and manifest.get("ecalib", {}).get("intensity_correction") is True
    and manifest.get("ecalib", {}).get("input_base") == expected_calibration
    and "-I" in manifest.get("ecalib", {}).get("command", [])
    and manifest.get("wave_lambda0", {}).get("backend") == "gpu"
    and "-g" in manifest.get("wave_lambda0", {}).get("command", [])
)
raise SystemExit(0 if valid else 1)
PY
}

run_reconstruction() {
    if reconstruction_reusable; then
        echo "Reusing completed GPU intensity-corrected reconstruction: $RECON_DIR/manifest.json"
        return
    fi
    if [[ -e "$RECON_DIR" ]]; then
        echo "Existing reconstruction is incomplete or has different provenance: $RECON_DIR" >&2
        exit 2
    fi
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
        --bart "$BART_EXECUTABLE" \
        --bart-input-dir "$BART_INPUT_DIR" \
        --calibration-base "$CALIBRATION_BASE" \
        --output-dir "$RECON_DIR" \
        --twix "$TWIX" \
        --sequence "$SEQUENCE" \
        --measurement-index 1 \
        --subject 20260817product-ecalib-intensity-c050 \
        --ecalib-crop 0.5 \
        --ecalib-intensity-correction \
        --cg-iterations 300 \
        --cg-tolerance 1e-3
}

run_comparison() {
    reconstruction_reusable || {
        echo "A complete GPU intensity-corrected reconstruction is required first." >&2
        exit 2
    }
    if [[ -f "$COMPARISON_DIR/comparison_manifest.json" ]]; then
        echo "Reusing completed comparison: $COMPARISON_DIR/comparison_manifest.json"
        return
    fi
    if [[ -e "$COMPARISON_DIR" ]]; then
        echo "Existing comparison output is incomplete: $COMPARISON_DIR" >&2
        exit 2
    fi
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/compare_ecalib_intensity_pilot.py" \
        --pilot-manifest "$RECON_DIR/manifest.json" \
        --metrics-provenance "$METRICS_PROVENANCE" \
        --raw-dicom-nifti "$WAVE_ROOT/nifti/dicom_reference/dicom_unfiltered_nd.nii.gz" \
        --grappa-nifti "$DATASET_ROOT/grappa_5x5x5_ncc12/grappa_5x5x5_ncc12_rss_ras.nii.gz" \
        --sense-nifti "$SENSE_NIFTI" \
        --output-dir "$COMPARISON_DIR"
}

case "$STAGE" in
    reconstruct) run_reconstruction ;;
    compare) run_comparison ;;
    all)
        run_reconstruction
        run_comparison
        ;;
esac
