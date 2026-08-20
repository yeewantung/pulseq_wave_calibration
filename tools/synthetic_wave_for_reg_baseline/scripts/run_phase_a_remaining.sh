#!/usr/bin/env bash
# Resumable Phase-A runner for the 2026-08-17 R3 product experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$TOOL_DIR/../.." && pwd)"

STAGE="all-before-review"
CONFIRM_VISUAL_QC=0
PHASE_A_ROOT="/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12_r3x2_phase_a"

usage() {
    cat <<'EOF'
Usage: run_phase_a_remaining.sh [options]

Stages:
  reconstruct       Run/resume the focused GPU Wavelet and corrected-LLR cases.
  prepare-qc        Convert normalized DICOM, create BET mask, and create QC figures.
  approve-qc        Record explicit approval after the user reviews the QC figures.
  evaluate          Run masked evaluation and generate the compact presentation package.
  all-before-review Run reconstruct and prepare-qc, then stop for visual review (default).
  all-after-review  Run approve-qc and evaluate; requires explicit confirmation.

Options:
  --stage NAME
  --phase-a-root PATH
  --confirm-reviewed-mask-and-lr
  -h, --help

Run `all-before-review` in tmux first. Inspect the two figures printed at the end.
Only then run `all-after-review --confirm-reviewed-mask-and-lr`.
EOF
}

while (($#)); do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --phase-a-root)
            PHASE_A_ROOT="$(realpath -m "$2")"
            shift 2
            ;;
        --confirm-reviewed-mask-and-lr)
            CONFIRM_VISUAL_QC=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$STAGE" in
    reconstruct|prepare-qc|approve-qc|evaluate|all-before-review|all-after-review) ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/user-mpl-phase-a}"
export FSLDIR=/path/to/software/packages/fsl/6.0.6
. "${FSLDIR}/etc/fslconf/fsl.sh"

BART_EXECUTABLE="$(command -v bart)"
PYTHON_EXECUTABLE="$(command -v python)"
SOURCE_ROOT="/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12_r3x2"
DATASET_ROOT="/path/to/data/20260817_product"
SYNTHESIS_ROOT="$DATASET_ROOT/synthetic_wave_grappa_5x5x5_ncc12"
MAPS="$SYNTHESIS_ROOT/bart_lambda0_ecalib_c050/coil_sens_bart"
MAPS_SHA256="95af12014c6fabab358e4177e091f8085a00619d93354385f32f4482be3e115d"
LAMBDA_ZERO="$SOURCE_ROOT/bart_lambda0_existing_csm_c050/image_wave"
TWIX="$DATASET_ROOT/meas_MID00345_FID35555_t1_mprage_sag_p2.dat"
SEQUENCE="/path/to/user_workspace/scan_protocols/20260817_integrated/v151/mprage_3d_wave_FOV256x256x256_res1x1x1_ETL256_R1-1_R2-3_os4_amp8_cyc10_SAG_prisma_v151.seq"
DICOM_DIR="$DATASET_ROOT/mprage_product_unfiltered_normalize"
DICOM_UID="1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0"
WRAPPER="$REPO_ROOT/external/wave-mprage/recon/bart/run_wave_recon.sh"

mkdir -p "$PHASE_A_ROOT/regularization" "$PHASE_A_ROOT/evaluation" "$PHASE_A_ROOT/logs"
if [[ ! -e "$PHASE_A_ROOT/bart_inputs" ]]; then
    ln -s "$SOURCE_ROOT/bart_inputs" "$PHASE_A_ROOT/bart_inputs"
fi
if [[ ! -e "$PHASE_A_ROOT/bart_lambda0_existing_csm_c050" ]]; then
    ln -s "$SOURCE_ROOT/bart_lambda0_existing_csm_c050" "$PHASE_A_ROOT/bart_lambda0_existing_csm_c050"
fi

LOG_PATH="$PHASE_A_ROOT/logs/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "Phase A stage: $STAGE"
echo "Python: $PYTHON_EXECUTABLE"
echo "BART: $BART_EXECUTABLE"
echo "Log: $LOG_PATH"

json_field() {
    "$PYTHON_EXECUTABLE" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

require_gpu_split_validation() {
    local manifest="$PHASE_A_ROOT/split_complex_lambda0_validation/manifest.json"
    [[ -f "$manifest" ]] || {
        echo "Missing split-complex validation manifest: $manifest" >&2
        exit 2
    }
    [[ "$(json_field "$manifest" status)" == "accepted" ]] || {
        echo "Split-complex validation is not accepted: $manifest" >&2
        exit 2
    }
}

run_case() {
    local regularizer="$1"
    local lambda_value="$2"
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/run_bart_regularization.py" \
        --wrapper "$WRAPPER" \
        --bart "$BART_EXECUTABLE" \
        --python "$PYTHON_EXECUTABLE" \
        --bart-input-dir "$SOURCE_ROOT/bart_inputs" \
        --maps "$MAPS" \
        --expected-maps-sha256 "$MAPS_SHA256" \
        --lambda-zero-base "$LAMBDA_ZERO" \
        --output-root "$PHASE_A_ROOT/regularization" \
        --twix "$TWIX" \
        --sequence "$SEQUENCE" \
        --regularizer "$regularizer" \
        --lambda-value "$lambda_value" \
        --block-size 8 \
        --iterations 100 \
        --tolerance 1e-6 \
        --max-eigenvalue 6.70e7 \
        --backend gpu \
        --subject 20260817product-r3x2-phaseA \
        --resume
}

run_reconstructions() {
    require_gpu_split_validation
    local value
    for value in 1e-6 1e-5 1e-4; do
        run_case wavelet "$value"
    done
    # The accepted block-8 pilot at 2e-4 justified this focused corrected-LLR range.
    for value in 2e-5 5e-5 1e-4 2e-4 5e-4; do
        run_case llr "$value"
    done
}

prepare_qc() {
    local evaluation_manifest="$PHASE_A_ROOT/evaluation/evaluation_inputs_manifest.json"
    if [[ ! -f "$evaluation_manifest" ]]; then
        "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/prepare_regularization_evaluation.py" \
            --recon-root "$PHASE_A_ROOT" \
            --dicom-dir "$DICOM_DIR" \
            --dicom-series-uid "$DICOM_UID" \
            --dicom-series-description t1_mprage_sag_p2_ND \
            --expected-cases 9 \
            --expected-dicom-count 256
    else
        echo "Reusing evaluation input manifest: $evaluation_manifest"
    fi

    local reference
    reference="$($PYTHON_EXECUTABLE -c 'import json,sys; print(json.load(open(sys.argv[1]))["dicom_reference_nifti"]["path"])' "$evaluation_manifest")"
    local mask_dir="$PHASE_A_ROOT/evaluation/brain_mask"
    if [[ ! -f "$mask_dir/brain_mask_manifest.json" ]]; then
        "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/prepare_reference_brain_mask.py" \
            --reference "$reference" \
            --output-dir "$mask_dir" \
            --bet "$(command -v bet)" \
            --fractional-threshold 0.25
    else
        echo "Reusing BET mask manifest: $mask_dir/brain_mask_manifest.json"
    fi

    local orientation_dir="$PHASE_A_ROOT/evaluation/orientation_qc"
    if [[ ! -f "$orientation_dir/orientation_report.json" ]]; then
        "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/review_regularization_orientation.py" \
            --dicom-nifti "$reference" \
            --lambda0-nifti "$SOURCE_ROOT/bart_lambda0_existing_csm_c050/nifti/sub-20260817product-r3x2-lambda0/sub-20260817product-r3x2-lambda0_part-mag_BARTWaveLambda0.nii.gz" \
            --output-dir "$orientation_dir"
    else
        echo "Reusing orientation report: $orientation_dir/orientation_report.json"
    fi

    echo
    echo "STOP FOR VISUAL REVIEW"
    echo "BET boundary: $mask_dir/reference_brain_mask_qc.png"
    echo "L/R mapping:  $orientation_dir/orientation_signed_axis_choice.png"
    echo "After reviewing both, run:"
    echo "  bash $SCRIPT_DIR/run_phase_a_remaining.sh --stage all-after-review --confirm-reviewed-mask-and-lr"
}

approve_qc() {
    if ((CONFIRM_VISUAL_QC != 1)); then
        echo "approve-qc requires --confirm-reviewed-mask-and-lr after actual visual review." >&2
        exit 2
    fi
    local evaluation_manifest="$PHASE_A_ROOT/evaluation/evaluation_inputs_manifest.json"
    local reference
    reference="$($PYTHON_EXECUTABLE -c 'import json,sys; print(json.load(open(sys.argv[1]))["dicom_reference_nifti"]["path"])' "$evaluation_manifest")"
    local orientation_dir="$PHASE_A_ROOT/evaluation/orientation_qc"
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/review_regularization_orientation.py" \
        --dicom-nifti "$reference" \
        --lambda0-nifti "$SOURCE_ROOT/bart_lambda0_existing_csm_c050/nifti/sub-20260817product-r3x2-lambda0/sub-20260817product-r3x2-lambda0_part-mag_BARTWaveLambda0.nii.gz" \
        --output-dir "$orientation_dir" \
        --accept-best-signed-axis
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/record_phase_a_qc_approval.py" \
        --brain-mask-manifest "$PHASE_A_ROOT/evaluation/brain_mask/brain_mask_manifest.json" \
        --orientation-report "$orientation_dir/orientation_report.json" \
        --confirm-reviewed-mask-and-lr
}

evaluate_and_present() {
    local mask_manifest="$PHASE_A_ROOT/evaluation/brain_mask/brain_mask_manifest.json"
    local orientation_report="$PHASE_A_ROOT/evaluation/orientation_qc/orientation_report.json"
    [[ "$(json_field "$mask_manifest" status)" == "approved" ]] || {
        echo "BET mask has not been explicitly approved: $mask_manifest" >&2
        exit 2
    }
    [[ "$(json_field "$orientation_report" status)" == "orientation_approved" ]] || {
        echo "Orientation has not been explicitly approved: $orientation_report" >&2
        exit 2
    }
    local metrics_dir="$PHASE_A_ROOT/evaluation/volume_metrics"
    if [[ ! -f "$metrics_dir/metrics_provenance.json" ]]; then
        "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/evaluate_regularization_volume.py" \
            --input-manifest "$PHASE_A_ROOT/evaluation/evaluation_inputs_manifest.json" \
            --orientation-report "$orientation_report" \
            --brain-mask "$PHASE_A_ROOT/evaluation/brain_mask/reference_brain_mask.nii.gz" \
            --output-dir "$metrics_dir"
    else
        echo "Reusing metrics provenance: $metrics_dir/metrics_provenance.json"
    fi

    local presentation_dir="$PHASE_A_ROOT/presentation"
    if [[ ! -f "$presentation_dir/presentation_manifest.json" ]]; then
        "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/generate_phase_a_presentation.py" \
            --metrics-provenance "$metrics_dir/metrics_provenance.json" \
            --grappa-nifti "$DATASET_ROOT/grappa_5x5x5_ncc12/grappa_5x5x5_ncc12_rss_ras.nii.gz" \
            --sense-nifti "$DATASET_ROOT/sense_no_wave_ncc12/recon_lambda0/nifti/sub-20260817product_part-mag_NoWaveSENSELambda0.nii.gz" \
            --output-dir "$presentation_dir"
    else
        echo "Reusing presentation manifest: $presentation_dir/presentation_manifest.json"
    fi
    echo "Phase A presentation: $presentation_dir/phase_a_method_comparison.png"
}

case "$STAGE" in
    reconstruct) run_reconstructions ;;
    prepare-qc) prepare_qc ;;
    approve-qc) approve_qc ;;
    evaluate) evaluate_and_present ;;
    all-before-review)
        run_reconstructions
        prepare_qc
        ;;
    all-after-review)
        approve_qc
        evaluate_and_present
        ;;
esac
