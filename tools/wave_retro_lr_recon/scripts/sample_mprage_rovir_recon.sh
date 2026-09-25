#!/usr/bin/env bash
set -euo pipefail

# Reconstruct prepared Wave-MPRAGE data with reviewed ROVir coils.

usage() {
    cat <<'EOF'
Usage:
  sample_mprage_rovir_recon.sh inspect TWIX.dat OUTPUT_ROOT SEQUENCE.seq
  sample_mprage_rovir_recon.sh run TWIX.dat OUTPUT_ROOT SEQUENCE.seq --virtual-coils N [ROI options] [run options]

ROI options (choose exactly one mode):
  --use-recommended
  --null-box "ro=A:B,lin=C:D,par=E:F"   repeat without a count limit
  --null-box-file ROI_BOXES.json

Run options:
  --ecalib-crop VALUE    override normal ecalib crop or the MPRAGE default (0.6)
  -g                     use GPU for BART Wave only

BART must already be on PATH. The three source arguments must exactly match the
accepted prepared MPRAGE inputs recorded below OUTPUT_ROOT.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
STAGE="$1"
shift
if [[ "$STAGE" == "-h" || "$STAGE" == "--help" ]]; then usage; exit 0; fi
[[ "$STAGE" == "inspect" || "$STAGE" == "run" ]] || {
    echo "Error: command must be inspect or run." >&2
    usage >&2
    exit 2
}
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
RECONSTRUCTION_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3
[[ -d "$RECONSTRUCTION_ROOT" ]] || {
    echo "Error: completed reconstruction root does not exist: $RECONSTRUCTION_ROOT" >&2
    exit 2
}
RECONSTRUCTION_ROOT="$(cd -- "$RECONSTRUCTION_ROOT" && pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW="$SCRIPT_DIR/mprage_rovir_workflow.py"
ROVIR_ROOT="$RECONSTRUCTION_ROOT/normal/rovir"
FEASIBILITY_ROOT="$ROVIR_ROOT/feasibility"
echo "ROVir reconstruction root: $RECONSTRUCTION_ROOT"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }
python "$WORKFLOW" validate-invocation "$TWIX_FILE" "$RECONSTRUCTION_ROOT" "$SEQUENCE_FILE" >/dev/null
# Use the validated canonical source paths recorded by the normal manifest.
TWIX_FILE="$(python "$WORKFLOW" context "$RECONSTRUCTION_ROOT" --field twix)"
SEQUENCE_FILE="$(python "$WORKFLOW" context "$RECONSTRUCTION_ROOT" --field sequence)"

if [[ "$STAGE" == "inspect" ]]; then
    [[ $# -eq 0 ]] || { echo "Error: inspect accepts exactly TWIX.dat OUTPUT_ROOT SEQUENCE.seq." >&2; exit 2; }
    python "$WORKFLOW" inspect-prepare "$RECONSTRUCTION_ROOT" >/dev/null
    CALIBRATION="$FEASIBILITY_ROOT/inputs/physical_calibration"
    LOGS="$FEASIBILITY_ROOT/logs"
    mkdir -p "$LOGS"
    INPUT_HDR_HASH="$(sha256sum "$CALIBRATION/physical_set4_kspace.hdr" | awk '{print $1}')"
    INPUT_CFL_HASH="$(sha256sum "$CALIBRATION/physical_set4_kspace.cfl" | awk '{print $1}')"
    IMAGE_COMMAND_RECORD="$LOGS/calibration_image_commands.txt"
    EXPECTED_IMAGE_COMMAND_RECORD="input_header_sha256=$INPUT_HDR_HASH
input_payload_sha256=$INPUT_CFL_HASH
bart fft -iu 7 $CALIBRATION/physical_set4_kspace $CALIBRATION/physical_set4_coil_images
bart rss 8 $CALIBRATION/physical_set4_coil_images $CALIBRATION/physical_set4_rss"
    GENERATED_IMAGES=false
    if [[ -f "$CALIBRATION/physical_set4_coil_images.hdr" && -f "$CALIBRATION/physical_set4_coil_images.cfl" && -f "$CALIBRATION/physical_set4_rss.hdr" && -f "$CALIBRATION/physical_set4_rss.cfl" ]]; then
        [[ -f "$IMAGE_COMMAND_RECORD" && "$(<"$IMAGE_COMMAND_RECORD")" == "$EXPECTED_IMAGE_COMMAND_RECORD" ]] || {
            echo "Error: existing ACS IFFT/RSS images lack the matching source/command record." >&2
            exit 2
        }
        echo "Reusing complete ACS IFFT/RSS images."
    elif [[ -e "$CALIBRATION/physical_set4_coil_images.hdr" || -e "$CALIBRATION/physical_set4_coil_images.cfl" || -e "$CALIBRATION/physical_set4_rss.hdr" || -e "$CALIBRATION/physical_set4_rss.cfl" ]]; then
        echo "Error: incomplete ACS IFFT/RSS output; refusing implicit cleanup." >&2
        exit 2
    else
        bart fft -iu 7 "$CALIBRATION/physical_set4_kspace" "$CALIBRATION/physical_set4_coil_images"
        bart rss 8 "$CALIBRATION/physical_set4_coil_images" "$CALIBRATION/physical_set4_rss"
        printf '%s\n' "$EXPECTED_IMAGE_COMMAND_RECORD" >"$IMAGE_COMMAND_RECORD"
        GENERATED_IMAGES=true
    fi
    if [[ "$GENERATED_IMAGES" == true ]]; then
        bart version >"$LOGS/bart_version_images.txt" 2>&1
    else
        [[ -f "$LOGS/bart_version_images.txt" ]] || { echo "Error: reused ACS images lack a BART version record." >&2; exit 2; }
    fi
    python "$WORKFLOW" inspect-finish "$RECONSTRUCTION_ROOT" "$LOGS/bart_version_images.txt" >/dev/null
    echo "Inspect figures: $FEASIBILITY_ROOT/diagnostics/calibration_views"
    echo "ROI recommendation: $FEASIBILITY_ROOT/diagnostics/roi_recommendation/roi_recommendation.json"
    exit 0
fi

VIRTUAL_COILS=""
ECALIB_CROP=""
USE_GPU=false
USE_RECOMMENDED=false
NULL_BOX_FILE=""
NULL_BOXES=()
while (($#)); do
    case "$1" in
        --virtual-coils) VIRTUAL_COILS="$2"; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --use-recommended) USE_RECOMMENDED=true; shift ;;
        --null-box) NULL_BOXES+=("$2"); shift 2 ;;
        --null-box-file) NULL_BOX_FILE="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$VIRTUAL_COILS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --virtual-coils N is required." >&2; exit 2; }
ROI_MODES=0
[[ "$USE_RECOMMENDED" == true ]] && ROI_MODES=$((ROI_MODES + 1))
[[ ${#NULL_BOXES[@]} -gt 0 ]] && ROI_MODES=$((ROI_MODES + 1))
[[ -n "$NULL_BOX_FILE" ]] && ROI_MODES=$((ROI_MODES + 1))
[[ $ROI_MODES -eq 1 ]] || { echo "Error: choose exactly one ROI input mode." >&2; exit 2; }

# Ensure inspect is complete before creating a reviewed candidate.
[[ -f "$FEASIBILITY_ROOT/manifests/calibration_images.json" ]] || {
    echo "Error: run 'inspect' and review its indexed ACS figures first." >&2
    exit 2
}
CANDIDATE_ARGS=()
if [[ ${#NULL_BOXES[@]} -gt 0 ]]; then
    for box in "${NULL_BOXES[@]}"; do CANDIDATE_ARGS+=(--null-box "$box"); done
elif [[ -n "$NULL_BOX_FILE" ]]; then
    CANDIDATE_ARGS+=(--null-box-file "$NULL_BOX_FILE")
else
    CANDIDATE_ARGS+=(--use-recommended)
fi
CANDIDATE_ID="$(python "$WORKFLOW" candidate "$RECONSTRUCTION_ROOT" "${CANDIDATE_ARGS[@]}" --id-only)"
OVERLAY="$FEASIBILITY_ROOT/masks/candidates/$CANDIDATE_ID/review_union_outline.png"
echo "Optional troubleshooting overlay: $OVERLAY"
echo "Candidate ID: $CANDIDATE_ID"
mkdir -p "$FEASIBILITY_ROOT/logs" "$FEASIBILITY_ROOT/transforms/rovir_full"
python "$WORKFLOW" approve "$RECONSTRUCTION_ROOT" "$CANDIDATE_ID" >/dev/null
POSITIVE="$FEASIBILITY_ROOT/inputs/rovir/positive_signal_images"
NEGATIVE="$FEASIBILITY_ROOT/inputs/rovir/negative_interference_images"
TRANSFORM="$FEASIBILITY_ROOT/transforms/rovir_full/transform"
ROVIR_COMMAND="bart rovir $POSITIVE $NEGATIVE $TRANSFORM"
GENERATED_TRANSFORM=false
if [[ -f "$TRANSFORM.hdr" && -f "$TRANSFORM.cfl" ]]; then
    [[ -f "$FEASIBILITY_ROOT/transforms/rovir_full/rovir_command.txt" ]] || { echo "Error: existing transform lacks command record." >&2; exit 2; }
    [[ "$(<"$FEASIBILITY_ROOT/transforms/rovir_full/rovir_command.txt")" == "$ROVIR_COMMAND" ]] || { echo "Error: existing transform used another command." >&2; exit 2; }
    echo "Reusing complete BART ROVir transform."
elif [[ -e "$TRANSFORM.hdr" || -e "$TRANSFORM.cfl" ]]; then
    echo "Error: incomplete BART ROVir transform." >&2
    exit 2
else
    bart rovir "$POSITIVE" "$NEGATIVE" "$TRANSFORM"
    printf '%s\n' "$ROVIR_COMMAND" >"$FEASIBILITY_ROOT/transforms/rovir_full/rovir_command.txt"
    GENERATED_TRANSFORM=true
fi
if [[ "$GENERATED_TRANSFORM" == true ]]; then
    bart version >"$FEASIBILITY_ROOT/logs/bart_version_rovir.txt" 2>&1
else
    [[ -f "$FEASIBILITY_ROOT/logs/bart_version_rovir.txt" ]] || { echo "Error: reused transform lacks a BART version record." >&2; exit 2; }
fi
python "$WORKFLOW" transform-prepare "$RECONSTRUCTION_ROOT" "$FEASIBILITY_ROOT/logs/bart_version_rovir.txt" "$VIRTUAL_COILS" >/dev/null

if [[ -z "$ECALIB_CROP" ]]; then
    ECALIB_CROP="$(python "$WORKFLOW" context "$RECONSTRUCTION_ROOT" --field ecalib_crop)"
fi
INPUTS="$ROVIR_ROOT/bart_inputs"
BART_OUTPUT="$ROVIR_ROOT/bart_output"
NIFTI_OUTPUT="$ROVIR_ROOT/nifti/fista_r0"
mkdir -p "$BART_OUTPUT/fista_r0" "$NIFTI_OUTPUT"
printf -v ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
ECALIB_COMMAND="${ECALIB_COMMAND% }"
if [[ -f "$BART_OUTPUT/coil_sens.hdr" && -f "$BART_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$BART_OUTPUT/ecalib_command.txt" && "$(<"$BART_OUTPUT/ecalib_command.txt")" == "$ECALIB_COMMAND" ]] || { echo "Error: existing ROVir CSM used another ecalib command." >&2; exit 2; }
else
    [[ ! -e "$BART_OUTPUT/coil_sens.hdr" && ! -e "$BART_OUTPUT/coil_sens.cfl" ]] || { echo "Error: incomplete ROVir CSM." >&2; exit 2; }
    bart ecalib -m 1 -c "$ECALIB_CROP" "$INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
    printf '%s\n' "$ECALIB_COMMAND" >"$BART_OUTPUT/ecalib_command.txt"
fi
if [[ "$USE_GPU" == true ]]; then
    WAVE_ARGS=(-g -w -f -r 0 -i 100 -t 1e-6)
else
    WAVE_ARGS=(-w -f -r 0 -i 100 -t 1e-6)
fi
printf -v WAVE_COMMAND '%q ' bart wave "${WAVE_ARGS[@]}" "$BART_OUTPUT/coil_sens" "$INPUTS/psf" "$INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
WAVE_COMMAND="${WAVE_COMMAND% }"
if [[ -f "$BART_OUTPUT/fista_r0/image_wave.hdr" && -f "$BART_OUTPUT/fista_r0/image_wave.cfl" ]]; then
    [[ -f "$BART_OUTPUT/fista_r0/wave_command.txt" && "$(<"$BART_OUTPUT/fista_r0/wave_command.txt")" == "$WAVE_COMMAND" ]] || { echo "Error: existing ROVir Wave output used another command." >&2; exit 2; }
else
    [[ ! -e "$BART_OUTPUT/fista_r0/image_wave.hdr" && ! -e "$BART_OUTPUT/fista_r0/image_wave.cfl" ]] || { echo "Error: incomplete ROVir Wave output." >&2; exit 2; }
    bart wave "${WAVE_ARGS[@]}" "$BART_OUTPUT/coil_sens" "$INPUTS/psf" "$INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
    printf '%s\n' "$WAVE_COMMAND" >"$BART_OUTPUT/fista_r0/wave_command.txt"
fi
mapfile -d '' -t MAGNITUDE_NIFTIS < <(
    find "$NIFTI_OUTPUT" -type f -name '*_part-mag_*.nii.gz' -print0
)
mapfile -d '' -t PHASE_NIFTIS < <(
    find "$NIFTI_OUTPUT" -type f -name '*_part-phase_*.nii.gz' -print0
)
if [[ ${#MAGNITUDE_NIFTIS[@]} -eq 1 && ${#PHASE_NIFTIS[@]} -eq 1 \
      && -f "${MAGNITUDE_NIFTIS[0]%.nii.gz}.json" \
      && -f "${PHASE_NIFTIS[0]%.nii.gz}.json" ]]; then
    echo "Reusing complete nested ROVir magnitude/phase NIfTI outputs."
elif [[ -n "$(find "$NIFTI_OUTPUT" -type f -print -quit)" ]]; then
    echo "Error: existing ROVir NIfTI output is incomplete or ambiguous." >&2
    exit 2
else
    python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
        --bart-inputs "$INPUTS" --image "$BART_OUTPUT/fista_r0/image_wave" \
        --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT" \
        --suffix "BARTWaveMPRAGEROVir${VIRTUAL_COILS}FISTAR0"
fi
python "$WORKFLOW" finalize "$RECONSTRUCTION_ROOT" "$VIRTUAL_COILS" >/dev/null
echo "Completed canonical ROVir branch: $ROVIR_ROOT"
echo "Contract: $ROVIR_ROOT/manifest.json"
echo "QC directory: $ROVIR_ROOT/qc"
