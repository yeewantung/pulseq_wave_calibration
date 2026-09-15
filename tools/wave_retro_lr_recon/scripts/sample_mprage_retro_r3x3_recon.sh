#!/usr/bin/env bash
# Prepare and reconstruct only native-grid retrospective R3x3 Wave-MPRAGE.

set -euo pipefail

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE] [-g]"
    echo "       [--psf-coefficient-processing smooth|sine-line] (default: automatic sine-line)"
    echo "       [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
    echo "       [--psf-fit-y-min INDEX --psf-fit-y-max INDEX]"
    echo "       [--psf-fit-z-min INDEX --psf-fit-z-max INDEX]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
OUTPUT_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3

ECALIB_CROP="0.6"
USE_GPU=false
PSF_COEFFICIENT_PROCESSING="sine-line"
PSF_FIT_KX_MIN=""
PSF_FIT_KX_MAX=""
PSF_FIT_Y_MIN=""
PSF_FIT_Y_MAX=""
PSF_FIT_Z_MIN=""
PSF_FIT_Z_MAX=""
while (($#)); do
    case "$1" in
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --psf-coefficient-processing) PSF_COEFFICIENT_PROCESSING="$2"; shift 2 ;;
        --psf-fit-kx-min) PSF_FIT_KX_MIN="$2"; shift 2 ;;
        --psf-fit-kx-max) PSF_FIT_KX_MAX="$2"; shift 2 ;;
        --psf-fit-y-min) PSF_FIT_Y_MIN="$2"; shift 2 ;;
        --psf-fit-y-max) PSF_FIT_Y_MAX="$2"; shift 2 ;;
        --psf-fit-z-min) PSF_FIT_Z_MIN="$2"; shift 2 ;;
        --psf-fit-z-max) PSF_FIT_Z_MAX="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NORMAL_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
NORMAL_OUTPUT="$OUTPUT_ROOT/normal/bart_output"
CASE_ROOT="$OUTPUT_ROOT/retro/native_r3x3"
CASE_INPUTS="$CASE_ROOT/bart_inputs"
ECALIB_RECORD="$NORMAL_OUTPUT/ecalib_command.txt"
R3X3_LAMBDA="4.5e-2"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

SPATIAL_ARGS=()
if [[ -n "$PSF_FIT_Y_MIN" && -n "$PSF_FIT_Y_MAX" ]]; then
    SPATIAL_ARGS+=(--psf-fit-y-min "$PSF_FIT_Y_MIN" --psf-fit-y-max "$PSF_FIT_Y_MAX")
elif [[ -n "$PSF_FIT_Y_MIN" || -n "$PSF_FIT_Y_MAX" ]]; then
    echo "Error: manual PSF y fitting requires both bounds." >&2; exit 2
fi
if [[ -n "$PSF_FIT_Z_MIN" && -n "$PSF_FIT_Z_MAX" ]]; then
    SPATIAL_ARGS+=(--psf-fit-z-min "$PSF_FIT_Z_MIN" --psf-fit-z-max "$PSF_FIT_Z_MAX")
elif [[ -n "$PSF_FIT_Z_MIN" || -n "$PSF_FIT_Z_MAX" ]]; then
    echo "Error: manual PSF z fitting requires both bounds." >&2; exit 2
fi

# Preparation reuses compatible normal inputs and the identical calibrated PSF.
if [[ "$PSF_COEFFICIENT_PROCESSING" == "smooth" ]]; then
    [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]] || {
        echo "Error: PSF kx bounds require sine-line processing." >&2; exit 2;
    }
    python "$SCRIPT_DIR/prepare_mprage_retro_r3x3.py" \
        "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
        --psf-coefficient-processing smooth "${SPATIAL_ARGS[@]}"
elif [[ "$PSF_COEFFICIENT_PROCESSING" == "sine-line" ]]; then
    if [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_mprage_retro_r3x3.py" \
            "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
            --psf-coefficient-processing sine-line "${SPATIAL_ARGS[@]}"
    elif [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_mprage_retro_r3x3.py" \
            "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
            --psf-coefficient-processing sine-line \
            --psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX" \
            "${SPATIAL_ARGS[@]}"
    else
        echo "Error: manual sine-line processing requires both PSF kx bounds." >&2
        exit 2
    fi
else
    echo "Error: PSF coefficient processing must be smooth or sine-line." >&2
    exit 2
fi

# The native CSM is estimated once and is shared unchanged by R3x3.
mkdir -p "$NORMAL_OUTPUT"
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$NORMAL_OUTPUT/coil_sens.hdr" && -f "$NORMAL_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
    [[ "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || {
        echo "Error: existing CSM was generated with a different ecalib command." >&2; exit 2;
    }
    echo "Reusing recorded ecalib result: $NORMAL_OUTPUT/coil_sens"
elif [[ -e "$NORMAL_OUTPUT/coil_sens.hdr" || -e "$NORMAL_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM or ecalib command record in $NORMAL_OUTPUT" >&2
    exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" \
        "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi

mkdir -p \
    "$CASE_ROOT/bart_output/fista_r0" "$CASE_ROOT/nifti/fista_r0" \
    "$CASE_ROOT/bart_output/optimal_wavelet" "$CASE_ROOT/nifti/optimal_wavelet"

# Preserve the unregularized control.
if [[ "$USE_GPU" == true ]]; then
    bart wave -g -w -f -r 0 -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/fista_r0/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/fista_r0/image_wave"
else
    bart wave -w -f -r 0 -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/fista_r0/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -w -f -r 0 -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/fista_r0/image_wave"
fi
printf '%s\n' "${WAVE_COMMAND% }" > "$CASE_ROOT/bart_output/fista_r0/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
    --bart-inputs "$CASE_INPUTS" \
    --image "$CASE_ROOT/bart_output/fista_r0/image_wave" \
    --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" \
    --output "$CASE_ROOT/nifti/fista_r0" \
    --suffix BARTWaveMPRAGENativeR3x3FISTAR0

# Apply the manually selected synthetic pure-mask Wavelet regularization.
if [[ "$USE_GPU" == true ]]; then
    bart wave -g -w -f -r "$R3X3_LAMBDA" -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/optimal_wavelet/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r "$R3X3_LAMBDA" -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/optimal_wavelet/image_wave"
else
    bart wave -w -f -r "$R3X3_LAMBDA" -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/optimal_wavelet/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -w -f -r "$R3X3_LAMBDA" -i 100 -t 1e-6 \
        "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" \
        "$CASE_ROOT/bart_output/optimal_wavelet/image_wave"
fi
printf '%s\n' "${WAVE_COMMAND% }" > "$CASE_ROOT/bart_output/optimal_wavelet/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
    --bart-inputs "$CASE_INPUTS" \
    --image "$CASE_ROOT/bart_output/optimal_wavelet/image_wave" \
    --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" \
    --output "$CASE_ROOT/nifti/optimal_wavelet" \
    --suffix BARTWaveMPRAGENativeR3x3OptimalWavelet

echo "Native R3x3 retrospective MPRAGE reconstructions complete: $CASE_ROOT"
