#!/usr/bin/env bash
set -euo pipefail

# Prepare and reconstruct every echo of measured native R3x1 Wave-GRE.
# Python never launches BART; every ecalib/wave command is explicit below.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE] [-g]"
    echo "       [--psf-coefficient-processing smooth|sine-line]"
    echo "       [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
OUTPUT_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3

ECALIB_CROP="0.6"
USE_GPU=false
PSF_COEFFICIENT_PROCESSING="smooth"
PSF_FIT_KX_MIN=""
PSF_FIT_KX_MAX=""
GRE_SHARED_WAVELET_LAMBDA="0.015"
while (($#)); do
    case "$1" in
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --psf-coefficient-processing) PSF_COEFFICIENT_PROCESSING="$2"; shift 2 ;;
        --psf-fit-kx-min) PSF_FIT_KX_MIN="$2"; shift 2 ;;
        --psf-fit-kx-max) PSF_FIT_KX_MAX="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BART_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
BART_OUTPUT="$OUTPUT_ROOT/normal/bart_output"
NIFTI_OUTPUT="$OUTPUT_ROOT/normal/nifti"
ECALIB_RECORD="$BART_OUTPUT/ecalib_command.txt"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

if [[ "$PSF_COEFFICIENT_PROCESSING" == "smooth" ]]; then
    [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]] || { echo "Error: kx bounds require sine-line." >&2; exit 2; }
    python "$SCRIPT_DIR/prepare_gre_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE"
elif [[ "$PSF_COEFFICIENT_PROCESSING" == "sine-line" ]]; then
    if [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_gre_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --psf-coefficient-processing sine-line
    elif [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_gre_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --psf-coefficient-processing sine-line --psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX"
    else
        echo "Error: manual sine-line fitting requires both bounds." >&2; exit 2
    fi
else
    echo "Error: PSF coefficient processing must be smooth or sine-line." >&2; exit 2
fi

mkdir -p "$BART_OUTPUT"
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$BART_OUTPUT/coil_sens.hdr" && -f "$BART_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" && "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing native CSM command differs." >&2; exit 2; }
elif [[ -e "$BART_OUTPUT/coil_sens.hdr" || -e "$BART_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete native CSM state in $BART_OUTPUT" >&2; exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi
ECHO_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["echoes"]))' "$BART_INPUTS/manifest.json")"
[[ "$ECHO_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "Error: invalid GRE echo count: $ECHO_COUNT" >&2; exit 2; }

run_branch() {
    local branch="$1"
    local shared_lambda="$2"
    local suffix="$3"
    local branch_root="$BART_OUTPUT/$branch"
    local echo_number echo_label image_base
    local -a conversion_args=(--bart-inputs "$BART_INPUTS")
    mkdir -p "$NIFTI_OUTPUT/$branch"
    for ((echo_number = 1; echo_number <= ECHO_COUNT; echo_number++)); do
        printf -v echo_label 'echo-%02d' "$echo_number"
        image_base="$branch_root/$echo_label/image_wave"
        mkdir -p "$branch_root/$echo_label"
        if [[ "$USE_GPU" == true ]]; then
            bart wave -g -w -f -r "$shared_lambda" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image_base"
            printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r "$shared_lambda" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image_base"
        else
            bart wave -w -f -r "$shared_lambda" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image_base"
            printf -v WAVE_COMMAND '%q ' bart wave -w -f -r "$shared_lambda" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image_base"
        fi
        printf '%s\n' "${WAVE_COMMAND% }" > "$branch_root/$echo_label/wave_command.txt"
        conversion_args+=(--image "$image_base")
    done
    python "$SCRIPT_DIR/convert_gre_bart_to_nifti.py" "${conversion_args[@]}" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT/$branch" --suffix "$suffix"
}

run_branch fista_r0 0 BARTWaveGRENativeR3x1FISTAR0
run_branch selected_wavelet "$GRE_SHARED_WAVELET_LAMBDA" BARTWaveGRENativeR3x1SelectedWavelet

echo "Normal multi-echo GRE reconstruction complete: $OUTPUT_ROOT/normal"
