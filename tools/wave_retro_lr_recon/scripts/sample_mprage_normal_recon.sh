#!/usr/bin/env bash
set -euo pipefail

# Prepare measured Wave-MPRAGE data, estimate one BART ESPIRiT map set, and
# export the unregularized-FISTA control plus selected Wavelet reconstruction.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE] [--r3-lambda VALUE] [-g]"
    echo "       [--psf-coefficient-processing smooth|sine-line] (default: automatic sine-line)"
    echo "       [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
    echo "       [--psf-fit-y-min INDEX --psf-fit-y-max INDEX] [--psf-fit-z-min INDEX --psf-fit-z-max INDEX]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
OUTPUT_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3

ECALIB_CROP="0.6"
R3_LAMBDA="3.5e-2"
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
        --r3-lambda) R3_LAMBDA="$2"; shift 2 ;;
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
BART_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
BART_OUTPUT_ROOT="$OUTPUT_ROOT/normal/bart_output"
NIFTI_OUTPUT_ROOT="$OUTPUT_ROOT/normal/nifti"
ECALIB_RECORD="$BART_OUTPUT_ROOT/ecalib_command.txt"

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

# Sine-line processing selects its range automatically by default. Providing
# both bounds is a reproducible manual override; smooth remains available.
if [[ "$PSF_COEFFICIENT_PROCESSING" == "smooth" ]]; then
    [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]] || { echo "Error: PSF kx bounds require sine-line processing." >&2; exit 2; }
    python "$SCRIPT_DIR/prepare_mprage_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --psf-coefficient-processing smooth "${SPATIAL_ARGS[@]}"
elif [[ "$PSF_COEFFICIENT_PROCESSING" == "sine-line" ]]; then
    if [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_mprage_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --psf-coefficient-processing sine-line "${SPATIAL_ARGS[@]}"
    elif [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_mprage_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --psf-coefficient-processing sine-line --psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX" "${SPATIAL_ARGS[@]}"
    else
        echo "Error: manual sine-line processing requires both PSF kx bounds; omit both for automatic selection." >&2
        exit 2
    fi
else
    echo "Error: PSF coefficient processing must be smooth or sine-line." >&2
    exit 2
fi
mkdir -p "$BART_OUTPUT_ROOT" "$NIFTI_OUTPUT_ROOT"

# Estimate one sensitivity-map set from the integrated ACS k-space. BART
# ecalib has no -g option; Wave uses CPU unless the user explicitly passes -g.
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT_ROOT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$BART_OUTPUT_ROOT/coil_sens.hdr" && -f "$BART_OUTPUT_ROOT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
    [[ "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM was generated with a different ecalib command." >&2; exit 2; }
    echo "Reusing recorded ecalib result: $BART_OUTPUT_ROOT/coil_sens"
elif [[ -e "$BART_OUTPUT_ROOT/coil_sens.hdr" || -e "$BART_OUTPUT_ROOT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM or ecalib command record in $BART_OUTPUT_ROOT" >&2
    exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT_ROOT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi

SAMPLING_CLASS="$(<"$BART_INPUTS/sampling_class.txt")"
if [[ "$SAMPLING_CLASS" == "R3x1" ]]; then
    # Unregularized FISTA is retained as the case-matched ablation control.
    mkdir -p "$BART_OUTPUT_ROOT/fista_r0" "$NIFTI_OUTPUT_ROOT/fista_r0"
    if [[ "$USE_GPU" == true ]]; then
        bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
    else
        bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
    fi
    printf '%s\n' "${WAVE_COMMAND% }" > "$BART_OUTPUT_ROOT/fista_r0/wave_command.txt"
    python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$BART_INPUTS" --image "$BART_OUTPUT_ROOT/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT_ROOT/fista_r0" --suffix BARTWaveMPRAGENormalFISTAR0

    # The selected pure-lattice rerun value is the default positive Wavelet arm.
    mkdir -p "$BART_OUTPUT_ROOT/optimal_wavelet" "$NIFTI_OUTPUT_ROOT/optimal_wavelet"
    if [[ "$USE_GPU" == true ]]; then
        bart wave -g -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/optimal_wavelet/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/optimal_wavelet/image_wave"
    else
        bart wave -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/optimal_wavelet/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/optimal_wavelet/image_wave"
    fi
    printf '%s\n' "${WAVE_COMMAND% }" > "$BART_OUTPUT_ROOT/optimal_wavelet/wave_command.txt"
    python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$BART_INPUTS" --image "$BART_OUTPUT_ROOT/optimal_wavelet/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT_ROOT/optimal_wavelet" --suffix BARTWaveMPRAGENormalOptimalWavelet
elif [[ "$SAMPLING_CLASS" == "R1" ]]; then
    # No R1 Wavelet optimum was selected; do not transfer the R3x1 lambda.
    mkdir -p "$BART_OUTPUT_ROOT/fista_r0" "$NIFTI_OUTPUT_ROOT/fista_r0"
    if [[ "$USE_GPU" == true ]]; then
        bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
    else
        bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
        printf -v WAVE_COMMAND '%q ' bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT_ROOT/fista_r0/image_wave"
    fi
    printf '%s\n' "${WAVE_COMMAND% }" > "$BART_OUTPUT_ROOT/fista_r0/wave_command.txt"
    python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$BART_INPUTS" --image "$BART_OUTPUT_ROOT/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT_ROOT/fista_r0" --suffix BARTWaveMPRAGENormalFISTAR0
else
    echo "Error: unsupported prepared sampling class $SAMPLING_CLASS" >&2
    exit 2
fi

echo "Normal MPRAGE reconstruction complete: $OUTPUT_ROOT/normal"
if [[ -f "$OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png" ]]; then
    echo "PSF visual-assessment plot: $OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png"
    echo "PSF full-range plot: $OUTPUT_ROOT/normal/PSF_COEFFICIENTS_FULL_RANGE.png"
    echo "PSF plane comparison: $OUTPUT_ROOT/normal/PSF_PLANE_COMPARISON.png"
    echo "For unexpected reconstruction artifacts, review that plot and tools/wave_retro_lr_recon/TROUBLESHOOTING.md."
fi
