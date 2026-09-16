#!/usr/bin/env bash
# Prepare and resumably reconstruct only native-grid retrospective R3x3 Wave-GRE.

set -euo pipefail

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE] [-g]"
    echo "       [--psf-coefficient-processing smooth|sine-line] (default: automatic sine-line)"
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
PSF_COEFFICIENT_PROCESSING="sine-line"
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
NORMAL_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
NORMAL_OUTPUT="$OUTPUT_ROOT/normal/bart_output"
CASE_ROOT="$OUTPUT_ROOT/retro/native_r3x3"
CASE_INPUTS="$CASE_ROOT/bart_inputs"
ECALIB_RECORD="$NORMAL_OUTPUT/ecalib_command.txt"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

# Retrospective compatibility may attest to legacy normal artifacts but never
# rewrites their historical manifest or recalibrates their measured PSFs.
if [[ "$PSF_COEFFICIENT_PROCESSING" == "smooth" ]]; then
    [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]] || {
        echo "Error: kx bounds require sine-line processing." >&2; exit 2;
    }
    python "$SCRIPT_DIR/prepare_gre_retro_r3x3.py" \
        "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
        --psf-coefficient-processing smooth
elif [[ "$PSF_COEFFICIENT_PROCESSING" == "sine-line" ]]; then
    if [[ -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_gre_retro_r3x3.py" \
            "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
            --psf-coefficient-processing sine-line
    elif [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then
        python "$SCRIPT_DIR/prepare_gre_retro_r3x3.py" \
            "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" \
            --psf-coefficient-processing sine-line \
            --psf-fit-kx-min "$PSF_FIT_KX_MIN" \
            --psf-fit-kx-max "$PSF_FIT_KX_MAX"
    else
        echo "Error: manual sine-line fitting requires both bounds." >&2
        exit 2
    fi
else
    echo "Error: PSF coefficient processing must be smooth or sine-line." >&2
    exit 2
fi

# Native CSM provenance remains strict and independent of PSF compatibility.
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

ECHO_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["echo_count"])' "$CASE_INPUTS/manifest.json")"
[[ "$ECHO_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "Error: invalid GRE echo count: $ECHO_COUNT" >&2; exit 2; }

run_branch() {
    local branch="$1"
    local lambda_value="$2"
    local suffix="$3"
    local branch_root="$CASE_ROOT/bart_output/$branch"
    local nifti_root="$CASE_ROOT/nifti/$branch"
    local base echo_number echo_label psf kspace echo_root image record
    local expected_cpu expected_gpu
    local -a cpu_command gpu_command run_command
    local -a conversion_args=(--bart-inputs "$CASE_INPUTS")
    mkdir -p "$nifti_root"
    for ((echo_number = 1; echo_number <= ECHO_COUNT; echo_number++)); do
        printf -v echo_label 'echo-%02d' "$echo_number"
        psf="$CASE_INPUTS/psf_$echo_label"
        kspace="$CASE_INPUTS/wave_kspace_$echo_label"
        echo_root="$branch_root/$echo_label"
        image="$echo_root/image_wave"
        record="$echo_root/wave_command.txt"
        for base in "$psf" "$kspace"; do
            [[ -f "$base.hdr" && -f "$base.cfl" ]] || {
                echo "Error: required complete R3x3 echo input is missing: $base.{hdr,cfl}" >&2
                exit 2
            }
        done
        conversion_args+=(--image "$image")
        cpu_command=(bart wave -w -f -r "$lambda_value" -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$psf" "$kspace" "$image")
        gpu_command=(bart wave -g -w -f -r "$lambda_value" -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$psf" "$kspace" "$image")
        printf -v expected_cpu '%q ' "${cpu_command[@]}"
        printf -v expected_gpu '%q ' "${gpu_command[@]}"
        expected_cpu="${expected_cpu% }"
        expected_gpu="${expected_gpu% }"

        if [[ -f "$record" ]]; then
            [[ -f "$image.hdr" && -f "$image.cfl" ]] || {
                echo "Error: $branch/$echo_label has a command record but incomplete output." >&2
                exit 2
            }
            if [[ "$(<"$record")" != "$expected_cpu" && "$(<"$record")" != "$expected_gpu" ]]; then
                echo "Error: completed $branch/$echo_label used a different Wave command." >&2
                exit 2
            fi
            echo "Skipping completed $branch/$echo_label: $record"
            continue
        fi
        if [[ -e "$image.hdr" || -e "$image.cfl" ]]; then
            echo "Found unrecorded partial output for $branch/$echo_label; BART will replace it."
        fi
        mkdir -p "$echo_root"
        if [[ "$USE_GPU" == true ]]; then
            run_command=("${gpu_command[@]}")
        else
            run_command=("${cpu_command[@]}")
        fi
        "${run_command[@]}"
        printf -v WAVE_COMMAND '%q ' "${run_command[@]}"
        printf '%s\n' "${WAVE_COMMAND% }" > "$record"
    done
    python "$SCRIPT_DIR/convert_gre_bart_to_nifti.py" \
        "${conversion_args[@]}" \
        --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" \
        --output "$nifti_root" --suffix "$suffix"
}

run_branch fista_r0 0 BARTWaveGRENativeR3x3FISTAR0
run_branch selected_wavelet "$GRE_SHARED_WAVELET_LAMBDA" BARTWaveGRENativeR3x3SelectedWavelet

echo "Native R3x3 retrospective multi-echo GRE reconstruction complete: $CASE_ROOT"
