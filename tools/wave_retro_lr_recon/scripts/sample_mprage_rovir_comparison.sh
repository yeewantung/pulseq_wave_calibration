#!/usr/bin/env bash
set -euo pipefail

# Prepare, reconstruct, or review matched ROVir-24 and ROVir-48 controls.

usage() {
    echo "Usage: $0 STAGE TWIX.dat SEQUENCE.seq ACCEPTED_NORMAL_ROOT FEASIBILITY_ROOT OUTPUT_ROOT [--channel-counts CSV] [--ecalib-crop VALUE] [-g]"
    echo "Stages: prepare, reconstruct, qc"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 6 ]] || { usage >&2; exit 2; }
STAGE="$1"
TWIX_FILE="$2"
SEQUENCE_FILE="$3"
ACCEPTED_NORMAL_ROOT="${4%/}"
FEASIBILITY_ROOT="${5%/}"
OUTPUT_ROOT="${6%/}"
shift 6

ECALIB_CROP=0.1
CHANNEL_COUNTS_CSV="24,48"
USE_GPU=false
while (($#)); do
    case "$1" in
        --channel-counts) CHANNEL_COUNTS_CSV="$2"; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
IFS=',' read -r -a CHANNEL_COUNTS <<<"$CHANNEL_COUNTS_CSV"
[[ ${#CHANNEL_COUNTS[@]} -ge 1 ]] || { echo "Error: channel-count list is empty." >&2; exit 2; }
for count in "${CHANNEL_COUNTS[@]}"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Error: invalid channel count: $count" >&2; exit 2; }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

validate_inputs() {
    python "$SCRIPT_DIR/prepare_mprage_rovir_comparison.py" \
        "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" \
        "$FEASIBILITY_ROOT" "$OUTPUT_ROOT" --channel-counts "${CHANNEL_COUNTS[@]}" >/dev/null
}

case "$STAGE" in
    prepare)
        python "$SCRIPT_DIR/prepare_mprage_rovir_comparison.py" \
            "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" \
            "$FEASIBILITY_ROOT" "$OUTPUT_ROOT" --channel-counts "${CHANNEL_COUNTS[@]}"
        ;;
    reconstruct)
        command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }
        [[ -f "$OUTPUT_ROOT/shared/manifest.json" ]] || {
            echo "Error: run the prepare stage first." >&2
            exit 2
        }
        echo "Validating source, transform, and all prepared-input hashes."
        validate_inputs
        mkdir -p "$OUTPUT_ROOT/logs"
        bart version >"$OUTPUT_ROOT/logs/bart_version.txt" 2>&1
        for count in "${CHANNEL_COUNTS[@]}"; do
            branch="$OUTPUT_ROOT/rovir_ncc${count}"
            inputs="$branch/bart_inputs"
            bart_output="$branch/bart_output"
            nifti_output="$branch/nifti/fista_r0"
            mkdir -p "$bart_output" "$bart_output/fista_r0" "$nifti_output"

            printf -v ecalib_command '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" \
                "$inputs/kspace_calib" "$bart_output/coil_sens"
            ecalib_command="${ecalib_command% }"
            if [[ -f "$bart_output/coil_sens.hdr" && -f "$bart_output/coil_sens.cfl" ]]; then
                [[ -f "$bart_output/ecalib_command.txt" ]] || { echo "Error: existing ROVir-${count} CSM has no command record." >&2; exit 2; }
                [[ "$(<"$bart_output/ecalib_command.txt")" == "$ecalib_command" ]] || { echo "Error: existing ROVir-${count} CSM used a different command." >&2; exit 2; }
                echo "Reusing complete ROVir-${count} CSM."
            elif [[ -e "$bart_output/coil_sens.hdr" || -e "$bart_output/coil_sens.cfl" || -e "$bart_output/ecalib_command.txt" ]]; then
                echo "Error: incomplete existing ROVir-${count} CSM output." >&2
                exit 2
            else
                bart ecalib -m 1 -c "$ECALIB_CROP" "$inputs/kspace_calib" "$bart_output/coil_sens"
                printf '%s\n' "$ecalib_command" >"$bart_output/ecalib_command.txt"
            fi

            if [[ "$USE_GPU" == true ]]; then
                printf -v wave_command '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 \
                    "$bart_output/coil_sens" "$inputs/psf" "$inputs/wave_kspace" \
                    "$bart_output/fista_r0/image_wave"
            else
                printf -v wave_command '%q ' bart wave -w -f -r 0 -i 100 -t 1e-6 \
                    "$bart_output/coil_sens" "$inputs/psf" "$inputs/wave_kspace" \
                    "$bart_output/fista_r0/image_wave"
            fi
            wave_command="${wave_command% }"
            if [[ -f "$bart_output/fista_r0/image_wave.hdr" && -f "$bart_output/fista_r0/image_wave.cfl" ]]; then
                [[ -f "$bart_output/fista_r0/wave_command.txt" ]] || { echo "Error: existing ROVir-${count} Wave output has no command record." >&2; exit 2; }
                [[ "$(<"$bart_output/fista_r0/wave_command.txt")" == "$wave_command" ]] || { echo "Error: existing ROVir-${count} Wave output used a different command." >&2; exit 2; }
                echo "Reusing complete ROVir-${count} FISTA-r0 output."
            elif [[ -e "$bart_output/fista_r0/image_wave.hdr" || -e "$bart_output/fista_r0/image_wave.cfl" || -e "$bart_output/fista_r0/wave_command.txt" ]]; then
                echo "Error: incomplete existing ROVir-${count} Wave output." >&2
                exit 2
            else
                if [[ "$USE_GPU" == true ]]; then
                    bart wave -g -w -f -r 0 -i 100 -t 1e-6 \
                        "$bart_output/coil_sens" "$inputs/psf" "$inputs/wave_kspace" \
                        "$bart_output/fista_r0/image_wave"
                else
                    bart wave -w -f -r 0 -i 100 -t 1e-6 \
                        "$bart_output/coil_sens" "$inputs/psf" "$inputs/wave_kspace" \
                        "$bart_output/fista_r0/image_wave"
                fi
                printf '%s\n' "$wave_command" >"$bart_output/fista_r0/wave_command.txt"
            fi
            python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
                --bart-inputs "$inputs" --image "$bart_output/fista_r0/image_wave" \
                --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$nifti_output" \
                --suffix "BARTWaveMPRAGEROVir${count}FISTAR0"
        done
        ;;
    qc)
        [[ "$CHANNEL_COUNTS_CSV" == "24,48" ]] || {
            echo "Error: built-in qc stage requires --channel-counts 24,48." >&2
            exit 2
        }
        mapfile -t rovir24_matches < <(find "$OUTPUT_ROOT/rovir_ncc24/nifti/fista_r0" -type f -name '*_part-mag_*.nii.gz' -print)
        mapfile -t rovir48_matches < <(find "$OUTPUT_ROOT/rovir_ncc48/nifti/fista_r0" -type f -name '*_part-mag_*.nii.gz' -print)
        [[ ${#rovir24_matches[@]} -eq 1 ]] || { echo "Error: expected exactly one ROVir-24 magnitude NIfTI." >&2; exit 2; }
        [[ ${#rovir48_matches[@]} -eq 1 ]] || { echo "Error: expected exactly one ROVir-48 magnitude NIfTI." >&2; exit 2; }
        python "$SCRIPT_DIR/mprage_rovir_comparison_qc.py" \
            "${rovir24_matches[0]}" "${rovir48_matches[0]}" "$OUTPUT_ROOT/qc"
        echo "Review: $OUTPUT_ROOT/qc/rovir_ncc24_vs_ncc48_fixed_window.png"
        ;;
    *)
        echo "Error: stage must be prepare, reconstruct, or qc." >&2
        usage >&2
        exit 2
        ;;
esac
