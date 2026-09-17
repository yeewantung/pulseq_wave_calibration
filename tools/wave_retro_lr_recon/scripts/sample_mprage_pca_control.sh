#!/usr/bin/env bash
set -euo pipefail

# Prepare, reconstruct, or review one higher-channel standard ACS-PCA control.

usage() {
    echo "Usage: $0 STAGE TWIX.dat SEQUENCE.seq ACCEPTED_NORMAL_ROOT FEASIBILITY_ROOT OUTPUT_ROOT [--virtual-coils COUNT] [--ecalib-crop VALUE] [-g]"
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

VIRTUAL_COILS=24
ECALIB_CROP=0.1
USE_GPU=false
while (($#)); do
    case "$1" in
        --virtual-coils) VIRTUAL_COILS="$2"; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BART_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
BART_OUTPUT="$OUTPUT_ROOT/normal/bart_output"
NIFTI_OUTPUT="$OUTPUT_ROOT/normal/nifti/fista_r0"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }

case "$STAGE" in
    prepare)
        python "$SCRIPT_DIR/prepare_mprage_pca_control.py" \
            "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" \
            "$FEASIBILITY_ROOT" "$OUTPUT_ROOT" \
            --virtual-coils "$VIRTUAL_COILS"
        ;;
    reconstruct)
        command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }
        [[ -f "$BART_INPUTS/manifest.json" ]] || {
            echo "Error: run the prepare stage first." >&2
            exit 2
        }
        echo "Validating source and all prepared-input hashes before reconstruction."
        python "$SCRIPT_DIR/prepare_mprage_pca_control.py" \
            "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" \
            "$FEASIBILITY_ROOT" "$OUTPUT_ROOT" \
            --virtual-coils "$VIRTUAL_COILS" >/dev/null
        recorded_ncc="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["coil_compression"]["virtual_coils"])' "$BART_INPUTS/manifest.json")"
        [[ "$recorded_ncc" == "$VIRTUAL_COILS" ]] || {
            echo "Error: prepared Ncc=$recorded_ncc differs from requested Ncc=$VIRTUAL_COILS." >&2
            exit 2
        }
        mkdir -p "$BART_OUTPUT" "$NIFTI_OUTPUT" "$OUTPUT_ROOT/logs"
        printf -v ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
        ECALIB_COMMAND="${ECALIB_COMMAND% }"
        if [[ -f "$BART_OUTPUT/coil_sens.hdr" && -f "$BART_OUTPUT/coil_sens.cfl" ]]; then
            [[ -f "$BART_OUTPUT/ecalib_command.txt" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
            [[ "$(<"$BART_OUTPUT/ecalib_command.txt")" == "$ECALIB_COMMAND" ]] || { echo "Error: existing CSM used a different ecalib command." >&2; exit 2; }
            echo "Reusing complete Ncc=$VIRTUAL_COILS CSM."
        elif [[ -e "$BART_OUTPUT/coil_sens.hdr" || -e "$BART_OUTPUT/coil_sens.cfl" || -e "$BART_OUTPUT/ecalib_command.txt" ]]; then
            echo "Error: incomplete existing CSM output." >&2
            exit 2
        else
            bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
            printf '%s\n' "$ECALIB_COMMAND" > "$BART_OUTPUT/ecalib_command.txt"
        fi

        mkdir -p "$BART_OUTPUT/fista_r0"
        if [[ "$USE_GPU" == true ]]; then
            printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
        else
            printf -v WAVE_COMMAND '%q ' bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
        fi
        WAVE_COMMAND="${WAVE_COMMAND% }"
        if [[ -f "$BART_OUTPUT/fista_r0/image_wave.hdr" && -f "$BART_OUTPUT/fista_r0/image_wave.cfl" ]]; then
            [[ -f "$BART_OUTPUT/fista_r0/wave_command.txt" ]] || { echo "Error: existing Wave output has no command record." >&2; exit 2; }
            [[ "$(<"$BART_OUTPUT/fista_r0/wave_command.txt")" == "$WAVE_COMMAND" ]] || { echo "Error: existing Wave output used a different command." >&2; exit 2; }
            echo "Reusing complete Ncc=$VIRTUAL_COILS FISTA-r0 output."
        elif [[ -e "$BART_OUTPUT/fista_r0/image_wave.hdr" || -e "$BART_OUTPUT/fista_r0/image_wave.cfl" || -e "$BART_OUTPUT/fista_r0/wave_command.txt" ]]; then
            echo "Error: incomplete existing Wave output." >&2
            exit 2
        else
            if [[ "$USE_GPU" == true ]]; then
                bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
            else
                bart wave -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/fista_r0/image_wave"
            fi
            printf '%s\n' "$WAVE_COMMAND" > "$BART_OUTPUT/fista_r0/wave_command.txt"
        fi
        bart version > "$OUTPUT_ROOT/logs/bart_version.txt" 2>&1
        python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
            --bart-inputs "$BART_INPUTS" \
            --image "$BART_OUTPUT/fista_r0/image_wave" \
            --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" \
            --output "$NIFTI_OUTPUT" \
            --suffix "BARTWaveMPRAGENormalPCA${VIRTUAL_COILS}FISTAR0"
        ;;
    qc)
        mapfile -t baseline_matches < <(find "$ACCEPTED_NORMAL_ROOT/normal/nifti/fista_r0" -type f -name '*_part-mag_*.nii.gz' -print)
        mapfile -t control_matches < <(find "$NIFTI_OUTPUT" -type f -name '*_part-mag_*.nii.gz' -print)
        [[ ${#baseline_matches[@]} -eq 1 ]] || { echo "Error: expected exactly one baseline magnitude NIfTI." >&2; exit 2; }
        [[ ${#control_matches[@]} -eq 1 ]] || { echo "Error: expected exactly one control magnitude NIfTI." >&2; exit 2; }
        python "$SCRIPT_DIR/mprage_pca_control_qc.py" \
            "${baseline_matches[0]}" "${control_matches[0]}" "$OUTPUT_ROOT/qc"
        echo "Review: $OUTPUT_ROOT/qc/fixed_window_ncc12_vs_ncc24.png"
        ;;
    *)
        echo "Error: stage must be prepare, reconstruct, or qc." >&2
        usage >&2
        exit 2
        ;;
esac
