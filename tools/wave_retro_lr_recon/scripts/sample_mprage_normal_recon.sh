#!/usr/bin/env bash
set -euo pipefail

# Prepare measured Wave-MPRAGE data, estimate one BART ESPIRiT map set, run
# native-resolution BART Wave reconstruction, and export magnitude plus phase.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE] [--r3-lambda VALUE]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
OUTPUT_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3

ECALIB_CROP="0.6"
R3_LAMBDA="2.2e-2"
while (($#)); do
    case "$1" in
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --r3-lambda) R3_LAMBDA="$2"; shift 2 ;;
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
command -v bart >/dev/null || { echo "Error: source bart_startup.sh before running." >&2; exit 2; }

python "$SCRIPT_DIR/prepare_mprage_normal.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE"
mkdir -p "$BART_OUTPUT" "$NIFTI_OUTPUT"

# Estimate one sensitivity-map set from the integrated ACS k-space. The pinned
# BART ecalib command has no -g option; every BART wave command below uses GPU.
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$BART_OUTPUT/coil_sens.hdr" && -f "$BART_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
    [[ "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM was generated with a different ecalib command." >&2; exit 2; }
    echo "Reusing recorded ecalib result: $BART_OUTPUT/coil_sens"
elif [[ -e "$BART_OUTPUT/coil_sens.hdr" || -e "$BART_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM or ecalib command record in $BART_OUTPUT" >&2
    exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi

SAMPLING_CLASS="$(<"$BART_INPUTS/sampling_class.txt")"
if [[ "$SAMPLING_CLASS" == "R3x1" ]]; then
    # Measured R3x1: Wavelet/FISTA, with a user-overridable default lambda.
    bart wave -g -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r "$R3_LAMBDA" -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/image_wave"
elif [[ "$SAMPLING_CLASS" == "R1" ]]; then
    # Measured R1: unregularized FISTA, represented by zero Wavelet weight.
    bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/image_wave"
    printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$BART_OUTPUT/coil_sens" "$BART_INPUTS/psf" "$BART_INPUTS/wave_kspace" "$BART_OUTPUT/image_wave"
else
    echo "Error: unsupported prepared sampling class $SAMPLING_CLASS" >&2
    exit 2
fi
printf '%s\n' "${WAVE_COMMAND% }" > "$BART_OUTPUT/wave_command.txt"

python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$BART_INPUTS" --image "$BART_OUTPUT/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_OUTPUT" --suffix BARTWaveMPRAGENormal

echo "Normal MPRAGE reconstruction complete: $OUTPUT_ROOT/normal"
