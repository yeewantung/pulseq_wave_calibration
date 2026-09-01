#!/usr/bin/env bash
set -euo pipefail

# Prepare and sequentially reconstruct native plus three direct-crop LR R3x2
# cases. The same native ACS map estimate is reused for every target grid.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--ecalib-crop VALUE]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"
OUTPUT_ROOT="${2%/}"
SEQUENCE_FILE="$3"
shift 3

ECALIB_CROP="0.6"
while (($#)); do
    case "$1" in
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NORMAL_INPUTS="$OUTPUT_ROOT/normal/bart_inputs"
NORMAL_OUTPUT="$OUTPUT_ROOT/normal/bart_output"
RETRO_ROOT="$OUTPUT_ROOT/retro"
ECALIB_RECORD="$NORMAL_OUTPUT/ecalib_command.txt"

command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: source bart_startup.sh before running." >&2; exit 2; }

# This reuses compatible normal inputs, or prepares them from TWIX when absent.
python "$SCRIPT_DIR/prepare_mprage_retro.py" "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE"
mkdir -p "$NORMAL_OUTPUT"

# Run ecalib exactly once unless the matching normal workflow already produced it.
# The pinned BART ecalib command has no -g option; all BART wave commands use GPU.
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$NORMAL_OUTPUT/coil_sens.hdr" && -f "$NORMAL_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
    [[ "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM was generated with a different ecalib command." >&2; exit 2; }
    echo "Reusing recorded ecalib result: $NORMAL_OUTPUT/coil_sens"
elif [[ -e "$NORMAL_OUTPUT/coil_sens.hdr" || -e "$NORMAL_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM or ecalib command record in $NORMAL_OUTPUT" >&2
    exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi

# Produce same-FOV LR map grids from that single accepted ecalib result.
python "$SCRIPT_DIR/prepare_mprage_retro_maps.py" "$OUTPUT_ROOT"

# Native 1x1x1-mm-class R3x2: FISTA r=0 control and selected Wavelet 3.5e-2.
mkdir -p "$RETRO_ROOT/native_r3x2/bart_output/fista_r0" "$RETRO_ROOT/native_r3x2/nifti/fista_r0"
bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$RETRO_ROOT/native_r3x2/bart_inputs/psf" "$RETRO_ROOT/native_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/native_r3x2/bart_output/fista_r0/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$RETRO_ROOT/native_r3x2/bart_inputs/psf" "$RETRO_ROOT/native_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/native_r3x2/bart_output/fista_r0/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/native_r3x2/bart_output/fista_r0/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/native_r3x2/bart_inputs" --image "$RETRO_ROOT/native_r3x2/bart_output/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/native_r3x2/nifti/fista_r0" --suffix BARTWaveMPRAGENativeR3x2FISTAR0
mkdir -p "$RETRO_ROOT/native_r3x2/bart_output/optimal_wavelet" "$RETRO_ROOT/native_r3x2/nifti/optimal_wavelet"
bart wave -g -w -f -r 3.5e-2 -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$RETRO_ROOT/native_r3x2/bart_inputs/psf" "$RETRO_ROOT/native_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/native_r3x2/bart_output/optimal_wavelet/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 3.5e-2 -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$RETRO_ROOT/native_r3x2/bart_inputs/psf" "$RETRO_ROOT/native_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/native_r3x2/bart_output/optimal_wavelet/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/native_r3x2/bart_output/optimal_wavelet/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/native_r3x2/bart_inputs" --image "$RETRO_ROOT/native_r3x2/bart_output/optimal_wavelet/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/native_r3x2/nifti/optimal_wavelet" --suffix BARTWaveMPRAGENativeR3x2OptimalWavelet

# Approximately 1.5x1x1 mm: FISTA r=0 control and selected Wavelet 2.5e-2.
mkdir -p "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/fista_r0" "$RETRO_ROOT/lr_x_1p5mm_r3x2/nifti/fista_r0"
bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/fista_r0/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/fista_r0/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/fista_r0/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_x_1p5mm_r3x2/nifti/fista_r0" --suffix BARTWaveMPRAGELRX1p5mmR3x2FISTAR0
mkdir -p "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/optimal_wavelet" "$RETRO_ROOT/lr_x_1p5mm_r3x2/nifti/optimal_wavelet"
bart wave -g -w -f -r 2.5e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 2.5e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/optimal_wavelet/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_x_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_x_1p5mm_r3x2/nifti/optimal_wavelet" --suffix BARTWaveMPRAGELRX1p5mmR3x2OptimalWavelet

# Approximately 1x1.5x1 mm: FISTA r=0 control and selected Wavelet 2.5e-2.
mkdir -p "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/fista_r0" "$RETRO_ROOT/lr_y_1p5mm_r3x2/nifti/fista_r0"
bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/fista_r0/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/fista_r0/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/fista_r0/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_y_1p5mm_r3x2/nifti/fista_r0" --suffix BARTWaveMPRAGELRY1p5mmR3x2FISTAR0
mkdir -p "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/optimal_wavelet" "$RETRO_ROOT/lr_y_1p5mm_r3x2/nifti/optimal_wavelet"
bart wave -g -w -f -r 2.5e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 2.5e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/optimal_wavelet/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_y_1p5mm_r3x2/bart_output/optimal_wavelet/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_y_1p5mm_r3x2/nifti/optimal_wavelet" --suffix BARTWaveMPRAGELRY1p5mmR3x2OptimalWavelet

# Approximately 1.25x1.25x1 mm: FISTA r=0 control and Wavelet 2.2e-2.
mkdir -p "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/fista_r0" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/nifti/fista_r0"
bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/fista_r0/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 0 -i 100 -t 1e-6 "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/fista_r0/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/fista_r0/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/fista_r0/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_xy_1p25mm_r3x2/nifti/fista_r0" --suffix BARTWaveMPRAGELRXY1p25mmR3x2FISTAR0
mkdir -p "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/optimal_wavelet" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/nifti/optimal_wavelet"
bart wave -g -w -f -r 2.2e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf -v WAVE_COMMAND '%q ' bart wave -g -w -f -r 2.2e-2 -i 100 -t 1e-6 "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/coil_sens" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/psf" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs/wave_kspace" "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/optimal_wavelet/image_wave"
printf '%s\n' "${WAVE_COMMAND% }" > "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/optimal_wavelet/wave_command.txt"
python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_inputs" --image "$RETRO_ROOT/lr_xy_1p25mm_r3x2/bart_output/optimal_wavelet/image_wave" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$RETRO_ROOT/lr_xy_1p25mm_r3x2/nifti/optimal_wavelet" --suffix BARTWaveMPRAGELRXY1p25mmR3x2OptimalWavelet

echo "Retrospective MPRAGE reconstructions complete: $RETRO_ROOT"
