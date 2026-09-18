#!/usr/bin/env bash
set -euo pipefail

# Reconstruct all standard MPRAGE retro cases with the canonical normal ROVir basis.

usage() {
    echo "Usage: $0 RECONSTRUCTION_ROOT [-g]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
ROOT="${1%/}"
shift
USE_GPU=false
while (($#)); do
    case "$1" in
        -g) USE_GPU=true; shift ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
ROOT="$(cd -- "$ROOT" && pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }
echo "Validating and hashing the canonical normal ROVir source contract; this can be quiet for several minutes."
python "$SCRIPT_DIR/prepare_mprage_rovir_retro.py" "$ROOT" >/dev/null
TWIX_FILE="$(python "$SCRIPT_DIR/mprage_rovir_workflow.py" context "$ROOT" --field twix)"
SEQUENCE_FILE="$(python "$SCRIPT_DIR/mprage_rovir_workflow.py" context "$ROOT" --field sequence)"

CASES=(native_r3x2 lr_x_1p5mm_r3x2 lr_y_1p5mm_r3x2 lr_xy_1p25mm_r3x2 native_r3x3)
LAMBDAS=(3e-2 2.5e-2 2.5e-2 2.2e-2 4.5e-2)
SUFFIXES=(NativeR3x2 LRX1p5mmR3x2 LRY1p5mmR3x2 LRXY1p25mmR3x2 NativeR3x3)

run_method() {
    local case_name="$1"
    local branch="$2"
    local lambda="$3"
    local suffix="$4"
    local inputs="$ROOT/retro/$case_name/rovir/bart_inputs"
    local output="$ROOT/retro/$case_name/rovir/bart_output/$branch"
    local nifti="$ROOT/retro/$case_name/rovir/nifti/$branch"
    mkdir -p "$output" "$nifti"
    local wave_args
    if [[ "$USE_GPU" == true ]]; then
        wave_args=(-g -w -f -r "$lambda" -i 100 -t 1e-6)
    else
        wave_args=(-w -f -r "$lambda" -i 100 -t 1e-6)
    fi
    local command
    printf -v command '%q ' bart wave "${wave_args[@]}" "$inputs/coil_sens" "$inputs/psf" "$inputs/wave_kspace" "$output/image_wave"
    command="${command% }"
    if [[ -f "$output/image_wave.hdr" && -f "$output/image_wave.cfl" ]]; then
        [[ -f "$output/wave_command.txt" && "$(<"$output/wave_command.txt")" == "$command" ]] || {
            echo "Error: existing $case_name/$branch output used another command." >&2
            exit 2
        }
    elif [[ -e "$output/image_wave.hdr" || -e "$output/image_wave.cfl" || -e "$output/wave_command.txt" ]]; then
        echo "Error: incomplete $case_name/$branch ROVir output." >&2
        exit 2
    else
        bart wave "${wave_args[@]}" "$inputs/coil_sens" "$inputs/psf" "$inputs/wave_kspace" "$output/image_wave"
        printf '%s\n' "$command" >"$output/wave_command.txt"
    fi
    python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" \
        --bart-inputs "$inputs" --image "$output/image_wave" \
        --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$nifti" \
        --suffix "BARTWaveMPRAGE${suffix}ROVir${branch}"
}

for index in "${!CASES[@]}"; do
    run_method "${CASES[$index]}" fista_r0 0 "${SUFFIXES[$index]}"
    run_method "${CASES[$index]}" optimal_wavelet "${LAMBDAS[$index]}" "${SUFFIXES[$index]}"
done
echo "Completed ROVir retrospective branches below: $ROOT/retro"
