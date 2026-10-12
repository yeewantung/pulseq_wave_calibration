#!/usr/bin/env bash
set -euo pipefail

# Reconstruct selected MPRAGE retro cases from the canonical normal ROVir branch.

usage() {
    echo "Usage: $0 RECONSTRUCTION_ROOT [--reg-full | --wavelet-only | --fista-only] [--include-legacy-lr] [--n4-bias-correction] [-g]"
    echo "       default: FISTA-r0 for native R3x2, LR-Y R3x2, and native R3x3"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
ROOT="${1%/}"
shift
USE_GPU=false
N4_BIAS_CORRECTION=false
RECON_PROFILE="fista-only"
PROFILE_EXPLICIT=false
INCLUDE_LEGACY_LR=false

select_profile() {
    [[ "$PROFILE_EXPLICIT" == false ]] || {
        echo "Error: --reg-full, --wavelet-only, and --fista-only are mutually exclusive." >&2
        exit 2
    }
    RECON_PROFILE="$1"
    PROFILE_EXPLICIT=true
}

while (($#)); do
    case "$1" in
        --reg-full) select_profile reg-full; shift ;;
        --wavelet-only) select_profile wavelet-only; shift ;;
        --fista-only) select_profile fista-only; shift ;;
        --include-legacy-lr) INCLUDE_LEGACY_LR=true; shift ;;
        --n4-bias-correction) N4_BIAS_CORRECTION=true; shift ;;
        -g) USE_GPU=true; shift ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
ROOT="$(cd -- "$ROOT" && pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

run_optional_n4() {
    [[ "$N4_BIAS_CORRECTION" == true ]] || return 0
    python "$SCRIPT_DIR/apply_mprage_n4_bias_correction.py" \
        "$1" "$2" --reference-derivative "$N4_REFERENCE"
}
CASES=(native_r3x2 lr_y_1p5mm_r3x2 native_r3x3)
if [[ "$INCLUDE_LEGACY_LR" == true ]]; then
    CASES+=(lr_x_1p5mm_r3x2 lr_xy_1p25mm_r3x2)
fi
declare -A LAMBDAS=(
    [native_r3x2]=3.5e-2
    [lr_x_1p5mm_r3x2]=2.5e-2
    [lr_y_1p5mm_r3x2]=2.5e-2
    [lr_xy_1p25mm_r3x2]=2.2e-2
    [native_r3x3]=4.5e-2
)
declare -A SUFFIXES=(
    [native_r3x2]=NativeR3x2
    [lr_x_1p5mm_r3x2]=LRX1p5mmR3x2
    [lr_y_1p5mm_r3x2]=LRY1p5mmR3x2
    [lr_xy_1p25mm_r3x2]=LRXY1p25mmR3x2
    [native_r3x3]=NativeR3x3
)
PREPARE_ARGS=("$ROOT")
for case_name in "${CASES[@]}"; do PREPARE_ARGS+=(--case "$case_name"); done
echo "Preparing selected pure-mask cases from the canonical normal ROVir contract; validation can be quiet for several minutes."
python "$SCRIPT_DIR/prepare_mprage_rovir_retro.py" "${PREPARE_ARGS[@]}" >/dev/null
TWIX_FILE="$(python "$SCRIPT_DIR/mprage_rovir_workflow.py" context "$ROOT" --field twix)"
SEQUENCE_FILE="$(python "$SCRIPT_DIR/mprage_rovir_workflow.py" context "$ROOT" --field sequence)"
SAMPLING_CLASS="$(python "$SCRIPT_DIR/mprage_rovir_workflow.py" context "$ROOT" --field sampling_class)"
N4_REFERENCE=""
if [[ "$N4_BIAS_CORRECTION" == true ]]; then
    if [[ "$SAMPLING_CLASS" == "R3x1" ]]; then
        N4_REFERENCE="$ROOT/normal/rovir/n4_bias_corrected_nifti/optimal_wavelet"
    else
        N4_REFERENCE="$ROOT/normal/rovir/n4_bias_corrected_nifti/fista_r0"
    fi
    [[ -f "$N4_REFERENCE/metadata/n4_manifest.json" ]] || {
        echo "Error: retro ROVir N4 requires the completed normal reference derivative: $N4_REFERENCE" >&2
        echo "Run sample_mprage_rovir_recon.sh run ... --n4-bias-correction first." >&2
        exit 2
    }
fi

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
    run_optional_n4 \
        "$nifti" \
        "$ROOT/retro/$case_name/rovir/n4_bias_corrected_nifti/$branch"
}

for case_name in "${CASES[@]}"; do
    if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == fista-only ]]; then
        run_method "$case_name" fista_r0 0 "${SUFFIXES[$case_name]}"
    fi
    if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == wavelet-only ]]; then
        run_method "$case_name" optimal_wavelet "${LAMBDAS[$case_name]}" "${SUFFIXES[$case_name]}"
    fi
done
echo "Completed selected ROVir retrospective branches below: $ROOT/retro ($RECON_PROFILE)"
