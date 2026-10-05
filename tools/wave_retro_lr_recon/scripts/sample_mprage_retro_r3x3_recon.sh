#!/usr/bin/env bash
set -euo pipefail

# Prepare and resumably reconstruct only native-grid R3x3 Wave-MPRAGE.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--virtual-coils N] [--ecalib-crop VALUE] [-g]"
    echo "       [--reg-full | --wavelet-only | --fista-only] (default: --fista-only)"
    echo "       [--psf-coefficient-processing smooth|sine-line]"
    echo "       [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
    echo "       [--psf-fit-y-min INDEX --psf-fit-y-max INDEX] [--psf-fit-z-min INDEX --psf-fit-z-max INDEX]"
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"; OUTPUT_ROOT="${2%/}"; SEQUENCE_FILE="$3"; shift 3
VIRTUAL_COILS=24; ECALIB_CROP="0.6"; USE_GPU=false
RECON_PROFILE="fista-only"; PROFILE_EXPLICIT=false
PSF_COEFFICIENT_PROCESSING="sine-line"
PSF_FIT_KX_MIN=""; PSF_FIT_KX_MAX=""; PSF_FIT_Y_MIN=""; PSF_FIT_Y_MAX=""; PSF_FIT_Z_MIN=""; PSF_FIT_Z_MAX=""
R3X3_LAMBDA="4.5e-2"

select_profile() {
    [[ "$PROFILE_EXPLICIT" == false ]] || { echo "Error: reconstruction-profile flags are mutually exclusive." >&2; exit 2; }
    RECON_PROFILE="$1"; PROFILE_EXPLICIT=true
}
while (($#)); do
    case "$1" in
        --virtual-coils) VIRTUAL_COILS="$2"; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --reg-full) select_profile reg-full; shift ;;
        --wavelet-only) select_profile wavelet-only; shift ;;
        --fista-only) select_profile fista-only; shift ;;
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
[[ "$VIRTUAL_COILS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --virtual-coils must be a positive integer." >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STANDARD_ROOT="$OUTPUT_ROOT/vcc$VIRTUAL_COILS"
NORMAL_INPUTS="$STANDARD_ROOT/normal/bart_inputs"; NORMAL_OUTPUT="$STANDARD_ROOT/normal/bart_output"
CASE_ROOT="$STANDARD_ROOT/retro/native_r3x3"; CASE_INPUTS="$CASE_ROOT/bart_inputs"
ECALIB_RECORD="$NORMAL_OUTPUT/ecalib_command.txt"
command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

PREP_ARGS=("$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --virtual-coils "$VIRTUAL_COILS" --psf-coefficient-processing "$PSF_COEFFICIENT_PROCESSING")
for axis in KX Y Z; do
    min_name="PSF_FIT_${axis}_MIN"; max_name="PSF_FIT_${axis}_MAX"
    minimum="${!min_name}"; maximum="${!max_name}"
    if [[ -n "$minimum" && -n "$maximum" ]]; then
        lower="${axis,,}"
        PREP_ARGS+=("--psf-fit-$lower-min" "$minimum" "--psf-fit-$lower-max" "$maximum")
    elif [[ -n "$minimum" || -n "$maximum" ]]; then
        echo "Error: manual PSF ${axis,,} fitting requires both bounds." >&2; exit 2
    fi
done
if [[ "$PSF_COEFFICIENT_PROCESSING" == smooth && -n "$PSF_FIT_KX_MIN" ]]; then echo "Error: kx bounds require sine-line." >&2; exit 2; fi
python "$SCRIPT_DIR/prepare_mprage_retro_r3x3.py" "${PREP_ARGS[@]}"

mkdir -p "$NORMAL_OUTPUT"
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$NORMAL_OUTPUT/coil_sens.hdr" && -f "$NORMAL_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" && "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM command mismatch." >&2; exit 2; }
elif [[ -e "$NORMAL_OUTPUT/coil_sens.hdr" || -e "$NORMAL_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM state." >&2; exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi

run_branch() {
    local branch="$1" method="$2" lambda_value="$3" suffix="$4"
    local branch_root="$CASE_ROOT/bart_output/$branch" nifti_root="$CASE_ROOT/nifti/$branch"
    local image="$branch_root/image_wave" record="$branch_root/wave_command.txt" run_manifest="$branch_root/reconstruction_manifest.json"
    local expected status; local -a command state_args
    if [[ "$USE_GPU" == true ]]; then
        command=(bart wave -g -w -f -r "$lambda_value" -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" "$image")
    else
        command=(bart wave -w -f -r "$lambda_value" -i 100 -t 1e-6 "$NORMAL_OUTPUT/coil_sens" "$CASE_INPUTS/psf" "$CASE_INPUTS/wave_kspace" "$image")
    fi
    printf -v expected '%q ' "${command[@]}"; expected="${expected% }"
    state_args=(--run-manifest "$run_manifest" --prepared-manifest "$CASE_INPUTS/manifest.json" --normal-manifest "$NORMAL_INPUTS/manifest.json" --profile "$RECON_PROFILE" --case native_r3x3 --branch "$branch" --method "$method" --regularization "$lambda_value" --maps "$NORMAL_OUTPUT/coil_sens" --psf "$CASE_INPUTS/psf" --kspace "$CASE_INPUTS/wave_kspace" --image "$image" --command-record "$record" --expected-command "$expected" --nifti-directory "$nifti_root")
    status="$(python "$SCRIPT_DIR/manage_standard_reconstruction.py" status "${state_args[@]}")"
    if [[ "$status" == complete ]]; then echo "Reusing completed native_r3x3/$branch"; return; fi
    if [[ "$status" == run ]]; then mkdir -p "$branch_root"; "${command[@]}"; printf '%s\n' "$expected" > "$record"; fi
    if [[ "$status" == run || "$status" == convert ]]; then mkdir -p "$nifti_root"; python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$CASE_INPUTS" --image "$image" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$nifti_root" --suffix "$suffix"; fi
    python "$SCRIPT_DIR/manage_standard_reconstruction.py" record "${state_args[@]}" >/dev/null
}

if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == fista-only ]]; then run_branch fista_r0 fista 0 BARTWaveMPRAGENativeR3x3FISTAR0; fi
if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == wavelet-only ]]; then run_branch wavelet_transferred_vcc12 wavelet "$R3X3_LAMBDA" BARTWaveMPRAGENativeR3x3WaveletTransferredFromVCC12; fi
echo "Native R3x3 retrospective MPRAGE complete: $CASE_ROOT ($RECON_PROFILE)"
