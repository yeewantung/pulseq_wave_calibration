#!/usr/bin/env bash
set -euo pipefail

# Prepare and resumably reconstruct one count-specific multi-echo GRE variant.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--virtual-coils N] [--ecalib-crop VALUE] [-g]"
    echo "       [--reg-full | --wavelet-only | --fista-only] (default: --wavelet-only)"
    echo "       [--wavelet-lambda VALUE] (one shared value for every echo)"
    echo "       [--psf-coefficient-processing smooth|sine-line] [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"; OUTPUT_ROOT="${2%/}"; SEQUENCE_FILE="$3"; shift 3
VIRTUAL_COILS=24; ECALIB_CROP="0.6"; USE_GPU=false
RECON_PROFILE="wavelet-only"; PROFILE_EXPLICIT=false
PSF_COEFFICIENT_PROCESSING="sine-line"; PSF_FIT_KX_MIN=""; PSF_FIT_KX_MAX=""
GRE_SHARED_WAVELET_LAMBDA="0.015"
GRE_WAVELET_LAMBDA_EXPLICIT=false
select_profile() { [[ "$PROFILE_EXPLICIT" == false ]] || { echo "Error: reconstruction-profile flags are mutually exclusive." >&2; exit 2; }; RECON_PROFILE="$1"; PROFILE_EXPLICIT=true; }
while (($#)); do
    case "$1" in
        --virtual-coils) VIRTUAL_COILS="$2"; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; shift 2 ;;
        --reg-full) select_profile reg-full; shift ;;
        --wavelet-only) select_profile wavelet-only; shift ;;
        --fista-only) select_profile fista-only; shift ;;
        --wavelet-lambda) GRE_SHARED_WAVELET_LAMBDA="$2"; GRE_WAVELET_LAMBDA_EXPLICIT=true; shift 2 ;;
        --psf-coefficient-processing) PSF_COEFFICIENT_PROCESSING="$2"; shift 2 ;;
        --psf-fit-kx-min) PSF_FIT_KX_MIN="$2"; shift 2 ;;
        --psf-fit-kx-max) PSF_FIT_KX_MAX="$2"; shift 2 ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$VIRTUAL_COILS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --virtual-coils must be a positive integer." >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STANDARD_ROOT="$OUTPUT_ROOT/vcc$VIRTUAL_COILS"; BART_INPUTS="$STANDARD_ROOT/normal/bart_inputs"
BART_OUTPUT_ROOT="$STANDARD_ROOT/normal/bart_output"; NIFTI_OUTPUT_ROOT="$STANDARD_ROOT/normal/nifti"
ECALIB_RECORD="$BART_OUTPUT_ROOT/ecalib_command.txt"
command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }
command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

PREP_ARGS=("$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --virtual-coils "$VIRTUAL_COILS" --psf-coefficient-processing "$PSF_COEFFICIENT_PROCESSING")
if [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then PREP_ARGS+=(--psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX"); elif [[ -n "$PSF_FIT_KX_MIN" || -n "$PSF_FIT_KX_MAX" ]]; then echo "Error: both kx bounds are required." >&2; exit 2; fi
if [[ "$PSF_COEFFICIENT_PROCESSING" == smooth && -n "$PSF_FIT_KX_MIN" ]]; then echo "Error: kx bounds require sine-line." >&2; exit 2; fi
python "$SCRIPT_DIR/prepare_gre_normal.py" "${PREP_ARGS[@]}"

mkdir -p "$BART_OUTPUT_ROOT"
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT_ROOT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$BART_OUTPUT_ROOT/coil_sens.hdr" && -f "$BART_OUTPUT_ROOT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" && "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM command mismatch." >&2; exit 2; }
elif [[ -e "$BART_OUTPUT_ROOT/coil_sens.hdr" || -e "$BART_OUTPUT_ROOT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then echo "Error: incomplete CSM state." >&2; exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$BART_INPUTS/kspace_calib" "$BART_OUTPUT_ROOT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi
ECHO_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["echoes"]))' "$BART_INPUTS/manifest.json")"
[[ "$ECHO_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "Error: invalid GRE echo count." >&2; exit 2; }

run_branch() {
    local branch="$1" method="$2" lambda_value="$3" suffix="$4"
    local branch_root="$BART_OUTPUT_ROOT/$branch" nifti_root="$NIFTI_OUTPUT_ROOT/$branch"
    local run_manifest="$branch_root/reconstruction_manifest.json" echo_number echo_label image record expected status
    local -a expected_commands=() images=() records=() state_args conversion_args=(--bart-inputs "$BART_INPUTS") command
    for ((echo_number=1; echo_number<=ECHO_COUNT; echo_number++)); do
        printf -v echo_label 'echo-%02d' "$echo_number"
        image="$branch_root/$echo_label/image_wave"; record="$branch_root/$echo_label/wave_command.txt"
        if [[ "$USE_GPU" == true ]]; then command=(bart wave -g -w -f -r "$lambda_value" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image"); else command=(bart wave -w -f -r "$lambda_value" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf_$echo_label" "$BART_INPUTS/wave_kspace_$echo_label" "$image"); fi
        printf -v expected '%q ' "${command[@]}"; expected="${expected% }"
        expected_commands+=("$expected"); images+=("$image"); records+=("$record"); conversion_args+=(--image "$image")
    done
    state_args=(--run-manifest "$run_manifest" --prepared-manifest "$BART_INPUTS/manifest.json" --normal-manifest "$BART_INPUTS/manifest.json" --profile "$RECON_PROFILE" --case normal --branch "$branch" --method "$method" --regularization "$lambda_value" --maps "$BART_OUTPUT_ROOT/coil_sens" --nifti-directory "$nifti_root")
    for ((echo_number=1; echo_number<=ECHO_COUNT; echo_number++)); do printf -v echo_label 'echo-%02d' "$echo_number"; state_args+=(--psf "$BART_INPUTS/psf_$echo_label" --kspace "$BART_INPUTS/wave_kspace_$echo_label" --image "${images[echo_number-1]}" --command-record "${records[echo_number-1]}" --expected-command "${expected_commands[echo_number-1]}"); done
    status="$(python "$SCRIPT_DIR/manage_standard_reconstruction.py" status "${state_args[@]}")"
    if [[ "$status" == complete ]]; then echo "Reusing completed normal/$branch"; return; fi
    if [[ "$status" == run ]]; then
        for ((echo_number=1; echo_number<=ECHO_COUNT; echo_number++)); do mkdir -p "$(dirname "${images[echo_number-1]}")"; if [[ "$USE_GPU" == true ]]; then bart wave -g -w -f -r "$lambda_value" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf_$(printf 'echo-%02d' "$echo_number")" "$BART_INPUTS/wave_kspace_$(printf 'echo-%02d' "$echo_number")" "${images[echo_number-1]}"; else bart wave -w -f -r "$lambda_value" -i 100 -t 1e-6 "$BART_OUTPUT_ROOT/coil_sens" "$BART_INPUTS/psf_$(printf 'echo-%02d' "$echo_number")" "$BART_INPUTS/wave_kspace_$(printf 'echo-%02d' "$echo_number")" "${images[echo_number-1]}"; fi; printf '%s\n' "${expected_commands[echo_number-1]}" > "${records[echo_number-1]}"; done
    fi
    if [[ "$status" == run || "$status" == convert ]]; then mkdir -p "$nifti_root"; python "$SCRIPT_DIR/convert_gre_bart_to_nifti.py" "${conversion_args[@]}" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$nifti_root" --suffix "$suffix"; fi
    python "$SCRIPT_DIR/manage_standard_reconstruction.py" record "${state_args[@]}" >/dev/null
}

if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == fista-only ]]; then run_branch fista_r0 fista 0 BARTWaveGRENormalFISTAR0; fi
if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == wavelet-only ]]; then
    if [[ "$GRE_WAVELET_LAMBDA_EXPLICIT" == true ]]; then
        run_branch wavelet_candidate wavelet "$GRE_SHARED_WAVELET_LAMBDA" BARTWaveGRENormalWaveletCandidate
    else
        run_branch wavelet_transferred_vcc12 wavelet "$GRE_SHARED_WAVELET_LAMBDA" BARTWaveGRENormalWaveletTransferredFromVCC12
    fi
fi
echo "Normal multi-echo GRE reconstruction complete: $STANDARD_ROOT/normal ($RECON_PROFILE)"
