#!/usr/bin/env bash
set -euo pipefail

# Reconstruct the reduced standard MPRAGE retro set or canonical legacy ROVir.

usage() {
    echo "Usage: $0 TWIX.dat OUTPUT_ROOT SEQUENCE.seq [--virtual-coils N] [--ecalib-crop VALUE] [-g]"
    echo "       [--reg-full | --wavelet-only | --fista-only] (default: --fista-only)"
    echo "       [--psf-coefficient-processing smooth|sine-line] (default: automatic sine-line)"
    echo "       [--psf-fit-kx-min INDEX --psf-fit-kx-max INDEX]"
    echo "       [--psf-fit-y-min INDEX --psf-fit-y-max INDEX] [--psf-fit-z-min INDEX --psf-fit-z-max INDEX]"
    echo "       [--rovir]  use only the completed canonical root-level ROVir branch"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
TWIX_FILE="$1"; OUTPUT_ROOT="${2%/}"; SEQUENCE_FILE="$3"; shift 3

VIRTUAL_COILS=24
VIRTUAL_COILS_EXPLICIT=false
ECALIB_CROP="0.6"
ECALIB_CROP_EXPLICIT=false
RECON_PROFILE="fista-only"
PROFILE_EXPLICIT=false
USE_GPU=false
USE_ROVIR=false
PSF_COEFFICIENT_PROCESSING="sine-line"
PSF_FIT_KX_MIN=""; PSF_FIT_KX_MAX=""
PSF_FIT_Y_MIN=""; PSF_FIT_Y_MAX=""
PSF_FIT_Z_MIN=""; PSF_FIT_Z_MAX=""

select_profile() {
    [[ "$PROFILE_EXPLICIT" == false ]] || {
        echo "Error: --reg-full, --wavelet-only, and --fista-only are mutually exclusive." >&2
        exit 2
    }
    RECON_PROFILE="$1"; PROFILE_EXPLICIT=true
}

while (($#)); do
    case "$1" in
        --virtual-coils) VIRTUAL_COILS="$2"; VIRTUAL_COILS_EXPLICIT=true; shift 2 ;;
        --ecalib-crop) ECALIB_CROP="$2"; ECALIB_CROP_EXPLICIT=true; shift 2 ;;
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
        --rovir) USE_ROVIR=true; shift ;;
        -g) USE_GPU=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$VIRTUAL_COILS" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: --virtual-coils must be a positive integer." >&2; exit 2;
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v python >/dev/null || { echo "Error: python is not on PATH." >&2; exit 2; }

if [[ "$USE_ROVIR" == true ]]; then
    [[ "$VIRTUAL_COILS_EXPLICIT" == false ]] || {
        echo "Error: --virtual-coils configures standard PCA and is incompatible with --rovir; use the canonical ROVir count." >&2; exit 2;
    }
    [[ "$PROFILE_EXPLICIT" == false ]] || {
        echo "Error: standard reconstruction-profile flags are incompatible with --rovir." >&2; exit 2;
    }
    [[ "$ECALIB_CROP_EXPLICIT" == false ]] || {
        echo "Error: --rovir reuses its canonical CSM; do not supply --ecalib-crop." >&2; exit 2;
    }
    [[ "$PSF_COEFFICIENT_PROCESSING" == "sine-line" && -z "$PSF_FIT_KX_MIN" && -z "$PSF_FIT_KX_MAX" && -z "$PSF_FIT_Y_MIN" && -z "$PSF_FIT_Y_MAX" && -z "$PSF_FIT_Z_MIN" && -z "$PSF_FIT_Z_MAX" ]] || {
        echo "Error: --rovir reuses its canonical PSF; do not supply PSF overrides." >&2; exit 2;
    }
    command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }
    python "$SCRIPT_DIR/mprage_rovir_workflow.py" validate-invocation "$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" >/dev/null
    python "$SCRIPT_DIR/mprage_rovir_workflow.py" finalize-existing "$OUTPUT_ROOT" >/dev/null
    ROVIR_ARGS=("$OUTPUT_ROOT"); [[ "$USE_GPU" == true ]] && ROVIR_ARGS+=(-g)
    bash "$SCRIPT_DIR/sample_mprage_rovir_retro_recon.sh" "${ROVIR_ARGS[@]}"
    exit 0
fi

command -v bart >/dev/null || { echo "Error: bart is not on PATH; follow SETUP.md." >&2; exit 2; }

STANDARD_ROOT="$OUTPUT_ROOT/vcc$VIRTUAL_COILS"
NORMAL_INPUTS="$STANDARD_ROOT/normal/bart_inputs"
NORMAL_OUTPUT="$STANDARD_ROOT/normal/bart_output"
RETRO_ROOT="$STANDARD_ROOT/retro"
ECALIB_RECORD="$NORMAL_OUTPUT/ecalib_command.txt"

PREP_ARGS=("$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --virtual-coils "$VIRTUAL_COILS" --psf-coefficient-processing "$PSF_COEFFICIENT_PROCESSING")
if [[ -n "$PSF_FIT_KX_MIN" && -n "$PSF_FIT_KX_MAX" ]]; then
    PREP_ARGS+=(--psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX")
elif [[ -n "$PSF_FIT_KX_MIN" || -n "$PSF_FIT_KX_MAX" ]]; then
    echo "Error: manual PSF kx fitting requires both bounds." >&2; exit 2
fi
if [[ "$PSF_COEFFICIENT_PROCESSING" == "smooth" && -n "$PSF_FIT_KX_MIN" ]]; then
    echo "Error: PSF kx bounds require sine-line processing." >&2; exit 2
fi
if [[ -n "$PSF_FIT_Y_MIN" && -n "$PSF_FIT_Y_MAX" ]]; then
    PREP_ARGS+=(--psf-fit-y-min "$PSF_FIT_Y_MIN" --psf-fit-y-max "$PSF_FIT_Y_MAX")
elif [[ -n "$PSF_FIT_Y_MIN" || -n "$PSF_FIT_Y_MAX" ]]; then
    echo "Error: manual PSF y fitting requires both bounds." >&2; exit 2
fi
if [[ -n "$PSF_FIT_Z_MIN" && -n "$PSF_FIT_Z_MAX" ]]; then
    PREP_ARGS+=(--psf-fit-z-min "$PSF_FIT_Z_MIN" --psf-fit-z-max "$PSF_FIT_Z_MAX")
elif [[ -n "$PSF_FIT_Z_MIN" || -n "$PSF_FIT_Z_MAX" ]]; then
    echo "Error: manual PSF z fitting requires both bounds." >&2; exit 2
fi
python "$SCRIPT_DIR/prepare_mprage_retro.py" "${PREP_ARGS[@]}"

mkdir -p "$NORMAL_OUTPUT"
printf -v EXPECTED_ECALIB_COMMAND '%q ' bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
EXPECTED_ECALIB_COMMAND="${EXPECTED_ECALIB_COMMAND% }"
if [[ -f "$NORMAL_OUTPUT/coil_sens.hdr" && -f "$NORMAL_OUTPUT/coil_sens.cfl" ]]; then
    [[ -f "$ECALIB_RECORD" ]] || { echo "Error: existing CSM has no command record." >&2; exit 2; }
    [[ "$(<"$ECALIB_RECORD")" == "$EXPECTED_ECALIB_COMMAND" ]] || { echo "Error: existing CSM used a different ecalib command." >&2; exit 2; }
elif [[ -e "$NORMAL_OUTPUT/coil_sens.hdr" || -e "$NORMAL_OUTPUT/coil_sens.cfl" || -e "$ECALIB_RECORD" ]]; then
    echo "Error: incomplete CSM or ecalib command record in $NORMAL_OUTPUT" >&2; exit 2
else
    bart ecalib -m 1 -c "$ECALIB_CROP" "$NORMAL_INPUTS/kspace_calib" "$NORMAL_OUTPUT/coil_sens"
    printf '%s\n' "$EXPECTED_ECALIB_COMMAND" > "$ECALIB_RECORD"
fi
python "$SCRIPT_DIR/prepare_mprage_retro_maps.py" "$OUTPUT_ROOT" --virtual-coils "$VIRTUAL_COILS"

run_case_branch() {
    local case_id="$1" maps="$2" branch="$3" method="$4" lambda_value="$5" suffix="$6"
    local inputs="$RETRO_ROOT/$case_id/bart_inputs"
    local branch_root="$RETRO_ROOT/$case_id/bart_output/$branch"
    local nifti_root="$RETRO_ROOT/$case_id/nifti/$branch"
    local image="$branch_root/image_wave" command_record="$branch_root/wave_command.txt"
    local run_manifest="$branch_root/reconstruction_manifest.json" expected status
    local -a command state_args
    if [[ "$USE_GPU" == true ]]; then
        command=(bart wave -g -w -f -r "$lambda_value" -i 100 -t 1e-6 "$maps" "$inputs/psf" "$inputs/wave_kspace" "$image")
    else
        command=(bart wave -w -f -r "$lambda_value" -i 100 -t 1e-6 "$maps" "$inputs/psf" "$inputs/wave_kspace" "$image")
    fi
    printf -v expected '%q ' "${command[@]}"; expected="${expected% }"
    state_args=(--run-manifest "$run_manifest" --prepared-manifest "$inputs/manifest.json" --normal-manifest "$NORMAL_INPUTS/manifest.json" --profile "$RECON_PROFILE" --case "$case_id" --branch "$branch" --method "$method" --regularization "$lambda_value" --maps "$maps" --psf "$inputs/psf" --kspace "$inputs/wave_kspace" --image "$image" --command-record "$command_record" --expected-command "$expected" --nifti-directory "$nifti_root")
    status="$(python "$SCRIPT_DIR/manage_standard_reconstruction.py" status "${state_args[@]}")"
    if [[ "$status" == "complete" ]]; then echo "Reusing completed $case_id/$branch"; return; fi
    if [[ "$status" == "run" ]]; then mkdir -p "$branch_root"; "${command[@]}"; printf '%s\n' "$expected" > "$command_record"; fi
    if [[ "$status" == "run" || "$status" == "convert" ]]; then
        mkdir -p "$nifti_root"
        python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$inputs" --image "$image" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$nifti_root" --suffix "$suffix"
    fi
    python "$SCRIPT_DIR/manage_standard_reconstruction.py" record "${state_args[@]}" >/dev/null
}

for case_id in native_r3x2 lr_y_1p5mm_r3x2; do
    maps="$NORMAL_OUTPUT/coil_sens"; [[ "$case_id" == lr_y_1p5mm_r3x2 ]] && maps="$RETRO_ROOT/$case_id/bart_inputs/coil_sens"
    wavelet_lambda=3.5e-2; [[ "$case_id" == lr_y_1p5mm_r3x2 ]] && wavelet_lambda=2.5e-2
    if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == fista-only ]]; then
        run_case_branch "$case_id" "$maps" fista_r0 fista 0 "BARTWaveMPRAGE${case_id}FISTAR0"
    fi
    if [[ "$RECON_PROFILE" == reg-full || "$RECON_PROFILE" == wavelet-only ]]; then
        run_case_branch "$case_id" "$maps" wavelet_transferred_vcc12 wavelet "$wavelet_lambda" "BARTWaveMPRAGE${case_id}WaveletTransferredFromVCC12"
    fi
done

R3X3_ARGS=("$TWIX_FILE" "$OUTPUT_ROOT" "$SEQUENCE_FILE" --virtual-coils "$VIRTUAL_COILS" --ecalib-crop "$ECALIB_CROP" --psf-coefficient-processing "$PSF_COEFFICIENT_PROCESSING" "--$RECON_PROFILE")
[[ -n "$PSF_FIT_KX_MIN" ]] && R3X3_ARGS+=(--psf-fit-kx-min "$PSF_FIT_KX_MIN" --psf-fit-kx-max "$PSF_FIT_KX_MAX")
[[ -n "$PSF_FIT_Y_MIN" ]] && R3X3_ARGS+=(--psf-fit-y-min "$PSF_FIT_Y_MIN" --psf-fit-y-max "$PSF_FIT_Y_MAX")
[[ -n "$PSF_FIT_Z_MIN" ]] && R3X3_ARGS+=(--psf-fit-z-min "$PSF_FIT_Z_MIN" --psf-fit-z-max "$PSF_FIT_Z_MAX")
[[ "$USE_GPU" == true ]] && R3X3_ARGS+=(-g)
bash "$SCRIPT_DIR/sample_mprage_retro_r3x3_recon.sh" "${R3X3_ARGS[@]}"

echo "Retrospective MPRAGE reconstructions complete: $RETRO_ROOT ($RECON_PROFILE)"
