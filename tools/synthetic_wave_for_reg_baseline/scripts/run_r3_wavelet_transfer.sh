#!/usr/bin/env bash
# Apply the frozen R1-selected Wavelet setting to R3 for qualitative review only.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: run_r3_wavelet_transfer.sh

Required environment variables:
  R3_TRANSFER_SELECTION_MANIFEST  Frozen R1 selection record
  R3_TRANSFER_BART_INPUT_DIR      R3 synthetic-Wave BART inputs
  R3_TRANSFER_CALIBRATION_BASE    R3 calibration-kspace CFL basename
  R3_TRANSFER_TWIX                R3 Siemens measurement
  R3_TRANSFER_SEQUENCE            Matching MPRAGE Pulseq file
  R3_TRANSFER_OUTPUT_ROOT         New output directory

Copy run_r3_wavelet_transfer.example.sh to the ignored .local.sh filename for
machine-specific environment activation and paths. This runner performs only a
qualitative R3 transfer assessment and never ranks or retunes lambda.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if (($#)); then
    usage >&2
    exit 2
fi

required_variables=(
    R3_TRANSFER_SELECTION_MANIFEST
    R3_TRANSFER_BART_INPUT_DIR
    R3_TRANSFER_CALIBRATION_BASE
    R3_TRANSFER_TWIX
    R3_TRANSFER_SEQUENCE
    R3_TRANSFER_OUTPUT_ROOT
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Error: required environment variable is unset: $variable_name" >&2
        echo "Copy run_r3_wavelet_transfer.example.sh to an ignored .local.sh launcher." >&2
        exit 2
    fi
done

PYTHON_EXECUTABLE="$(command -v python)" || {
    echo "Error: python is unavailable; activate the configured environment." >&2
    exit 2
}
BART_EXECUTABLE="$(command -v bart)" || {
    echo "Error: bart is unavailable; source the host BART startup script." >&2
    exit 2
}
WRAPPER="$REPOSITORY_ROOT/external/wave-mprage/recon/bart/run_wave_recon.sh"
SELECTION_MANIFEST="$(realpath "$R3_TRANSFER_SELECTION_MANIFEST")"
BART_INPUT_DIR="$(realpath "$R3_TRANSFER_BART_INPUT_DIR")"
CALIBRATION_BASE="$(realpath -m "$R3_TRANSFER_CALIBRATION_BASE")"
TWIX="$(realpath "$R3_TRANSFER_TWIX")"
SEQUENCE="$(realpath "$R3_TRANSFER_SEQUENCE")"
OUTPUT_ROOT="$(realpath -m "$R3_TRANSFER_OUTPUT_ROOT")"
MEASUREMENT_INDEX="${R3_TRANSFER_MEASUREMENT_INDEX:-1}"
SUBJECT="${R3_TRANSFER_SUBJECT:-mprage-r3-transfer}"
LAMBDA_ZERO_DIR="$OUTPUT_ROOT/ecalib_crop-0p6_lambda0"
REGULARIZATION_ROOT="$OUTPUT_ROOT/regularization"
REVIEW_DIR="$OUTPUT_ROOT/qualitative_review"

readarray -t selection_fields < <(
    "$PYTHON_EXECUTABLE" -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
selection = record["selection"]
if record.get("status") != "frozen_for_cross_dataset_transfer":
    raise SystemExit("selection is not frozen for transfer")
if selection.get("regularizer") != "wavelet":
    raise SystemExit("frozen selection is not Wavelet")
if float(selection.get("lambda", -1)) != 0.015:
    raise SystemExit("frozen Wavelet lambda is not 1.5e-2")
if record.get("frozen_transfer_rule", {}).get("retune_on_r3") is not False:
    raise SystemExit("selection record does not prohibit R3 retuning")
print(selection["lambda"])
print(selection["lambda_label"])
print(selection["iterations"])
print(selection["tolerance"])
' "$SELECTION_MANIFEST"
)
SELECTED_LAMBDA="${selection_fields[0]}"
SELECTED_LABEL="${selection_fields[1]}"
ITERATIONS="${selection_fields[2]}"
TOLERANCE="${selection_fields[3]}"

echo "Python: $PYTHON_EXECUTABLE"
echo "BART: $BART_EXECUTABLE"
echo "Output: $OUTPUT_ROOT"
echo "Frozen Wavelet lambda: $SELECTED_LAMBDA"
echo "Purpose: qualitative R3 transfer assessment only; no metric ranking or retuning."

"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
    --bart "$BART_EXECUTABLE" \
    --bart-input-dir "$BART_INPUT_DIR" \
    --calibration-base "$CALIBRATION_BASE" \
    --output-dir "$LAMBDA_ZERO_DIR" \
    --twix "$TWIX" \
    --sequence "$SEQUENCE" \
    --measurement-index "$MEASUREMENT_INDEX" \
    --subject "$SUBJECT-lambda0" \
    --ecalib-crop 0.6 \
    --cg-iterations 300 \
    --cg-tolerance 1e-3 \
    --resume

LAMBDA_ZERO_MANIFEST="$LAMBDA_ZERO_DIR/manifest.json"
readarray -t lambda_zero_fields < <(
    "$PYTHON_EXECUTABLE" -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if record.get("status") != "lambda0_complete_awaiting_visual_review":
    raise SystemExit("R3 lambda-zero calibration is incomplete")
if "-g" not in record.get("wave_lambda0", {}).get("command", []):
    raise SystemExit("R3 lambda zero did not use BART GPU -g")
print(record["ecalib"]["output_base"])
print(record["ecalib"]["output_cfl_sha256"])
print(record["wave_lambda0"]["output_base"])
print(record["wave_lambda0"]["maximum_eigenvalue"])
' "$LAMBDA_ZERO_MANIFEST"
)
MAPS="${lambda_zero_fields[0]}"
MAPS_SHA256="${lambda_zero_fields[1]}"
LAMBDA_ZERO_BASE="${lambda_zero_fields[2]}"
MAXIMUM_EIGENVALUE="${lambda_zero_fields[3]}"

run_wavelet_case() {
    local lambda_value="$1"
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/run_bart_regularization.py" \
        --source-lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
        --wrapper "$WRAPPER" \
        --bart "$BART_EXECUTABLE" \
        --python "$PYTHON_EXECUTABLE" \
        --bart-input-dir "$BART_INPUT_DIR" \
        --maps "$MAPS" \
        --expected-maps-sha256 "$MAPS_SHA256" \
        --lambda-zero-base "$LAMBDA_ZERO_BASE" \
        --output-root "$REGULARIZATION_ROOT" \
        --twix "$TWIX" \
        --sequence "$SEQUENCE" \
        --regularizer wavelet \
        --lambda-value "$lambda_value" \
        --iterations "$ITERATIONS" \
        --tolerance "$TOLERANCE" \
        --max-eigenvalue "$MAXIMUM_EIGENVALUE" \
        --backend gpu \
        --subject "$SUBJECT" \
        --resume
}

# A solver-matched FISTA zero makes the visual comparison attributable to lambda.
run_wavelet_case 0
run_wavelet_case "$SELECTED_LAMBDA"

"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/review_regularization_sweep.py" \
    --lambda-zero-manifest "$LAMBDA_ZERO_MANIFEST" \
    --sweep-root "$REGULARIZATION_ROOT" \
    --output-dir "$REVIEW_DIR" \
    --regularizer wavelet \
    --lambda-labels 0 "$SELECTED_LABEL" \
    --qualitative-transfer-only

echo "R3 qualitative transfer package complete."
echo "Review: $REVIEW_DIR/wavelet_sweep_common_window.png"
echo "No R3 metric ranking or lambda selection was performed."
