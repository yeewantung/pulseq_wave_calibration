#!/usr/bin/env bash
# Copy this file to run_r3_wavelet_transfer.local.sh and fill private paths there.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"

export R3_TRANSFER_SELECTION_MANIFEST="/path/to/selection_manifest.json"
export R3_TRANSFER_BART_INPUT_DIR="/path/to/r3/bart_inputs"
export R3_TRANSFER_CALIBRATION_BASE="/path/to/r3/kspace_calib"
export R3_TRANSFER_TWIX="/path/to/r3_measurement.dat"
export R3_TRANSFER_SEQUENCE="/path/to/mprage_sequence.seq"
export R3_TRANSFER_OUTPUT_ROOT="/path/to/new/r3_transfer_output"
export R3_TRANSFER_MEASUREMENT_INDEX=1
export R3_TRANSFER_SUBJECT="mprage-r3-transfer"

exec "$SCRIPT_DIR/run_r3_wavelet_transfer.sh"
