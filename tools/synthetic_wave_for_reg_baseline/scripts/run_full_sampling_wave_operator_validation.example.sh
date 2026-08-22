#!/usr/bin/env bash
# Copy to the ignored .local.sh name and fill in machine-specific paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SETUP:?Set CONDA_SETUP in the local copy.}"
conda activate "${SYNTHETIC_WAVE_CONDA_ENV:?Set SYNTHETIC_WAVE_CONDA_ENV.}"

: "${R1_DATASET_MANIFEST:?Set R1_DATASET_MANIFEST in the local copy.}"
exec python "$SCRIPT_DIR/validate_full_sampling_wave_operator.py" \
    --dataset-manifest "$R1_DATASET_MANIFEST" \
    --resume
