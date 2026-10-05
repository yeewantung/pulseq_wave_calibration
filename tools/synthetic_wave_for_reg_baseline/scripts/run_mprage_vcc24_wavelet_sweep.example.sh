#!/usr/bin/env bash
# Copy to .local.sh and set the private dataset-manifest path.

set -euo pipefail

REPOSITORY_ROOT="/path/to/pulseq_wave_calibration"
DATASET_MANIFEST="$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/configs/mprage_vcc24_wavelet_sweep.local.json"
APPROVED_BRAIN_MASK_MANIFEST="/path/to/approved_same_subject_brain_mask_manifest.json"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"

exec bash "$REPOSITORY_ROOT/tools/synthetic_wave_for_reg_baseline/scripts/run_mprage_vcc24_wavelet_sweep.sh" \
    "$DATASET_MANIFEST" "$APPROVED_BRAIN_MASK_MANIFEST" --confirm-run
