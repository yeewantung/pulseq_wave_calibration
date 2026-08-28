#!/usr/bin/env bash
# Copy to .local.sh, set private paths, and run in tmux.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SOURCE_DATASET_ROOT:?Set SOURCE_DATASET_ROOT to the accepted R1 dataset root.}"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the new target root.}"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"

exec python "$SCRIPT_DIR/export_bart_wave_target_branch.py" \
    --source-synthesis-manifest "$SOURCE_DATASET_ROOT/synthetic_wave/full_encoding/manifest.json" \
    --source-bart-input-manifest "$SOURCE_DATASET_ROOT/synthetic_wave/target_sampling/bart_inputs/manifest.json" \
    --operator-validation-manifest "$SOURCE_DATASET_ROOT/evaluation/full_sampling_wave_operator_validation/operator_validation_manifest.json" \
    --output-dir "$SYNTHETIC_WAVE_R3X1_ROOT/target_sampling" \
    --target-id synthetic-wave-r3x1 \
    --pe1-acceleration 3 \
    --pe2-acceleration 1 \
    --pe1-residue 1 \
    --pe2-residue 0 \
    --acs-pe1-start 115 \
    --acs-pe1-stop-exclusive 139 \
    --confirm-full-wave-reviewed \
    --resume
