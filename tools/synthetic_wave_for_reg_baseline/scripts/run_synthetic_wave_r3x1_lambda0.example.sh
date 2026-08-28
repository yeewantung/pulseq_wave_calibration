#!/usr/bin/env bash
# Copy to .local.sh, set private paths, and run only after target preparation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SOURCE_TWIX:?Set SOURCE_TWIX to the fully sampled source TWIX.}"
: "${WAVE_SEQUENCE:?Set WAVE_SEQUENCE to the accepted Wave sequence.}"
: "${SOURCE_SUBJECT:?Set SOURCE_SUBJECT to the subject in the source dataset manifest.}"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the target root.}"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"
source "/path/to/bart_startup.sh"

echo "Backend: run_bart_wave_lambda0.run -> BART ecalib, then BART wave CG -g"
python "$SCRIPT_DIR/run_bart_wave_lambda0.py" \
    --bart "$(command -v bart)" \
    --bart-input-dir "$SYNTHETIC_WAVE_R3X1_ROOT/target_sampling/bart_inputs" \
    --output-dir "$SYNTHETIC_WAVE_R3X1_ROOT/reconstructions/lambda0_ecalib_crop-0p6" \
    --twix "$SOURCE_TWIX" \
    --sequence "$WAVE_SEQUENCE" \
    --measurement-index 1 \
    --subject "$SOURCE_SUBJECT" \
    --ecalib-crop 0.6 \
    --cg-iterations 300 \
    --cg-tolerance 1e-3 \
    --resume

echo "Before the sweep, inspect lambda0_nifti_central_slices.png and both ESPIRiT-map montages."
