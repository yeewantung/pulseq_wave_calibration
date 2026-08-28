#!/usr/bin/env bash
# Copy to .local.sh and set private target/reference paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SYNTHETIC_WAVE_R3X1_ROOT:?Set SYNTHETIC_WAVE_R3X1_ROOT to the target root.}"
: "${METRICS_REFERENCE_MANIFEST:?Set METRICS_REFERENCE_MANIFEST to the approved direct-FFT manifest.}"
: "${METRICS_REFERENCE_MANIFEST_SHA256:?Set its approved SHA-256 digest.}"

source "/path/to/conda/etc/profile.d/conda.sh"
conda activate "your-conda-environment"

exec bash "$SCRIPT_DIR/evaluate_synthetic_wave_r3x1_regularization.sh"
