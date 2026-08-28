# Retrospective low-resolution Wave-MPRAGE reconstruction

This tool creates lower phase-encoding resolution Wave-MPRAGE cases from an
explicit source manifest. It is integrated into the parent repository and uses
the pinned `external/wave-mprage` NIfTI exporter. The imported two-commit
history remains available in Git; the legacy internal-NPY/Torch-CG program has
been replaced by this BART workflow.

## Scientific contract

For validated sagittal MPRAGE, logical `(RO, LIN, PAR)` maps to physical
`(Z, Y, X)`. The tool changes LIN and/or PAR only. It never crops readout.

Each case performs these operations in order:

1. center-crop the full no-wave k-space in LIN/PAR;
2. inverse FFT on the requested target matrix at unchanged physical FOV;
3. preserve and center-embed the original readout in the oversampled Wave FOV;
4. rebuild the final source PSF phase planes on the target PE grid;
5. apply Wave encoding and transform PE back to k-space; and
6. apply the cropped acquisition/ACS mask.

The program deliberately does not center-crop already Wave-encoded k-space.
The included operator test demonstrates that the two orderings are generally
not equivalent.

Before preparing any target case, two gates must pass:

- extracted PSF phase planes must regenerate the source PSF within strict
  complex-error tolerances; and
- crop-first Wave synthesis on the native grid must reproduce the supplied
  source Wave k-space.

## Reconstruction and export

Sensitivity maps are interpolated only in PE and RSS-renormalized. Calibration
k-space is center-cropped only in PE and retained in each case's BART input
contract. The primary backend is `bart wave`; every reconstruction command
always includes GPU option `-g`. LLR uses BART's split-complex `-v` form and is
recombined using the previously validated rule.

After BART reconstruction, the actual retrospective Wave-k-space norm is
restored. The pinned Wave-MPRAGE exporter writes normalized magnitude and
phase directly in canonical RAS with target achieved voxel sizes.

## Configuration

Use one JSON file containing explicit source, companion, output, case, and
reconstruction settings. See
[`configs/retrospective_low_resolution.example.json`](configs/retrospective_low_resolution.example.json).
Source discovery is intentionally shallow: the source BART `manifest.json` is
authoritative for Wave k-space and PSF basenames, while maps, calibration
k-space, no-wave k-space, sequence, and TWIX are explicit companion inputs.

Requested resolution uses physical `[X, Y, Z]` millimetres. Each target PE
matrix is rounded to its nearest multiple of four; readout remains unchanged.
Requested and matrix-achieved resolutions are both stored. Output folders use
the achieved resolution and acceleration, for example:

| Requested XYZ resolution (mm) | Logical RO, LIN, PAR matrix | Achieved XYZ resolution (mm) |
| --- | --- | --- |
| 1.5 x 1.0 x 1.0 | 256 x 256 x 172 | 1.488372 x 1.0 x 1.0 |
| 1.0 x 1.5 x 1.0 | 256 x 172 x 256 | 1.0 x 1.488372 x 1.0 |
| 1.25 x 1.25 x 1.0 | 256 x 204 x 204 | 1.254902 x 1.254902 x 1.0 |

```text
retrospective_low_resolution/
├── batch_manifest.json
├── res1.49x1x1mm_R3x2/
│   ├── case_manifest.json
│   ├── bart_inputs/
│   ├── bart_output/
│   ├── bart_wave.log
│   └── nifti/
└── res1.25x1.25x1mm_R3x2/
```

Sampling masks are used in memory and are not copied into result trees.
Source and companion manifest hashes, geometry, crop bounds, requested and
achieved resolution, matrix, FOV, BART inputs, commands, norms, and canonical
outputs are recorded in manifests. Existing non-empty output trees are refused
unless `--resume` finds matching case metadata.

For solver-only comparisons on the same target grids, set
`prepared_cases_root` to a completed `retrospective_low_resolution` workflow.
The pipeline verifies the prior batch, case geometry, source provenance, and
BART-input hashes, then links those immutable inputs into the new output tree
instead of synthesizing them again. The completed preparation's finite PSF and
source-operator gates are reused with that hash-bound batch, avoiding repeated
full-volume operator checks during solver sweeps. Wavelet mode permits `lambda: 0` so the
effective GPU command uses BART Wave FISTA (`-w -f -r 0 -g`) as an
unregularized solver control.

## Commands

Activate the appropriate Python environment and BART build first. Keep the
machine-specific startup locations in an ignored local launcher:

```bash
source "$CONDA_SETUP"
conda activate "$WAVE_RECON_CONDA_ENV"
source "$BART_STARTUP"
```

Structural validation reads JSON, sequence metadata, NPY headers, and CFL
headers only; it creates no output:

```bash
python tools/wave_retro_lr_recon/scripts/run_retro_lr.py \
    --config /path/to/config.json \
    --validate-only
```

Prepare BART inputs and run the source-operator gates without reconstruction:

```bash
python tools/wave_retro_lr_recon/scripts/run_retro_lr.py \
    --config /path/to/config.json \
    --prepare-only
```

Run or resume the complete GPU workflow:

```bash
python tools/wave_retro_lr_recon/scripts/run_retro_lr.py \
    --config /path/to/config.json \
    --resume
```

## Tests

From this tool directory:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The historical `wave-retro-lr-us-TODO.md` records the design analysis that led
to the current manifest/BART implementation; it is not the active run guide.
