# Synthetic Wave regularization baseline

This directory contains the phased no-wave-to-Wave baseline experiment described in `R3x1_no_wave_to_wave_BART_regularization_TODO.md`.

## Layout

```text
scripts/       Data inspection, GRAPPA, and theoretical Wave synthesis programs
requirements/  Incremental Python dependency sets for each phase
tests/         Unit and reference-oracle tests
```

Machine-local dataset notes and generated reconstruction artifacts remain at this directory's root. They are ignored by git; paths and scan filenames stay command-line inputs rather than source constants.

## Current baseline

The selected no-wave baseline is joint-coil 5×5×5 GRAPPA at Ncc=12. It removes
the aliasing retained by shallower kernels, preserves the full visual anatomy,
and directly supplies completed multi-coil k-space for Wave synthesis. Tested
SENSE/ESPIRiT variants either crop the nose or admit a detached weak-support
noise shell and require an additional `F(Sx)` back-projection. SENSE remains a
deferred secondary comparison; its apparently worse effective g-factor is a
qualitative visual observation, not yet a formal measurement.

The next run will regenerate synthetic Wave inputs from the accepted 5×5×5
k-space in a new output tree, visually gate one reusable hard-crop 0.5 ESPIRiT
map set with λ=0, then run coarse wavelet (`1e-4`, `1e-3`, `1e-2`) and LLR
(block 8; `2e-4`, `2e-3`, `2e-2`) pilots. Fine sweeps are deferred. Final
ranking will combine visual review with consistently registered and normalized
DICOM-referenced SSIM, PSNR, NRMSE, noise/CNR, sharpness, aliasing, and anatomy
coverage metrics.

The positive-λ pilot has its own troubleshooting gate: run wavelet `1e-3` and
LLR block-8 `2e-3` first, export both, and wait for explicit visual approval
before running the four remaining coarse endpoints.

## Setup and tests

From the repository root:

```bash
git submodule update --init external/wave-mprage
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/phase-e.txt
python -m unittest discover \
    -s tools/synthetic_wave_for_reg_baseline/tests \
    -p 'test_*.py'
```

Emergency Phase G additionally uses:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/sense.txt
```

Wave-MPRAGE is pinned as a submodule because the current scripts import its
BART CFL and TWIX-to-NIfTI utilities at runtime. Wave-GRE remains an optional
reference and is not a submodule because no current code depends on it.

Run a script with `--help` for its dataset-independent CLI, for example:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/phase_c_grappa.py --help
```
