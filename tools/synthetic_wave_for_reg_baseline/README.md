# Synthetic Wave regularization baseline

This directory contains the no-wave-to-Wave baseline experiment described in
`R3x1_no_wave_to_wave_BART_regularization_TODO.md`.

## Layout

```text
scripts/       Reconstruction programs and reusable algorithm/I/O modules
requirements/  Incremental Python dependency sets grouped by task
docs/          Maintenance indexes, including the old-to-new filename map
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

## Which programs matter

The current production path is:

1. `inspect_product_dataset.py` audits a new product TWIX/DICOM dataset.
2. `estimate_coil_compression.py` estimates the Ncc=12 compression basis.
3. `reconstruct_no_wave_grappa_3d.py` completes the no-wave k-space with the
   accepted joint-coil 5×5×5 GRAPPA kernel.
4. `synthesize_wave_kspace.py` applies the theoretical sequence PSF.
5. `export_bart_wave_inputs.py` masks and exports the synthetic Wave data.
6. `export_bart_calibration_acs.py` exports measured no-wave ACS for one
   reusable BART ESPIRiT calibration.
7. `run_bart_wave_lambda0.py` runs the unregularized acceptance reconstruction.

`reconstruct_no_wave_grappa_2d.py`, `prepare_no_wave_sense.py`, and
`run_no_wave_sense.py` are retained diagnostic alternatives, not the selected
production path. `export_multicoil_nifti.py` and `export_grappa_rss.py` are
visualization helpers. The old-to-new filename dictionary and the reason each
program was retained are in [`docs/script_name_map.md`](docs/script_name_map.md).

## Setup and tests

From the repository root:

```bash
git submodule update --init external/wave-mprage
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/bart_reconstruction.txt
python -m unittest discover \
    -s tools/synthetic_wave_for_reg_baseline/tests \
    -p 'test_*.py'
```

The optional SENSE diagnostic additionally uses:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/sense_diagnostics.txt
```

Wave-MPRAGE is pinned as a submodule because the current scripts import its
BART CFL and TWIX-to-NIfTI utilities at runtime. Wave-GRE remains an optional
reference and is not a submodule because no current code depends on it.

The NIfTI orientation/export helpers and the in-memory BART CFL writer are
called directly from that submodule. Coil-compression covariance accumulation
remains a local streaming adapter because the product refscan is read in PE2
chunks rather than materialized as the four-dimensional array expected by the
reference utility. The theoretical-trajectory adapter likewise adds strict
integrated-tail/sequence-definition checks and lightweight provenance without
importing the upstream all-in-one reconstruction program and its unrelated
GPU/SENSE dependencies. Regularized production runs should call the pinned
`external/wave-mprage/recon/bart/run_wave_recon.sh` wrapper directly; the local
lambda-zero runner is retained for its timing, map montage, and acceptance
manifest checks.

Run a script with `--help` for its dataset-independent CLI, for example:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/reconstruct_no_wave_grappa_3d.py --help
```
