# Wave retrospective reconstruction

This repository tool prepares measured Wave data for BART reconstruction. The
current supported user workflow is sagittal integrated Wave-MPRAGE; GRE is
not yet implemented while its reviewed axial multi-echo adapter is planned.
The confirmed future GRE contract uses native-grid normal R3x1 plus
native-grid and LIN-cropped retrospective R3x2, with explicit unregularized
FISTA (`-w -f -r 0`) for every case. These statements describe planned support
and do not make the current GRE placeholder runnable.

The user-facing input is a Wave-encoded Siemens TWIX file plus its matching
Pulseq sequence. Python validates and prepares BART inputs. The sample Bash
scripts show every `bart ecalib` and `bart wave` command explicitly; Python
does not launch BART in the new MPRAGE workflow.

## Data and operator contract

For sagittal MPRAGE, logical `(RO, LIN, PAR)` corresponds to physical
`(Z, Y, X)`. Readout and physical-Z resolution are never cropped.

Accepted measured sampling is determined from TWIX MDH coordinates:

- duplicate-free fully sampled `R1`; or
- a complete regular factor-three logical-LIN lattice for every PAR partition
  (`R3x1`).

Other, ambiguous, duplicated, incomplete, or out-of-range sampling is
rejected. A valid factor-three image lattice may omit the exact logical center
when the separate integrated ACS contains it; its measured LIN residue is
preserved. The sequence trajectory must contain both Wave axes.

Native and retrospective PSFs are calculated directly on their requested PE
grid from:

- the two sequence-derived Wave trajectory displacements; and
- the integrated calibration phase-plane coefficients `a`, `b`, and `c`.

The coefficients use the upstream nine-sample smoothing mode by default. The
sample scripts also expose the upstream `sine-line` model
`A*sin(w*kx+phi)+C1*kx+C2`; that mode requires inclusive `kx-min` and exclusive
`kx-max` readout indices, defining a half-open `[min, max)` fit interval. The
selected processing mode and interval are recorded in the normal-input
manifest and must match before prepared inputs can be reused.

For native R3x1 input, preparation also writes the processed coefficients used
by reconstruction as an immediately visible diagnostic:
`OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png`. The plot shows
`a`, `b`, and `c` against the oversampled kx readout index, marks the readout
center, and shades the selected sine-line fit interval when applicable. It is
also backfilled when compatible older BART inputs are reused. The sample
scripts print a reminder to inspect it when reconstruction has unexpected
artifacts. See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for interpretation;
the plot is diagnostic and does not automatically accept or reject a PSF.

LR PSFs are neither cropped nor interpolated. Measured-Wave LR k-space is
created by direct centered LIN/PAR cropping, preservation of the measured LIN
residue, and explicit factor-two selection on PAR, without interpolation,
forward simulation, or inferred ACS rows. The nearest PE matrices divisible
by four are used, and manifests record the achieved resolution.

The older crop-first operation for a no-Wave dataset remains available as
`wave_retro_lr.retrospective.synthesize_wave_from_no_wave_crop`. It is an
explicit utility for `synthetic_wave_for_reg_baseline`, not a user-facing
measured-data mode.

## Reconstruction defaults

The normal MPRAGE sample performs one BART `ecalib` with crop `0.6` by default.
For measured R3x1 it reconstructs both the unregularized FISTA control
(`-w -f -r 0`) and the selected Wavelet/FISTA result at lambda `3.5e-2`; the
crop and R3 Wavelet lambda may be overridden on the command line. R1 remains
FISTA-r0-only because the five-case rerun did not select an R1 Wavelet value.

The validated BART v1.0 `ecalib` command does not provide a `-g` option, so its
explicit command is `bart ecalib -m 1 -c ...`. The sample scripts run BART
`wave` on CPU by default; pass `-g` to select their explicit GPU commands.

The retrospective sample estimates native CSMs once and runs four sequential
R3x2 cases:

| Case | Requested physical XYZ resolution | FISTA control | Selected Wavelet |
| --- | --- | --- | --- |
| native | source resolution | `-w -f -r 0` | `-w -f -r 3.5e-2` |
| LR-X | `1.5 x 1.0 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| LR-Y | `1.0 x 1.5 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| LR-XY | `1.25 x 1.25 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.2e-2` |

These values come from the completed corrected pure-image-lattice synthetic
R3x1/R3x2 rerun and explicit user visual/metric review. The hash-bound
selection manifest has SHA-256
`07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`.
Historical ACS-union selections are not carried forward.

The complete dual-branch normal, retrospective, NIfTI-conversion, and shared
head-mask collection workflow passed representative real measured-MPRAGE
visual validation on 2026-09-01. This closes the MPRAGE gate before GRE work.

LR CSMs are derived from the one accepted native ecalib map set by centered
Fourier resampling in PE at unchanged FOV, followed by coil-RSS
renormalization. Readout maps are not resized. Every reconstruction exports
magnitude and phase NIfTI files. Exact shell-escaped `ecalib` and `wave`
commands are retained beside the BART outputs and copied into the NIfTI JSON
sidecars; an existing CSM is reused only when its command record matches.

Reconstruction and presentation masking are separate workflows. The normal
and retrospective reconstruction scripts write canonical NIfTIs below the
`fista_r0` and `optimal_wavelet` branches. The optional collection script
creates byte-identical canonical copies and
whole-head-masked derivatives beneath `OUTPUT_ROOT/nifti_collection`. The
canonical files under `normal/nifti/*` and `retro/*/nifti/*` remain unchanged
and are the scientific source of record.

The mask is estimated once from the normal optimal-Wavelet magnitude and is
applied identically to both reconstruction branches. For the R1-only workflow,
the normal FISTA-r0 magnitude is the documented fallback source. The mask
supports a high-confidence head core with distance-limited low-threshold
growth, optional physical opening, physical closing, the largest 26-connected
3D component, 3D hole filling, and optional physical dilation. BET is not used. On LR grids,
this same mask is mapped by nearest-neighbor interpolation in NIfTI physical
space; it is never re-estimated from a noisier R3x2 image. These masked
derivatives are for viewing and background suppression, not for
regularization-sweep evaluation.

## Environment

Follow [`SETUP.md`](SETUP.md) for the recommended standard-venv installation,
the single continued-work reactivation procedure, optional uv usage, CPU-only
or CUDA-enabled BART compilation, and runtime validation. The sample scripts
resolve `python` and `bart` from `PATH`; complete that setup before using the
commands below.

## Normal measured-data command

Choose a new output root, then run:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

Optional numerical overrides are:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    --ecalib-crop 0.55 \
    --r3-lambda 1.8e-2
```

To replace the default coefficient smoothing with the upstream sine-plus-line
fit, provide both half-open readout bounds:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing sine-line \
    --psf-fit-kx-min START_INDEX \
    --psf-fit-kx-max END_INDEX
```

Replace `START_INDEX` and `END_INDEX` with integers satisfying
`0 <= min < max <=` the oversampled readout length. Use a new output root when
changing the PSF processing settings; incompatible prepared inputs are rejected
rather than overwritten.

Both examples use CPU BART. Add `-g` after the other arguments to request GPU
execution from a CUDA-enabled BART build:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    -g
```

The shared inputs and CSM remain directly below `normal/bart_inputs` and
`normal/bart_output`. Reconstructed images and NIfTIs are separated into
`fista_r0` and `optimal_wavelet` subdirectories. R1 creates only `fista_r0`.
The native R3x1 PSF coefficient diagnostic is placed directly under `normal/`
so it is not hidden among BART arrays.

## Retrospective R3x2 and LR command

Use the same TWIX, output root, and sequence values as the normal command:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_retro_lr_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

This command uses CPU BART by default. Append `-g` to reconstruct all eight
Wave branches with a CUDA-enabled BART build. The retrospective script accepts
the same three `--psf-*` options as the normal script. If normal inputs already
exist, their recorded coefficient-processing settings must match.

Compatible normal inputs and maps are reused. If absent, the inputs are
prepared from TWIX and ecalib is run once. Each of the four canonical cases
contains `fista_r0` and `optimal_wavelet` reconstruction/NIfTI branches beneath
`OUTPUT_ROOT/retro/`.

## Optional whole-head-masked NIfTI collection

After running the normal reconstruction and any desired retrospective cases,
build the separate presentation collection with:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_nifti_collection.sh \
    /path/to/output_root \
    --require-retro
```

Omit `--require-retro` when only the normal reconstruction should be
collected. The real-data-validated starting defaults are low threshold `0.02`,
core threshold `0.05`, maximum core-growth distance `12 mm`, smoothing `1 mm`,
opening `0 mm`, closing `1.5 mm`, and dilation `0 mm`. Subject-specific
overrides are accepted by the separate script and recorded in the manifest;
they never complicate or change the reconstruction command.

The collection layout is:

```text
OUTPUT_ROOT/
├── normal/nifti/<branch>/                # canonical, unmasked source
├── retro/<case>/nifti/<branch>/          # canonical, unmasked source
└── nifti_collection/
    ├── original_nifti/
    │   └── <branch>/
    │       ├── normal/
    │       └── retro/<case>/
    ├── head_masked_nifti/
    │   └── <branch>/
    │       ├── normal/
    │       └── retro/<case>/
    ├── masks/
    └── manifest.json
```

Here `<case>` is `native_r3x2`, `lr_x_1p5mm_r3x2`,
`lr_y_1p5mm_r3x2`, or `lr_xy_1p25mm_r3x2`.
`<branch>` is `fista_r0` or `optimal_wavelet`. One mask derived from the normal
optimal-Wavelet magnitude is reused for both branches so masked visual
comparisons never differ because of candidate-specific support.

This script never runs k-space preparation, ecalib, or Wave reconstruction.
Its optional parameters are listed by
`scripts/build_mprage_nifti_collection.py --help`. Actual subject paths and
preferred overrides may be kept in an ignored `.local.sh` wrapper.

## Implementation map

- `wave_retro_lr/mprage.py`: measured MPRAGE preparation and orchestration;
- `wave_retro_lr/sampling.py`: MDH sampling classification and the canonical
  pure Cartesian image-lattice mask/validation contract;
- `wave_retro_lr/psf.py`: direct calibrated PSF evaluation;
- `wave_retro_lr/retrospective.py`: measured-Wave crop, CSM resampling, and the
  explicitly named synthetic no-Wave utility;
- `wave_retro_lr/gre.py`: non-runnable GRE adapter placeholder pending
  real-data MPRAGE validation;
- `wave_retro_lr/bart_io.py`: bounded BART CFL I/O, logical hashing, and
  split-complex output recombination;
- `wave_retro_lr/nifti_collection.py`: byte-identical canonical collection,
  normal-derived whole-head mask, physical-grid mask mapping, and provenance;
- `wave_retro_lr/core.py`: geometry, grids, FFT, masks, and compatibility
  primitives.

`wave_retro_lr/pipeline.py` and `scripts/run_retro_lr.py` temporarily preserve
the old config-driven no-Wave interface used by the synthetic tool. They are
not the measured-data MPRAGE entry points and remain only until the synthetic
cleanup migrates its consumers.

The small `pyproject.toml` is retained as a dependency and Python-version
contract for this tool; it does not make the directory an independent nested
repository.

## Tests

From the parent repository:

```bash
python -m unittest discover \
    -s tools/wave_retro_lr_recon/tests \
    -p 'test_*.py'
```
