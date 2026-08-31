# Wave retrospective reconstruction

This repository tool prepares measured Wave data for BART reconstruction. The
current supported user workflow is sagittal integrated Wave-MPRAGE; GRE is
intentionally deferred until MPRAGE has passed real-data validation.

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
For measured R3x1 it uses Wavelet/FISTA with lambda `2.2e-2`; the crop and R3
lambda may be overridden on the command line. R1 uses unregularized FISTA
(`-w -f -r 0`).

The pinned `macha` BART `ecalib` command does not provide a `-g` option, so its
explicit command is `bart ecalib -m 1 -c ...`. BART `wave` does support GPU,
and every reconstruction command requires `-g`.

The retrospective sample estimates native CSMs once and runs four sequential
R3x2 cases:

| Case | Requested physical XYZ resolution | BART Wave solver |
| --- | --- | --- |
| native | source resolution | Wavelet/FISTA, locked `1.5e-2` |
| LR-X | `1.5 x 1.0 x source-Z` mm | unregularized FISTA |
| LR-Y | `1.0 x 1.5 x source-Z` mm | unregularized FISTA |
| LR-XY | `1.25 x 1.25 x source-Z` mm | unregularized FISTA |

LR CSMs are derived from the one accepted native ecalib map set by centered
Fourier resampling in PE at unchanged FOV, followed by coil-RSS
renormalization. Readout maps are not resized. Every reconstruction exports
magnitude and phase NIfTI files. Exact shell-escaped `ecalib` and `wave`
commands are retained beside the BART outputs and copied into the NIfTI JSON
sidecars; an existing CSM is reused only when its command record matches.

## Environment

Activate the repository environment and the host-compatible BART build. Keep
their actual locations in an ignored local launcher, for example:

```bash
source "$CONDA_SETUP"
conda activate "$WAVE_RECON_ENV"
source "$BART_STARTUP"
```

The scripts resolve `python` and `bart` from `PATH`. Do not record a
machine-specific executable or dataset path in a tracked file; use an ignored
`.local.sh` launcher when desired.

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

Outputs are grouped under `OUTPUT_ROOT/normal/{bart_inputs,bart_output,nifti}`.

## Retrospective R3x2 and LR command

Use the same TWIX, output root, and sequence values as the normal command:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_retro_lr_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

If compatible normal inputs or maps already exist, they are reused. Otherwise
the script prepares normal inputs from TWIX and runs ecalib once. The four
cases are written beneath `OUTPUT_ROOT/retro/`.

## Implementation map

- `wave_retro_lr/mprage.py`: measured MPRAGE preparation and orchestration;
- `wave_retro_lr/sampling.py`: MDH sampling classification;
- `wave_retro_lr/psf.py`: direct calibrated PSF evaluation;
- `wave_retro_lr/retrospective.py`: measured-Wave crop, CSM resampling, and the
  explicitly named synthetic no-Wave utility;
- `wave_retro_lr/gre.py`: non-runnable GRE adapter placeholder pending
  real-data MPRAGE validation;
- `wave_retro_lr/bart_io.py`: bounded BART CFL I/O;
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
