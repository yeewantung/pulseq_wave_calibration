# Wave retrospective reconstruction

This repository tool prepares measured Wave data for explicit BART
reconstruction. It supports the validated sagittal integrated Wave-MPRAGE
workflow and a transverse Wave-GRE workflow for any positive echo count,
including single echo. It supports normal native R3x1, retrospective native
R3x2, and exact LIN-cropped R3x2 inputs. GRE code and unit contracts are
complete, but GRE output is not claimed as real-data validated until the user
visually reviews the generated magnitude and phase NIfTIs.

The user-facing input is a Wave-encoded Siemens TWIX file plus its matching
Pulseq sequence. Python validates and prepares BART inputs. The sample Bash
scripts show every `bart ecalib` and `bart wave` command explicitly; Python
preparation and conversion modules never launch BART.

## Data and operator contract

GRE uses logical `(RO, LIN, PAR)` roles `(readout, phase, slice)`. Both TWIX
and sequence are required to describe matrix `250 x 250 x 72`, nominal FOV
`220 x 220 x 180 mm`, and Wave grid `1000 x 250 x 72`. The echo count must be
positive, Eco counters must be consecutive from zero, and sequence/TWIX echo
counts and ordered TE values must agree exactly within the recorded tolerance.
Historical encoded-slab-thickness, target-FOV, and `lPartitions` tags are not
geometry inputs. Pulseq TWIX can retain `lPartitions=1` for a valid 3D scan;
the raw value is recorded while sequence `Nz` is checked against the measured
MDH PAR range. Every measured echo must contain the same complete residue-2
R3x1 Cartesian lattice. The sparse Siemens phase-extent tag is likewise
recorded but does not replace the logical 250 matrix.

GRE fits one integrated-refscan `a/b/c` calibration solution and records one
hash identity shared by all echoes. Each echo retains its own sequence
trajectory and receives its own calibrated PSF. Native CSMs are estimated once
and shared across echoes. The `250 x 148 x 72` case uses the exact centered LIN
crop `[51:199]`; its CSMs are produced by centered Fourier PE resampling at
unchanged FOV followed by coil-RSS normalization. All retrospective GRE
k-space comes from direct measured-Wave cropping and pure Cartesian masking,
never forward simulation from no-Wave data.

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
`A*sin(w*kx+phi)+C1*kx+C2`. Omitting both kx bounds requests automatic range
selection; providing both inclusive `kx-min` and exclusive `kx-max` indices is
a reproducible manual override using the half-open interval `[min, max)`. The
request, selected interval, algorithm diagnostics, and pinned implementation
identity are recorded in the normal-input manifest and must match before
prepared inputs can be reused. Automatic `a/b` validation remains strict. If
only the weaker `c` fit fails, validated `a/b` frequencies must agree with one
another and with the sequence trajectory before a fixed-common-frequency `c`
fit is attempted under relaxed safety gates. If that fit fails, the accepted
hybrid uses sine-line `a/b` and the upstream nine-point smooth `c`; this
fallback is explicit in the manifest and coefficient PNG. Other automatic
selection or fit failures stop preparation. A fully rejected automatic
candidate is still written as raw-sample/candidate-curve PNG and numerical
JSON diagnostics under `OUTPUT_ROOT/normal`, explicitly labeled as not used
for reconstruction; see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

For native R3x1 input, preparation also writes the processed coefficients used
by reconstruction as complementary fixed- and full-range diagnostics:
`OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png`. The plot shows
`a`, `b`, and `c` against the oversampled kx readout index, marks the readout
center, and shades the selected sine-line fit interval when applicable. Its
y limits remain fixed at `[-2*pi, 2*pi]`; `PSF_COEFFICIENTS_FULL_RANGE.png`
autoscales each coefficient so clipped branch changes and blow-up values remain
visible. `PSF_PLANE_COMPARISON.png` compares theoretical, directly measured,
fitted, and residual phase for the kx-y and kx-z calibration planes. It is
created with each new preparation; the coefficient plots are also backfilled
when compatible BART inputs are reused. The sample scripts print a reminder to
inspect them when reconstruction has unexpected artifacts. See
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for interpretation; the plots are
diagnostic and do not automatically accept or reject a PSF.

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

## Measured single- and multi-echo GRE workflow

The adapter imports the reviewed upstream calibration implementation from
`external/wave-gre-flow-comp` commit `d3772bd`. It does not modify or duplicate
that external submodule.

The multi-echo calibration contract is:

- fit one shared `a`, `b`, and `c` coefficient solution from the integrated
  projection refscan and reuse it for every echo;
- do not independently refit `a/b/c` for later echoes, because that could
  absorb the inter-echo phase evolution needed for delta-B0 analysis;
- retain echo-specific sequence-derived theoretical trajectories when flow
  compensation makes them differ, and combine each trajectory with the one
  shared coefficient solution to produce the corresponding per-echo PSF; and
- validate identical image sampling, echo/TE ordering, finite inputs, and the
  shared calibration identity before exporting any per-echo BART inputs.

The upstream coefficient-processing interface matches the reviewed MPRAGE
policy: nine-sample `smooth` remains the default, `sine-line` without
bounds requests automatic range selection, and a complete half-open
`[kx_min, kx_max)` pair is the reproducible manual override. Automatic
selection receives calibration-quality evidence, records its selected range
and fit diagnostics, and fails explicitly instead of silently reverting to
smooth.

The default must not move from smooth to automatic sine-line until both paths
have been tested on representative real multi-echo GRE data.

LR CSMs are derived from the one accepted native ecalib map set by centered
Fourier resampling in PE at unchanged FOV, followed by coil-RSS
renormalization. Readout maps are not resized. Every reconstruction exports
magnitude and phase NIfTI files. Exact shell-escaped `ecalib` and `wave`
commands are retained beside the BART outputs and copied into the NIfTI JSON
sidecars; an existing CSM is reused only when its command record matches.

Each case and echo has both `fista_r0` and `selected_wavelet` branches. The
Wavelet method and lambda `0.015` originate from
`wavelet_shared_echo_selection.json` with SHA-256
`0c43a9d31672e90ad851decfca66c253c362cbd67ca5ba97c4fd8ef1f5a61afd`.
The runtime policy applies that shared value unchanged to every validated
echo:

| GRE case | Shared lambda for every echo |
| --- | ---: |
| native R3x1 | `0.015` |
| native R3x2 | `0.015` |
| LIN-low-resolution R3x2 | `0.015` |

The shared value does not couple echoes: BART runs separately with distinct
PSF, measured k-space, TE, output, phase, and NIfTI provenance for each echo.
No LLR selection is inferred. The converter restores BART output with
`amplitude = kspace_norm * sqrt(extended_RO * LIN * PAR)` and
`phase = 1j * (-1)**(LIN//2)`. Restored quantitative complex arrays are saved
separately from display-normalized magnitude and wrapped-phase NIfTIs. GRE
first applies logical roles `(readout, phase, slice)` and the GRE orientation-
sweep-validated flips `(False, True, False)`, then uses only axis
permutation/flips to store canonical RAS without interpolation. No
reconstruction-stage brain mask or BET step is included.

## GRE head-mask parameter derivation

The MPRAGE NIfTI collection command is intentionally not used for GRE. Before
adding the GRE collection, derive and visually approve a separate GRE default
from the corrected canonical-RAS normal/native-R3x1, selected-Wavelet, echo-1
magnitude. The derivation command creates an unranked parameter grid; each
candidate receives a NIfTI mask and a nine-panel overlay covering 25%, 50%,
and 75% positions in all three anatomical planes:

```bash
scripts/derive_gre_head_mask_parameters.py \
    /exact/path/to/normal/nifti/selected_wavelet/echo-01_part-mag.nii.gz \
    /exact/new/gre_head_mask_parameter_sweep
```

The destination must be a new user-confirmed directory. The command never
selects a winner. Review every file under `overlays/`, then record one explicit
candidate ID as the GRE default. A later GRE-specific collection will derive
one mask on this normal echo-1 grid, apply it identically to every echo and
reconstruction branch, and map it to different retrospective grids only with
nearest-neighbor physical-space mask resampling.

## MPRAGE presentation masking

MPRAGE reconstruction and presentation masking are separate workflows. The
normal and retrospective reconstruction scripts write canonical NIfTIs below the
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

## GRE commands awaiting visual validation

After confirming a new exact output root, run native R3x1 in the configured
tmux shell:

```bash
scripts/sample_gre_normal_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/confirmed_output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

Run both retrospective R3x2 cases with the same inputs and output root:

```bash
scripts/sample_gre_retro_lr_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/confirmed_output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

Omit `-g` for CPU BART Wave. Both samples accept the same optional smooth or
sine-line PSF settings documented below for MPRAGE. Inspect every echo and
both branches before describing GRE as real-data validated.

## MPRAGE normal measured-data command

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

If the default smooth-coefficient PNG looks unreliable, request automatic
sine-plus-line fitting by omitting the optional bounds:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing sine-line
```

If the automatic result is also unsatisfactory, override it with a reviewed
half-open readout interval:

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
rather than overwritten. Always inspect the newly generated coefficient PNG;
automatic sine-line is optional and does not change the default from smooth.

Projection-space region selection is global and independent of the kx
sine-line interval. Every preparation compares each full y/z projection with
a center-containing core. A clean full-region fit is preserved exactly; only
a materially inconsistent plane searches for the widest stable inner region.
The selected half-open calibration-image indices are recorded in the manifest
and drawn on `PSF_PLANE_COMPARISON.png`. A reviewed manual override is:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-fit-y-min Y_START --psf-fit-y-max Y_END \
    --psf-fit-z-min Z_START --psf-fit-z-max Z_END
```

The original full-FOV normalized coordinates are retained after selecting an
inner region. The constant coefficient is aligned only by integer multiples of
`2*pi` before smoothing or sine-line fitting; this is exactly invariant in
complex phase. Raw coefficients, aligned processing inputs, and integer branch
turns are stored separately.

If a nominally clean full-y fit later makes automatic kx selection reject
sustained coefficient corruption, preparation retries once with the central
50% of the y calibration plane. For the standard 72-sample calibration this is
the exact half-open interval `[18, 54)`. The cached central fit is reused, so
the integrated refscan is not loaded again. The retry must pass the unchanged
kx and coefficient gates; otherwise preparation still fails. Explicit manual
y bounds disable this fallback, and every accepted retry is recorded under
`processing_diagnostics.automatic_spatial_fallback`.

These examples use CPU BART. Add `-g` after the other arguments to request GPU
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
- `wave_retro_lr/gre.py`: measured multi-echo GRE geometry, sampling, shared
  calibration, direct retrospective crop, CSM, command, and normalization
  contracts;
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

On `macha`, activate `cuda133py312-macha` and source `bart_startup.sh` first,
as shown in `SETUP.md`.
