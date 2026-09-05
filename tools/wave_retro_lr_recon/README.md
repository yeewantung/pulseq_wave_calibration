# Wave retrospective reconstruction

This tool prepares measured Wave data for explicit BART reconstruction. It
supports the validated sagittal integrated Wave-MPRAGE workflow and a
transverse single- or multi-echo Wave-GRE workflow. Both workflows use the
same three stages:

1. normal native R3x1 reconstruction;
2. retrospective R3x2 and low-resolution reconstruction; and
3. a separate NIfTI collection step.

The user-facing inputs are a Wave-encoded Siemens TWIX file and its matching
Pulseq sequence. Python validates and prepares BART inputs. The sample Bash
scripts show every `bart ecalib` and `bart wave` command explicitly; the Python
preparation and conversion modules never launch BART.

The sections below describe the fully validated MPRAGE workflow first. The GRE
section then summarizes the corresponding three stages and highlights only the
differences.

## MPRAGE workflow

For sagittal MPRAGE, logical `(RO, LIN, PAR)` corresponds to physical
`(Z, Y, X)`. Readout and physical-Z resolution are never cropped. TWIX MDH
coordinates must describe either duplicate-free fully sampled R1 data or a
complete regular factor-three logical-LIN lattice for every PAR partition
(`R3x1`). Ambiguous, duplicated, incomplete, or out-of-range sampling is
rejected. A valid R3x1 image lattice may omit the exact logical center when it
is present in the separate integrated ACS; the measured LIN residue is always
preserved. The sequence trajectory must contain both Wave axes.

The complete dual-branch normal, retrospective, NIfTI-conversion, and shared
head-mask collection workflow passed representative real measured-MPRAGE
visual validation on 2026-09-01.

### 1. Normal reconstruction

Choose a new output root, then run:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

The default MPRAGE normal run performs one BART `ecalib` with crop `0.6`. For
R3x1 data it reconstructs both the unregularized FISTA control
`fista_r0` (`-w -f -r 0`) and the selected Wavelet/FISTA branch
`optimal_wavelet` (`-w -f -r 3.5e-2`). R1 data creates only `fista_r0`, because
the five-case rerun did not select an R1 Wavelet value. The crop and R3
Wavelet lambda can be overridden explicitly:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    --ecalib-crop 0.55 \
    --r3-lambda 1.8e-2
```

Native and retrospective PSFs are evaluated directly on the requested PE grid
from the two sequence-derived Wave trajectory displacements and the integrated
calibration phase-plane coefficients `a`, `b`, and `c`. Automatic `sine-line`
coefficient processing is the default and selects its kx interval when no
bounds are supplied. The upstream nine-sample `smooth` mode remains available
as an explicit fallback:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing smooth
```

The sine-line model is `A*sin(w*kx+phi)+C1*kx+C2`. A reviewed manual half-open
readout interval `[min, max)` can instead be supplied with both bounds:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing sine-line \
    --psf-fit-kx-min START_INDEX \
    --psf-fit-kx-max END_INDEX
```

Use integers satisfying `0 <= min < max <=` the oversampled readout length.
Changing coefficient-processing settings requires a new output root;
incompatible prepared inputs are rejected rather than overwritten. The
request, selected interval, diagnostics, and pinned implementation identity
are recorded in the normal-input manifest.

Automatic `a/b` validation is strict. If only the weaker `c` fit fails,
validated `a/b` frequencies must agree with one another and with the sequence
trajectory before a fixed-common-frequency `c` fit is attempted under relaxed
safety gates. If that fit also fails, the accepted hybrid uses sine-line `a/b`
and upstream nine-point smooth `c`. This fallback is explicit in the manifest
and coefficient plot. Other automatic-selection or fitting failures stop
preparation. Rejected candidates are still written as labeled PNG and JSON
diagnostics under `OUTPUT_ROOT/normal`; see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

Projection-space y/z selection is independent of the kx sine-line interval.
Every preparation compares the full projection with a center-containing core.
A clean full-region fit is retained; a materially inconsistent plane searches
for the widest stable inner region. A reviewed manual override is:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-fit-y-min Y_START --psf-fit-y-max Y_END \
    --psf-fit-z-min Z_START --psf-fit-z-max Z_END
```

The original full-FOV normalized coordinates remain unchanged after selecting
an inner region. The constant coefficient is aligned only by integer multiples
of `2*pi`, which is invariant in complex phase. Raw coefficients, aligned
processing inputs, and integer branch turns are stored separately.

If a nominally clean full-y fit later causes automatic kx selection to reject
sustained coefficient corruption, preparation retries once with the central
50% of the y calibration plane. For the standard 72-sample calibration this is
the exact half-open interval `[18, 54)`. The cached central fit is reused. The
retry must pass the unchanged gates, explicit manual y bounds disable it, and
any accepted retry is recorded in
`processing_diagnostics.automatic_spatial_fallback`.

Each new preparation writes these diagnostics under `OUTPUT_ROOT/normal`:

- `PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png`, with fixed y limits
  `[-2*pi, 2*pi]`;
- `PSF_COEFFICIENTS_FULL_RANGE.png`, with independently autoscaled
  coefficients; and
- `PSF_PLANE_COMPARISON.png`, comparing theoretical, measured, fitted, and
  residual phase for the kx-y and kx-z calibration planes.

The plots are diagnostic and do not automatically accept or reject a PSF.
Always inspect them when reconstruction has unexpected artifacts.

BART `wave` runs on CPU by default. Add `-g` to request GPU execution from a
CUDA-enabled BART build:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    -g
```

The validated BART v1.0 `ecalib` command itself has no `-g` option and remains
`bart ecalib -m 1 -c ...`. Shared inputs and CSMs are stored under
`normal/bart_inputs` and `normal/bart_output`; reconstructed arrays and NIfTIs
are separated into `normal/{bart_output,nifti}/{fista_r0,optimal_wavelet}`.

### 2. Retrospective R3x2 and low-resolution reconstruction

Use the same TWIX, output root, and sequence as the normal command:

```bash
scripts/sample_mprage_retro_lr_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

The script reuses compatible normal inputs and native CSMs. If they are absent,
it prepares them and runs ecalib once. It then reconstructs four sequential
R3x2 cases:

| Case | Requested physical XYZ resolution | FISTA control | Selected Wavelet |
| --- | --- | --- | --- |
| `native_r3x2` | source resolution | `-w -f -r 0` | `-w -f -r 3.5e-2` |
| `lr_x_1p5mm_r3x2` | `1.5 x 1.0 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| `lr_y_1p5mm_r3x2` | `1.0 x 1.5 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| `lr_xy_1p25mm_r3x2` | `1.25 x 1.25 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.2e-2` |

These values come from the corrected pure-image-lattice synthetic rerun and
explicit visual/metric review. The hash-bound selection manifest has SHA-256
`07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`;
historical ACS-union selections are not carried forward.

Measured-Wave LR k-space is created by direct centered LIN/PAR cropping,
preserving the measured LIN residue and selecting factor two on PAR. It does
not interpolate, forward-simulate, or infer ACS rows. LR PSFs are evaluated
directly on the target PE grid, never cropped or interpolated. The nearest PE
matrices divisible by four are used, and manifests record the achieved
resolution.

The retrospective script accepts the same `--psf-*` settings as the normal
script, and existing normal settings must match. It uses CPU by default; append
`-g` to run all eight Wave branches on GPU. Outputs are stored beneath
`OUTPUT_ROOT/retro/<case>/{bart_output,nifti}/{fista_r0,optimal_wavelet}`.

The older crop-first operation for a no-Wave dataset remains available as
`wave_retro_lr.retrospective.synthesize_wave_from_no_wave_crop`. It is an
explicit `synthetic_wave_for_reg_baseline` utility, not a measured-data mode.

### 3. NIfTI collection

After normal reconstruction and any desired retrospective cases, build the
separate presentation collection from the same output root:

```bash
scripts/sample_mprage_nifti_collection.sh \
    /path/to/output_root \
    --require-retro
```

Omit `--require-retro` to collect only the normal reconstruction. This script
never runs k-space preparation, ecalib, or Wave reconstruction.

MPRAGE reconstruction and presentation masking remain separate. The collection
copies canonical NIfTIs byte-for-byte and creates whole-head-masked derivatives
without modifying the scientific source files under `normal/nifti` and
`retro/<case>/nifti`.

The mask is estimated once from the normal `optimal_wavelet` magnitude and
applied identically to both reconstruction branches. Normal `fista_r0` is the
documented fallback for an R1-only workflow. The mask uses a high-confidence
head core with distance-limited low-threshold growth, optional physical
opening, physical closing, the largest 26-connected 3D component, 3D hole
filling, and optional physical dilation. BET is not used. The same normal mask
is mapped to LR grids by nearest-neighbor interpolation in NIfTI physical
space; it is never re-estimated from a noisier R3x2 image. Masked outputs are
for viewing and background suppression, not regularization evaluation.

The validated defaults are low threshold `0.02`, core threshold `0.05`,
maximum core-growth distance `12 mm`, smoothing `1 mm`, opening `0 mm`, closing
`1.5 mm`, and dilation `0 mm`. Subject-specific overrides are accepted by the
collection script and recorded in its manifest. See
`scripts/build_mprage_nifti_collection.py --help` for all options.

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

Here `<branch>` is `fista_r0` or `optimal_wavelet`, and `<case>` is one of the
four case names in the table above.

## GRE workflow

GRE mirrors the same normal, retrospective, and collection stages. The main
differences are multi-echo validation, a LIN-only low-resolution case, the
branch name `selected_wavelet`, and the absence of head masking.

The adapter imports the reviewed upstream calibration implementation from the
read-only `external/wave-gre-flow-comp` submodule at commit `d3772bd`. Logical
`(RO, LIN, PAR)` corresponds to `(readout, phase, slice)`. TWIX and sequence
must describe matrix `250 x 250 x 72`, nominal FOV `220 x 220 x 180 mm`, and
Wave grid `1000 x 250 x 72`. Any positive echo count is accepted when Eco
counters are consecutive from zero, ordered TE values agree, and every echo
has the same complete residue-2 R3x1 Cartesian lattice.

One integrated-refscan `a/b/c` calibration solution is shared by every echo;
later echoes are never independently refit. Each echo retains its own
sequence-derived trajectory and receives its own calibrated PSF. Native CSMs
are estimated once and shared. Automatic `sine-line` is the default; the same
manual kx-bound override and explicit `smooth` fallback described for MPRAGE
are available.

### 1. Normal reconstruction

```bash
scripts/sample_gre_normal_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

Omit `-g` for CPU BART Wave. Every echo is reconstructed independently in both
`fista_r0` (`-w -f -r 0`) and `selected_wavelet` (`-w -f -r 0.015`) branches.
The shared Wavelet value comes from `wavelet_shared_echo_selection.json`, with
SHA-256
`0c43a9d31672e90ad851decfca66c253c362cbd67ca5ba97c4fd8ef1f5a61afd`.
There is no joint-echo or inferred LLR reconstruction.

The converter restores BART output using
`amplitude = kspace_norm * sqrt(extended_RO * LIN * PAR)` and
`phase = 1j * (-1)**(LIN//2)`. Quantitative complex arrays are stored
separately from display-normalized magnitude and wrapped-phase NIfTIs. GRE uses
the orientation-sweep-validated flips `(False, True, False)` followed only by
axis permutation/flips to store canonical RAS, without interpolation.

### 2. Retrospective R3x2 and LIN-low-resolution reconstruction

Use the same three inputs and output root:

```bash
scripts/sample_gre_retro_lr_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

This creates two retrospective cases, each with `fista_r0` and
`selected_wavelet` branches and the same shared `0.015` lambda for every echo:

| Case | Matrix | Construction |
| --- | --- | --- |
| `native_r3x2` | `250 x 250 x 72` | native measured data with an R3x2 Cartesian mask |
| `lin_low_resolution_r3x2` | `250 x 148 x 72` | exact centered LIN crop `[51:199]` followed by the R3x2 mask |

All retrospective k-space comes from direct measured-Wave cropping and pure
Cartesian masking, never from no-Wave forward simulation. The LIN-low CSM is
derived from the accepted native map by centered Fourier PE resampling at
unchanged FOV followed by coil-RSS normalization; readout maps are not resized.

### 3. NIfTI collection

GRE needs no head mask. Collect canonical magnitude and wrapped-phase NIfTIs
from the same output root:

```bash
scripts/sample_gre_nifti_collection.sh \
    /path/to/output_root \
    --require-retro
```

Omit `--require-retro` to collect normal outputs plus any complete
retrospective geometries already present. Every included geometry must contain
both reconstruction branches and a magnitude/phase NIfTI and JSON pair for
every echo. Before copying, the script validates conversion manifests, echo
times, canonical RAS geometry, shared-Wavelet provenance, and echo-specific
BART command records.

The collection is written to `OUTPUT_ROOT/nifti_collection`. NIfTIs and JSON
sidecars are copied byte-for-byte and recorded by SHA-256, along with each
branch conversion manifest and a top-level `manifest.json`. An existing
collection is refreshed only when its builder and all owned-file hashes match.
It creates no mask, masked derivative, synthetic evaluation output, or copy of
the quantitative complex `.npy` arrays.

```text
OUTPUT_ROOT/nifti_collection/
├── original_nifti/
│   ├── fista_r0/
│   │   ├── normal/
│   │   └── retro/<case>/
│   └── selected_wavelet/
│       ├── normal/
│       └── retro/<case>/
└── manifest.json
```

GRE code and unit contracts are complete, but GRE output should not be
described as real-data validated until every echo and both branches have been
visually reviewed.

## Environment

Follow [`SETUP.md`](SETUP.md) for the recommended standard-venv installation,
continued-work reactivation, optional uv usage, CPU-only or CUDA-enabled BART
compilation, and runtime validation. The sample scripts resolve `python` and
`bart` from `PATH`; complete that setup before running the commands above.

On `macha`, activate `cuda133py312-macha` and select the compatible BART build:

```bash
source ~/cluster/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source ~/cluster/bart/bart_startup.sh
```

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
- `wave_retro_lr/gre_nifti_collection.py`: strict byte-identical GRE magnitude
  and phase collection with no masking or quantitative-complex duplication;
- `wave_retro_lr/core.py`: geometry, grids, FFT, masks, and compatibility
  primitives.

`wave_retro_lr/pipeline.py` and `scripts/run_retro_lr.py` temporarily preserve
the old config-driven no-Wave interface used by the synthetic tool. They are
not measured-data MPRAGE entry points and remain only until the synthetic
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
