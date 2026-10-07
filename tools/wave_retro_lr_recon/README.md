# Wave retrospective reconstruction

This tool prepares measured Wave data for explicit BART reconstruction. It
supports the validated sagittal integrated Wave-MPRAGE workflow and a
transverse single- or multi-echo Wave-GRE workflow. Both workflows use the
same three stages:

1. normal native R3x1 reconstruction;
2. retrospective R3x2, low-resolution, and native R3x3 reconstruction; and
3. a separate NIfTI collection step.

The user-facing inputs are a Wave-encoded Siemens TWIX file and its matching
Pulseq sequence. Python validates and prepares BART inputs. The sample Bash
scripts show every `bart ecalib` and `bart wave` command explicitly; the Python
preparation and conversion modules never launch BART.

The sections below describe the fully validated MPRAGE workflow first. The GRE
section then summarizes the corresponding three stages and highlights only the
differences.

## Current standard-PCA contract

All new non-ROVir runs default to 24 virtual coils and accept an explicit
`--virtual-coils N`. A run with count `N` is isolated below
`OUTPUT_ROOT/vccN/{normal,retro}`; this includes an explicit Ncc=12 run. The
historical root-level `normal/` and `retro/` trees are legacy VCC=12 products
and are never overwritten, moved, or relabeled. Preparation rejects Ncc above
the measured physical receive-coil count and records retained energy, PCA-basis
identity, source and geometry provenance, and prepared-artifact hashes.

The four main launchers accept mutually exclusive profiles: `--reg-full`
(FISTA-r0 plus Wavelet), `--wavelet-only`, and `--fista-only`. Normal defaults
to Wavelet-only; retrospective reconstruction defaults to FISTA-only. The
reviewed MPRAGE Ncc=24 native-R3x1 sweep selected `lambda=3e-2`, so default
VCC24 normal reconstruction uses `wavelet_selected_vcc24`. Explicit lambda
overrides use `wavelet_candidate`. The reviewed two-echo GRE Ncc=24 native-R3x1
sweep independently retained the shared `lambda=1.5e-2`, so its default VCC24
normal reconstruction also uses `wavelet_selected_vcc24`. Counts other than 24
and unswept retrospective cases retain their Ncc=12 values under
`wavelet_transferred_vcc12`. MPRAGE R1 rejects profiles containing Wavelet
because no positive R1 value has been approved.

The default MPRAGE retrospective batch is `native_r3x2`,
`lr_y_1p5mm_r3x2`, and `native_r3x3`. LR-X and LR-XY remain explicit options
but are not prepared by default. GRE keeps `native_r3x2`,
`lin_low_resolution_r3x2`, and `native_r3x3`.

There is exactly one `OUTPUT_ROOT/nifti_collection/`. It places complete
count-specific products below `nifti_collection/vccN/` and canonical MPRAGE
ROVir below `nifti_collection/rovir/`. Variants are additive, and normal and
retro may expose different branches. Every new `vccN` collection entry is
bound to preparation and branch-run hashes, basis, Ncc, source, geometry,
method, lambda, and copied-file hashes.

The provenance of the MPRAGE Wavelet defaults, including the synthetic
pure-mask coarse-to-fine sweep, manual review gates, presentation artifacts,
and the boundary between parameter selection and measured reconstruction, is
documented in
[`synthetic_mprage_regularization_pipeline.md`](../synthetic_wave_for_reg_baseline/docs/synthetic_mprage_regularization_pipeline.md).

The completed synthetic two-echo GRE native-R3x3 coarse-to-fine sweep,
shared-lambda decision, presentation contract, and measured-reconstruction
handoff boundary are documented in
[`synthetic_gre_regularization_pipeline.md`](../synthetic_wave_for_reg_baseline/docs/synthetic_gre_regularization_pipeline.md).

## MPRAGE workflow

For sagittal MPRAGE, logical `(RO, LIN, PAR)` corresponds to physical
`(Z, Y, X)`. Readout and physical-Z resolution are never cropped. TWIX MDH
coordinates must describe either duplicate-free fully sampled R1 data or a
complete regular factor-three logical-LIN lattice for every PAR partition
(`R3x1`). Ambiguous, duplicated, incomplete, or out-of-range sampling is
rejected. A valid R3x1 image lattice may omit the exact logical center when it
is present in the separate integrated ACS; the measured LIN residue is always
preserved. The sequence trajectory must contain both Wave axes.

The complete dual-branch normal, retrospective, NIfTI-conversion, and optional
shared-head-mask collection workflow passed representative real
measured-MPRAGE visual validation on 2026-09-01.

Coil calibration removes readout oversampling from integrated set-4 ACS by a
centered full-readout IFFT, central nominal-FOV image crop, and centered FFT.
Direct readout k-space striding is forbidden because it aliases extended-FOV
signal into the head. The Wave image k-space and PSF remain on the oversampled
readout grid required by the forward model. Normal manifests record this
versioned calibration contract; older stride-derived normal inputs are not
eligible for exact or compatibility reuse.

### 1. Normal reconstruction

Choose a new output root, then run:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

The default MPRAGE normal run performs one BART `ecalib` with crop `0.6` and
uses the Wavelet-only profile for R3x1. At the default Ncc=24 it reconstructs
`wavelet_selected_vcc24` with the reviewed `lambda=3e-2`. Use `--reg-full` to
add the unregularized `fista_r0` control or `--fista-only` for that control
alone. R1 requires `--fista-only`. The crop, Ncc, and R3 Wavelet lambda can be
overridden explicitly:

```bash
scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    --virtual-coils 24 \
    --reg-full \
    --ecalib-crop 0.55 \
    --r3-lambda 1.8e-2
```

An explicit `--r3-lambda` is stored under `wavelet_candidate` rather than the
default selected or transferred branch. Run one candidate per approved output
root; the software records it but does not rank or select a winner.

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
diagnostics under `OUTPUT_ROOT/vccN/normal`; see
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

Each new preparation writes these diagnostics under `OUTPUT_ROOT/vccN/normal`:

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
`vccN/normal/bart_inputs` and `vccN/normal/bart_output`; reconstructed arrays
and NIfTIs are separated into `vccN/normal/{bart_output,nifti}/<branch>`.

### 2. Retrospective R3x2 and low-resolution reconstruction

Use the same TWIX, output root, and sequence as the normal command:

```bash
scripts/sample_mprage_retro_lr_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq
```

The script reuses compatible normal inputs and native CSMs in the selected
`vccN` tree. If absent, it prepares them and runs ecalib once. The default
FISTA-only batch contains native R3x2, LR-Y R3x2, and native R3x3. The table
also lists explicit compatibility cases and transferred VCC=12 Wavelet values
available through `--wavelet-only` or `--reg-full`:

| Case | Requested physical XYZ resolution | FISTA control | Transferred Wavelet |
| --- | --- | --- | --- |
| `native_r3x2` | source resolution | `-w -f -r 0` | `-w -f -r 3.5e-2` |
| `lr_x_1p5mm_r3x2` | `1.5 x 1.0 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| `lr_y_1p5mm_r3x2` | `1.0 x 1.5 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.5e-2` |
| `lr_xy_1p25mm_r3x2` | `1.25 x 1.25 x source-Z` mm | `-w -f -r 0` | `-w -f -r 2.2e-2` |
| `native_r3x3` | source resolution | `-w -f -r 0` | `-w -f -r 4.5e-2` |

These values come from the corrected pure-image-lattice synthetic rerun and
explicit visual/metric review. The selection remains hash-bound in its source
manifest; historical ACS-union selections are not carried forward.

Measured-Wave LR k-space is created by direct centered LIN/PAR cropping,
preserving the measured LIN residue and selecting factor two on PAR. It does
not interpolate, forward-simulate, or infer ACS rows. LR PSFs are evaluated
directly on the target PE grid, never cropped or interpolated. The nearest PE
matrices divisible by four are used, and manifests record the achieved
resolution.

The retrospective script accepts the same `--psf-*` settings as the normal
script. It uses CPU by default; append `-g` to run every Wave branch on GPU.
Outputs are stored beneath
`OUTPUT_ROOT/vccN/retro/<case>/{bart_output,nifti}/<branch>`.

Strict source, coefficient-processing, Ncc, basis, and artifact-hash matching
is mandatory when reusing normal inputs. Historical root-level VCC=12 inputs
are not promoted into a new `vccN` tree. Within an existing count-specific
tree, retrospective preparation reuses its hash-bound PSF and trajectory;
manual fit overrides still require an exact metadata match, and source
mismatches remain fatal.

The older crop-first operation for a no-Wave dataset remains available as
`wave_retro_lr.retrospective.synthesize_wave_from_no_wave_crop`. It is an
explicit `synthetic_wave_for_reg_baseline` utility, not a measured-data mode.

### 2a. Native R3x3 retrospective undersampling

For measured native R1 or regular single-residue R3x1 Wave-MPRAGE data, the
tracked global script below applies the reviewed native R3x3 image lattice:

```bash
scripts/sample_mprage_retro_r3x3_recon.sh \
    /path/to/measured_r1_or_r3x1_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    -g
```

The standard retrospective command above now includes this case by default.
The focused script remains available when only native R3x3 is wanted. It
rejects incompatible accelerated sources. For R1 it explicitly
selects residue 1 on LIN; for R3x1 it inherits the measured LIN residue so the
R3x3 lattice is an exact subset of available samples. PAR uses the
center-aligned residue `(Npar // 2) mod 3`. Consequently, the exact mask count
and hash are geometry- and source-residue-specific and are recorded in each
case manifest. For the reviewed synthetic `256 x 256` grid with residue
`(1, 2)`, the count remains 7,225 coordinates. It reuses the native calibrated
PSF and CSM without changing geometry, keeps
calibration k-space separate, and writes only
`retro/native_r3x3/{bart_inputs,bart_output,nifti}`. The two reconstruction
branches are FISTA-r0 and the explicitly reviewed Wavelet lambda `4.5e-2`.
CPU remains the default and `-g` selects GPU BART.

Automatic `sine-line` PSF coefficient processing is already the default, so
`--psf-coefficient-processing sine-line` does not need to be repeated. When a
legacy normal preparation is accepted through the retrospective compatibility
path, the script deliberately uses the PSF already stored in that root; the
current default does not rewrite or regenerate it. Use a new output root when
a newly calibrated sine-line PSF is required.

The ecalib crop is a separate CSM provenance constraint. If the existing
normal CSM was generated with crop `0.1`, pass the same value to the focused or
combined retrospective command:

```bash
scripts/sample_mprage_retro_r3x3_recon.sh \
    /path/to/measured_r1_or_r3x1_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    --ecalib-crop 0.1
```

The default is `0.6`. An existing CSM must have a matching
`normal/bart_output/ecalib_command.txt`; PSF metadata compatibility never
relaxes this check.

### 3. NIfTI collection

After normal reconstruction and any desired retrospective cases, build the
separate presentation collection from the same output root:

```bash
scripts/sample_mprage_nifti_collection.sh \
    /path/to/output_root \
    --require-retro
```

Omit `--require-retro` to collect every currently available normal and
retrospective reconstruction. For new `vccN` trees, `--require-retro` requires
the reduced default cases but accepts any complete branch per case; normal and
retro branch names need not match. Historical root-level layouts remain
discoverable under their legacy requirements. This script never runs k-space
preparation, ecalib, or Wave reconstruction.

The default collection contains only byte-identical canonical files below
`original_nifti`. Add `--head-mask` to explicitly request the historical
whole-head mask and `head_masked_nifti` presentation derivatives. Mask
parameter overrides apply only to that explicit mode.

Discovery is directory-backed rather than case-list-backed. Standard
`nifti/<branch>` and ROVir `rovir/nifti/<branch>` results are retained as
separate collection branches, including when they have the same case,
resolution, acceleration, and reconstruction method. Normal collection builds
never delete or modify source reconstruction trees.

Rerunning the builder validates the existing tool-owned collection and its
hashes, then atomically synchronizes it with the source tree. A newly available
ROVir result is appended without removing its standard collection entry. Any
source disappearance remains a hard error rather than silently removing an
entry. The legacy `synchronization.rovir_replacements` field remains present
but is always empty under this additive policy.

MPRAGE reconstruction and presentation masking remain separate. The collection
always copies canonical NIfTIs byte-for-byte. Whole-head-masked derivatives are
created only when `--head-mask` is supplied and never modify scientific source
files under `normal/nifti` and `retro/<case>/nifti`.

When explicitly requested, each collection variant estimates its mask from its own normal magnitude;
canonical ROVir and every `vccN` subtree remain independent. Within a variant,
Wavelet is preferred over FISTA when present, followed by the first other
normal branch. The mask is applied identically to that variant's selected
branches. The mask uses a high-confidence
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
├── normal/rovir/nifti/<branch>/          # collected alongside standard output
├── retro/<case>/nifti/<branch>/          # canonical, unmasked source
├── retro/<case>/rovir/nifti/<branch>/    # collected alongside standard output
└── nifti_collection/
    ├── original_nifti/
    │   └── <branch>/
    │       ├── normal/
    │       └── retro/<case>/
    ├── head_masked_nifti/                 # only with --head-mask
    │   └── <branch>/
    │       ├── normal/
    │       └── retro/<case>/
    ├── masks/                              # only with --head-mask
    └── manifest.json
```

Here `<branch>` and `<case>` are discovered from the populated source tree;
standard examples include `fista_r0`, `wavelet_selected_vcc24`,
`wavelet_transferred_vcc12`, the reduced default cases, and explicitly
requested legacy-compatible LR cases.

For an explicitly finalized archive, source NIfTI/JSON pairs may be removed
only after their collection copies pass exact SHA-256 verification:

```bash
python scripts/archive_collection_source_niftis.py /path/to/output_root --dry-run
python scripts/archive_collection_source_niftis.py /path/to/output_root
```

The archival command preserves collection copies, GRE conversion manifests,
and quantitative complex arrays. It writes `source_nifti_archival.json` and
updates the collection manifest with the removed paths, hashes, byte counts,
and the fact that branch-level reconstruction resumability is intentionally
disabled. It must therefore be used only after reconstruction and collection
are final.

Existing MPRAGE collections with historical mask products can be migrated in
place before archival:

```bash
python scripts/prune_mprage_collection_head_masks.py /path/to/output_root
```

Use `--validate-hashes` to reread and verify every collection payload. For a
large reviewed batch, `archive_collection_source_niftis.py` also accepts
`--trust-manifest-hashes`; that mode still requires an exact owned-file set,
matching recorded source/copy hashes, and both files to be present, but should
only be used after representative collections pass full SHA-256 verification.

### Optional MPRAGE troubleshooting features

Higher-channel PCA controls and ROVir shoulder-wrap suppression are optional
diagnostic/recovery tools, not stages of the standard workflow. See
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for PCA controls and failure
diagnosis, and
[`mprage_rovir_reconstruction.md`](docs/mprage_rovir_reconstruction.md) for
the user-facing ROVir pipeline.

ROVir is currently implemented only for MPRAGE. It is not exposed for GRE;
the GRE acquisition FOV normally avoids the extended-body shoulder-wrap
failure mode that motivated this feature.

For R3x1 source data, the normal ROVir workflow retains both FISTA-r0 and the
standard-selection Wavelet `lambda=3.5e-2` comparison. The latter is labeled as
reused rather than ROVir-optimized. The Wavelet block is directly visible in
the main ROVir launcher and is resumable with the rest of that workflow. It
reuses the canonical ROVir transform, CSM, PSF, FISTA output, and manifest.

## GRE workflow

GRE mirrors the same normal, retrospective, and collection stages. The main
differences are multi-echo validation, a LIN-only low-resolution case, and the
absence of head masking.

The adapter imports the reviewed upstream calibration implementation from the
pinned read-only `external/wave-gre-flow-comp` submodule. Logical
`(RO, LIN, PAR)` corresponds to `(readout, phase, slice)`. Readout and slice
remain fixed at `250 x 72`, with readout FOV `220 mm`, slice FOV `180 mm`, and
fourfold Wave readout oversampling. The sequence defines the LIN matrix and
phase FOV, which must match TWIX. Supported examples include the adult
`250 x 250 x 72`, `220 x 220 x 180 mm` grid and the pediatric
`250 x 196 x 72`, `220 x 172 x 180 mm` grid. Any positive echo count is
accepted when Eco counters are consecutive from zero, ordered TE values agree,
and every echo has the same complete residue-2 R3x1 Cartesian lattice.

One integrated-refscan `a/b/c` calibration solution is shared by every echo;
later echoes are never independently refit. Each echo retains its own
sequence-derived trajectory and receives its own calibrated PSF. Native CSMs
are estimated once and shared. Automatic `sine-line` is the default; the same
manual kx-bound override and explicit `smooth` fallback described for MPRAGE
are available.

GRE coil calibration removes readout oversampling from integrated set-4 ACS
with a centered IFFT, central nominal-FOV image crop, and centered FFT before
PCA compression or BART `ecalib`. Direct readout k-space striding is forbidden
because it aliases outside-FOV signal into the head. Wave image k-space and
echo-specific PSFs retain the extended readout required by the forward model.
Normal manifests record the versioned crop and exact readout geometry.

Normal GRE preparation writes the same two shared-coefficient diagnostics as
MPRAGE under `OUTPUT_ROOT/vccN/normal`: the fixed `[-2*pi, 2*pi]`
`PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png` and the independently autoscaled
`PSF_COEFFICIENTS_FULL_RANGE.png`. Retrospective preparation reuses this normal
calibration and backfills both plots when compatible prepared inputs are
reused; it does not generate redundant per-case copies.

Retrospective entry points also support a narrow legacy-PSF-metadata
compatibility path. It requires exact TWIX and sequence identities, corrected
alias-free coil-calibration provenance, and validates the measured
R3x1 mask, native geometry, calibration grid, and every echo k-space/PSF grid,
and requires finite PSFs. It never refits coefficients or regenerates PSFs,
never rewrites a legacy normal manifest, and records the decision in
`normal/NORMAL_INPUT_REUSE_ATTESTATION.json`. Manual kx bounds remain strict.

### 1. Normal reconstruction

```bash
scripts/sample_gre_normal_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

Omit `-g` for CPU BART Wave. At the default Ncc=24, every echo is reconstructed
independently only in `wavelet_selected_vcc24` at the reviewed shared lambda
`0.015`. Use `--reg-full` to add `fista_r0` (`-w -f -r 0`) or
`--fista-only` for the control alone. Counts other than 24 retain the same
numeric value as an explicitly labeled `wavelet_transferred_vcc12` branch.
There is no joint-echo or inferred LLR reconstruction.

An explicit `--wavelet-lambda VALUE` override is applied identically to every
echo and written under `wavelet_candidate`; it does not replace or relabel the
reviewed branch. The Ncc=24 decision used magnitude, phase, inter-echo scaling,
and delta-B0 metrics plus fixed-window magnitude review. The shared-lambda
evaluation-manifest and CSV SHA-256 values are
`fecbac4b66c6ba3b6d8723e4e47cc67bb2e40ead6bf88a9cc509914c111544f6`
and `bbcd3dc2245d4a309b4550bb60b099294b7ba350525ecbef66d089c9642ab656`.

The converter restores BART output using
`amplitude = kspace_norm * sqrt(extended_RO * LIN * PAR)` and
`phase = 1j * (-1)**(LIN//2)`. Quantitative complex arrays are stored
separately from display-normalized magnitude and wrapped-phase NIfTIs. GRE uses
the orientation-sweep-validated flips `(False, True, False)` followed only by
axis permutation/flips to store canonical RAS, without interpolation.

### 2. Retrospective R3x2, LIN-low-resolution, and native-R3x3 reconstruction

Use the same three inputs and output root:

```bash
scripts/sample_gre_retro_lr_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

This creates three retrospective cases using FISTA-r0 by default. If
`--wavelet-only` or `--reg-full` is requested, every echo uses the same shared
transferred `0.015` value:

| Case | Matrix | Construction |
| --- | --- | --- |
| `native_r3x2` | `250 x sequence-Ny x 72` | native measured data with an R3x2 Cartesian mask |
| `lin_low_resolution_r3x2` | `250 x target-Ny x 72` | centered LIN crop nearest to 1.5 mm and divisible by four, followed by the R3x2 mask |
| `native_r3x3` | `250 x sequence-Ny x 72` | native measured data with a source-derived R3x3 Cartesian mask |

All retrospective k-space comes from direct measured-Wave cropping and pure
Cartesian masking, never from no-Wave forward simulation. The LIN-low CSM is
derived from the accepted native map by centered Fourier PE resampling at
unchanged FOV followed by coil-RSS normalization; readout maps are not resized.
For the adult grid, `target-Ny=148` and the crop is `[51:199]`; for the
`Ny=196`, `172 mm` pediatric grid, `target-Ny=116` and the crop is `[40:156]`.

The native-R3x3 case inherits its LIN residue from the validated measured R3x1
image stream and center-aligns the PAR residue. For the reviewed adult grid the
residue is `(2, 0)`, the exact mask contains 1992 coordinates, and its logical
SHA-256 is
`e57069cd4f3cc9af4a78e70cb10f66b79ad1ceb35efac966c871df3822febefa`.
ACS/refscan calibration remains separate. Each echo preserves its acquired
samples bitwise, is exactly zero outside the R3x3 mask, links its own existing
measured PSF without recalibration, and reuses the native CSM unchanged.

To add only R3x3 to an older completed GRE root, use the focused resumable
entry point:

```bash
scripts/sample_gre_retro_r3x3_recon.sh \
    /path/to/measured_wave_gre.dat \
    /path/to/existing_output_root \
    /path/to/matching_wave_gre.seq \
    -g
```

Completed branches with matching commands, preparation manifests, PCA basis,
CSM, PSFs, k-space, finite outputs, NIfTIs, and hashes are reused. A mismatch
or incomplete recorded result fails instead of replacing accepted data.
The standard retrospective script invokes this focused implementation after
the two established R3x2 cases.

### 3. NIfTI collection

GRE needs no head mask. Collect canonical magnitude and wrapped-phase NIfTIs
from the same output root:

```bash
scripts/sample_gre_nifti_collection.sh \
    /path/to/output_root \
    --require-retro
```

Omit `--require-retro` to collect normal outputs plus any complete
retrospective geometries already present. `--require-retro` remains compatible
with legacy roots and accepts asymmetric complete branches, including the
default VCC24 normal `wavelet_selected_vcc24` plus retrospective `fista_r0`.
Every included branch must contain a magnitude/phase NIfTI and JSON pair for every echo.
Before copying, the script validates conversion manifests, echo
times, canonical RAS geometry, shared-Wavelet provenance, and echo-specific
BART command records.

The collection is written to `OUTPUT_ROOT/nifti_collection`. NIfTIs and JSON
sidecars are copied byte-for-byte and recorded by SHA-256, along with each
branch conversion manifest and a top-level `manifest.json`. An existing
collection is refreshed only when its builder and all owned-file hashes match.
Refreshing performs an atomic source sync: newly completed R3x3 groups are
appended to the rebuilt collection, while a previously collected group that is
no longer discoverable causes a hard failure rather than silent removal.
It creates no mask, masked derivative, synthetic evaluation output, or copy of
the quantitative complex `.npy` arrays.

```text
OUTPUT_ROOT/nifti_collection/
├── vccN/
│   ├── original_nifti/<branch>/
│   │   ├── normal/
│   │   └── retro/<case>/
│   └── manifest.json
└── manifest.json
```

GRE code and unit contracts are complete, but GRE output should not be
described as real-data validated until every echo and requested branch has
been visually reviewed.

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
  calibration, legacy compatibility, native-R3x3 preparation, direct
  retrospective crop, CSM, command, and normalization contracts;
- `wave_retro_lr/bart_io.py`: bounded BART CFL I/O, logical hashing, and
  split-complex output recombination;
- `wave_retro_lr/nifti_collection.py`: byte-identical canonical collection,
  optional normal-derived whole-head mask, physical-grid mask mapping, and
  provenance;
- `wave_retro_lr/gre_nifti_collection.py`: strict byte-identical GRE magnitude
  and phase collection with no masking or quantitative-complex duplication;
- `wave_retro_lr/collection_archive.py`: explicit mask-product pruning and
  hash-bound source-NIfTI archival after collection finalization;
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
