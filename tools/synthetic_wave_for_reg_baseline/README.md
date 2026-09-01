# Synthetic Wave regularization baseline

This directory contains the R3 presentation-optimization, R1 parameter-
refinement, and cross-dataset transfer workflow for synthetic Wave-MPRAGE
reconstruction.

Start with:

- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for the active scientific plan;
- [`HANDOVER.md`](HANDOVER.md) for exact cross-session state;
- [`docs/r1_dataset_processing_todo.md`](docs/r1_dataset_processing_todo.md)
  for the measured 2026-08-21 R1 execution checklist; and
- [`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`](docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md)
  only when historical R3x1 development detail is needed.

## Layout

```text
configs/       Portable dataset-contract examples
scripts/       Reconstruction programs and reusable algorithm/I/O modules
requirements/  Incremental Python dependency sets grouped by task
docs/          Maintenance indexes, including the old-to-new filename map
tests/         Unit and reference-oracle tests
```

Machine-local dataset notes and generated reconstruction artifacts are ignored
by git. Large artifacts should live in dataset output trees rather than at this
directory's root; paths and scan filenames remain configuration/CLI inputs.
Tracked launchers contain no machine-specific absolute paths. For transfer
runs, copy the tracked `.example.sh` launcher to the ignored `.local.sh` name
and put private paths only in that local copy.

## Corrected pure-mask rerun

The corrected rerun is isolated from the frozen R1/R3 output trees. Copy
`configs/pure_mask_regularization_rerun.example.json` to the ignored
`configs/pure_mask_regularization_rerun.local.json` name and replace every
placeholder path, hash, and explicit provenance assertion. The route has
separate review-gated entry points:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/prepare_pure_mask_rerun.py --help
python tools/synthetic_wave_for_reg_baseline/scripts/run_pure_mask_sweeps.py --help
python tools/synthetic_wave_for_reg_baseline/scripts/evaluate_pure_mask_sweeps.py --help
python tools/synthetic_wave_for_reg_baseline/scripts/render_pure_mask_shortlist.py --help
python tools/synthetic_wave_for_reg_baseline/scripts/record_pure_mask_selections.py --help
```

For server execution, put the confirmed paths in the ignored
`scripts/run_pure_mask_rerun.local.sh`. From one user-created tmux session,
invoke exactly one review-gated operation at a time:

```bash
RUNNER=tools/synthetic_wave_for_reg_baseline/scripts/run_pure_mask_rerun.local.sh
$RUNNER validate-sources
$RUNNER materialize-sources
$RUNNER validate-inputs
```

The runner does not create or manage tmux sessions and never chains
preparation, reconstruction, evaluation, or review actions.

The accepted full no-Wave and full synthetic-Wave NPY files were intentionally
deleted as rebuildable intermediates during the frozen-output cleanup. The
`validate-sources` action validates the retained raw-data, coil-basis,
theoretical-PSF, source-report, and cleanup-inventory bindings without writing.
After review, `materialize-sources` reproduces both arrays inside the new run
tree and requires their file hashes to exactly match the archived accepted
hashes before preparation can continue.

Run `--validate-only` first. It checks the accepted full no-Wave/full-Wave
sources, approved BET mask, case-matched CSMs and theoretical PSFs, exact FOV
and dimensions, finite values, CSM RSS normalization, PSF unit magnitude,
artifact hashes, and named provenance assertions without writing outputs.
Preparation and every later stage additionally require
`--confirm-output-root` to exactly equal the reviewed local-config root.

The five cases are native R3x1, native R3x2, LR-X R3x2, LR-Y R3x2, and LR-XY
R3x2. Every mask is `pure_cartesian_image_lattice`; calibration is absent from
the Wave reconstruction input. The 2026-08-21 synthetic route reuses the
accepted theoretical sequence PSF because its Wave data are synthesized from
no-Wave data. Refscan calibration applies to measured-Wave acquisitions and is
never unioned with image k-space in either route.
Preparation creates resolution-matched
direct-FFT references, reuses the accepted case CSM/PSF pairs without ecalib
or PSF calibration, and verifies bitwise acquired-sample equality plus exact
zeros outside the mask. The coarse runner creates one FISTA lambda-zero
control per case, the approved Wavelet grid, and corrected split-complex LLR
grids for blocks 4, 8, and 16. Fine settings must be explicitly listed after
coarse evaluation. Evaluation reads only sweep manifests, reports separate
metric leaders, and never selects a composite or winner automatically.
Orthogonal review figures apply the same per-candidate, BET-restricted LSQ
intensity alignment used by the metrics and then use one direct-FFT-derived
display window. Each Wavelet and LLR-block family also receives separate
metric-versus-lambda curves. LR curves show native-grid and matched-1-mm
results as distinct series; native cases show only the native grid. FISTA
lambda zero is a horizontal control rather than a point on the logarithmic
lambda axis.

If an evaluation from the earlier display derivation already exists, refresh
only its manifest-owned CSV, masks, and figures with the explicit dispatcher
action:

```bash
$RUNNER refresh-coarse-evaluation
```

The refresh first verifies the existing sweep binding, recorded hashes, and
absence of unowned files. It does not run BART or alter any reconstruction.
After final manual selection, `build_pure_mask_presentation.py` can export a
manifested package containing the five FISTA controls and five approved
Wavelet magnitudes as canonical-RAS NIfTIs, three center-slice TIFFs per
reconstruction, and the corresponding native/matched metric rows. The NIfTIs
retain raw reconstruction intensity; TIFF comparisons reuse evaluation LSQ
scales and one resolution-matched direct-FFT window per case.
After reviewing those metrics, edit only the local `fine_sweep` and
`manual_shortlist` sections. Those decision-only fields are deliberately
excluded from the immutable preparation-contract hash. The shortlist renderer
accepts only manifest-listed candidates, and the final recorder requires an
explicit visual-review acknowledgement plus one manual selection per case.

The corrected rerun is complete. Explicit manual visual/metric review selected
Wavelet `3.5e-2` for native R3x1 and native R3x2, `2.5e-2` for LR-X and LR-Y
R3x2, and `2.2e-2` for LR-XY R3x2. Its selection manifest SHA-256 is
`07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`.
The measured-data MPRAGE tool uses these as the positive-Wavelet ablation arm
while retaining one FISTA-r0 control per case; it does not transfer them to
R1 normal reconstruction or GRE.

## Current experiment

The current execution target is the fully sampled 2026-08-21 MID00198 R1
dataset and its retrospective synthetic R3x2 Wave reconstruction. It uses
image-derived 64-to-12 coil compression and direct source assembly, with no
GRAPPA or SENSE interpolation. DICOM ranking is disabled because Prescan
Normalize was enabled. The `ecalib -I` pilot remains a negative result.

Wave NIfTI export now stores magnitude and phase directly in canonical RAS
using the product-DICOM-validated affine-axis convention. Downstream manual
signed-axis correction is no longer required for newly exported results.

Retrospective low-resolution code support is integrated under
`tools/wave_retro_lr_recon/`. It changes phase encoding only, rounds each PE
matrix to the nearest multiple of four, reconstructs with GPU `bart wave -g`,
and exports through the same canonical-RAS path. For the historical product
study, copy
`requirements/retrospective_low_resolution_product.example.json` to the
ignored `.local.json` name, fill in private paths, and run the tmux-friendly
`scripts/run_retrospective_low_resolution.sh` launcher. No large retrospective
outputs are created by validation alone.

For the final R1 study, use
`requirements/retrospective_low_resolution_r1.example.json` and
`scripts/run_r1_retrospective_low_resolution.example.sh` to create their
ignored local copies. The R1 runner validates the frozen Wavelet
`lambda=1.5e-2` selection before delegating to the same reconstruction tool.
After a non-writing `--validate-only` preflight, run in tmux with:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_r1_retrospective_low_resolution.local.sh \
    --resume
```

For the follow-up retrospective-resolution Wavelet sweep, copy
`requirements/retrospective_low_resolution_wavelet_sweep.example.json` and
`scripts/run_r1_retrospective_wavelet_sweep.example.sh` to their ignored
`.local.*` names. That reconstruction launcher calls the existing
retrospective pipeline for each new lambda and reuses declared completed
lambda-zero and `1.5e-2` controls.

Evaluate the completed sweep separately with the original retrospective
matched-grid calculation:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_r1_retrospective_wavelet_sweep_matched_grid_evaluation.local.sh
```

This evaluation-only launcher linearly resamples every candidate to the
original 1 mm RAS grid and reuses the exact hash-bound full-resolution
FISTA-lambda-zero/direct-FFT references and fixed BET brain/edge masks from the
accepted retrospective analysis. It does not call reconstruction or create a
native-grid reference. The evaluator reports separate per-metric leaders for
each resolution and does not collapse them into one automatic selection.

The retrospective corrected-LLR follow-up has separate reconstruction and
evaluation launchers. The default configurable grid tests block 4 at
`2e-3, 5e-3, 6e-3, 1e-2` and blocks 8/16 at
`2e-3, 5e-3, 1e-2, 2e-2`, for 36 reconstructions across three geometries.
Every reconstruction calls the same retrospective pipeline with BART
`wave -l -v -b <block> -f -r <lambda> -g` and saves magnitude and phase.

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_r1_retrospective_llr_sweep.local.sh
```

After that sweep completes, run the independent matched-grid evaluator:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_r1_retrospective_llr_sweep_matched_grid_evaluation.local.sh
```

The evaluator never calls reconstruction. It uses the same original 1 mm
references and fixed BET brain/edge masks as the Wavelet evaluation, includes
the completed retrospective FISTA-lambda-zero cases as controls, and produces
one metric CSV plus separate block-4, block-8, and block-16 plots.

MID00198 has passed manifest-backed qualification, source preparation,
synthetic-Wave reconstruction, refinement, and fixed-reference evaluation.
Wavelet `lambda=1.5e-2` is frozen and its qualitative R3 transfer is approved.
The older 2021 R1 scans remain partial-readout fallbacks only.

New acquisitions use one portable dataset manifest for input/output paths,
logical geometry, acquired and synthetic-Wave sampling, reconstruction
settings, and evaluation-reference policy. See
[`docs/dataset_manifest.md`](docs/dataset_manifest.md) and copy
`configs/incoming_r1_dataset.example.json`. The contract enforces GPU BART and
metrics-only mask use. Its inspector integration records a hashed, fully
resolved contract snapshot and checks declared acquisition expectations against
TWIX/DICOM metadata, including raw/post-OS readout sizes and MDH center column.
Coil compression, direct fully sampled R1 source assembly, the existing R3x1
GRAPPA source path, full Wave synthesis, and target BART input export now
consume that passed contract. Coil compression can
explicitly use either the image or refscan stream, so R1 does not depend on a
PAT refscan being present. The source sampling gates are mutually exclusive,
preventing either interpolation of R1 or direct copying of accelerated R3 data.
The target mask is applied only after full Wave encoding and written to a
separate output tree with sample-by-sample validation. The remaining
consumer-by-consumer work is tracked in
[`docs/dataset_portability_audit.md`](docs/dataset_portability_audit.md).

For the current dataset, use the two-mode tmux wrapper. Preparation stops at
the full-Wave visual gate; reconstruction requires explicit approval:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_synthetic_wave_dataset.sh prepare

tools/synthetic_wave_for_reg_baseline/scripts/run_synthetic_wave_dataset.sh \
    reconstruct \
    "$R1_SYNTHETIC_WAVE_ROOT/dataset_manifest.json" \
    --confirm-full-wave-reviewed
```

To create a second synthetic-Wave R3x1 target from that accepted full-Wave
encoding, use the four matching `.example.sh`/ignored `.local.sh` launchers.
The operations are deliberately separated so the lambda-zero image and maps
are visually assessed before spending GPU time on the sweep:

```bash
scripts/prepare_synthetic_wave_r3x1_target.local.sh
scripts/run_synthetic_wave_r3x1_lambda0.local.sh
# Review the central-slice image and ESPIRiT-map montages here.
scripts/run_synthetic_wave_r3x1_regularization_sweep.local.sh
scripts/evaluate_synthetic_wave_r3x1_regularization.local.sh
```

Preparation branches only the post-Wave mask (`R3x1`, residue `[1,0]`, the
same full-PE2 24-line ACS); it does not rerun full-Wave synthesis. Lambda zero
uses the existing BART CG acceptance runner. The sweep uses the existing GPU
BART Wavelet/corrected-LLR case runner and first gates split-complex LLR lambda
zero. Evaluation is a separate exact-grid calculation against the approved
direct-FFT RSS and fixed metrics-only BET mask, with no registration,
interpolation, or automatic parameter choice.

The completed R3x1 coarse evaluation places the Wavelet NRMSE/SSIM optimum at
`2e-2`, with the next upper sample only at `5e-2`. Refine that interval without
rerunning the existing controls using:

```bash
scripts/run_synthetic_wave_r3x1_wavelet_refinement.local.sh
scripts/evaluate_synthetic_wave_r3x1_wavelet_refinement.local.sh
```

The first launcher adds only `1.6e-2, 1.8e-2, 2.2e-2, 2.5e-2, 3e-2, 4e-2`.
The second combines those cases with the original sweep and remains pinned to
the same approved direct-FFT/BET package.

The refinement is complete. The explicit user choice is Wavelet
`lambda=2.2e-2`, which has the lowest refined brain NRMSE while remaining
effectively tied for the best 3D SSIM. `record_regularization_selection.py`
binds that decision to the metric table, selected reconstruction, and
solver-matched FISTA lambda-zero control; it records the decision as explicit,
not automatic.

After approving the crop-`0.6` ESPIRiT maps and lambda zero, the resumable
GPU-FISTA Wavelet sweep is:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_wavelet_sweep.sh \
    --confirm-crop-0p6-reviewed
```

It runs solver-matched lambda zero and the five frozen positive lambdas, then
writes a DICOM-free, common-window review under the sweep's `review` directory.

For the compact block-8 LLR presentation sweep, use:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_llr_sweep.sh \
    --confirm-crop-0p6-reviewed
```

It first gates split-complex LLR lambda zero against native-complex FISTA
lambda zero, then runs `2e-5`, `1e-4`, and `5e-4`. All reconstructions use
GPU BART and the same approved crop-`0.6` maps. The output includes a
DICOM-free common-window review and a manifest for each case.

`scripts/prepare_r1_reference_comparison.py` audits and converts an exact
matched unfiltered enhanced-DICOM pair, exports direct FFT RSS from fully
sampled no-Wave multicoil k-space, and creates a qualitative side-by-side
review. The review deliberately uses independent positive-voxel p99 display
scaling and performs no registration, BET masking, intensity matching, or
ranking. An unfiltered ACC, Normalize-on series can be added explicitly with
`--acc-normalize-on-dicom`. Pass `--resume` to hash-validate and update or
reuse an already complete output.

After R1 refinement, apply the selected parameter back to R3 without retuning
as a cross-dataset transfer check. Because R3 is used for the preliminary
optimization, it is not an untouched independent validation dataset.

The transfer is qualitative only. Prepare the private launcher with:

```bash
cp tools/synthetic_wave_for_reg_baseline/scripts/run_r3_wavelet_transfer.example.sh \
   tools/synthetic_wave_for_reg_baseline/scripts/run_r3_wavelet_transfer.local.sh
```

Edit only the ignored `.local.sh` copy, then run it in tmux. The tracked generic
runner validates the frozen selection, recalibrates dataset-specific crop-`0.6`
maps, reconstructs solver-matched FISTA lambda zero and Wavelet `1.5e-2` with
GPU `-g`, and creates a common-window qualitative review. It performs no R3
metrics, ranking, or lambda selection.

## Which programs matter

The current production path is:

1. `inspect_product_dataset.py` audits a new product TWIX/DICOM dataset,
   preferably through one `--dataset-manifest` contract.
2. `estimate_coil_compression.py` estimates the Ncc=12 compression basis and
   can derive its dataset state from the passed manifest.
3. The source-sampling branch is explicit:
   `assemble_fully_sampled_no_wave_kspace.py` directly assembles and compresses
   a complete R1 source with no interpolation, while
   `reconstruct_no_wave_grappa_3d.py` uses the accepted joint-coil 5×5×5
   GRAPPA kernel only for a compatible measured R3x1 source. Both derive matrix
   allocation from the manifest and produce the same downstream array layout.
4. `synthesize_wave_kspace.py` applies the theoretical sequence PSF.
5. `validate_full_sampling_wave_operator.py` gates the real source with a
   `PSF=1` no-Wave identity and an all-coil full-sampling Wave inverse check.
6. `export_bart_wave_inputs.py` and `export_bart_wave_target_branch.py` are
   historical ACS-union exporters retained for frozen provenance. They must
   not prepare the corrected pure-mask rerun.
7. `export_bart_calibration_acs.py` exports measured no-wave ACS for one
   reusable BART ESPIRiT calibration. Its manifest route explicitly selects
   direct fully sampled image data or a measured refscan.
8. `run_bart_wave_lambda0.py` runs the unregularized acceptance reconstruction.
9. `run_bart_regularization.py` calls the pinned upstream wrapper for one
   hashed, resumable wavelet or LLR case.
10. `prepare_regularization_evaluation.py` consolidates complete NIfTI pairs,
   selects one exact DICOM series by UID and unfiltered `ND` metadata, and
   records hashes and conversion provenance.
11. `review_regularization_orientation.py` canonicalizes both inputs to RAS,
    audits all signed axis mappings, and produces explicitly labeled L/R QC
    figures without accepting a correction or running registration.
12. `evaluate_regularization_volume.py` requires that recorded approval,
    estimates one proper rigid transform from lambda zero, applies it unchanged
    to every magnitude volume, and writes whole-volume metrics and plots.

The manifest-backed preparation wrapper runs the operator validator immediately
after full Wave synthesis. It verifies all virtual coils, uses no BART
reconstruction or presentation processing, and writes
`evaluation/full_sampling_wave_operator_validation/operator_validation_manifest.json`
below the dataset output root. It must pass before the visual-review and target
mask gates.

The retrospective-resolution implementation is maintained separately in
`tools/wave_retro_lr_recon/`; its CLI consumes an explicit source/config
contract instead of duplicating reconstruction algorithms in this baseline
tool. For the current product dataset, run its wrapper from the repository
root:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_retrospective_low_resolution.sh \
    --resume
```

Use `--validate-only` for a non-writing structural check or `--prepare-only`
to create target BART inputs without starting reconstruction.

After reconstruction, generate the physical-coordinate visual review with:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_retrospective_low_resolution_review.sh
```

`review_retrospective_low_resolution.py` reads the completed batch/case
manifests and creates a native-grid comparison plus an explicitly labeled
matched-1-mm comparison. It enforces common RAS center/FOV geometry and the
corrected Wave canonical-export sidecar contract. The native figure selects
the nearest slice to one shared RAS location and uses physical-mm extents; the
matched figure linearly resamples only for alignment. Per-volume positive
p99.5 scaling is display-only. Neither DICOM nor BET is used, and no
quantitative ranking is performed.

After visual review, generate the descriptive resolution-tradeoff analysis:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_retrospective_low_resolution_analysis.sh
```

`analyze_retrospective_low_resolution.py` maps the approved fixed BET mask into
the untouched reconstruction grids using the frozen shared transform. It
reports native physical-mm sharpness, a fixed smooth-brain
signal/local-residual proxy, and matched-grid NRMSE/SSIM/NCC against both the
same-regularizer full-resolution result and temporary GRAPPA reference.
Background values are separate QC because BART air support is nearly zero.
The manifest explicitly excludes DICOM intensities, true-SNR/CNR claims,
candidate-specific registration, composite ranking, and automatic resolution
selection.

For the current R3 presentation optimization, `run_r3_presentation_optimization.sh` provides a
resumable tmux-friendly orchestration entry point. Its default
`all-before-review` stage runs the focused GPU reconstructions and prepares the
normalized-DICOM, BET-mask, and orientation QC package. It stops before the
visual gate. After reviewing the printed BET and L/R figures, the separate
`all-after-review --confirm-reviewed-mask-and-lr` stage records that explicit
decision, evaluates the fixed-mask sweep, and creates the compact comparison
package. `validate_bart_split_complex.py` records the required lambda-zero
equivalence of native and recombined `wave -l -v` representations.

For the full-head normalized DICOM mask/presentation source, the runner invokes
`prepare_reference_brain_mask.py` with BET robust center estimation and a fixed
fractional threshold of `0.55`. The manifest records both settings.
The same utility accepts `--mask-dilation-voxels` when a metric mask needs a
small, controlled outward margin. It preserves native BET outputs, writes the
expanded mask under the canonical mask filename, overlays both boundaries in
the QC figure, and always leaves visual approval unset.
For MID00198 R1 evaluation, the approved direct FFT RSS reference and approved
`f=0.59` metrics-only mask are bound by hashes in
`evaluation/direct_fft_reference/metrics_reference_manifest.json` under the
dataset output root. Evaluation code should consume that record rather than
selecting among BET candidate directories.

Before metrics, `validate_metrics_geometry.py` hash-validates that reference,
its approved mask, and an explicitly required set of regularization cases. It
requires exact RAS shape, voxel size, and affine equality, verifies GPU BART
provenance, and writes a gate report without registration or interpolation.
`evaluate_direct_fft_regularization.py` then evaluates those exact cases using
the approved fixed mask. It undoes documented NIfTI export scaling, fits one
mask-restricted LSQ scalar, writes CSV/JSON metrics, and creates method-specific
common-window figures. It reports per-metric leaders but deliberately performs
no composite ranking or parameter selection.

The resumable higher-lambda and multi-block launcher is:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_regularization_refinement.sh \
    --confirm-approved-reference-and-mask
```

The launcher is resumable. The 28 missing cases have completed: it adds
Wavelet `2e-3`, `5e-3`, `2e-2`, and `5e-2`;
LLR blocks `4` and `16` use a 1-2-5 grid from `2e-5` through `1e-2`, while
block `8` runs only values absent from its retained coarse sweep. Every case
uses the approved crop-`0.6` maps and GPU BART `-g`.

The retained coarse and new refinement manifests now pass one exact-grid gate
as a 38-case set. The report is
`evaluation/direct_fft_reference/geometry_validation/refined_grid_geometry_validation.json`
under the dataset output root. The canonical combined evaluation is
`evaluation/direct_fft_reference/regularization_refinement_metrics`; it writes
`regularization_metrics.csv`, hash-bound JSON provenance, per-metric leaders,
LLR block-size-by-lambda plots, and a shared direct-FFT display window for each
method/block. It performs no composite ranking or parameter selection.

The small follow-up sweep can be run or resumed in tmux with:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_regularization_targeted_sweep.sh \
    --confirm-approved-reference-and-mask
```

This resumable launcher writes to the separate
`regularization_targeted_ecalib_crop-0p6` reconstruction directory. It adds
Wavelet `1.5e-2`, four block-4 LLR values around `5e-3`, and LLR block-8 and
block-16 boundary extensions at `1.5e-2`, `2e-2`, and `3e-2`. It runs 11 cases
through the same manifest-backed GPU BART path and never replaces accepted
reconstructions.

All 11 follow-up cases are complete. Combined with the retained cases, the
49-case exact-grid report is
`evaluation/direct_fft_reference/geometry_validation/targeted_grid_geometry_validation.json`,
and the current metric package is
`evaluation/direct_fft_reference/regularization_targeted_metrics`. The LLR
heatmap labels deliberately unrun cells in the scientifically ragged grid.
Wavelet `1.5e-2` has the lowest NRMSE, while block-4 LLR peaks near `6e-3` and
blocks 8/16 turn after `1e-2`.

Wavelet `lambda=1.5e-2` is now the frozen MPRAGE choice. Its hash-bound decision
record is
`evaluation/direct_fft_reference/regularization_selection/selection_manifest.json`
under the R1 dataset root. It binds the selected NIfTI and reconstruction
manifest to the approved reference, exact-grid report, and metric package. The
next experiment applies that configuration unchanged to R3; R3 is a transfer
check and must not be used for retuning.

`run_ecalib_intensity_pilot.sh` is the isolated tmux entry point for testing
BART `ecalib -I`. It runs one GPU Wave lambda-zero reconstruction and creates a
brain-median-scaled intensity-profile comparison. It does not run or select
regularization, and it treats no-wave GRAPPA/SENSE only as comparison
references.

The measured ACS calibration is shared from the parent full-Wave BART inputs;
the R3x2 directory supplies the retrospectively masked k-space and linked PSF.

For acceleration comparisons,
`export_bart_wave_inputs_retrospective.py` reuses the validated full synthetic
Wave volume and theoretical PSF, builds an explicit Cartesian PE1×PE2 lattice
plus fully sampled ACS, and exports a separate historical BART input tree. It
does not repeat GRAPPA completion or Wave encoding, but its union mask is not a
valid input for the corrected pure-mask rerun.

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

The DICOM-referenced evaluation tools additionally use:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/evaluation.txt
```

The optional SENSE diagnostic additionally uses:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/sense_diagnostics.txt
```

## Presentation magnitude collection

`build_presentation_nifti_collection.py` materializes a private, manifested
folder of magnitude NIfTIs from accepted outputs. It copies available sources
byte-for-byte, validates finite canonical-RAS storage, preserves native
retrospective-resolution grids, and uses explicit JSON records rather than
fake NIfTIs for pending reconstructions. Display order lives in the manifest;
filenames remain descriptive and stable.

The presentation-only no-Wave R3x1 comparison is prepared with
`run_no_wave_r3x1_pics_sweep.py`. It masks the fully sampled no-Wave source on
the product R3x1 PE1 lattice plus 24-line ACS, runs a GPU `bart pics -g -S`
CG-SENSE control, and runs a compact Wavelet/FISTA sweep with random cycle
spinning disabled. Its lambda values are not assumed to be numerically
interchangeable with the custom `bart wave` operator. Every completed case is
evaluated on the unchanged approved direct-FFT R1 grid and mask; the standard
metric set is saved under `direct_fft_metrics.metrics` in its case manifest.
Fresh cases export both canonical-RAS magnitude and phase NIfTIs. On an older
completed case, `--resume` backfills a missing phase from the hash-validated
complex BART image without rerunning PICS or replacing the accepted magnitude.

`run_no_wave_r3x1_grappa.py` prepares the matching retrospective GRAPPA
comparison from that same fully sampled no-Wave source and R3x1-plus-ACS mask.
It reuses the accepted local joint-coil 5x5x5, Ncc=12 implementation and frozen
regularization `0.01`; it does not define a new GRAPPA algorithm. Calibration
and reconstruction are chunked and resumable, acquired samples are checked
bit-for-bit, and the conventional coil-RSS magnitude is paired with a
sensitivity-aligned phase. The accepted maps provide only the phase reference,
not missing-sample reconstruction or magnitude weighting. The case manifest
saves exact-grid direct-FFT metrics and both canonical-RAS NIfTI components.
Use the matching ignored `.local.sh` launcher for the tmux run; unlike the PICS
runner this NumPy/SciPy GRAPPA path does not invoke BART.

`run_previous_non_bart_wave_cg_sense.py` is a presentation comparison adapter,
not a new reconstruction algorithm. It calls the existing Torch PCG-SENSE
operator in `external/wave-mprage`, reuses the accepted maps and R3x2 inputs,
and exports canonical-RAS magnitude and phase NIfTIs. On an older completed
case, `--resume` similarly backfills phase from the saved complex image without
rerunning CG-SENSE. Copy the matching `.example.sh`
launchers to ignored `.local.sh` files and run the long jobs in tmux. Both
launchers require the approved metrics-reference manifest, accept
`--validate-only`, and safely reuse only cases carrying metrics from the same
hash-bound reference. Install both the BART reconstruction and evaluation
requirements for these presentation jobs.
See [`docs/presentation_nifti_collection.md`](docs/presentation_nifti_collection.md).

`build_presentation_metrics_csv.py` writes a presentation-ordered metric table
and provenance manifest beside the collection. It keeps exact-grid metrics and
matched-grid retrospective-resolution metrics explicitly labelled and leaves
DICOM rows qualitative-only. Repeat `--regularization-metrics` to combine
hash-bound tables from multiple reconstruction experiments without dropping
existing presentation metrics. `export_presentation_orientation_tiffs.py`
exports sagittal, coronal, and axial 16-bit TIFFs using per-volume display
scaling. Standard 256-cubed cases use index 128. Retrospective-resolution cases
use the center index of each physical RAS orientation independently, so reduced
axes do not inherit an off-center index. Copy its `.example.sh` launcher to the
ignored `.local.sh` form so private collection paths remain local.

`evaluate_no_wave_r3x1_sweep.py` validates the hash-bound case metrics and
plots Wavelet metric curves against the fully sampled direct-FFT RSS reference,
with CG-SENSE and GRAPPA controls. It reports per-metric leaders without making
an automatic lambda selection. The current evaluation shows `1e-3` leading
brain NRMSE, 3D SSIM, and edge-ratio closeness among tested values, while
`1e-4` leads edge-gradient NCC; `1.5e-2` is worse on all four plotted metrics.

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
manifest checks. `run_bart_regularization.py` provides the publishable CLI,
provenance, validation, and safe completed-run reuse around that direct call.

Evaluation intentionally has a hard orientation gate. Run the preparation and
orientation-review programs, inspect `orientation_signed_axis_choice.png`, and
record the user's L/R decision before registering volumes or calculating
metrics. The signed-axis search is diagnostic only; its top-scoring mapping is
not silently applied.

Run a script with `--help` for its dataset-independent CLI, for example:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/reconstruct_no_wave_grappa_3d.py --help
```
