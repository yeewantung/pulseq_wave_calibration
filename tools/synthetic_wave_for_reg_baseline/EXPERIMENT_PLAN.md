# R3 presentation optimization, R1 refinement, and retrospective-resolution plan

Updated: 2026-09-01

This is the active experiment plan. The original R3x1-centered tracker is
preserved as a historical record in
`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`.
The dataset-specific execution checklist is
`docs/r1_dataset_processing_todo.md`.

## Corrected pure-lattice rerun addendum

The five-case corrected pure-image-lattice rerun is complete. Explicit manual
visual and metric tradeoff review selected Wavelet `3.5e-2` for native R3x1,
`3.5e-2` for native R3x2, `2.5e-2` for LR-X R3x2, `2.5e-2` for LR-Y R3x2,
and `2.2e-2` for LR-XY R3x2. The hash-bound selection manifest SHA-256 is
`07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`.
These settings supersede the older ACS-union settings only for the cleaned
measured-MPRAGE tool defaults. The historical completed-study statements below
remain records of their original experiments and are not retroactively edited.
The cleaned tool produces both the FISTA-r0 control and selected Wavelet arm
for ablation; the approved BET mask remains evaluation-only.

The active execution order is now:

1. preserve the completed fully sampled R1 preparation, direct-FFT-reference
   parameter evaluation, frozen Wavelet `lambda=1.5e-2` decision, and approved
   qualitative-only R3 transfer;
2. preserve the completed three-case R1 retrospective low-resolution
   reconstruction batch;
3. preserve the generalized, explicitly approved R1 native-grid and
   matched-grid visual review; and
4. preserve the completed descriptive R1 resolution-tradeoff analysis, which
   makes no automatic resolution selection; and
5. preserve the passed all-coil real-data `PSF=1` identity and full-sampling
   Wave inverse gates.

The earlier R3 parameters and GRAPPA-referenced evaluation remain historical
development results. The R1-selected setting is frozen and must not be retuned
on R3 or independently for the retrospective low-resolution cases.

As of 2026-08-21, the fully sampled R1 source preparation, synthetic-Wave
encoding, GPU lambda-zero/coarse/refined/targeted reconstructions, approved-mask
metric support, combined 49-case exact-grid evaluation, frozen Wavelet
selection, qualitative R3 transfer, all three R1 retrospective low-resolution
reconstructions, the approved native/matched review, and the descriptive R1
resolution analysis are complete. The real-data `PSF=1` identity and
full-sampling Wave inverse gates also pass on all 12 virtual coils. No
retrospective resolution was selected automatically.

## Current implementation and execution status

Completed:

- [x] R3 orientation correction and canonical-RAS NIfTI export;
- [x] retrospective low-resolution reconstruction, visual QC, and descriptive
  analysis support;
- [x] provisional R3 presentation optimization using no-wave GRAPPA as the
  temporary metric reference;
- [x] portable dataset contract and measured TWIX/DICOM inspection;
- [x] image/refscan coil-compression support;
- [x] direct, interpolation-free fully sampled R1 source assembly and the
  compatible R3 GRAPPA source branch;
- [x] manifest-backed full Wave encoding and separate post-Wave target mask;
- [x] manifest-backed measured ACS export from direct image data or refscan;
- [x] manifest-backed GPU lambda-zero reconstruction and tmux runner;
- [x] combined coarse/refined exact-grid validation and fixed-mask evaluation;
- [x] frozen R1 Wavelet selection and qualitative-only R3 transfer approval;
- [x] structurally validated R1 retrospective low-resolution configuration and
  private/public launcher split;
- [x] completed all three R1 retrospective low-resolution Wavelet
  reconstructions with case-specific maximum-eigenvalue estimation and GPU
  `-g`.
- [x] generalized and explicitly approved the R1 native/matched review;
- [x] completed the approval-gated descriptive R1 resolution analysis without
  a true-SNR claim, composite rank, or automatic selection;
- [x] passed the real-data `PSF=1` identity and full-sampling Wave inverse
  gates on all 12 virtual coils without reconstruction or presentation
  processing.

Checklist status (the sole unchecked item is explicitly deferred):

- [x] identify the fully sampled R1 TWIX and candidate Wave sequence;
- [x] create its concrete no-DICOM-reference manifest and pass measured
  acquisition inspection;
- [ ] designate an independent confirmation scan if one is acquired; this is
  deferred at the user's request and the current R1 parameter selection is not
  independently confirmed;
- [x] run and visually qualify the real R1 preparation path through measured
  ACS export;
- [x] make lambda-zero BART reconstruction manifest-aware;
- [x] make regularized BART reconstruction manifest-aware and GPU-only;
- [x] pass the R1 lambda-zero scaling, orientation, geometry, and GPU gates;
- [x] make evaluation preparation and metrics reference-neutral;
- [x] run the R1 Wavelet coarse sweep with a solver-matched FISTA lambda zero;
- [x] pass the split-complex lambda-zero gate and run the compact R1 block-8
  LLR sweep;
- [x] run the higher-lambda Wavelet and full blocks `4`, `8`, `16` LLR
  refinement grid;
- [x] combine retained and new cases for exact-grid validation and direct-FFT
  metrics without registration, interpolation, or DICOM intensity ranking;
- [x] freeze Wavelet `lambda=1.5e-2` as the R1-selected MPRAGE regularization;
- [x] apply it unchanged to R3 as a qualitative cross-dataset transfer check;
  do not calculate selection metrics or retune on R3.
- [x] run the three retrospective low-resolution R1 cases with frozen Wavelet
  `lambda=1.5e-2` and case-specific maximum-eigenvalue estimation;
- [x] generalize the native/matched physical-coordinate review for the R1
  direct-FFT, full-resolution Wavelet, and low-resolution Wavelet inputs;
- [x] generate the R1 review package and obtain explicit visual approval;
- [x] after approval, generalize and run descriptive R1 resolution-tradeoff
  analysis without an automatic winner.

Intentionally deferred or excluded: no-wave BART PICS, completion of the older
partial-readout R1 scans, DICOM-intensity ranking for the current R1 dataset,
and use of BET outside metric calculation.

The selected fully sampled source is MID00198 in the 2026-08-21 product
folder. It has a complete 256 cubed logical grid, 64 coils, and no refscan;
therefore it uses image-derived 64-to-12 coil compression and direct measured
k-space assembly. MID00196 is measured R3x1 and is explicitly excluded from
the direct R1 route. Prescan Normalize was enabled for the available DICOM, so
the initial R1 experiment has no DICOM or other intensity-ranking baseline.

### Historical R3 reference decision and final R1 reference

The BART `ecalib -I` pilot did not correct the relevant receive-profile
mismatch. Its brain-core-to-shell median ratio was `0.895`, compared with
`0.966` for the previous Wave lambda-zero reconstruction and approximately
`0.975` for both no-wave GRAPPA and no-wave SENSE. The normalized and raw IDEA
DICOM ratios were `1.347` and `0.655`, respectively. Retain the pilot and its
manifest as a negative result, but do not use `ecalib -I` for the production
path or continue tuning regularization against either DICOM intensity profile.

The completed preliminary R3 work used no-wave GRAPPA as its single temporary
metric reference. No GRAPPA-versus-SENSE comparison or agreement gate was
required. SENSE remains an optional historical diagnostic, and the no-wave
BART PICS branch remains deferred.

For the qualified fully sampled R1 experiment, direct FFT RSS of the compressed
no-Wave k-space is the approved quantitative reference. GRAPPA, SENSE, and
DICOM are not R1 ranking references. Apply the approved BET `f=0.59` plus
one-voxel mask only while calculating metrics; do not use it to alter
reconstruction or display. DICOM-reference support may remain configurable for
a future qualified dataset, but it is disabled for this experiment.

## 1. Scientific roles of the datasets

### Fully sampled R1 data used for final parameter refinement

The R1 dataset was collected, inspected, and accepted as the final baseline.
Its sampling, readout completeness, DICOM processing, coil configuration,
geometry, and sequence definition were measured rather than inherited from R3.
The acquisition did not establish a separate confirmation scan, so do not
describe the parameter selection as independently confirmed.

The older R1 candidates below are retained as documented fallbacks, not the
active final-selection dataset.

The original two R1 MPRAGE scans are under the private path recorded as
`$FALLBACK_R1_ROOT` in the ignored local dataset notes:

```text
$FALLBACK_R1_ROOT
```

Two additional R1 TWIX datasets are similarly referenced as:

```text
$ADDITIONAL_R1_ROOT/subject6
$ADDITIONAL_R1_ROOT/subject7
```

All four scans contain the complete 256 x 192 PE1 x PE2 grid, so none needs
GRAPPA. However, all four also use the same asymmetric readout: 404 stored
samples with center 148 on a nominal 512-point oversampled grid. They are R1
in PE but are not fully measured along readout.

Consequently, these scans are not an unqualified true baseline. Readout
partial-Fourier completion can affect
sharpness, phase, noise texture, and the Wave forward model, which can confound
regularization selection.

Before the now-completed fully sampled acquisition, no better R1 dataset was
available. The
inspected 2026-06-01 MID00077 `pulseq_mprage_nowave_full` scan is nominally
256 x 256 x 192 but is actually R2x2 in PE with a central 32 x 32 calibration
region, so it cannot replace the R1 baseline. Defer implementation of the
older scans' readout completion. Reactivate the 2021-05-10 fallback only after
freezing and validating one completion method, stating the limitation
explicitly, and reserving at
least one independent scan for confirmation.

### Preliminary optimization, presentation, and later transfer check

Use the R3x1 product dataset referenced locally as:

```text
$R3_PRODUCT_ROOT
```

This accelerated dataset is not a true parameter-selection baseline. It had
two explicitly different roles:

1. provisional R3-specific optimization and presentation results; and
2. a cross-dataset transfer check with the R1-selected parameter and no
   further tuning. Both roles are complete.

The historical development comparison used:

1. R3x1 no-wave GRAPPA as the single temporary reference;
2. R3x2 synthetic-Wave BART reconstruction with the appropriate provisional
   or R1-selected regularizer; and
3. the selected retrospective low-resolution R3x2 cases.

No-wave SENSE and no-wave BART PICS are outside the active development path.

Show raw and normalized IDEA DICOMs only as qualitative presentation context
during this temporary stage. Their receive-profile corrections make them
unsuitable for ranking regularization.

The qualitative DICOM selector must require both `ND` and `NORM` and reject
`DIS2D`/`DIS3D`. The accepted normalized series UID is:

```text
1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0
```

The earlier evaluations against normalized and non-normalized DICOM are
historical presentation pilots, not parameter-selection references.

## 2. Reproducible environment

Use environment variables supplied by an ignored machine-local launcher:

```bash
source "$CONDA_SETUP"
conda activate "$SYNTHETIC_WAVE_CONDA_ENV"
source "$BART_STARTUP"
```

Use the `bart` resolved by `command -v bart`. Always use `-g` for every BART
reconstruction. Do not silently fall back to CPU; stop and document any
command-specific incompatibility.

For FSL BET:

```bash
: "${FSLDIR:?Set FSLDIR in the private local environment}"
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

## 3. Phase A: preliminary R3x1 optimization and presentation — complete

This completed phase used the existing R3x1 product and synthetic R3x2 Wave
dataset. It used no-wave GRAPPA for temporary quantitative development and
reserved DICOM for qualitative display.

1. retain the approved fixed BET mask and L/R orientation, applying the mask
   only when calculating metrics;
2. rank provisional candidates against the single GRAPPA reference;
3. keep DICOM-based ranking disabled for this dataset but configurable so it
   can be enabled after the new R1 dataset is qualified;
4. compare DICOM, no-wave GRAPPA, and selected synthetic R3x2 Wave
   reconstructions visually with fixed display settings; and
5. label every resulting choice as preliminary, R3-dataset-specific, and
   subject to replacement by the new R1 experiment.

Do not overwrite the accepted historical sweep. Store the new corrected and
masked-evaluation results in a separately manifested output tree.

## 4. Phase B: qualify the R1 acquisition — complete

This hard gate passed before R1 Wave synthesis and final parameter refinement.
Measured inspection confirmed a fully sampled `256^3` logical grid, complete
usable readout, 64 coils, and no refscan. The image stream supplied the frozen
64-to-12 coil-compression basis and direct no-Wave source. No separate
confirmation scan has yet been designated.

### Deferred fallback: older R1 partial-readout completion

The R1 files have a nominal 512-sample oversampled readout grid but 404 stored
samples. The MDH k-space center is sample 148, so the acquired samples map to
indices 108:512 of the nominal 512-point grid. This is asymmetric/readout
partial-Fourier sampling, not a complete 202-point readout that should simply
be accepted as the final image matrix.

Only if the older R1 fallback is reactivated, implement and compare the
following while preserving the MDH center:

1. embed the 404 samples into the nominal 512-point oversampled grid;
2. test zero-filled and homodyne/POCS partial-Fourier completion;
3. inverse FFT on the oversampled grid and crop the central 256-pixel FOV;
4. verify the equivalent ordering of partial-Fourier completion, 2x readout
   oversampling removal, and 202-to-256 Fourier interpolation;
5. compare geometry, sharpness, phase behavior, and normalized-DICOM agreement;
6. freeze one method and record all centering/crop indices in the manifest.

Do not crop the 404 raw k-space samples directly to 256. That would discard
high-frequency support and change the intended field of view/resolution.

Fallback acceptance requires the chosen development and holdout scans to
produce a 256 x 256 x 192 reference with correct anatomy and a verified L/R
convention. The parameter-selection claim must then be qualified as conditional
on the frozen readout completion.

## 5. Phase C: prepare the R1 synthetic-Wave experiment — complete

The completed R1 development scan followed this preparation contract:

1. load the full raw multi-coil volume using the accepted readout method;
2. estimate and record the coil-compression basis and retained energy;
3. estimate ESPIRiT maps from the fully sampled central k-space;
4. form a fully sampled raw-derived no-wave reference without GRAPPA;
5. generate the matching theoretical Wave PSF;
6. Wave-encode the full multi-coil data;
7. apply the R3x2 sampling mask only after Wave encoding;
8. run lambda zero and verify the forward model, scaling, and orientation.

Required synthetic checks are complete:

- `PSF=1` reproduces the no-Wave path on all 12 real-data virtual coils; the
  maximum relative complex L2 error is `2.7991e-7`.
- Full-sampling Wave forward/inversion is internally consistent on all 12
  real-data virtual coils; the maximum relative complex L2 error is
  `3.0099e-7` and maximum exterior energy fraction is `4.7121e-14`.
- The R3x2 mask is applied only after Wave encoding.
- Acquired samples are unchanged and missing samples are exact zero.
- Any future confirmation scan must use the same conventions but independently
  estimated maps and coil-compression basis.

The hash-bound operator record is
`evaluation/full_sampling_wave_operator_validation/operator_validation_manifest.json`
below the private dataset output root. The standard preparation wrapper runs
this gate after full Wave synthesis and before visual review, masking, or BART
export.

## 6. Phase D: freeze the scaling contract — complete

The custom `bart wave` command unconditionally divides k-space by its global
L2 norm. It has no PICS-style `-w` or `-S` option. Therefore:

- record the pre-normalization global L2 norm for every Wave input;
- verify invariance by reconstructing a small test after multiplying k-space
  by a known scalar;
- restore output amplitude with the recorded norm when an original-scale
  output is needed;
- use one documented positive LSQ intensity match inside the fixed brain mask
  for vendor-DICOM comparisons;
- do not interpret `wave -e` as intensity scaling; it is the normal-operator
  maximum eigenvalue used for the iterative step size.

For no-wave `bart pics` validation runs, use and record its automatic/default
data scale and `-S` output rescaling. Do not assume that a numerical lambda is
directly interchangeable between `pics` and the custom `wave` command.

## 7. Phase E: final regularization refinement on R1 — complete

### Wavelet coarse sweep

The development scan began with:

```text
lambda = 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
```

GPU FISTA used otherwise frozen settings, and lambda zero remained in the plots
as a control even though it is not regularized. Coarse review led to the
completed refined and targeted grids documented in the R1 checklist.

### LLR correctness gate

The completed pilot used `wave -l` without `-v`. In the current BART source,
the real/imaginary ITER dimension becomes size two only with `-v`; without it,
the LLR matrix has a singleton second dimension and behaves like block-vector
shrinkage rather than the intended real/imaginary low-rank penalty.

Before any new LLR sweep:

1. verify the upstream intended `-l -v` usage;
2. add/validate conversion of the split real/imaginary output;
3. compare lambda zero with and without `-v` to establish equivalence after
   recombination;
4. run a small block-8 lambda pilot and confirm that the proximal operation is
   non-degenerate; and
5. only then define a new lambda/block-size sweep.

The R1 block-8 correctness gate passed on 2026-08-21: recombined split-complex
lambda zero differed from native-complex FISTA lambda zero by relative L2
`2.73366e-6`, below the fixed `1e-5` limit. The presentation-oriented coarse
cases `2e-5`, `1e-4`, and `5e-4`, and the later block-size refinement over
blocks 4, 8, and 16, are complete.

Do not use the old LLR sweep to select parameters. A brain mask may make small
differences easier to measure, but it does not correct the singleton-dimension
configuration.

### Selection rule

The intended design was one development scan plus independent confirmation.
Only the development scan has been designated, so the frozen Wavelet setting
must not be described as independently confirmed. The method and parameter
were nevertheless locked before the completed qualitative R1-to-R3 transfer,
and no retuning occurred on R3.

## 8. Phase F: standardized evaluation

The completed historical R3 development evaluation:

1. canonicalize no-wave GRAPPA and all candidates to the same RAS grid and
   verify orientation explicitly;
2. use the approved BET mask only during metric calculation;
3. estimate any required geometry transform once from lambda zero and reuse it
   unchanged across the candidate sweep;
4. apply one positive LSQ intensity scale for each candidate against GRAPPA
   inside the fixed mask;
5. calculate masked NRMSE, RMSE, MAE, SSIM, NCC, and gradient/detail metrics
   against GRAPPA; and
6. keep DICOM intensity metrics out of regularization selection, while retaining
   raw and normalized DICOM for clearly labelled qualitative figures.

The qualified R1 dataset replaced that temporary contract as follows:

1. use direct FFT RSS of the fully sampled compressed no-Wave k-space as the
   quantitative reference; exclude DICOM, GRAPPA, and SENSE intensities;
2. use only the visually approved BET `f=0.59` plus one-voxel mask, and only
   for metrics;
3. require exact canonical-RAS geometry for the full-resolution reference and
   candidates; the 49-case gate passed without registration or interpolation;
4. undo recorded candidate export scaling before fitting one positive LSQ
   scalar inside the fixed mask;
5. calculate brain-mask NRMSE, RMSE, MAE, SSIM, NCC, and gradient/detail
   metrics without histogram matching or bias correction;
6. retain background and missed-anatomy measures as separately labelled QC;
7. save CSV, JSON provenance, common-window figures, and input/output hashes;
   and
8. perform no composite scoring or automatic selection.

The shared Wave exporter now produces canonical RAS with affine-axis flips
`[true, false, true]`; R1 full-resolution geometry was explicitly validated.
Retrospective low-resolution native grids differ by design, so compare native
images in physical coordinates and label any matched-grid interpolation.

## 9. Phase G: retrospective low-resolution reconstruction

The separate repository was imported with its two-commit history under:

```text
tools/wave_retro_lr_recon/
```

The integration is complete. The legacy internal-NPY/Torch-CG implementation
was replaced with a tested library and dataset-independent CLI that consumes
the current BART-input manifest contract, uses the parent's pinned
`external/wave-mprage` exporter, and reconstructs with GPU `bart wave -g`.
The synthetic-Wave workflow calls it through:

```text
requirements/retrospective_low_resolution_product.example.json
scripts/run_retrospective_low_resolution.sh
```

The source PSF identity gate passed on the real product PSF with relative
complex L2 `4.406439186e-08` and maximum complex error `1.685873912e-07`.
The all-coil native-grid Wave-operator gate passed with relative L2
`2.150150032e-07` and maximum complex error `1.317088993e-09`. Structural
validation of the product configuration also passed without writing output.
Production preparation and all three corrected-LLR reconstructions completed
on 2026-08-21. Their manifests, finite-value checks, matrices, voxel sizes,
canonical-RAS affines, and GPU BART logs pass.

The separate fully sampled R1 batch also completed on 2026-08-21 local time.
It uses the frozen Wavelet `lambda=1.5e-2`, FISTA with 100 iterations and
tolerance `1e-6`, case-specific maximum-eigenvalue estimation, and mandatory
GPU `-g`. Its batch and all three case manifests report `complete`; all
canonical-RAS NIfTI exports are finite and have the expected physical XYZ
shapes `172x256x256`, `256x172x256`, and `204x204x256`.

The user subsequently authorized a separate Wavelet-lambda sweep on these
three target grids. Evaluate it with the already accepted retrospective
matched-grid calculation: linearly resample candidates to the original 1 mm
RAS grid and reuse the exact hash-bound full-resolution FISTA-lambda-zero and
direct-FFT references plus the fixed approved BET brain/edge masks. Do not
create or substitute a new native-grid reference for this selection. The
sweep reuses the completed lambda-zero and `1.5e-2` controls and makes no
automatic choice.

A corrected-LLR follow-up uses the same three target grids and prepared inputs.
Its configurable default tests block 4 at `2e-3, 5e-3, 6e-3, 1e-2` and blocks
8/16 at `2e-3, 5e-3, 1e-2, 2e-2`. Reconstruction must retain split-complex
`-l -v`, FISTA `-f`, mandatory GPU `-g`, magnitude/phase export, and resume
manifests. Evaluation is a separate program using the exact original matched
1 mm references, BET brain mask, edge mask, and LSQ scaling; retrospective
FISTA lambda zero is a control, not an LLR reconstruction. Make no automatic
block/lambda selection.

The current tool's phase-encoding-only crop is the intended behavior. For
sagittal MPRAGE, physical X maps to logical partition/PE2, physical Y maps to
logical line/PE1, and physical Z maps to logical readout. Never crop the
readout dimension in the retrospective-resolution experiment.

At locked R3x2 Wave regularization, generate these physical-XYZ resolutions:

```text
1.5 x 1.0 x 1.0 mm
1.0 x 1.5 x 1.0 mm
1.25 x 1.25 x 1.0 mm
```

Use unchanged FOV and integer center crops of PE1 and/or PE2 only. Keep the
readout matrix and resolution exactly unchanged. Record the achieved PE
resolution when the requested matrix is not integral. Target PE matrices are
rounded to the nearest multiple of four. For the 256 mm product FOV, the three
logical `(RO, LIN, PAR)` matrices are `256 x 256 x 172`, `256 x 172 x 256`,
and `256 x 204 x 204`, with achieved changed-axis resolutions `1.488372` mm
and `1.254902` mm. Crop the no-wave source in PE before Wave encoding and
rebuild the target PE-grid PSF; the focused operator test confirms that
cropping already Wave-encoded data is not an equivalent ordering.

Evaluate descriptive signal/local-residual, fidelity, and sharpness tradeoffs.
Do not call a background-derived quantity true SNR/CNR because BART air support
is nearly zero. Preserve native-resolution images and create explicitly
labelled reference-grid resamples only for matched visual/metric comparisons.

The historical product corrected-LLR visual-review package provides both
representations:
native grids use the nearest slice to one shared RAS location with physical-mm
extents and no spatial resampling; matched grids use linear interpolation onto
the full-resolution 1 mm RAS grid. Both use display-only per-volume positive
p99.5 scaling. Neither DICOM nor BET participates. The initial attempt exposed
an older full-resolution LLR NIfTI that predated the canonical-RAS exporter;
that review is retained under a clearly rejected diagnostics directory. The
existing BART result was re-exported without reconstruction, and the review
code now rejects Wave NIfTIs lacking the corrected exporter sidecar contract.
That corrected historical package was visually approved before its descriptive
analysis.

The manifested quantitative tradeoff analysis is complete. It transfers the
approved BET mask into untouched reconstruction space using the frozen shared
rigid transform, then keeps native-grid and matched-grid measurements separate.
No candidate-specific registration, DICOM intensity, true-SNR/CNR claim,
composite rank, or automatic resolution selection is used.

Relative to full-resolution corrected LLR, native total edge-gradient ratios
were `0.981`, `0.980`, and `0.964` for the lower-X, lower-Y, and balanced cases.
The expected directional losses were visible: lower-X retained X-gradient
`0.924`, lower-Y retained Y-gradient `0.917`, and the balanced case retained
X/Y gradients `0.953/0.946`. The fixed smooth-brain signal/local-residual proxy
increased from `13.71` at 1 mm to `15.22`, `15.57`, and `15.67`; this is a
noise-plus-residual-anatomy proxy, not SNR. After linear alignment to the 1 mm
grid, brain NRMSE versus full-resolution LLR was `0.0663`, `0.0695`, and
`0.0719`, with mean axial SSIM `0.916`, `0.911`, and `0.896`. These results
describe the expected gain/loss curve and do not identify a single best case.

An initial background-based apparent-SNR summary was rejected because BART's
reconstructed background is nearly zero and the resulting ratios were
scientifically uninformative. It is retained under diagnostics. Background
statistics remain separately labelled QC; the accepted summary uses the fixed
smooth-brain signal/local-residual proxy.

For the R1 Wavelet batch, do not reuse the historical wrapper unchanged: its
inputs, titles, and manifest scope name GRAPPA and corrected LLR. Generalize the
review interface while retaining historical compatibility. The R1 review must
include direct FFT RSS, the full-resolution frozen Wavelet result, and the
three low-resolution Wavelet results; it must produce native physical-coordinate
and explicitly resampled matched-grid figures. User approval of those figures
is required before adapting or running the R1 descriptive analysis.

## 10. Phase H: R1-to-R3 parameter transfer check

This check is complete. The frozen R1-selected Wavelet `lambda=1.5e-2` was
applied to R3 without retuning and the user approved the visibly smoother
regularized result. Because R3 had already been used for preliminary
optimization, this remains a qualitative transfer check rather than untouched
independent validation. No R3 selection metrics were calculated.

## 11. Presentation design gates

Presentation packaging resumed on 2026-08-24. The private magnitude-NIfTI
collection is manifested with all 20 requested outputs plus three supplemental
retrospective FISTA-lambda-zero controls present as byte-identical copies. No
spatial resampling or cross-volume intensity normalization is performed while
collecting files.

The previous non-BART R3x2 Wave PCG-SENSE reconstruction and no-Wave R3x1
PICS CG-SENSE/Wavelet sweep have completed. The no-Wave sweep uses its own
`pics -g -S` scaling contract and retains the originally requested `1.5e-2`
case, but the presentation entry now uses the explicit `1e-3` choice. It does
not assume numerical lambda transfer from the custom Wave operator. The no-Wave
R3x1 GRAPPA reconstruction subsequently completed with
the accepted joint-coil 5x5x5/Ncc=12 implementation and bitwise preservation of
acquired samples. All reconstruction launcher workflows calculate and save the
existing standard metric set
against the approved direct-FFT R1 RSS and BET mask on the exact grid. They
restore export normalization, permit only the metric contract's one least-
squares intensity scale, and perform neither registration nor interpolation.

All workflows export magnitude and phase, including completed resume-mode
phase backfills from hash-validated complex images. The presentation collection
copies magnitude only and now contains 23 NIfTIs. Its ordered
metric CSV distinguishes exact-grid, matched-grid retrospective, native
descriptive, and qualitative-only rows. Sixty-nine manifested orthogonal TIFFs
cover all available NIfTIs. Standard cases use index 128; the six
retrospective-resolution cases use each physical orientation's center index.
GRAPPA has brain NRMSE `0.046644`, 3D
brain-bounding-box SSIM `0.963763`, and axial mean SSIM `0.976415` against the
approved exact-grid direct-FFT reference.

The no-Wave R3x1 Wavelet sweep now has a dedicated metric CSV and lambda-curve
plot with CG-SENSE and GRAPPA controls. Among tested values, `1e-3` leads brain
NRMSE (`0.038955`), 3D SSIM (`0.972682`), and edge-ratio closeness (`1.004790`),
whereas `1e-4` leads edge-gradient NCC (`0.993260`). The presentation's
`1.5e-2` case is worse on all four plotted metrics. This evidence rejects
assuming numerical transfer from the custom-Wave R3x2 value. The user explicitly
selected `1e-3` for the no-Wave presentation entry; this does not change the
separately frozen custom-Wave R3x2 setting.

Retain the compact R3 panel as historical development/transfer context:
qualitative DICOM, no-wave GRAPPA, the R1-selected Wavelet transfer result,
corrected LLR if useful, and restrained difference maps. Do not present R3
metrics as independent validation.

For the R1/final presentation package, proposed minimal outputs are:

1. one pipeline/data-role diagram;
2. the completed R1 log-lambda plots with brain NRMSE/SSIM and one
   sharpness/detail tradeoff measure, without claiming an independent
   confirmation scan;
3. fixed-slice Wavelet comparison panels with identical windowing;
4. an LLR heatmap only if corrected LLR proves scientifically useful;
5. a resolution-versus-signal/local-residual/sharpness plot for the three
   retrospective-LR cases plus the 1 mm baseline, explicitly avoiding a true
   SNR claim;
6. native-resolution and matched-grid zoom panels for the resolution study;
7. one R1-to-R3 qualitative transfer panel, with DICOM clearly labelled as
   presentation context rather than an intensity reference; and
8. restrained difference maps for the final candidates, not every sweep case.

Freeze slice positions, zoom boxes, display percentiles, metric definitions,
and ranking rules before generating presentation figures.

## 12. Final repository and output cleanup

The experiments and requested presentation outputs are now frozen, and this is
the next authorized work for a fresh session. Begin with a non-mutating
inventory and proposal. Do not move, archive, merge, or delete outputs until
their manifests, hashes, canonical consumers, and downstream references have
been checked.

- Reorganize the tool into clear public entry points, reusable `utils`,
  configuration/examples, focused tests, and `archive` material.
- Remove tests only when their covered behavior is removed or duplicated;
  retain tests for all production functionality.
- Archive diagnostic scripts that remain useful for provenance but are not
  public entry points.
- Polish docstrings, CLI help, examples, and maintenance comments.
- Keep one output index per dataset with canonical runs, hashes, status, and
  rebuildable/deletable classifications.
- Remove large intermediates only after downstream references and hashes have
  been checked.

## 13. Immediate next step

Perform Section 12 in a fresh session. No further reconstruction, sweep,
evaluation, or presentation generation is pending. The first cleanup turn must
read the applicable `AGENTS.md` files and `HANDOVER.md`, inspect the dirty
worktree without resetting it, and classify private outputs before proposing
changes. Prefer reversible archive moves and obtain approval before any
irreversible deletion.

The synthetic-Wave R3x1 experiment is complete on the same exact R1 source and
1 mm grid. It reused the accepted full-Wave encoding, branched only the R3x1
sampling target, passed the lambda-zero visual gate, ran the coarse Wavelet and
corrected-LLR sweeps, and refined Wavelet at
`1.6e-2, 1.8e-2, 2.2e-2, 2.5e-2, 3e-2, 4e-2`. All reconstructions used GPU BART
and retained magnitude and phase.

The user explicitly selected Wavelet `lambda=2.2e-2`; no automatic selection
was performed. Its brain NRMSE is `0.0275119`, 3D SSIM is `0.985167`, brain NCC
is `0.996493`, gradient NCC is `0.995822`, and edge ratio is `1.004955` on the
unchanged direct-FFT grid with the approved fixed BET mask. The selection is
hash-bound in the R3x1 refinement evaluation tree. The presentation collection
contains the solver-matched FISTA lambda-zero control and selected Wavelet case,
with magnitude NIfTIs, three central-orientation TIFFs each, and metric rows.

The R1 review was explicitly approved on 2026-08-21 and its separate approval
record binds both figures and the review manifest. The path-agnostic analysis
interface then completed using the approved BET mask on the exact shared RAS
grid. The fixed auxiliary edge, smooth-brain, and background supports derive
from the full-resolution frozen Wavelet image; direct FFT RSS contains genuine
noise throughout the FOV and therefore cannot define thresholded air support.
Background values remain QC only.

For lower-X, lower-Y, and balanced cases respectively, native total
edge-gradient ratios to full-resolution Wavelet are `0.9779`, `0.9773`, and
`0.9627`; signal/local-residual proxies are `21.01`, `21.33`, and `21.47`
versus `18.77` at full resolution; matched brain NRMSE is `0.06236`, `0.06260`,
and `0.06552`; and mean axial SSIM is `0.9265`, `0.9280`, and `0.9150`.
Directional losses occur on the expected changed axes. These are descriptive
tradeoffs, not a selected winner.

After the separate matched-grid Wavelet and corrected-LLR sweeps, the user made
three explicit presentation choices: lower-X Wavelet `lambda=1e-2`, lower-Y
Wavelet `lambda=5e-3`, and balanced unregularized FISTA `lambda=0`. The private
selection record hash-binds those cases. These are per-grid regularization
choices and do not select a preferred retrospective resolution.

The presentation collection, metric table, and orthogonal TIFF exports are
current and jointly validated: 28 NIfTIs, 28 metric rows, no placeholders, and
84 TIFFs. The no-Wave presentation-specific Wavelet entry now uses the explicit
user choice `1e-3`. This PICS-specific decision does not reopen the frozen
custom-Wave R3x2 setting. Independent confirmation remains deferred, and no
preferred retrospective resolution is inferred automatically.

## 14. Remaining work and deferred branches

1. **Retrospective Wavelet sweep — complete:** all 27 lambda/resolution
   combinations and the fresh matched-grid evaluation are available. The
   attempted native-grid evaluator is superseded diagnostics for this task.
2. **Retrospective corrected-LLR sweep — complete:** the reconstruction sweep,
   low-lambda extension, combined matched-grid evaluation, and block-specific
   metric plots are available.
3. **R1 retrospective visual review — complete:** the generalized review and
   hash-bound explicit approval are recorded.
4. **R1 retrospective descriptive analysis — complete:** native and matched
   metrics are manifested, with no automatic resolution selection.
5. **Independent confirmation scan — deferred:** at the user's request, no
   separate confirmation scan is being pursued now. Do not call the selected
   parameter independently confirmed.
6. **DICOM ranking — disabled:** the direct FFT RSS is the active R1 reference.
   Retain DICOM for metadata and qualitative context only unless a future
   dataset is explicitly qualified for intensity ranking.
7. **Reconstruction branches:** the presentation-only no-Wave R3x1 PICS sweep
   and previous non-BART Wave PCG-SENSE reconstruction and phase exports are
   complete. The no-Wave R3x1 GRAPPA reconstruction, phase export, and metrics
   are also complete. Older partial-readout R1 fallbacks remain out of scope.
8. **Acceptance gates — retain:** preserve PSF/no-Wave identity,
   full-sampling Wave consistency, mask-after-Wave ordering, exact acquired
   sample preservation, lambda-zero scaling, norm restoration, orientation,
   GPU `-g`, provenance, and resumability checks. The two source-operator gates
   are now productionized and passed on every R1 virtual coil.
9. **Presentation package — complete:** the manifested collection has 28
   available magnitude NIfTIs, an ordered 28-row metric CSV, and 84 orthogonal
   TIFFs. Their manifests are hash-bound to the refreshed collection; no-Wave
   Wavelet uses `1e-3`, and synthetic-Wave R3x1 includes FISTA lambda zero and
   selected Wavelet `2.2e-2`.
10. **Repository/output cleanup — next:** audit the dirty worktree, hashes, and
   downstream references; choose canonical runs; merge indexes and redundant
   sweep organization; archive superseded material; and consider deletion only
   after an explicit reviewed proposal.
