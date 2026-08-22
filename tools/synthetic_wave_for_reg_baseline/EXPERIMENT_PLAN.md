# R3 presentation optimization, R1 refinement, and retrospective-resolution plan

Updated: 2026-08-21

This is the active experiment plan. The original R3x1-centered tracker is
preserved as a historical record in
`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`.
The dataset-specific execution checklist is
`docs/r1_dataset_processing_todo.md`.

The execution order is now:

1. use the no-wave GRAPPA reconstruction from the existing R3x1 product scan
   as the single temporary development reference;
2. finish the reusable infrastructure needed by later stages: retain the
   completed NIfTI orientation correction and retrospective low-resolution
   support, then remove R3-specific geometry/configuration assumptions from
   the R1 path;
3. qualify the newly acquired R1 dataset; and
4. switch the final parameter-selection and DICOM-ranking workflow to that R1
   dataset, then apply the R1-selected settings back to R3 without retuning.

The first-stage R3 parameters are provisional and may be tuned to that
dataset. Only the later R1 development/confirmation experiment is intended to
select the more transferable final parameters. Results selected against the
temporary R3 reference must be labelled developmental and repeated after the
new R1 dataset is available.

As of 2026-08-21, the fully sampled R1 source preparation, synthetic-Wave
encoding, GPU lambda-zero/coarse/refined/targeted reconstructions, approved-mask
metric support, and combined 49-case exact-grid evaluation are complete. The next
decision is visual and quantitative finalist selection, not more general
infrastructure work.

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
- [x] combined coarse/refined exact-grid validation and fixed-mask evaluation.

Pending:

- [x] identify the fully sampled R1 TWIX and candidate Wave sequence;
- [x] create its concrete no-DICOM-reference manifest and pass measured
  acquisition inspection;
- [ ] decide whether the acquisition supplies separate development and
  confirmation scans;
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
- [ ] apply it unchanged to R3 as a qualitative cross-dataset transfer check;
  do not calculate selection metrics or retune on R3.

Intentionally deferred or excluded: no-wave BART PICS, completion of the older
partial-readout R1 scans, DICOM-intensity ranking before the new R1 DICOM is
qualified, and use of BET outside metric calculation.

The selected fully sampled source is MID00198 in the 2026-08-21 product
folder. It has a complete 256 cubed logical grid, 64 coils, and no refscan;
therefore it uses image-derived 64-to-12 coil compression and direct measured
k-space assembly. MID00196 is measured R3x1 and is explicitly excluded from
the direct R1 route. Prescan Normalize was enabled for the available DICOM, so
the initial R1 experiment has no DICOM or other intensity-ranking baseline.

### Decision recorded 2026-08-20: temporary references

The BART `ecalib -I` pilot did not correct the relevant receive-profile
mismatch. Its brain-core-to-shell median ratio was `0.895`, compared with
`0.966` for the previous Wave lambda-zero reconstruction and approximately
`0.975` for both no-wave GRAPPA and no-wave SENSE. The normalized and raw IDEA
DICOM ratios were `1.347` and `0.655`, respectively. Retain the pilot and its
manifest as a negative result, but do not use `ecalib -I` for the production
path or continue tuning regularization against either DICOM intensity profile.

The completed preliminary R3 work uses no-wave GRAPPA as its single temporary
metric reference. No GRAPPA-versus-SENSE comparison or agreement gate is
required. Keep SENSE as an existing optional diagnostic only, and defer the
no-wave BART PICS regularization branch. Apply the approved BET mask only while
calculating metrics; do not use it to alter reconstruction or display. DICOM
voxel intensities must not enter provisional R3 ranking, but the evaluator
must retain a configurable DICOM-reference mode for the new R1 dataset.

## 1. Scientific roles of the datasets

### New R1 data for final parameter refinement

A new R1 dataset has been collected and is the intended final baseline. Before
reconstruction, inspect its sampling, readout completeness, DICOM processing,
coil configuration, geometry, and sequence definitions rather than reusing
current assumptions. Assign independent development and confirmation scans
when the acquisition provides enough data.

The older R1 candidates below are retained as documented fallbacks, not the
active final-selection dataset.

The original two R1 MPRAGE scans are in:

```text
/path/to/data/2021_05_10_bay4_mprage_R1_subjects2and3
```

Two additional R1 TWIX files are in:

```text
/path/to/data/2021_05_14_bay4_subject6
/path/to/data/2021_05_14_bay4_subject7
```

All four scans contain the complete 256 x 192 PE1 x PE2 grid, so none needs
GRAPPA. However, all four also use the same asymmetric readout: 404 stored
samples with center 148 on a nominal 512-point oversampled grid. They are R1
in PE but are not fully measured along readout.

Consequently, these scans are not an unqualified true baseline. Readout
partial-Fourier completion can affect
sharpness, phase, noise texture, and the Wave forward model, which can confound
regularization selection.

Before the planned new acquisition, no better R1 dataset was available. The
inspected 2026-06-01 MID00077 `pulseq_mprage_nowave_full` scan is nominally
256 x 256 x 192 but is actually R2x2 in PE with a central 32 x 32 calibration
region, so it cannot replace the R1 baseline. Defer implementation of the
older scans' readout completion while the new acquisition is pending. If the
new dataset is not
usable, reactivate the 2021-05-10 fallback only after freezing and validating
one completion method, stating the limitation explicitly, and reserving at
least one independent scan for confirmation.

### Preliminary optimization, presentation, and later transfer check

Use the R3x1 product dataset in:

```text
/path/to/data/20260817_product
```

This accelerated dataset is not a true parameter-selection baseline. It has
two explicitly different roles:

1. use it now for provisional R3-specific optimization and presentation
   results; and
2. after R1 refinement, rerun the comparison with the R1-selected parameters
   as a cross-dataset transfer check without further tuning.

For current development, compare:

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

On `macha`:

```bash
source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
```

Use the `bart` resolved by `command -v bart`. Always use `-g` for every BART
reconstruction. Do not silently fall back to CPU; stop and document any
command-specific incompatibility.

For FSL BET:

```bash
export FSLDIR=/path/to/software/packages/fsl/6.0.6
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

## 3. Phase A: preliminary R3x1 optimization and presentation

Proceed first with the existing R3x1 product and synthetic R3x2 Wave dataset;
do not wait for the new R1 acquisition. Use no-wave GRAPPA for temporary
quantitative development and reserve DICOM for qualitative display.

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

## 4. Phase B: qualify the new R1 acquisition

This is the hard gate before later R1 Wave synthesis and final parameter
refinement. First audit the newly acquired R1 raw data and DICOMs. Confirm the
actual PE sampling, readout support and center, oversampling, matrix/FOV,
orientation, coil configuration, and vendor processing. Do not assume that it
matches either the 2026 R3 scan or the older R1 scans.

If the new R1 acquisition is fully sampled and has complete usable readout,
freeze it directly as the final development/confirmation source. If it has
partial readout, define and validate completion before any parameter selection.

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

## 5. Phase C: prepare the R1 synthetic-Wave experiment

For each selected development/holdout R1 scan:

1. load the full raw multi-coil volume using the accepted readout method;
2. estimate and record the coil-compression basis and retained energy;
3. estimate ESPIRiT maps from the fully sampled central k-space;
4. form a fully sampled raw-derived no-wave reference without GRAPPA;
5. generate the matching theoretical Wave PSF;
6. Wave-encode the full multi-coil data;
7. apply the R3x2 sampling mask only after Wave encoding;
8. run lambda zero and verify the forward model, scaling, and orientation.

Required synthetic checks:

- PSF=1 reproduces the no-wave path.
- Full-sampling Wave forward/reconstruction is internally consistent.
- The R3x2 mask is applied only after Wave encoding.
- Acquired samples are unchanged and missing samples are exact zero.
- Development and holdout scans use the same conventions but independent maps
  and coil-compression bases.

## 6. Phase D: freeze the scaling contract

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

## 7. Phase E: final regularization refinement on R1

### Wavelet coarse sweep

Run the development scan first with:

```text
lambda = 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
```

Use GPU FISTA with otherwise frozen settings. Include lambda zero in all plots
even though it is not a regularized case. Refine only around a genuine optimum
after the coarse metrics and images have been reviewed.

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
cases `2e-5`, `1e-4`, and `5e-4` are complete. Block-size refinement remains
deferred.

Do not use the old LLR sweep to select parameters. A brain mask may make small
differences easier to measure, but it does not correct the singleton-dimension
configuration.

### Selection rule

After the baseline dataset is chosen, designate one scan for development and
an independent scan for confirmation. Select a narrow candidate range on the
development scan and confirm it without retuning. Lock the method and parameter
before running the R1-to-R3 transfer comparison.

## 8. Phase F: standardized evaluation

During temporary R3 development:

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

After the new R1 dataset is qualified, replace this temporary reference
contract with a dataset-specific final contract. For each R1 subject/scan:

1. convert the normalized unfiltered DICOM to canonical RAS;
2. create one fixed BET brain mask from the reference only;
3. visually approve the BET boundary and L/R orientation;
4. estimate one proper rigid registration from lambda zero;
5. apply the exact same transform to all cases;
6. calculate brain-mask NRMSE, RMSE, MAE, SSIM, NCC, and gradient/detail
   metrics;
7. retain background-noise and missed-anatomy measures as separately labelled
   QC, not the primary parameter-selection score; and
8. save CSV, JSON provenance, and plots.

The approved R3 pilot orientation mapping was identity axis permutation with
RAS-grid flips `[true, false, true]`. Revalidate rather than silently assume
that mapping for the new R1 acquisition or an older fallback scan.

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
requirements/retrospective_low_resolution_product.json
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

Evaluate the expected SNR/CNR gain together with the loss in sharpness and
spatial resolution. Preserve native-resolution images and create explicitly
labelled reference-grid resamples only for matched visual/metric comparisons.

The first manifested visual-review package now provides both representations:
native grids use the nearest slice to one shared RAS location with physical-mm
extents and no spatial resampling; matched grids use linear interpolation onto
the full-resolution 1 mm RAS grid. Both use display-only per-volume positive
p99.5 scaling. Neither DICOM nor BET participates. The initial attempt exposed
an older full-resolution LLR NIfTI that predated the canonical-RAS exporter;
that review is retained under a clearly rejected diagnostics directory. The
existing BART result was re-exported without reconstruction, and the review
code now rejects Wave NIfTIs lacking the corrected exporter sidecar contract.
Visual approval of the corrected native/matched package is the gate before
implementing quantitative sharpness and noise/contrast-proxy analysis.

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

## 10. Phase H: R1-to-R3 parameter transfer check

Apply the R1-selected parameter without retuning. Because this R3 dataset was
already used for preliminary optimization, describe this as a transfer check,
not an untouched independent validation. Compare the normalized
unfiltered product DICOM, no-wave GRAPPA, full-resolution R3x2 Wave, and
retrospective-LR R3x2 cases. Keep the earlier R3-specific optimum visibly
separate from the R1-selected result.

## 11. Presentation design gates

For the near-term presentation, prioritize a compact R3 panel: normalized
DICOM, no-wave GRAPPA, selected Wavelet result, corrected
LLR result if useful, masked metric curves, and restrained difference maps.
Label all selected values as provisional R3-specific parameters.

Before the later R1/final batch, agree on the expanded presentation package.
Proposed minimal outputs are:

1. one pipeline/data-role diagram;
2. a log-lambda plot for the R1 development and confirmation scans with brain
   NRMSE/SSIM and one sharpness/noise tradeoff measure;
3. fixed-slice Wavelet comparison panels with identical windowing;
4. an LLR heatmap only if corrected LLR proves scientifically useful;
5. a resolution-versus-SNR/sharpness plot for the three retrospective-LR
   cases plus 1 mm baseline;
6. native-resolution and matched-grid zoom panels for the resolution study;
7. one final R1-to-R3 transfer panel with normalized DICOM, no-wave
   alternatives, full-resolution R3x2 Wave, and selected LR result; and
8. restrained difference maps for the final candidates, not every sweep case.

Freeze slice positions, zoom boxes, display percentiles, metric definitions,
and ranking rules before generating presentation figures.

## 12. Final repository and output cleanup

Do this only after the experiments and presentation outputs are frozen.

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

The NIfTI orientation correction and retrospective low-resolution code
integration are complete. The latter uses the current BART-input/manifest
contract, crop-first target-grid Wave synthesis, mandatory GPU BART
reconstruction, norm restoration, resumable manifests, and the same canonical
RAS exporter. Product configuration validation, the real source-operator
gates, and all three production reconstructions pass. The corrected
native-grid and matched-1-mm visual review is generated and awaits user
approval. No accepted historical output was overwritten.

The retrospective resolution tradeoff analysis is complete and intentionally
makes no automatic selection. The first dataset-portability layer is also
complete: one validated manifest now carries geometry, acquired versus
synthetic-Wave sampling, paths, reconstruction settings, and evaluation policy;
the dataset inspector consumes it and records measured contract checks. Coil
compression and the accepted R3x1 GRAPPA entry point now consume the passed
contract; GRAPPA allocations are matrix-derived and an exact measured-sampling
gate prevents R1 misuse. The direct fully sampled no-Wave source path is also
implemented: it requires a centered complete readout and duplicate-free PE
grid, permits explicitly image-derived coil compression when no PAT refscan is
present, applies no interpolation, and creates resumable provenance-bound
k-space. The next implementation step is manifest propagation
through Wave synthesis and BART input export.
Use GRAPPA as the temporary metric reference, apply BET only during metrics,
and keep DICOM ranking as a configuration mode to enable on the new R1
dataset. Defer no-wave BART PICS and the older R1 partial-readout work.

## 14. Remaining readiness gaps for the incoming R1 dataset

The regularization engine, split-complex LLR handling, resumable manifests,
and core metric functions are already available. The remaining infrastructure
work is:

1. **NIfTI orientation at source — complete:** the shared Wave exporter now
   writes canonical RAS data/affines directly and no longer requires the
   R3-only manual signed-axis correction. GRAPPA and SENSE already use the same
   validated affine. Retrospective-LR consumes this shared exporter.
2. **Retrospective low resolution — code complete:** the imported tool consumes
   current BART manifests, crops only PE dimensions to four-multiple matrices,
   rebuilds the target PSF, runs BART on GPU, restores the target k-space norm,
   and exports with the same orientation contract. Product preparation,
   reconstruction, and manifested native/matched visual review are complete;
   the descriptive quantitative resolution-tradeoff analysis is also complete
   without selecting a winning resolution.
3. **Dataset portability — in progress:** the shared manifest, validator,
   example, hard-code audit, inspection integration, coil compression, and
   both direct R1 and compatible R3x1 GRAPPA source preparation are complete in
   code. Propagate the contract through Wave/BART export, regularized
   reconstruction, and evaluation to replace fixed
   `256 x 256 x 256`, R3 sampling-line, subject, path, DICOM-count, and
   maximum-eigenvalue assumptions with manifest or sequence/TWIX metadata.
4. **Incoming-data qualification and reference construction:** provide one
   audit entry point that records sampling completeness, readout center and
   oversampling, matrix/FOV, coils, ACS, sequence match, DICOM series/tags,
   and the chosen fully sampled no-wave reference construction.
5. **Configurable evaluation reference:** support temporary GRAPPA ranking now
   and DICOM/reference ranking later without changing metric code. BET is
   applied only for metrics. DICOM mode must remain disabled for the current
   R3 development run and enabled only after the new dataset is qualified.
6. **End-to-end acceptance gates:** retain PSF=1/no-wave identity,
   full-sampling Wave consistency, mask-after-Wave ordering, exact acquired
   sample preservation, lambda-zero reconstruction, scaling/norm restoration,
   orientation, and resumability checks in one dataset-level manifest.
