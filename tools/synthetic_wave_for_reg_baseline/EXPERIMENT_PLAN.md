# R3 presentation optimization, R1 refinement, and retrospective-resolution plan

Updated: 2026-08-20

This is the active experiment plan. The original R3x1-centered tracker is
preserved as a historical record in
`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`.

The execution order is now:

1. use the no-wave GRAPPA reconstruction from the existing R3x1 product scan
   as the single temporary development reference;
2. finish the reusable infrastructure needed by later stages: correct NIfTI
   orientation at export, integrate retrospective low-resolution support, and
   remove R3-specific geometry/configuration assumptions from the R1 path;
3. acquire and qualify the new R1 dataset; and
4. switch the final parameter-selection and DICOM-ranking workflow to that R1
   dataset, then apply the R1-selected settings back to R3 without retuning.

The first-stage R3 parameters are provisional and may be tuned to that
dataset. Only the later R1 development/confirmation experiment is intended to
select the more transferable final parameters. Results selected against the
temporary R3 reference must be labelled developmental and repeated after the
new R1 dataset is available.

### Decision recorded 2026-08-20: temporary references

The BART `ecalib -I` pilot did not correct the relevant receive-profile
mismatch. Its brain-core-to-shell median ratio was `0.895`, compared with
`0.966` for the previous Wave lambda-zero reconstruction and approximately
`0.975` for both no-wave GRAPPA and no-wave SENSE. The normalized and raw IDEA
DICOM ratios were `1.347` and `0.655`, respectively. Retain the pilot and its
manifest as a negative result, but do not use `ecalib -I` for the production
path or continue tuning regularization against either DICOM intensity profile.

Until the new R1 acquisition arrives, use no-wave GRAPPA as the single
temporary metric reference. No GRAPPA-versus-SENSE comparison or agreement
gate is required. Keep SENSE as an existing optional diagnostic only, and
defer the no-wave BART PICS regularization branch. Apply the approved BET mask
only while calculating metrics; do not use it to alter reconstruction or
display. DICOM voxel intensities must not enter provisional regularization
ranking, but the evaluator must retain a configurable DICOM-reference mode for
the new R1 dataset.

## 1. Scientific roles of the datasets

### New R1 data for final parameter refinement

A new R1 dataset will be collected and is the intended final baseline. On
arrival, inspect its sampling, readout completeness, DICOM processing, coil
configuration, geometry, and sequence definitions before reusing any current
assumptions. Assign independent development and confirmation scans when the
acquisition provides enough data.

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

## 9. Phase G: integrate retrospective low-resolution reconstruction

The existing untested repository is:

```text
/path/to/user_workspace/sources/published_code/wave-retro-lr-recon
```

Complete this integration before the new R1 regularization run so the same
case interface and manifests can be exercised on current data first:

1. run and audit its current tests and validate-only workflow;
2. merge its history into this repository under
   `tools/wave_retro_lr_recon/` rather than keeping a nested Git repository;
3. reuse the parent repository's pinned `external/wave-mprage` dependency;
4. replace its legacy internal-NPY/Torch-CG path with the current exported
   BART-input manifest contract and GPU `bart wave -g` reconstruction;
5. expose a small tested library/API plus a dataset-independent CLI;
6. call that CLI/API from the synthetic-Wave baseline workflow through a
   manifest, without copying its algorithms into the caller; and
7. preserve requested/achieved resolution, matrix, crop bounds, FOV, scaling,
   and source hashes for every case.

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
resolution when the requested matrix is not integral. Crop the no-wave source
in PE before Wave encoding and rebuild the target PE-grid PSF unless a focused
operator test establishes an equivalent ordering; do not assume cropping an
already Wave-encoded volume is equivalent.

Evaluate the expected SNR/CNR gain together with the loss in sharpness and
spatial resolution. Preserve native-resolution images and create explicitly
labelled reference-grid resamples only for matched visual/metric comparisons.

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

Fix NIfTI orientation at the export boundary so all reconstruction methods
write geometry-correct canonical RAS outputs without a downstream manual
signed-axis correction. Then integrate the retrospective low-resolution code
with the current BART-input/manifest contract and GPU BART reconstruction.

In parallel with those two concrete gaps, remove hard-coded R3 geometry,
dataset paths, and DICOM-only reference assumptions from the R1-facing
orchestration. Use GRAPPA as the temporary metric reference, apply BET only
during metrics, and keep DICOM ranking as a configuration mode to enable on
the new R1 dataset. Defer no-wave BART PICS and the older R1 partial-readout
work.

## 14. Remaining readiness gaps for the incoming R1 dataset

The regularization engine, split-complex LLR handling, resumable manifests,
and core metric functions are already available. The remaining infrastructure
work is:

1. **NIfTI orientation at source:** unify GRAPPA, BART lambda zero,
   regularized Wave, and retrospective-LR exporters around one tested
   geometry contract; write canonical RAS data/affines directly; and remove
   reliance on the R3-only manual signed-axis correction.
2. **Retrospective low resolution:** integrate the separate repository,
   consume current BART manifests, crop only PE dimensions, rebuild the target
   PSF, run BART on GPU, restore the target k-space norm, and export with the
   same orientation contract.
3. **Dataset portability:** replace fixed `256 x 256 x 256`, R3 sampling-line,
   subject, path, DICOM-count, and maximum-eigenvalue assumptions with values
   derived from a dataset manifest or sequence/TWIX metadata.
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
