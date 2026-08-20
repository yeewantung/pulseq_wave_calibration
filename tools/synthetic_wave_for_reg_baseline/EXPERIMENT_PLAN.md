# R3 presentation optimization, R1 refinement, and retrospective-resolution plan

Updated: 2026-08-20

This is the active experiment plan. The original R3x1-centered tracker is
preserved as a historical record in
`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`.

The execution order is now:

1. optimize on the existing R3x1 product dataset and generate near-term
   presentation results;
2. accept the 2021-05-10 R1 data as the best available baseline, validate and
   freeze its partial-Fourier completion, then perform the more rigorous
   regularization refinement; and
3. apply the better R1-selected parameters back to the R3 experiment as a
   cross-dataset transfer check.

The first-stage R3 parameters are provisional and may be tuned to that
dataset. Only the later R1 development/confirmation experiment is intended to
select the more transferable final parameters.

## 1. Scientific roles of the datasets

### R1 data for final parameter refinement

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

No better R1 dataset is currently available. The inspected 2026-06-01
MID00077 `pulseq_mprage_nowave_full` scan is nominally 256 x 256 x 192 but is
actually R2x2 in PE with a central 32 x 32 calibration region, so it cannot
replace the R1 baseline. Use the 2021-05-10 partial-readout scans after
freezing and validating one completion method. State this limitation
explicitly and use at least one independent R1 scan as confirmation. The
2021-05-10 DICOM series remain the orientation and qualitative references.

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

For each stage, compare:

1. normalized unfiltered R3x1 product DICOM;
2. R3x1 no-wave BART GRAPPA reconstruction;
3. R3x1 no-wave BART SENSE/PICS reconstruction, with regularization if useful;
4. R3x2 synthetic-Wave BART reconstruction with the appropriate provisional
   or R1-selected regularizer; and
5. the selected retrospective low-resolution R3x2 cases.

The future DICOM selector must require both `ND` and `NORM` and reject
`DIS2D`/`DIS3D`. The accepted normalized series UID is:

```text
1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0
```

The earlier evaluation against the non-normalized DICOM is a historical pilot,
not the future validation reference.

## 2. Reproducible environment

On `macha`:

```bash
source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
```

Use the `bart` resolved by `command -v bart`. All production BART
reconstructions should use `-g` unless a command does not support the GPU.

For FSL BET:

```bash
export FSLDIR=/path/to/software/packages/fsl/6.0.6
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

## 3. Phase A: preliminary R3x1 optimization and presentation

Proceed first with the existing R3x1 product and synthetic R3x2 Wave dataset;
do not wait for R1 partial-Fourier completion. Use the updated normalized,
unfiltered DICOM reference and the standardized whole-volume evaluation.

1. create and visually approve a fixed BET brain mask and L/R orientation;
2. continue the Wavelet search toward smaller lambda, since the pilot RMSE
   decreased as lambda decreased;
3. correct LLR to use the intended `wave -l -v` formulation before drawing
   conclusions about lambda or block size;
4. select provisional R3-specific candidates using masked metrics together
   with fixed-slice visual assessment;
5. compare normalized DICOM, no-wave GRAPPA/SENSE alternatives, and the
   selected synthetic R3x2 Wave reconstructions; and
6. generate a compact presentation package with fixed display settings and
   clearly label the tuning as preliminary and R3-dataset-specific.

Do not overwrite the accepted historical sweep. Store the new corrected and
masked-evaluation results in a separately manifested output tree.

## 4. Phase B: freeze the R1 partial-Fourier completion

This is the hard gate before the later R1 Wave synthesis and final parameter
refinement. The 2021-05-10 data are now the accepted best available R1
baseline; freeze their readout completion before using them for selection.

The R1 files have a nominal 512-sample oversampled readout grid but 404 stored
samples. The MDH k-space center is sample 148, so the acquired samples map to
indices 108:512 of the nominal 512-point grid. This is asymmetric/readout
partial-Fourier sampling, not a complete 202-point readout that should simply
be accepted as the final image matrix.

Implement and compare the following while preserving the MDH center:

1. embed the 404 samples into the nominal 512-point oversampled grid;
2. test zero-filled and homodyne/POCS partial-Fourier completion;
3. inverse FFT on the oversampled grid and crop the central 256-pixel FOV;
4. verify the equivalent ordering of partial-Fourier completion, 2x readout
   oversampling removal, and 202-to-256 Fourier interpolation;
5. compare geometry, sharpness, phase behavior, and normalized-DICOM agreement;
6. freeze one method and record all centering/crop indices in the manifest.

Do not crop the 404 raw k-space samples directly to 256. That would discard
high-frequency support and change the intended field of view/resolution.

Acceptance requires the chosen
development and holdout scans to produce a 256 x 256 x 192 reference with
correct anatomy and a verified L/R convention. The parameter-selection claim
must then be qualified as conditional on the frozen readout completion.

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

For each subject/scan:

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
that mapping for the older R1 scans.

## 9. Phase G: integrate retrospective low-resolution reconstruction

The existing untested repository is:

```text
/path/to/user_workspace/sources/published_code/wave-retro-lr-recon
```

After the main R1 regularization path is stable:

1. run and audit its current tests and validate-only workflow;
2. merge its history into this repository under
   `tools/wave_retro_lr_recon/` rather than keeping a nested Git repository;
3. reuse the parent repository's pinned `external/wave-mprage` dependency;
4. expose a small tested library/API plus a dataset-independent CLI;
5. call that CLI/API from the synthetic-Wave baseline workflow through a
   manifest, without copying its algorithms into the caller; and
6. preserve requested/achieved resolution, matrix, crop bounds, FOV, scaling,
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
unfiltered product DICOM, no-wave GRAPPA, no-wave SENSE/PICS, regularized
no-wave reconstruction, full-resolution R3x2 Wave, and retrospective-LR R3x2
cases. Keep the earlier R3-specific optimum visibly separate from the
R1-selected result.

## 11. Presentation design gates

For the near-term presentation, prioritize a compact R3 panel: normalized
DICOM, no-wave GRAPPA/SENSE alternatives, selected Wavelet result, corrected
LLR result if useful, masked metric curves, and restrained difference maps.
Label all selected values as provisional R3-specific parameters.

Before the later R1/final batch, agree on the expanded presentation package.
Proposed minimal outputs are:

1. one pipeline/data-role diagram;
2. a log-lambda plot for both R1 scans with brain NRMSE/SSIM and one
   sharpness/noise tradeoff measure;
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

Resume optimization on the existing R3x1/R3x2 experiment for presentation:
add the fixed BET mask, extend Wavelet toward smaller lambda, correct LLR with
`-l -v`, and generate the standardized comparison figures. After those
results are ready, implement and visually validate the 2021-05-10 asymmetric
readout completion, including zero-fill versus homodyne/POCS comparison,
before beginning final R1 parameter refinement.
