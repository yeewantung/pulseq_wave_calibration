# Synthetic-Wave regularization handover

Updated: 2026-08-21, America/New_York

Read `EXPERIMENT_PLAN.md` first. It is the active scientific and implementation
plan. The old R3x1 tracker is historical only.

## Immediate next action

Apply the frozen R1-selected Wavelet `lambda=1.5e-2` configuration unchanged to
the existing R3 dataset as a qualitative portability check; do not calculate
selection metrics or retune on R3. Run the ignored machine-local launcher
`scripts/run_r3_wavelet_transfer.local.sh` in tmux. The tracked public template
is `scripts/run_r3_wavelet_transfer.example.sh`, and the path-agnostic
implementation is `scripts/run_r3_wavelet_transfer.sh`. Private absolute paths
must appear only in the ignored local copy. The hash-bound selection record is
`evaluation/direct_fft_reference/regularization_selection/selection_manifest.json`
under the R1 dataset root. It identifies the exact reconstruction, direct-FFT
reference, 49-case geometry report, metric package, and explicit user decision.
LLR is retained as a comparison but is not selected.

The 11-case targeted follow-up is complete under
`reconstructions/synthetic_wave/regularization_targeted_ecalib_crop-0p6`, and
every reconstruction records `bart wave -g`.

The direct FFT RSS of the fully sampled NCC=12 no-Wave k-space is the approved
quantitative reference, and DICOM remains qualitative only. The approved
metrics-only brain mask is the
user-confirmed robust-center BET `f=0.59` result with a one-voxel outward
dilation, bound together with the reference in
`evaluation/direct_fft_reference/metrics_reference_manifest.json`.

The combined exact-grid gate passed all 49 retained and targeted cases at exact `256^3`, 1-mm,
canonical RAS geometry with zero affine difference and no registration or
interpolation. Its canonical report is
`evaluation/direct_fft_reference/geometry_validation/targeted_grid_geometry_validation.json`.
The hash-bound metric package is
`evaluation/direct_fft_reference/regularization_targeted_metrics`; it has 49
finite rows, eight plots, a block-size-by-lambda heatmap, and separate shared-
window reviews for Wavelet and each LLR block size. Candidate export scaling is
undone before a single mask-restricted LSQ scalar is fitted. There is no DICOM,
bias correction, histogram matching, composite score, or automatic selection.

Wavelet `1.5e-2` leads NRMSE (`0.033252`), intensity NCC, and edge-ratio
closeness (`1.001633`), while `2e-2` retains the highest SSIM (`0.979635`) and
`2e-3` the highest edge-gradient NCC (`0.995591`). For LLR, block 4 at `6e-3`
has its lowest NRMSE (`0.037453`) and highest SSIM (`0.974612`); block 16 at
`1e-2` remains the overall LLR NRMSE leader (`0.037416`). Both block 8 and 16
worsen in NRMSE and SSIM beyond `1e-2`, so their upper ranges are bracketed.
Use the common-window figures to resolve these trade-offs rather than treating
any one metric leader as the selected result.

The qualitative R1 receive-profile comparison is complete under
`comparisons/dicom_sos_normalization_vs_no_wave_fft`. Series 11 is the matched
SOS, Prescan-Normalize-off, unfiltered ND DICOM; series 9 is the SOS,
Prescan-Normalize-on, unfiltered ND DICOM. Both were converted to canonical
RAS and compared with direct FFT RSS of the fully sampled NCC=12 no-Wave
k-space. The figure uses independent positive-voxel p99 display scaling and no
registration, BET mask, or intensity ranking. Normalize-on shows the expected
strong central brightening; normalize-off is visually closer to direct FFT
RSS. The two DICOM affines match exactly. The FFT affine retains a documented
0.500 mm center-convention offset in two axes and was not silently resampled.
Series 13 is unfiltered ND, Adaptive Combine, and Normalize-on. It is included
as an additional qualitative comparison column. Its repeated per-frame
`CC:SoS` history token conflicts with Phoenix `ucCoilCombineMode=2`; dcm2niix
correctly decodes the latter as `Adaptive Combine`. Series 9 and 11 use mode 1
and decode as `Sum of Squares`. No ACC, Normalize-off MPRAGE is available.

The NIfTI orientation fix is complete. The shared Wave exporter now writes
canonical RAS data with affine-axis flips `[true,false,true]`, matching the
approved manual R3 correction exactly for both magnitude and phase. Its affine
matches the accepted GRAPPA/SENSE affine. No historical output was overwritten.

Retrospective low-resolution code support is complete with the current BART
manifest, crop-first target-grid Wave synthesis, and mandatory GPU
reconstruction path. Product structural validation and both real-data source
operator gates pass. All three corrected-LLR production cases and their
canonical-RAS exports are complete. A manifested native/matched visual-review
package and descriptive quantitative resolution-tradeoff analysis are
complete. No automatic resolution selection was made. Dataset-portability work
is proceeding consumer by consumer through the R1-facing orchestration.

The first dataset-portability layer is complete. A validated JSON contract now
separates acquired sampling from the synthetic-Wave target and records paths,
geometry, reconstruction settings, reference mode, mandatory GPU BART, and
metrics-only mask use. `inspect_product_dataset.py --dataset-manifest ...`
resolves the contract, records its hash/snapshot, and checks measured
TWIX/DICOM metadata. Coil compression and compatible R3x1 GRAPPA now consume
that passed, hash-matched inspection. GRAPPA derives its allocations from the
manifest and rejects non-R3x1 sampling. Direct fully sampled R1 source assembly
is now implemented as the mutually exclusive alternative: it requires complete
centered readout and PE support, can derive the coil basis from the image stream
when no PAT refscan exists, performs no interpolation, and produces resumable
compressed k-space with bound provenance. Manifest propagation through Wave
synthesis and BART input export is also complete. Full Wave encoding now
derives its allocation, trajectory settings, diagnostics, and provenance from
the contract. Its separate exporter applies the declared retrospective target
lattice and full-PE2 ACS band only after Wave encoding, validates every masked
sample, and leaves the accepted synthesis untouched. Measured ACS export is
now manifest-aware as well. The incoming R1 route copies the declared ACS
support from validated, compressed, fully sampled image k-space without
interpolation or repeated compression; the compatible refscan route remains
available. The new R1 acquisition is ready, its manifest-backed metadata
inspection and sample probe pass, and the lambda-zero runner is manifest-aware.
Remaining consumers are listed in `docs/dataset_portability_audit.md`.

The new R1 dataset is under
`/path/to/data/20260821_product`. Metadata inspection
selects MID00198 `t1_mprage_sag_p2.dat` as the actual fully sampled source:
256 cubed logical matrix, 64 coils, complete duplicate-free PE support, and no
refscan. MID00196 `pulseq151fix_mprage.dat` is measured R3x1 and must not enter
the direct R1 path. Prescan Normalize was enabled for the available DICOM, so
do not use DICOM intensities as baseline or ranking input. The concrete
manifest and passed inspection live under
`20260821_product_synthetic_wave_r1_ncc12_r3x2`; follow
`docs/r1_dataset_processing_todo.md`. The intended source path is image-derived
64-to-12 coil compression, direct k-space assembly, full Wave encoding, a
separate R3x2 target mask, and image-derived measured ACS.

## Environment on macha

```bash
source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
```

FSL BET:

```bash
export FSLDIR=/path/to/software/packages/fsl/6.0.6
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

Always use GPU (`-g`) for every BART reconstruction. Do not silently fall back
to CPU; stop and document any command-specific incompatibility.

The fixed reference mask uses FSL BET's robust center estimation with a
fractional threshold of `0.55`. The earlier non-robust `0.25` mask included face
and neck and is retained only under `diagnostics/rejected_bet_loose_mask_f025`.

## R3 presentation-optimization continuation state

The isolated output tree is:

```text
/path/to/data/20260817_product/
  synthetic_wave_grappa_5x5x5_ncc12_r3x2_presentation_optimization
```

It references the accepted R3x2 BART inputs and lambda-zero reconstruction by
symlink; it does not overwrite the historical sweep. The corrected block-8 LLR
pilot at lambda `2e-4` is complete there and differs from accepted lambda zero
by relative L2 `0.0061681`, confirming a non-degenerate penalty.

The first `wave -v -g` attempt exposed a GPU-unsafe direct pointer access in
BART's complex-decomposition forward operator. BART commit `60dceb33` replaces
that loop with device-aware multidimensional primitives. The production
lambda-zero FISTA equivalence gate then passed: recombined split-complex versus
native-complex relative L2 was `1.36985e-6`, below the frozen `1e-5` limit.
The failed pre-fix CG attempt is retained under the presentation tree's `diagnostics/`
directory.

`run_r3_presentation_optimization.sh` is resumable and always passes `-g`. It runs three
Wavelet values (`1e-6`, `1e-5`, `1e-4`) and the focused corrected block-8 LLR
range (`2e-5`, `5e-5`, `1e-4`, `2e-4`, `5e-4`). The accepted pilot is reused.
The QC and evaluation stages require explicit mask and L/R approval; there is
no automatic approval path. Both approvals were recorded and the presentation
package completed on 2026-08-20.

The completed isolated intensity-profile test is driven by
`scripts/run_ecalib_intensity_pilot.sh`. It calibrates crop-0.5 maps with BART
`ecalib -I`, runs only Wave lambda zero with `-g`, and then compares the result
with both IDEA DICOM variants and the existing no-wave GRAPPA/SENSE and Wave
outputs. Those existing reconstructions are comparison references only, not a
true baseline, and the pilot does not select a regularization parameter. The
measured ACS comes explicitly from the parent full-Wave BART inputs; the R3x2
directory supplies the masked reconstruction k-space and linked PSF without
duplicating the 1.5 GB calibration CFL. The default output tree is:

```text
/path/to/data/20260817_product/
  ecalib_intensity_c050_wave_lambda0
```

The user rejected `ecalib -I` as a solution to the profile mismatch. The
brain-core-to-shell ratio changed from `0.966` for the prior Wave lambda-zero
case to `0.895` with `-I`; no-wave GRAPPA and SENSE were both approximately
`0.975`. Retain this tree as a manifested negative result. Do not use its maps
for production or use either raw or normalized DICOM intensities to rank the
provisional regularization sweep.

## Repository state

Repository:

```text
/path/to/user_workspace/sources/published_code/pulseq_wave_calibration
```

The accepted R3x2 evaluation is commit:

```text
28f3f12 Add R3x2 regularization sweep evaluation
```

This cleanup/plan update may be uncommitted. Check `git status --short` before
working. The historical tracker was moved to `docs/archive/`, and the active
plan is `EXPERIMENT_PLAN.md`.

## Dataset roles and baseline status

Original R1 candidates:

```text
/path/to/data/2021_05_10_bay4_mprage_R1_subjects2and3
```

- `meas_MID00792_FID57743_tfl_mprage_R1_Qiyuan.dat` / DICOM `0013`.
- `meas_MID00809_FID57760_tfl_mprage_R1_Qiyuan.dat` / DICOM `0019`.
- Both are 32-coil R1 scans with a full 256 x 192 PE grid but partial readout.
- Both DICOM series contain 192 normalized unfiltered `ND,NORM` images at
  1 mm isotropic resolution.

Additional R1 candidates inspected on 2026-08-20:

```text
/path/to/data/2021_05_14_bay4_subject6/
  meas_MID00786_FID58692_tfl_mprage_R1_Qiyuan.dat

/path/to/data/2021_05_14_bay4_subject7/
  meas_MID00805_FID58714_tfl_mprage_R1_Qiyuan.dat
```

Both are also 32-coil R1 scans with complete 256 x 192 PE coverage, but both
have the identical 404-sample, center-148 partial readout. They do not solve
the baseline problem. Subject 6 contains prior `.nii`/`.mat` reconstructions;
no DICOM files were found in either added folder during this inspection.

The user is collecting a new R1 dataset for final refinement. These older
partial-readout scans are now fallbacks only. Do not implement or freeze their
completion unless the new acquisition fails qualification and the fallback is
explicitly reactivated.

An additional candidate inspected on 2026-08-20 was:

```text
/path/to/data/20260601_cimax_mprage_invivo/
  meas_MID00077_FID25178_pulseq_mprage_nowave_full.dat
```

Despite the name and an R1x1 protocol-header value, its counters show R2x2 PE
sampling with a central 32 x 32 calibration region. Its nominal image matrix
is 256 x 256 x 192, but it is not an R1 baseline and was rejected for that
role.

R3 preliminary optimization and later parameter-transfer dataset:

```text
/path/to/data/20260817_product
```

Qualitative DICOM presentation source:

```text
/path/to/data/20260817_product/mprage_product_unfiltered_normalize
```

Select the 256-image `ND,NORM` series UID
`1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0` and reject the
`DIS2D/DIS3D` series. Do not use its intensity profile for temporary
regularization ranking.

The R3 data may be tuned now for preliminary presentation results. Since it
will no longer be an untouched holdout, the later application of R1-selected
parameters must be called a cross-dataset transfer check rather than an
independent validation.

## R1 readout finding

For all four inspected R1 TWIX files:

```text
nominal oversampled Fourier columns: 512
stored MDH samples per line:          404
MDH center column:                    148
readout OS factor:                    2
nominal final columns:                256
```

Mapping input center 148 to nominal center 256 places the acquired samples at
nominal indices 108:512. This is one-sided/asymmetric readout support. The
existing inspector's `flagRemoveOS=True` result of 202 samples is not itself a
valid final 256-pixel reconstruction. Preserve the 404 samples and the center
metadata; test partial-Fourier completion and crop the oversampled image FOV.

## Frozen regularization decisions

NIfTI orientation correction and retrospective low-resolution code integration
are complete. Remaining execution order is the user-run retrospective batch,
dataset-portability work, and then qualification/refinement on the new R1
acquisition. Parameters selected during R3 development are provisional and
R3-specific.

Wavelet coarse sweep after the new R1 development scan is qualified:

```text
lambda = 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
```

Correct LLR before rerunning it. Existing effective commands used `wave -l`
without `-v`. BART only creates the size-two real/imaginary ITER dimension
with `-v`; without it, the low-rank matrix has a singleton column dimension.
First validate `-l -v`, output recombination, and lambda-zero equivalence.

The custom `bart wave` internally divides input k-space by its global L2 norm
and does not expose PICS `-w`/`-S`. Record that norm and restore output scale in
the wrapper when needed. `-e` is maximum-eigenvalue/step-size information, not
intensity scaling.

## Evaluation decisions

- During temporary R3 development, use no-wave GRAPPA as the single metric
  reference.
- Keep DICOM intensity metrics out of provisional regularization selection.
- Preserve a configurable DICOM/reference mode for the incoming R1 dataset.
- Apply the approved DICOM-derived FSL BET mask only during metric
  calculation, not reconstruction or display.
- Visually approve mask and L/R orientation.
- Estimate one proper rigid transform from lambda zero and reuse it unchanged.
- Primary ranking uses GRAPPA-referenced brain-mask structural/error/detail
  metrics during temporary development.
- Background noise and missed anatomy remain separately reported QC.
- The approved historical R3 orientation used permutation `[0,1,2]` and RAS
  flips `[true,false,true]`; revalidate on R1.

## Retrospective low-resolution integration

The original two-commit repository history is merged under:

```text
tools/wave_retro_lr_recon/
```

The active tool is a tested library/CLI using the current BART manifest,
mandatory `bart wave -g`, target-k-space norm restoration, resumable case
manifests, and the shared canonical-RAS exporter. The baseline entry points are:

```text
requirements/retrospective_low_resolution_product.json
scripts/run_retrospective_low_resolution.sh
```

The real source PSF identity gate passed with relative complex L2
`4.406439186e-08`; the all-coil native Wave-operator gate passed with relative
L2 `2.150150032e-07`. Structural validation creates no output and has passed.
The three product reconstructions completed on 2026-08-21 with corrected LLR,
block size 8, lambda `2e-5`, 100 iterations, and GPU `-g`.

Never crop readout. Use these requested physical XYZ resolutions:

```text
1.5 x 1.0 x 1.0 mm
1.0 x 1.5 x 1.0 mm
1.25 x 1.25 x 1.0 mm
```

For sagittal MPRAGE, physical X and Y are the two PE directions and physical Z
is readout. Keep physical Z/readout at 1 mm and its matrix unchanged. Round
each PE matrix to the nearest multiple of four. The product cases therefore
use logical `(RO, LIN, PAR)` matrices `256 x 256 x 172`, `256 x 172 x 256`,
and `256 x 204 x 204`, achieving changed-axis resolutions `1.488372` mm and
`1.254902` mm. Crop no-wave PE support before Wave encoding and rebuild the
target PE-grid PSF; do not crop already Wave-encoded k-space.

The physical-coordinate review entry point is:

```text
scripts/run_retrospective_low_resolution_review.sh
```

It compares no-wave GRAPPA, the full-resolution corrected LLR result, and all
three LR cases. The native figure retains each matrix and selects slices by RAS
millimetres; the matched figure linearly resamples LR magnitudes to the 1 mm
full-resolution grid. Display uses independent positive p99.5 scaling and is
not an intensity/SNR comparison. BET and DICOM are not used.

This review caught the historical full-resolution LLR NIfTI orientation. The
old review is retained under
`diagnostics/rejected_visual_review_historical_orientation_reference`. The
launcher re-exported the existing BART CFL through the corrected shared NIfTI
producer without reconstructing and stores that canonical reference under
`full_resolution_reference/`. The review code requires
`NIfTICanonicalRAS=true` and affine flips `[true,false,true]` in every Wave
export sidecar, preventing reuse of the historical file.

The quantitative entry point is:

```text
scripts/run_retrospective_low_resolution_analysis.sh
```

Its canonical output is `resolution_tradeoff_analysis/`. The approved BET mask
is transferred into untouched reconstruction space using the frozen shared
rigid transform and is used only for metrics. Native-grid gradients use
physical-mm spacing; matched fidelity uses linear interpolation to the 1 mm
full-resolution LLR grid. DICOM intensities, candidate-specific registration,
true-SNR/CNR claims, composite ranking, and automatic selection are excluded.

Key lower-X/lower-Y/balanced results are: native total edge-gradient ratios
`0.981/0.980/0.964`; smooth-region signal/local-residual proxies
`15.22/15.57/15.67` versus `13.71` at full resolution; matched brain NRMSE
`0.0663/0.0695/0.0719`; and mean axial SSIM `0.916/0.911/0.896`. Directional
edge ratios behave as expected, with X `0.924` in the lower-X case, Y `0.917`
in the lower-Y case, and X/Y `0.953/0.946` in the balanced case. These are
descriptive tradeoffs, not a selected winner.

The first background-based apparent-SNR summary was scientifically rejected
because BART's reconstructed air background is nearly zero. It is preserved
under `diagnostics/rejected_resolution_analysis_background_noise_proxy`.
Background statistics remain QC only; the accepted summary uses a fixed
smooth-brain signal/local-residual proxy and explicitly does not call it SNR.

## Accepted historical R3 outputs

Accepted R3x1 5x5x5-derived synthetic Wave tree:

```text
/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12
```

Accepted R3x2 sweep/evaluation:

```text
/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12_r3x2
```

The complete 25-case R3x2 evaluation is retained as a historical pilot. Its
LLR interpretation is limited by the missing `-v`, and its DICOM predates the
normalized reference. Do not overwrite it.

Canonical accepted GRAPPA 5x5x5 results after cleanup:

```text
/path/to/data/20260817_product/grappa_5x5x5_ncc12
```

The dataset-level canonical/historical inventory and deletion record is:

```text
/path/to/data/20260817_product/OUTPUTS_INDEX.md
```

The old manifest path below is retained as a compatibility symlink:

```text
.../synthetic_wave_theoretical_ncc12/grappa_diagnostics
```

Do not alter `mprage_bart` or `gre_bart`.

## Verification commands

```bash
python -m unittest discover \
  -s tools/synthetic_wave_for_reg_baseline/tests \
  -p 'test_*.py'

git status --short
```

The last complete run before this documentation/cleanup pass had 68 passing
tests.
