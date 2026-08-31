# Synthetic-Wave regularization handover

Updated: 2026-08-30, America/New_York

Read the applicable workspace/server `AGENTS.md`, this tool's `AGENTS.md`, and
then `EXPERIMENT_PLAN.md`. The plan is the active scientific and implementation
record; the old R3x1 tracker is historical only.

## Immediate next action

Start the final **source-repository cleanup** in a fresh session. The separate
R1 and R3 private-output cleanups are complete, indexed, checksum-validated,
and outside the remaining task. Do not move, delete, relink, regenerate, or
otherwise normalize those output trees while cleaning the repository. Do not
launch reconstruction, sweep, evaluation, presentation, or other production
jobs. The experiment and presentation deliverables are frozen.

Begin the repository task read-only:

1. read the applicable `AGENTS.md` files, this handover,
   `EXPERIMENT_PLAN.md`, `docs/r1_dataset_processing_todo.md`, and the private
   cleanup summary;
2. inspect the complete repository structure, tracked scripts, tests,
   documentation, references, Git state, and ignored `.local.*` files;
3. identify redundant entry points and implementation, historical or
   diagnostic code, stale documentation, and every downstream test, import,
   launcher, and documentation consumer;
4. present a concrete repository-only move/archive/removal map before changing
   structure or deleting files; and
5. after approved implementation, update imports, compatibility entry points,
   tests, README/script maps, and this handover; run both test suites,
   `git diff --check`, and tracked/staged private-path audits.

The previously proposed direction is to move workflow example JSON into
`configs/`, keep matching machine-local JSON ignored, factor shared sweep and
matched-grid evaluation engines behind thin compatible CLIs, archive truly
superseded pilot/evaluator code with its tests and provenance, and clearly
separate retained diagnostics from active entry points. Treat this as the
starting proposal, not permission to remove a file without checking all
consumers. Do not reset or discard changes, commit without review, or push
unless explicitly requested. Actual machine paths must remain only in ignored
`.local.sh`/`.local.json` files or private generated manifests.

## Completed private-output cleanup

Both private dataset roots now have a current root `OUTPUTS_INDEX.md` that is
the authority for retained paths, classifications, relocations, tombstones,
and recovery records.

- R1 cleanup consolidated outputs into the no-Wave R3x1, synthetic-Wave R3x1,
  synthetic-Wave R3x2, and low-resolution R3x2 project trees. Approved NIfTIs,
  PSFs, CSMs, masks, sampling masks, the canonical full-grid calibration, and
  compact provenance were retained. Approved rebuildable intermediates,
  diagnostics, rejected masks, and unnecessary JSON were deleted.
- R3 cleanup retained the fixed raw/DICOM and BART workspaces, organized the
  remaining projects under the same project-oriented scheme, converted all
  retained internal links to relative links, and preserved the current
  presentation deliverable without creating new NIfTI links.
- The superseded theoretical R3x1 output project was removed completely at the
  user's direction. Its binaries were not archived and require reconstruction
  if ever needed again; its inventory, deletion ledger, and provenance remain
  in the private cleanup archive.
- The final R3 deletion removed 373 regular files and one symlink. The measured
  net allocated-space reduction was 82,512,261,120 bytes. The retained R3
  scientific tree passed full-file hashing with 1,680 regular files and 516
  relative, resolving symlinks.
- The R3 recovery archive is `_archive/cleanup_20260830/` below the private
  output root. The complete cross-dataset decision record is the private
  `clean_up_summary.md`; do not copy its machine paths into tracked files.

## Frozen completion state

The synthetic-Wave R3x1 target, visual gate, coarse Wavelet/LLR sweep, Wavelet
refinement, and exact-grid evaluation are complete. The user selected Wavelet
`lambda=2.2e-2`. It has the lowest refined brain NRMSE (`0.0275119`), 3D SSIM
`0.985167`, brain NCC `0.996493`, gradient NCC `0.995822`, and edge ratio
`1.004955`. The hash-bound choice is recorded under
`synthetic_wave_r3x1/evaluation/direct_fft_wavelet_refinement/selection` with
`automatic_selection_performed=false`.

The presentation collection now includes exactly the two requested R3x1
views: solver-matched unregularized BART Wave FISTA lambda zero and Wavelet
FISTA `lambda=2.2e-2`. Their magnitude NIfTIs, central triplanar TIFFs, and
exact-grid direct-FFT metrics are included. No R3x1 reconstruction or
presentation follow-up remains.

The retrospective-resolution Wavelet and corrected-LLR sweeps and their fresh
matched-grid evaluations are complete. The superseded native-grid Wavelet
evaluation output was deleted during the approved R1 cleanup; its tracked
evaluator code remains a repository-cleanup classification candidate and must
not be mistaken for the current matched-grid path.
The explicit presentation choices are recorded in
`retrospective_low_resolution_selected_cases/selection_decision.json`:

- lower-X (`1.49x1x1 mm` achieved): Wavelet `lambda=1e-2`;
- lower-Y (`1x1.49x1 mm` achieved): Wavelet `lambda=5e-3`; and
- balanced (`1.25x1.25x1 mm` requested): unregularized FISTA `lambda=0`.

Those choices are presentation decisions, not automatic metric winners and
not a redefinition of the frozen full-resolution Wavelet `lambda=1.5e-2`.

The requested presentation magnitude collection is materialized with 28
available slots, all containing finite canonical-RAS magnitude NIfTIs.
Canonical sources were copied byte-for-byte and were not moved, resampled, or
cross-normalized.

The no-Wave R3x1 PICS CG-SENSE/Wavelet sweep and previous non-BART R3x2
PCG-SENSE reconstruction have completed. Each completed case saves the
standard exact-grid direct-FFT metric dictionary under
`direct_fft_metrics.metrics` in its manifest, using
the approved reference/mask with no registration or interpolation. These runs
have magnitude and phase NIfTIs in their reconstruction trees; only magnitude
is copied into the presentation collection. `presentation_metrics.csv` lists
all 28 entries with explicit metric status and comparison-grid scope.
The `orientation_slices` subdirectory contains 84 validated 16-bit TIFFs
(three orientations for each available NIfTI) plus a hash-bound manifest.
Standard cases use index 128; the six retrospective-resolution cases use the
center index of each physical RAS orientation independently.
The no-Wave R3x1 GRAPPA reconstruction and presentation refresh are complete.
It used the accepted joint-coil 5x5x5/Ncc=12 implementation on the same
retrospective R3x1-plus-ACS source contract, preserved every acquired sample
bitwise, and saved magnitude/phase plus direct-FFT metrics. Its brain NRMSE is
`0.046644` and 3D brain-bounding-box SSIM is `0.963763`. The collection,
metric CSV, and TIFF manifests were revalidated together with 28/28 available
entries, 28 metric rows, and 84 TIFFs. The added FISTA-lambda-zero controls
have matched direct-FFT and native same-solver descriptive metrics.

A dedicated no-Wave R3x1 sweep evaluation now plots all six Wavelet lambdas
with CG-SENSE and GRAPPA controls. Among tested values, `1e-3` leads brain
NRMSE (`0.038955`), 3D SSIM (`0.972682`), and edge-ratio closeness (`1.004790`),
while `1e-4` leads edge-gradient NCC (`0.993260`). The current presentation
value is now explicitly selected as `1e-3`; the former `1.5e-2` entry is worse
on all four plotted metrics and has been replaced only in the presentation
collection. Its canonical sweep reconstruction remains intact.

Do not confuse the no-Wave PICS-specific `1e-3` choice with either the frozen
full-resolution custom-Wave R3x2 `1.5e-2` result or the new per-resolution
follow-up sweep; the operators and target-grid norms differ. Independent
confirmation remains deferred; do not infer an automatic preferred
retrospective resolution.

The hash-bound operator gate is
`evaluation/full_sampling_wave_operator_validation/operator_validation_manifest.json`
below the private R1 dataset output root. Maximum relative complex L2 error is
`2.7991e-7` for `PSF=1` no-Wave identity and `3.0099e-7` for full-sampling Wave
inversion; maximum recovered exterior energy fraction is `4.7121e-14`. The
gate covers every virtual coil and uses neither BART reconstruction nor
presentation processing.

The review and analysis interfaces now accept path-agnostic JSON configuration
while preserving the historical product launch behavior. The R1 outputs are
under the private retrospective Wavelet tree in `visual_review` and
`resolution_tradeoff_analysis`. A separate `visual_approval.json` binds the
approved native and matched figures to their review manifest.

For lower-X, lower-Y, and balanced cases respectively, native total
edge-gradient ratios to full-resolution Wavelet are `0.9779/0.9773/0.9627`;
the signal/local-residual proxy is `21.01/21.33/21.47` versus `18.77` at full
resolution; matched brain NRMSE is `0.06236/0.06260/0.06552`; and mean axial
SSIM is `0.9265/0.9280/0.9150`. Expected directional losses are X `0.9227`, Y
`0.9164`, and balanced X/Y `0.9514/0.9463`. These are descriptive results, not
a selected winner. The approved direct-FFT BET remains metrics-only and exact
grid; fixed auxiliary supports derive from full-resolution Wavelet because
direct FFT noise fills the FOV. Background quantities are QC, not SNR/CNR.

The three R1 retrospective reconstructions completed successfully on
2026-08-21 local time. The batch and all case manifests report `complete`;
every reconstruction used frozen Wavelet `lambda=1.5e-2`, FISTA with 100
iterations and tolerance `1e-6`, a case-specific estimated maximum eigenvalue,
and BART GPU `-g`. Canonical-RAS magnitude/phase exports are finite. The
logical `(RO, LIN, PAR)` matrices are `256x256x172`, `256x172x256`, and
`256x204x204`; their physical XYZ NIfTI shapes are `172x256x256`,
`256x172x256`, and `204x204x256`. Achieved changed-axis resolutions are
`1.488372` mm and `1.254902` mm. The ignored local configuration identifies
the private output root; do not copy its absolute paths into tracked files.
The tracked entry point is `scripts/run_r1_retrospective_low_resolution.sh`;
copyable path-agnostic templates are
`scripts/run_r1_retrospective_low_resolution.example.sh` and
`requirements/retrospective_low_resolution_r1.example.json`. The populated
`.local.*` counterparts are intentionally ignored.

The R3 transfer review is complete and explicitly approved. Its manifest
records that the frozen regularized result is visibly smoother, no metrics
were calculated, and lambda was not retuned on R3. LLR remains a comparison
but is not selected.

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
now manifest-aware as well. The active R1 route copies the declared ACS
support from validated, compressed, fully sampled image k-space without
interpolation or repeated compression; the compatible refscan route remains
available. The R1 acquisition passed manifest-backed metadata inspection and
sample probing, and the lambda-zero runner is manifest-aware.
The portability audit in `docs/dataset_portability_audit.md` is reconciled:
the active R1 preparation, reconstruction, and direct-reference evaluation
route has no remaining manifest-consumer gap. Historical DICOM and partial-
readout branches remain separate and out of scope.

The active R1 dataset is referenced locally as `$R1_PRODUCT_ROOT`. Metadata inspection
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
source "$CONDA_SETUP"
conda activate "$SYNTHETIC_WAVE_CONDA_ENV"
source "$BART_STARTUP"
```

FSL BET:

```bash
: "${FSLDIR:?Set FSLDIR in the private local environment}"
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

Always use GPU (`-g`) for every BART reconstruction. Do not silently fall back
to CPU; stop and document any command-specific incompatibility.

The approved metrics-only reference mask uses FSL BET's robust center
estimation with fractional threshold `0.59`, followed by a one-voxel outward
dilation. The user visually approved this boundary and its L/R orientation.
The `0.55` expanded candidate was too large and the `0.60` expanded candidate
was slightly too small. Their rejection decisions remain recorded, while the
rejected diagnostic payloads were removed in the completed R1 cleanup. Do not
use BET for reconstruction or display.

## R3 presentation-optimization continuation state

After output cleanup, the isolated project is organized as:

```text
$R3_PRODUCT_ROOT/synthetic_wave_r3x2/presentation_optimization
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
The failed pre-fix CG attempt's rejection remains in provenance, but its large
diagnostic payload was deleted during the approved R3 cleanup.

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
directory supplied the masked reconstruction k-space and linked PSF without
duplicating the 1.5 GB calibration CFL. The retained compact negative-result
provenance is organized under:

```text
$R3_PRODUCT_ROOT/synthetic_wave_r3x1/ecalib_intensity_diagnostic
```

The user rejected `ecalib -I` as a solution to the profile mismatch. The
brain-core-to-shell ratio changed from `0.966` for the prior Wave lambda-zero
case to `0.895` with `-I`; no-wave GRAPPA and SENSE were both approximately
`0.975`. Its large rejected diagnostic intermediates were deleted during the
approved R3 cleanup. Do not use its maps for production or use either raw or
normalized DICOM intensities to rank the provisional regularization sweep.

## Repository state

Repository:

```text
$REPOSITORY_ROOT
```

Before this handover-only edit, the tracked worktree and index were clean at:

```text
ef14968 Complete retrospective sweeps and R3x1 presentation
```

The local `origin/main` tracking ref also pointed to `ef14968`; no fetch was
performed to verify the server, and no push was performed during output
cleanup. This `HANDOVER.md` update is the only intended tracked change for the
next session. Check `git status --short --branch` before working and do not
assume any additional modification or untracked file is disposable. Inventory
the complete diff and form cleanup commits without resetting the tree. The
historical tracker is under `docs/archive/`, and the active plan is
`EXPERIMENT_PLAN.md`.

## Dataset roles and baseline status

Original R1 candidates:

```text
$FALLBACK_R1_ROOT
```

- `meas_MID00792_FID57743_tfl_mprage_R1_Qiyuan.dat` / DICOM `0013`.
- `meas_MID00809_FID57760_tfl_mprage_R1_Qiyuan.dat` / DICOM `0019`.
- Both are 32-coil R1 scans with a full 256 x 192 PE grid but partial readout.
- Both DICOM series contain 192 normalized unfiltered `ND,NORM` images at
  1 mm isotropic resolution.

Additional R1 candidates inspected on 2026-08-20:

```text
$ADDITIONAL_R1_ROOT/subject6/
  meas_MID00786_FID58692_tfl_mprage_R1_Qiyuan.dat

$ADDITIONAL_R1_ROOT/subject7/
  meas_MID00805_FID58714_tfl_mprage_R1_Qiyuan.dat
```

Both are also 32-coil R1 scans with complete 256 x 192 PE coverage, but both
have the identical 404-sample, center-148 partial readout. They do not solve
the baseline problem. Subject 6 contains prior `.nii`/`.mat` reconstructions;
no DICOM files were found in either added folder during this inspection.

The fully sampled R1 dataset has been collected, qualified, reconstructed, and
used to freeze the MPRAGE Wavelet selection. These older partial-readout scans
are fallbacks only. Do not implement or freeze their completion unless the
fallback is explicitly reactivated.

An additional candidate inspected on 2026-08-20 was:

```text
$REJECTED_R1_ROOT/
  meas_MID00077_FID25178_pulseq_mprage_nowave_full.dat
```

Despite the name and an R1x1 protocol-header value, its counters show R2x2 PE
sampling with a central 32 x 32 calibration region. Its nominal image matrix
is 256 x 256 x 192, but it is not an R1 baseline and was rejected for that
role.

R3 preliminary optimization and later parameter-transfer dataset:

```text
$R3_PRODUCT_ROOT
```

Qualitative DICOM presentation source:

```text
$R3_PRODUCT_ROOT/mprage_product_unfiltered_normalize
```

Select the 256-image `ND,NORM` series UID
`1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0` and reject the
`DIS2D/DIS3D` series. Do not use its intensity profile for temporary
regularization ranking.

The historical R3 development data were tuned for preliminary presentation
results and are not an untouched holdout. The later application of the frozen
R1-selected parameter is therefore a qualitative cross-dataset transfer check,
not independent validation; that transfer is complete and must not be retuned.

## Historical fallback R1 readout finding

For all four older fallback R1 TWIX files:

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
are complete. Fully sampled R1 preparation, parameter refinement, the frozen
Wavelet decision, qualitative-only R3 transfer, and the R1 retrospective
reconstructions are also complete. The R1 retrospective native/matched visual
review and approval-gated descriptive resolution-tradeoff analysis are
complete. The source-operator checks now pass on all virtual coils; no further
active gate remains. Parameters from the earlier R3 development sweep remain
historical and R3-specific.

The initial R1 Wavelet coarse grid was:

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

The final R1 refinement and targeted evaluation selected and froze Wavelet
`lambda=1.5e-2` for full-resolution MPRAGE. LLR remains a documented
comparison and is not selected. The user has now explicitly authorized a
separate per-target-grid Wavelet sweep for the retrospective low-resolution
presentation cases; it does not reopen the full-resolution selection or the R3
qualitative transfer.

## Evaluation decisions

- The approved R1 quantitative reference is direct FFT RSS of the fully
  sampled, compressed no-Wave k-space. GRAPPA, SENSE, and DICOM are not R1
  ranking references.
- DICOM remains metadata/qualitative context only; Prescan Normalize and coil
  combination can change its intensity profile.
- Apply only the approved BET `f=0.59` plus one-voxel mask during metric
  calculation, never during reconstruction or display.
- The direct FFT reference and full-resolution candidates passed exact-grid
  geometry, so their selection metrics use no registration or interpolation.
- For retrospective low-resolution review, preserve native grids and use
  physical coordinates; matched-grid resampling is explicitly display-only.
- After visual approval, calculate descriptive resolution tradeoffs without a
  composite score or automatic resolution selection. BET remains metrics-only.
- Background noise and missed anatomy remain separately reported QC; do not
  claim true SNR/CNR from nearly zero BART air support.
- Canonical Wave NIfTI exports require RAS orientation and recorded affine-axis
  flips `[true,false,true]`.

## Retrospective low-resolution integration

The original two-commit repository history is merged under:

```text
tools/wave_retro_lr_recon/
```

The active tool is a tested library/CLI using the current BART manifest,
mandatory `bart wave -g`, target-k-space norm restoration, resumable case
manifests, and the shared canonical-RAS exporter. The baseline entry points are:

```text
requirements/retrospective_low_resolution_product.example.json
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
rejected review payload was deleted during the approved R1 cleanup after its
decision was recorded. The launcher re-exported the existing BART CFL through
the corrected shared NIfTI producer without reconstructing and stores that
canonical reference under `full_resolution_reference/`. The review code requires
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
because BART's reconstructed air background is nearly zero. Its decision is
recorded, while its diagnostic payload was deleted during the approved R1
cleanup. Background statistics remain QC only; the accepted summary uses a
fixed smooth-brain signal/local-residual proxy and explicitly does not call it
SNR.

The fully sampled R1 retrospective batch uses the same target matrices but the
frozen Wavelet `lambda=1.5e-2` configuration instead of the historical
corrected LLR setting. All three cases, the generalized review, explicit visual
approval, and descriptive analysis are complete. Backward compatibility is
preserved and private paths remain in ignored local files.

## Current R3 output organization

The R3 root is frozen after cleanup. Its project-oriented scientific areas are:

```text
$R3_PRODUCT_ROOT/synthetic_no_wave_r3x1
$R3_PRODUCT_ROOT/synthetic_wave_r3x1
$R3_PRODUCT_ROOT/synthetic_wave_r3x2
$R3_PRODUCT_ROOT/synthetic_wave_r3x2_LR
$R3_PRODUCT_ROOT/presentation
```

The complete 25-case R3x2 evaluation remains under the historical-pilot
project. Its LLR interpretation is limited by the missing `-v`, and its DICOM
predates the normalized reference. The accepted GRAPPA, presentation
optimization, regularization-transfer, low-resolution, and compact diagnostic
provenance are classified by the current root index:

```text
$R3_PRODUCT_ROOT/OUTPUTS_INDEX.md
```

The superseded theoretical R3x1 project and its compatibility symlink no longer
exist. Do not recreate them during repository cleanup. Do not alter the root
raw `.dat`/`.dcm` material, product DICOM directories, `mprage_bart`, or
`gre_bart`.

## Verification commands

```bash
python -m unittest discover \
  -s tools/synthetic_wave_for_reg_baseline/tests \
  -p 'test_*.py'

python -m unittest discover \
  -s tools/wave_retro_lr_recon/tests \
  -p 'test_*.py'

git status --short --branch
```

Run both the synthetic-Wave baseline suite and the retrospective-resolution
suite before handing off code changes. Record the observed test counts in the
commit or handoff message rather than preserving a count here that can become
stale.
