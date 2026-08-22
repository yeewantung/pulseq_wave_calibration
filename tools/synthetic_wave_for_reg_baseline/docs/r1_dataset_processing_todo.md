# R1 dataset processing to-do

Updated: 2026-08-21

This checklist covers the raw-data-first synthetic R3x2 Wave experiment under
the private dataset root supplied by the local environment:

```text
$R1_PRODUCT_ROOT
```

The approved quantitative baseline is direct FFT root-sum-of-squares (RSS) of
the fully sampled no-Wave data after the frozen 64-to-12 coil compression. It
is derived from the same raw acquisition without GRAPPA, SENSE, Wave encoding,
or regularization. DICOM remains available for metadata and qualitative review,
but its voxel intensities must not enter parameter selection.

The active next gate is retrospective low-resolution visual review. Generalize
the historical product-specific review for the R1 Wavelet study, generate
native physical-coordinate and explicitly matched-grid figures, and obtain
user approval before calculating descriptive resolution-tradeoff metrics.

## Frozen source decision

- [x] Use the fully sampled no-Wave source:
  `meas_MID00198_FID90829_t1_mprage_sag_p2.dat`.
- [x] Do not use `meas_MID00196_FID90827_pulseq151fix_mprage.dat` as the R1
  source. Despite its header acceleration fields, its measured image counters
  contain only 85 PE1 lines with stride/residue `3/[2]`; it is R3x1 and its
  post-Siemens-OS readout is 512 rather than 256.
- [x] Do not run GRAPPA or SENSE to prepare the fully sampled source.
- [x] Compress the measured image stream from 64 physical coils to 12 virtual
  coils before synthetic Wave encoding.
- [x] Derive measured ESPIRiT ACS from the compressed image stream because the
  selected R1 measurement has no refscan.

## Verified preflight facts

For MID00198 measurement index 1:

- [x] matrix: 256 readout x 256 PE1 x 256 PE2;
- [x] sampling: 65,536 unique PE coordinates, no duplicates, complete PE1 and
  PE2 support, measured stride 1;
- [x] readout: 512 acquired samples, center column 256, and 256 samples after
  Siemens twofold oversampling removal;
- [x] receiver channels: 64;
- [x] refscan: absent;
- [x] raw file size: 17,482,650,624 bytes.

The candidate theoretical Wave sequence is:

```text
$SEQUENCE_ROOT/
  mprage_3d_wave_FOV256x256x256_res1x1x1_ETL256_R1-1_R2-3_
  os4_amp8_cyc10_SAG_prisma_v151.seq
```

Its definitions match 256 mm isotropic FOV, 256 cubed logical matrix, 1024
extended readout samples, sagittal orientation, 8 mT/m Wave amplitude, and 10
cycles. Its nominal acquisition definition is 1x3; that does not define this
experiment's retrospective R3x2 mask. The target mask must be applied
separately and only after full Wave encoding.

## Before large output generation

- [x] Extend the dataset contract to support `ranking_reference.kind: none`
  and disabled DICOM inspection/ranking. Do not insert a placeholder DICOM or
  GRAPPA reference merely to satisfy the schema.
- [x] Create a concrete dataset manifest for MID00198 with a descriptive output
  root outside the repository, such as `synthetic_wave_r1_ncc12_r3x2`.
- [x] Record the candidate Wave sequence path and its inspected identity.
- [x] Freeze logical axis mapping and confirm that target R3x2 means the
  intended TWIX PE1/PE2 axes.
- [x] Freeze the retrospective mask residues and ACS support. The current
  candidate is acceleration `[3,2]`, residues `[1,0]`, and PE1 ACS `[115,139)`.
- [x] Run manifest-backed metadata inspection and one small image-stream sample
  probe. Require complete centered readout, complete PE support, 64 coils, and
  no duplicate coordinates.
- [x] Choose a distinct, non-historical output root; never overwrite the R3
  results or any accepted dataset tree.

## Coil compression and direct source

- [x] Estimate the coil basis from the fully sampled image stream in bounded
  PE2 chunks, retaining the first 12 virtual coils.
- [x] Review cumulative covariance energy, basis orthogonality, finite probes,
  and input/output dimensions before continuing.
- [x] Directly assemble `[RO, PE1, PE2, coil] = [256,256,256,12]` complex64
  k-space with the accepted basis. Apply no interpolation, GRAPPA, SENSE, or
  partial-Fourier completion.
- [x] Verify every source chunk is finite, the complete grid is nonzero, and
  restart provenance matches the TWIX identity and coil-basis hash.
- [x] Export direct FFT RSS from the fully sampled NCC=12 no-Wave k-space.
- [x] Approve that direct FFT RSS volume as the quantitative reference for
  regularization metrics; no GRAPPA, SENSE, or DICOM baseline is required.

## Synthetic Wave R3x2 preparation

- [x] Generate the theoretical PSF from the frozen sequence and Wave settings.
- [x] Wave-encode the complete 12-channel source before applying any sampling
  mask.
- [x] Review the full-Wave direct-IFFT magnitude and phase diagnostics.
- [x] Apply the frozen R3x2 Cartesian lattice plus full-PE2 ACS band after Wave
  encoding.
- [x] Verify acquired samples are bitwise identical to full Wave k-space and
  all omitted samples are exact complex zero.
- [x] Export measured ACS from the direct compressed R1 image source, without
  repeated compression, and validate its exact-zero exterior.

## Reconstruction gates

- [x] Finish manifest support in the lambda-zero runner before launching it.
- [x] Require BART from the active local environment and use `-g` for every
  `bart wave` reconstruction.
- [x] Estimate the initial dataset-specific ESPIRiT map set from measured ACS.
  Do not
  use `ecalib -I` as the production solution; the prior pilot was negative.
- [x] Run synthetic R3x2 Wave lambda zero and restore BART's recorded internal
  k-space normalization.
- [x] Visually review the initial lambda-zero reconstruction made with ESPIRiT
  crop `0.8`; it is acceptable apart from testing a smaller map crop.
- [x] Repeat ESPIRiT calibration and lambda zero with crop `0.6` in the separate
  `ecalib_crop-0p6_lambda0` result directory. The run is finite, uses
  `bart wave -g`, retains the same measured maximum eigenvalue (`6.70e7`), and
  differs from crop `0.8` lambda zero by relative L2 `0.033998`.
- [x] Visually approve the crop `0.6` map montage and lambda-zero quicklook,
  then freeze those maps and lambda-zero output for every regularized candidate.
- [ ] Check finite outputs, map support, data consistency, orientation/LR,
  scaling, central slices, and full-volume geometry before regularization.
- [ ] Add full-sampling and PSF=1 operator checks where they can be performed
  without conflating reconstruction and presentation processing.

## Reference decision and DICOM review

- [x] Inventory the added IDEA offline DICOM reconstructions, series 10-14.
  Their repeated per-frame history token reports `CC:SoS`, but that token is
  not the authoritative final coil-combination label: dcm2niix decodes Phoenix
  `ucCoilCombineMode=1` as Sum of Squares and mode `2` as Adaptive Combine.
  Series 11 and 12 omit `NormalizeAlgo:PreScan`, whereas series 10, 13, and 14
  include it. Series 11 is the unfiltered SOS, Normalize-off candidate.
- [x] Keep all DICOM series out of metric calculation and regularization
  ranking. Their intensity processing no longer blocks quantitative evaluation.
- [x] Qualify a matched pair for qualitative receive-profile comparison only:
  series 11 is SOS, Prescan Normalize off, unfiltered ND; series 9 is SOS,
  Prescan Normalize on, unfiltered ND. Both contain 256 finite 256x256 frames,
  share the acquisition, study, frame of reference, protocol, and protocol
  coil-combine mode, and are converted to canonical RAS NIfTI. Series 10, 12,
  and 14 contain `DIS2D`/`DIS3D` distortion-filter markers. Series 13 is
  unfiltered ACC with Normalize on and is included as an additional qualitative
  comparison column. No ACC, Normalize-off MPRAGE is currently available.
  This does not enable DICOM ranking.
- [x] Preserve the audited DICOM series identities and reconstruction metadata
  for qualitative interpretation without changing the metric reference to a
  DICOM series.

## Regularization refinement and evaluation

- [x] Make regularized Wave reconstruction consume the same dataset contract
  and measure any dataset-specific maximum eigenvalue rather than copying R3.
- [x] Run the frozen GPU-FISTA Wavelet coarse sweep after lambda zero passes:
  solver-matched lambda zero plus `1e-6`, `1e-5`, `1e-4`, `1e-3`, and `1e-2`.
  All cases are complete, finite, canonical RAS, and bound to the approved
  crop-`0.6` maps and measured maximum eigenvalue.
- [x] Run the urgent compact block-8 LLR sweep with the verified `-l -v -g`
  path: solver-matched lambda zero plus `2e-5`, `1e-4`, and `5e-4`. The
  split/native FISTA lambda-zero gate passed at relative L2 `2.73366e-6`, and
  all cases are complete, finite, canonical RAS, and manifest-backed.
- [x] Register the direct FFT RSS volume and approved `f=0.59` mask in
  `evaluation/direct_fft_reference/metrics_reference_manifest.json`. This
  evaluation overlay records their hashes and preserves the immutable dataset
  manifest hash already bound to completed reconstructions.
- [x] Verify that the direct FFT reference and every completed coarse candidate
  have identical
  shape, voxel size, affine, and RAS axis convention. Because they arise from
  the same source grid, calculate metrics without image registration or
  interpolation; stop if the geometry differs. The manifested gate passed all
  10 cases with exact `256^3`, 1-mm RAS geometry and zero affine difference.
- [x] Reject the first expanded candidate made with robust-center BET threshold
  `0.55` plus one face-connected voxel: its 1,924,995-voxel boundary was judged
  too large. Preserve it for provenance and do not use it for metrics.
- [x] Reject the tighter candidate from the direct FFT reference using
  robust-center BET threshold `0.60`, followed by the same one-voxel outward
  margin. Its 1,752,607-voxel boundary was judged slightly too small; preserve
  it for provenance and do not use it for metrics.
- [x] Generate an intermediate candidate using robust-center BET threshold
  `0.59` plus the same one-voxel margin. It contains 1,818,144 voxels: 3.74%
  larger than the `0.60` candidate and 5.55% smaller than the `0.55` candidate.
  Its manifest is now `approved_for_metrics`.
- [x] Obtain explicit user visual approval of the `f=0.59` expanded mask
  boundary and labeled L/R orientation before calculating any metrics. Approval
  was received on 2026-08-21; only this mask may be used for the R1 metrics.
- [x] Evaluate the completed coarse Wavelet and block-8 LLR cases against the
  direct FFT reference. Keep the solver-matched lambda-zero cases as operator
  and convergence controls, not as the quantitative ground truth. The
  hash-bound package contains 10 cases, fixed-mask metrics, and common-window
  reviews without registration, interpolation, DICOM, or automatic selection.
- [x] For scale-dependent metrics, fit one documented scalar per candidate
  within the fixed reference mask. Use reference-derived scaling/windowing for
  SSIM and figures; do not use candidate-specific histogram matching or bias
  correction. Each NIfTI export normalization is first undone from its sidecar,
  and the fitted LSQ scalar is recorded in the metric table.
- [x] Refine Wavelet around the `1e-3` to `1e-2` trade-off by adding `2e-3`
  and `5e-3`; add outward endpoints `2e-2` and `5e-2` because NRMSE, SSIM, and
  intensity NCC still improved at the tested `1e-2` boundary while edge-gradient
  NCC peaked at `1e-3`. All four missing cases completed successfully.
- [x] Refine LLR over block sizes `4`, `8`, and `16` using the common lambda
  grid `2e-5`, `5e-5`, `1e-4`, `2e-4`, `5e-4`, `1e-3`, `2e-3`, `5e-3`, and
  `1e-2`. The higher points bridge rather than jump from the improving `5e-4`
  boundary. All missing cases completed, while the retained block-8 cases were
  reused without reconstruction.
- [x] Add the resumable tmux launcher `run_regularization_refinement.sh` for the
  28 missing cases. It sources the Macha environment and host BART build,
  requires the approved coarse-metrics provenance, and delegates every run to
  the manifest-backed GPU-only reconstruction entry point.
- [x] Combine the retained coarse cases and completed refinement manifests,
  then pass the exact-grid geometry/provenance gate for all 38 cases. The
  canonical report is
  `geometry_validation/refined_grid_geometry_validation.json`; all candidates
  are exact `256^3`, 1-mm RAS matches with no registration or interpolation.
- [x] Evaluate the combined grid with fixed-mask whole-volume metrics and a
  common reference-derived visual window. The canonical 38-row package is
  `regularization_refinement_metrics`, with metric definitions, per-metric
  leaders, input/output hashes, a block-size-by-lambda heatmap, and separate
  common-window reviews for Wavelet and each LLR block size. It deliberately
  performs no composite ranking or automatic selection.
- [x] Confirm the refined endpoint behavior. Wavelet `2e-2` leads NRMSE, SSIM,
  intensity NCC, and edge-ratio closeness, while gradient NCC peaks at `2e-3`;
  `5e-2` worsens the fidelity metrics and therefore brackets the Wavelet range.
  LLR has split leaders: block 16 at `1e-2` has the lowest NRMSE and closest
  edge ratio, block 4 at `5e-3` has the highest SSIM, and block 4 at `2e-3` has
  the highest gradient NCC.
- [x] Add the resumable `run_regularization_targeted_sweep.sh` launcher for the
  small grid requested after combined-metric review. It runs Wavelet `1.5e-2`;
  block-4 LLR `3e-3`, `4e-3`, `6e-3`, and `7.5e-3`; and block-8 and block-16
  LLR `1.5e-2`, `2e-2`, and `3e-2`. All 11 cases use the frozen crop-`0.6`
  maps and the GPU-only reconstruction entry point.
- [x] Run the 11-case targeted sweep and combine its manifests with the 38
  retained cases. The 49-case exact-grid gate passed with identical `256^3`,
  1-mm RAS geometry and no registration or interpolation. The new report is
  `geometry_validation/targeted_grid_geometry_validation.json`, and the
  hash-bound evaluation is `regularization_targeted_metrics`.
- [x] Confirm the targeted endpoints. Wavelet `1.5e-2` now has the lowest NRMSE
  (`0.033252`), highest intensity NCC, and closest edge ratio, while `2e-2`
  retains the highest SSIM (`0.979635`). Block-4 LLR NRMSE and SSIM peak at
  `6e-3`. Block-8 and block-16 NRMSE/SSIM both worsen above `1e-2`, so their
  useful ranges are now bracketed; block 16 at `1e-2` retains the lowest LLR
  NRMSE (`0.037416`).
- [x] Select and freeze Wavelet `lambda=1.5e-2` as the MPRAGE regularization
  choice after quantitative and common-window review. LLR remains a documented
  comparison and is not selected. The hash-bound decision record is
  `evaluation/direct_fft_reference/regularization_selection/selection_manifest.json`.
- [x] Apply the frozen Wavelet `lambda=1.5e-2` configuration unchanged to R3 as
  a qualitative cross-dataset transfer check. Do not calculate R3 selection
  metrics or retune against the historical R3 DICOM or GRAPPA results.
- [x] Add the path-agnostic tracked R3 transfer runner and copyable example
  launcher. Keep the real server paths only in
  `run_r3_wavelet_transfer.local.sh`, which is explicitly ignored by Git.
- [x] Run the ignored local launcher in tmux and obtain visual confirmation of
  the common-window FISTA-zero versus frozen-Wavelet transfer figure. The user
  approved the smoother regularized result, and the hash-validated review
  manifest is `qualitative_transfer_approved`.

## R1 retrospective low resolution

- [x] Add a tracked path-agnostic runner, copyable launcher/configuration
  examples, and ignored machine-local files for the R1 study.
- [x] Bind the configuration to the frozen selection manifest and require
  Wavelet `lambda=1.5e-2`, FISTA, 100 iterations, tolerance `1e-6`, and BART
  GPU `-g`; estimate maximum eigenvalue separately for each target matrix.
- [x] Pass non-writing structural validation for `256x256x172`,
  `256x172x256`, and `256x204x204` logical matrices.
- [x] Run the three R1 cases in tmux and retain their completed manifests. The
  batch and all case manifests report `complete`; all cases used frozen
  Wavelet `lambda=1.5e-2`, FISTA 100 iterations, tolerance `1e-6`,
  case-specific maximum-eigenvalue estimation, and BART GPU `-g`. Their
  logical `(RO, LIN, PAR)` matrices are `256x256x172`, `256x172x256`, and
  `256x204x204`; the canonical-RAS physical XYZ NIfTI shapes are
  `172x256x256`, `256x172x256`, and `204x204x256`.
- [ ] Generalize the historical product GRAPPA/corrected-LLR review interface
  for the R1 direct-FFT reference, full-resolution frozen Wavelet result, and
  three low-resolution Wavelet results. Keep real paths in ignored local
  configuration or launcher files and preserve historical compatibility.
- [ ] Generate and visually approve native-grid and matched-grid R1 review
  figures before calculating descriptive resolution-tradeoff metrics. Native
  panels must preserve each matrix and use physical RAS coordinates; matched
  panels must label display-only interpolation explicitly. Use neither BET nor
  DICOM for the visual review.
- [ ] After visual approval, generalize and run descriptive R1
  resolution-tradeoff analysis. Use the approved `f=0.59` plus one-voxel BET
  mask only for metrics, avoid true-SNR/CNR claims, and perform no composite
  ranking or automatic resolution selection.

## Repository privacy and reproducibility

- [x] Record the decision that R3 is rerun for qualitative visual transfer
  assessment only; it must not provide metrics or retune the R1-selected lambda.
- [x] Keep tracked launchers path-agnostic and provide copyable examples for
  machine-local environment and dataset paths.
- [x] Convert the retrospective low-resolution requirement to a tracked
  `.example.json` plus ignored `.local.json` configuration.
- [x] Remove private absolute paths and server startup locations from the
  current tracked source, documentation, tests, and configuration.
- [x] Rewrite all 79 reachable commits after creating a verified recovery
  bundle; the full-history audit contains zero private-path or identity matches.
- [x] Force-update the public branch with the rewritten history and verify that
  GitHub advertises the expected sanitized tip.
