# R1 dataset processing to-do

Updated: 2026-08-21

This checklist covers the raw-data-first synthetic R3x2 Wave experiment under:

```text
/path/to/data/20260821_product
```

The approved quantitative baseline is direct FFT root-sum-of-squares (RSS) of
the fully sampled no-Wave data after the frozen 64-to-12 coil compression. It
is derived from the same raw acquisition without GRAPPA, SENSE, Wave encoding,
or regularization. DICOM remains available for metadata and qualitative review,
but its voxel intensities must not enter parameter selection.

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
/path/to/user_workspace/scan_protocols/20260817_integrated/v151/
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
- [x] Source BART through `/path/to/user_workspace/bart/bart_startup.sh` and require `-g` for
  every `bart wave` reconstruction.
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
- [ ] Register the direct FFT RSS volume as `ranking_reference` in the dataset
  contract, including its source k-space, coil-basis hash, FFT/RSS convention,
  canonical-RAS NIfTI, and file hash.
- [ ] Verify that the direct FFT reference and every candidate have identical
  shape, voxel size, affine, and RAS axis convention. Because they arise from
  the same source grid, calculate metrics without image registration or
  interpolation; stop if the geometry differs.
- [ ] Create one fixed brain mask from the direct FFT reference and use BET
  only to define metric support. Do not replace reconstruction volumes with
  BET skull-stripped outputs or apply any bias correction.
- [ ] Evaluate the completed coarse Wavelet and block-8 LLR cases against the
  direct FFT reference. Keep the solver-matched lambda-zero cases as operator
  and convergence controls, not as the quantitative ground truth.
- [ ] For scale-dependent metrics, fit one documented scalar per candidate
  within the fixed reference mask. Use reference-derived scaling/windowing for
  SSIM and figures; do not use candidate-specific histogram matching or bias
  correction.
- [ ] Refine Wavelet lambda on a 1-2-5 logarithmic grid around the best coarse
  interval. Reuse all completed cases and add only missing intermediate values;
  do not expand beyond the existing `1e-6` to `1e-2` bracket unless the optimum
  lies at an endpoint.
- [ ] Refine block-8 LLR lambda similarly, initially reusing `2e-5`, `1e-4`,
  and `5e-4` and adding `5e-5` and `2e-4`. Extend one step outward only if the
  best result remains at a boundary. Keep block size fixed at 8 for this search.
- [ ] Rank the refined candidates with fixed-mask whole-volume metrics and a
  common reference-derived visual window. Record metric definitions, lambda,
  iteration/convergence information, input hashes, and output hashes in CSV
  and JSON.
- [ ] Select one Wavelet and one LLR finalist, then choose the R1 regularization
  using quantitative metrics plus visual review. DICOM remains qualitative and
  cannot break a metric tie.
- [ ] Freeze the R1-selected regularization and apply it unchanged to R3 as a
  cross-dataset transfer check.
