# R1 dataset processing to-do

Updated: 2026-08-21

This checklist covers the raw-data-first synthetic R3x2 Wave experiment under:

```text
/path/to/data/20260821_product
```

The initial reconstruction must not use DICOM intensity as a baseline or
ranking reference. Prescan Normalize was enabled and produces the known bright
brain-center bias. Keep the DICOM files untouched for possible metadata or
later qualitative review, but exclude their voxel values from selection.

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

- [ ] Estimate the coil basis from the fully sampled image stream in bounded
  PE2 chunks, retaining the first 12 virtual coils.
- [ ] Review cumulative covariance energy, basis orthogonality, finite probes,
  and input/output dimensions before continuing.
- [ ] Directly assemble `[RO, PE1, PE2, coil] = [256,256,256,12]` complex64
  k-space with the accepted basis. Apply no interpolation, GRAPPA, SENSE, or
  partial-Fourier completion.
- [ ] Verify every source chunk is finite, the complete grid is nonzero, and
  restart provenance matches the TWIX identity and coil-basis hash.
- [ ] Export a raw-derived no-Wave RSS quicklook for source/orientation QC only.
  It is not yet a regularization-ranking baseline.

## Synthetic Wave R3x2 preparation

- [ ] Generate the theoretical PSF from the frozen sequence and Wave settings.
- [ ] Wave-encode the complete 12-channel source before applying any sampling
  mask.
- [ ] Review the full-Wave direct-IFFT magnitude and phase diagnostics.
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
- [ ] Visually approve the crop `0.6` map montage and lambda-zero quicklook,
  then freeze those maps and lambda-zero output for every regularized candidate.
- [ ] Check finite outputs, map support, data consistency, orientation/LR,
  scaling, central slices, and full-volume geometry before regularization.
- [ ] Add full-sampling and PSF=1 operator checks where they can be performed
  without conflating reconstruction and presentation processing.

## Later refinement and evaluation

- [x] Inventory the added IDEA offline DICOM reconstructions, series 10-14.
  Their enhanced-DICOM private reconstruction metadata consistently reports
  `CC:SoS`. Series 11 and 12 omit `NormalizeAlgo:PreScan`, whereas series 10,
  13, and 14 include `NormalizeAlgo:PreScan` across their frames. Treat series
  11 and 12 as the candidate Prescan-Normalize-off, SOS-combined reconstructions.
- [ ] At the DICOM stage, fully qualify series 11 and 12 before enabling either
  one as a baseline: verify series UID and completeness, geometry/orientation,
  absence of other intensity filters, finite pixel data, and center-to-shell
  intensity behavior. Absence of the private `NormalizeAlgo:PreScan` string is
  necessary evidence but is not by itself approval for intensity ranking.
- [ ] Record the selected DICOM series UID and the decisive reconstruction
  metadata in the dataset manifest before changing
  `ranking_reference.kind` from `none` or enabling DICOM intensity ranking.
- [ ] Make regularized Wave reconstruction consume the same dataset contract
  and measure any dataset-specific maximum eigenvalue rather than copying R3.
- [ ] Run the frozen Wavelet coarse sweep only after lambda zero passes.
- [ ] Keep BET restricted to metric support and keep all DICOM intensities out
  of ranking until a suitable DICOM acquisition is available and qualified.
- [ ] Define a raw-derived no-Wave comparison reference only after its coil
  combination, scaling, and orientation contract is explicit; no GRAPPA or
  accelerated SENSE reference is needed for this fully sampled source.
- [ ] Freeze the R1-selected regularization and apply it unchanged to R3 as a
  cross-dataset transfer check.
