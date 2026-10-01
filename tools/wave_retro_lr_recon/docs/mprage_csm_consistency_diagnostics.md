# Set-4 coil-sensitivity consistency diagnostics

This optional MPRAGE troubleshooting workflow asks one calibration question:
is the integrated FLASH set-4 ACS locally consistent with the accepted
single-map ESPIRiT sensitivity model near metal? It supports the metal-fringe
investigation in
[`../to-do/METAL_FRINGE_INVESTIGATION_PLAN.md`](../to-do/METAL_FRINGE_INVESTIGATION_PLAN.md).
It is not a reconstruction stage, it changes no default, and its metrics never
select a winner.

## Scientific framing

FLASH set 4 and MPRAGE use the same readout gradient, polarity, 1024 samples,
and 5 us dwell, so their off-resonance readout displacement is the same:
5.12 mm/kHz along the readout. For an individual isochromat under a constant
readout gradient, off-resonance phase is equivalent to a readout-direction
shift, and Wave does not add EPI-like B0 distortion. Near metal, however,
non-invertible pile-up, intravoxel dephasing, excitation differences, signal
voids, and displaced or mixed coil sensitivities can still violate the
single-map SENSE model.

The R = 3 LIN alias-partner mechanism, in which a locally inconsistent metal
region leaks through unfolding to positions about `round(N_LIN / 3)` LIN away
(85 for 256), is a pre-registered testable hypothesis, not a conclusion.

## What runs and what does not

- The only BART command is the diagnostic
  `bart ecalib -m 2 -c 0 KSPACE_CALIB MAPS EIGENVALUES`, run by the shell entry
  point on the accepted PCA calibration k-space. Its two uncropped maps and
  eigenvalue maps are never used for Wave reconstruction; the accepted
  one-map `-m 1 -c 0` CSM remains the reconstruction baseline.
- Not run: Soft-SENSE or `ecalib -S`, physical-coil ESPIRiT, PCA-24, Wave
  reconstruction, `bart fft` or `bart rss`, refscan sets 0-3, or any PSF fit.
- Python modules never launch BART or any other process.
- The physical set-4 ACS is exported by the existing
  `rovir_feasibility.export_mprage_physical_calibration`, unchanged. Its
  manifest keeps the inherited status string
  `mprage_rovir_physical_calibration_ready`; the name reflects where the
  exporter originated, not a ROVir run.

## Commands

Activate the environment described in [`../SETUP.md`](../SETUP.md), confirm the
exact output root, then run the stages in order:

```bash
SCRIPT=tools/wave_retro_lr_recon/scripts/sample_mprage_csm_consistency.sh
ARGS=(/path/to/measured_wave_mprage.dat
      /path/to/matching_wave_mprage.seq
      /path/to/accepted_normal_root
      /path/to/new_csm_consistency_root)

"$SCRIPT" print-calibration-command "${ARGS[@]}"   # prints the BART command only
"$SCRIPT" prepare "${ARGS[@]}"
"$SCRIPT" calibrate "${ARGS[@]}"
"$SCRIPT" roi-template "${ARGS[@]}"
# Draw the five labels on OUTPUT_ROOT/rois/template, save the reviewed map, then:
"$SCRIPT" diagnose "${ARGS[@]}" \
    --roi-labels /path/to/new_csm_consistency_root/rois/reviewed/csm_consistency_roi_labels_reviewed.nii.gz
```

Inclusive BART-index boxes may replace the label map, for example
`--roi-box fringe=ro=a:b,lin=c:d,par=e:f`, one option per box.

### Accepted baseline contract

The accepted root must contain `normal/bart_inputs` with the alias-free
readout-crop record, `kspace_calib`, `psf`, and `wave_kspace`, and
`normal/bart_output` with the one-map CSM and the FISTA-r0 image. Both command
records are parsed exactly as the normal launcher writes them:

- `normal/bart_output/ecalib_command.txt`:
  `bart ecalib -m 1 -c 0 KSPACE_CALIB COIL_SENS`;
- `normal/bart_output/fista_r0/wave_command.txt`:
  `bart wave [-g] -w -f -r 0 -i 100 -t 1e-6 COIL_SENS PSF WAVE_KSPACE IMAGE`.

Every path in both records must be absolute and must not depend on the
calling process. The path is resolved one component at a time, following
every symbolic link. It is refused if any step lies on the file system mounted
at `/proc` or `/dev`, whatever the spelling: `/proc/self/cwd/...`,
`//proc/...`, `/./proc/...`, `/../proc/...`, `//dev/fd/N/...`, or a link into
`/proc`. The same stable-path rule applies to every recorded path (see
[Recorded paths](#recorded-paths)). Its basename must be the expected
artifact name, which also excludes BART's single-file endings `.ra`, `.coo`,
`.shm`, `.mem`, and `.fifo`. Its `.hdr` and `.cfl` must be
the same files (device and inode) as this root's corresponding artifact.
Symlink or automount aliases such as `/homes/...` versus `/autofs/...` pass;
a matching directory layout in another root, a relative path, a swapped role,
a regularized `-r`, or another branch's output fails. The PSF and wave k-space
are bound by file identity and header hash only; the diagnostics never read
them, and their multi-GB payloads are not hashed.

| Stage | Backend | Writes |
| --- | --- | --- |
| `prepare` | Python | physical set-4 export, noise covariance, map-1 reference, prepare manifest |
| `calibrate` | BART `ecalib -m 2 -c 0`, then Python validation | input hash record, two maps, eigenvalues, command record, logs, calibration manifest |
| `roi-template` | Python | canonical-RAS reference and empty label template |
| `diagnose` | Python | metrics, NIfTI views, figures, CSV, report, diagnostics manifest |

### Reuse and interrupted runs

Every stage revalidates the TWIX identity, the sequence file record, the
accepted arrays, manifest, and command records, and the manifests and outputs
of the stages it depends on. A stage returns its existing manifest only when
all of the following hold:

- every input it depends on is unchanged, including the identity of each
  upstream manifest, by size and SHA-256;
- every output it recorded is unchanged (size and SHA-256, or the full CFL
  record);
- each record names, by a stable absolute path, the file at its expected
  location in the current output or accepted root. A copied or relocated
  output root is therefore not reused.

Stage-specific rules:

- **Calibrate.** Two-map outputs are reused only through their calibration
  manifest. Just before `bart ecalib` runs, the shell writes the SHA-256 of
  the accepted `kspace_calib.hdr` and `kspace_calib.cfl` to
  `csm/map2_uncropped/ecalib_input.sha256`. The outputs are recorded only if
  that file names exactly the prepared accepted pair, by stable absolute
  paths, with its recorded hashes. Maps left over from another k-space, for example after the accepted
  root was regenerated, are therefore refused rather than recorded against the
  new inputs.
- **ROI template.** Reuse recomputes the descriptive comparison with the
  accepted FISTA-r0 NIfTI and requires it to equal the record. The recorded
  reviewed-label destination must equal the absolute path of
  `rois/reviewed/csm_consistency_roi_labels_reviewed.nii.gz` in the current
  output root.
- **Diagnose.** Reuse additionally requires:
  - the implementation hashes to match;
  - the ROI provenance to be identical, meaning the same source kind, the same
    label file bytes or the same boxes;
  - the recorded outputs to equal the fixed output contract exactly;
  - every file under the diagnose-owned directories to be recorded;
  - the summaries to cover exactly the 15 accepted-grid metrics and the 10
    native-grid summaries, each over the same nine ROI and control masks.

  The contract comes from the implementation, not from the manifest: 15
  accepted-grid metric arrays, 8 native local-rank arrays (2 coil bases, 2
  neighbourhoods, e1 and kappa2), 7 NIfTI views, 5 figures, and the CSV.
  Removing a summary together with its output record and file is therefore
  still detected. Both coil bases must yield local rank. A background-air ROI
  too small on the native ACS grid stops diagnose; it no longer drops those
  outputs.

  Output from older code or another ROI source is refused, not returned. The
  implementation identity covers every module of the `wave_retro_lr` package,
  both entry points, and the pinned upstream Wave-MPRAGE TWIX-import,
  coil-compression, NIfTI-geometry, and sequence-geometry sources.

The prepare, calibration, and ROI-template manifests are verified by content;
each records the implementation that produced it.

### Recorded paths

A stable path is absolute and does not resolve through `/proc` or `/dev` (the
component-wise rule above). A relative path depends on the working directory
and a `/proc` or `/dev` path on the calling process, so either could name the
expected file in one process only. Reuse therefore binds every recorded path
in two steps: the path must be stable, and it must name the expected file.

- **File and CFL records.** Every `path` and CFL `base` in the four manifests,
  every name in `ecalib_input.sha256`, and the environment-log record must be
  stable. Each must also name the same file (device and inode) as the expected
  artifact in the current roots, with an unchanged size and SHA-256 (or CFL
  record). A relative spelling is refused even when the current working
  directory makes it name the right file.
- **Prepare inputs.** The recorded sequence, accepted normal manifest,
  accepted manifest, and both accepted command records are complete file
  records, never checked by hash alone. A record that points at a nonexistent
  path, or at an identical copy elsewhere, is refused. The TWIX record must
  be stable and equal the current path, size, and modification time.
- **Derived fields.** Paths inside other records must equal the record
  computed now from stable absolute paths. These are the accepted command
  `argv` tokens and `bound_artifacts`, `accepted.root`, the calibration
  `accepted_kspace_calib` and command `argv`, the figure entries (each equal
  to its verified output record), and the reviewed-label file record.
- **Destinations.** The reviewed-label destination may not exist yet, so it
  must equal its expected stable absolute path exactly.

The shell resolves its four positional paths (TWIX, sequence, accepted root,
output root) and `--roi-labels` with `realpath`. `logs/bart_binary.sha256` names the canonical BART executable, so
no shell-written record depends on relative `PATH` entries or the working
directory.

**Environment logs.** Before each Python stage, the shell writes a new,
uniquely named log to `logs/environment/<stage>_<UTC>_<id>.txt`. A log is
never rewritten. A stage records the log of its own invocation only when it
writes a new manifest. On reuse, the recorded log must still exist,
unchanged, at its recorded relative path, or the stage stops. The logs of
reuse invocations remain as unrecorded invocation history.

Any mismatch or missing file stops the stage. Outputs that exist without their
stage manifest come from an interrupted run, and the stage refuses to run over
them. Partial CFL pairs and changed commands also fail closed. Nothing is
deleted or overwritten automatically; move the named files aside to rerun.

## Output tree

```text
OUTPUT_ROOT/
├── inputs/
│   ├── physical_calibration/      physical_set4_kspace (RO x NCALIB x NCALIB x physical coils)
│   └── noise/                     noise_covariance.npy (physical coils, complex128)
├── csm/
│   ├── map1/                      accepted_map1_reference.json (no copy of the accepted CSM)
│   ├── map2_uncropped/            coil_sens (RO, LIN, PAR, Ncc, 2), ecalib_command.txt, ecalib_input.sha256
│   └── eigenvalues/               ev_m2_c0 (RO, LIN, PAR, 1, 2)
├── rois/
│   ├── template/                  reference and empty label NIfTI, README.txt
│   └── reviewed/                  user-drawn label map
├── diagnostics/
│   ├── calibration_views/         log10 SNR energy
│   ├── coil_projection_residuals/ rho, conditional RNR, eigenvalues, map reproduction, figures
│   ├── local_rank/                native-grid e1 and kappa2, figure
│   ├── csm_coherence/             C1, C2, map switching, figure
│   └── roi_overlays/              fixed-window ROI overlay
├── manifests/                     physical_calibration, prepare, two_map_calibration, roi_template, diagnostics
├── reports/                       csm_consistency_report.md, roi_summary.csv
└── logs/                          environment/<stage>_<UTC>_<id>.txt (one per invocation), BART version and binary hash, ecalib log
```

## Diagnostics

All metrics are computed only inside the signal support and are NaN elsewhere.
Coil vectors `d(r)` are centered orthonormal IFFTs of the set-4 k-space.

- **Coil images.** Primary: DC-centered Hann apodization
  `w(k) = cos^2(pi k / N_acs)` over the measured ACS block; secondary: no
  window. The accepted PCA calibration k-space is used on the accepted image
  grid; physical-coil data use the native ACS grid without zero filling.
- **Support.** `SNR(r) = ||d(r)||^2 / tr(Psi_air)`, where `Psi_air` is the
  empirical background-air coil covariance; primary threshold 25, robustness
  10 and 100.
- **Projection residual (primary).**
  `rho_M = ||d - Q_M Q_M^H d|| / ||d||`, with `Q_M` an orthonormal basis of the
  leading `M` uncropped maps from a truncated SVD (relative tolerance 1e-3). It
  is invariant to map phase and positive map weights.
- **Residual-to-noise ratio (conditional).**
  `RNR_M = ||(I - P_M) d||^2 / tr[(I - P_M) Psi_air]`. It is interpretable only
  when the noise model below is compatible; otherwise it is reported as
  uncalibrated and never used to support or reject a hypothesis.
- **Eigenvalues.** Per-ROI quantiles of `lambda1`, `lambda2`, and their gap,
  fractions with `lambda2 >= 0.5, 0.8, 0.9, 0.95` and with `lambda1 >= 0.8`,
  and global counts of `|lambda| >= 1` and negative values.
- **Map-1 reproduction.** `|S1^H S_accepted|` where `lambda1 >= 0.9`; the
  median is expected to be at least 0.999.
- **Local coil-vector rank (model-free).** For 5-sample readout
  neighbourhoods (secondary 5 x 3 x 3) of unit-normalized coil vectors on the
  native grid: `e1 = mu1 / sum(max(mu_i, 0))` and `kappa2 = mu2 / mu1`, in the
  physical and the recovered accepted PCA bases, without noise debiasing.
- **Phase-free coherence.** `C1` is the minimum face-neighbour
  `|S1^H S1'|`; `C2` is the minimum `||Q2^H Q2'||_F^2 / 2`. Per-coil phase maps
  are never interpreted.
- **Alias-partner test.** Metal labels shifted by `+-round(N_LIN / 3)` LIN
  with a +-1 tolerance, PAR unchanged, optionally dilated along RO by
  0, 16, 32, 64 voxels or the full extent. The observed fringe overlap is
  compared with every non-partner LIN shift; the exceedance fraction is
  descriptive and is not a p-value.
- **Edge-matched control.** Head-mask voxels within 3 mm of the scalp/air
  boundary and more than 40 mm from the metal labels, with a deterministic
  subsample matched to the metal SNR distribution in 10 quantile bins. Every
  bin is scaled by one common factor so the histogram shape is kept; when a
  metal SNR bin has no eligible control voxel the subsample is empty, and the
  report says so and shows the unmatched edge control. The head mask comes
  from the existing `nifti_collection.create_whole_head_mask`.
- **FLASH-support mismatch.** Share of accepted FISTA-r0 energy per ROI where
  `lambda1 < 0.5`. It is circular by construction and descriptive only.

These thresholds are pre-registered descriptive defaults recorded in every
manifest and report. They do not select a winner or prove a mechanism.

## Noise model and conditional RNR

The noise source is the measurement-0 adjustment scan, whose dwell (4 us in
the reviewed data) differs from the 5 us set-4 ACS. FFT-scale metadata also
differ between measurements and are recorded but not applied. A white-noise
dwell ratio is reported only as an approximation.

Before RNR is labelled calibrated, two matrix-level checks must both pass:

1. the noise-scan covariance against the empirical air covariance of the
   native physical-coil set-4 image; and
2. the noise-scan covariance transformed into the recovered accepted PCA basis
   (`W^T Psi conj(W)`) against the Hann-apodized air covariance of the
   accepted PCA image, which is the covariance RNR actually uses.

Each check compares total trace, normalized covariance shape, per-channel
variance, off-diagonal correlation structure, and the generalized eigenvalue
spectrum. Shape limits (`twix_noise.DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS`)
are descriptive; the trace ratio is reported but is never a pass criterion.
A comparison that cannot be formed, for example because an air covariance is
singular, is recorded as `comparison_error` and counts as incompatible; it
does not stop the diagnostics. The manifest lists the failed checks in
`rnr_failed_checks`.

Finite samples alone can exceed the limits, so every check also records a
sampling-null reference: a Monte Carlo (16 draws, seed 0) of the estimator
that produced the air covariance. White complex Gaussian noise with the
reference covariance is drawn for every measured ACS sample, weighted by the
same k-space window, zero-filled onto the same grid, transformed with the
centered orthonormal inverse FFT, and averaged over the same air mask. The
Hann window and zero fill correlate neighbouring air voxels, so the record
also gives the effective number of independent air vectors,
`N_eff = M^2 K(0)^2 / sum_rows sum_D |K(D)|^2 A_row(D)`, with
`K = ifft2(|w|^2)` the LIN/PAR noise kernel, `A_row` the circular mask
autocorrelation in one readout row, and `M` the masked voxel count. `N_eff`
equals `M` on the native unapodized grid and `(sum w^2)^2 / sum w^4` per fully
masked readout row of the accepted Hann grid (about 271 for a 32 x 32 ACS).
The window power gain `K(0)` is the trace ratio expected for identical
per-sample noise, so `trace ratio / K(0)` approximates the ACS-to-noise-scan
variance ratio, including any export scaling. The gate stays conservative: a
failure leaves RNR uncalibrated even when it is plausibly caused by sample
size. A held-out split of the noise scan records the sampling variability of
the noise estimate itself.

Channel identity is checked twice: exact MDH channel-ID sequences in noise,
image, and every refscan set, and a block-aware comparison of
`aRxCoilSelectData[0]` element-to-ADC maps. Blocks are never merged.

## ROI contract

Labels are uint8 on the accepted image grid: 0 unassigned, 1 metal void or
null, 2 metal pile-up or bright displaced signal, 3 fringe, 4 preserved
anatomy, 5 background air. Draw air superior to the scalp so that no head
signal shares its readout rows. A reviewed label map must keep the exported
shape, affine, and RAS axis codes; the inverse orientation must round-trip
exactly. Fringe, preserved anatomy, air, and at least one metal label must be
nonempty. Stored values are validated exactly as read, before any integer
conversion. They must be finite integers from 0 to 5 after NIfTI scaling, so
a value such as 257 is rejected rather than wrapping into label 1. The
background-air label must also cover more native ACS voxels than there are
physical coils, or diagnose stops before writing any output.

## Manifest schema

All manifests carry `format_version` 1, `status`, `created_at_utc`,
`implementation` (SHA-256 of every package module, both entry points, and the
pinned upstream helper sources), `environment` (Python and package versions
plus `shell_environment_log` with the file record, `relative_path`, and exact
text of the log written for the invocation that created the manifest, or
`null` for direct Python calls), and `flags`
(`bart_launched_by_python`, `wave_reconstruction_launched`,
`soft_sense_used`, `psf_recalibrated`, `refscan_sets_0_to_3_used`,
`normal_defaults_changed`, `two_map_csm_used_for_reconstruction`, all false).

- `csm_consistency_prepare.json`: `environment`, `sources` (TWIX with SHA-256,
  and file records of the sequence and the accepted manifest),
  `sequence_contract`, `accepted` (manifest,
  geometry, coil counts, readout-crop record, CFL records of `kspace_calib`,
  `coil_sens`, `fista_r0_image`, identity records of `psf` and `wave_kspace`,
  and the `ecalib_command` and `fista_r0_command` records with `text`,
  `argv`, `bound_artifacts` (recorded, resolved, and accepted path per role),
  and for FISTA-r0 `gpu`), `physical_calibration`
  (exporter manifest, inherited status, CFL record, set index), `noise`
  (raid table, MDH walks, channel identity, set-4 lattice, block-0 coil
  select, metadata, covariance file and statistics, held-out split,
  comparison, and compatibility, dwell-ratio expectation,
  `absolute_acs_noise_calibration_claimed: false`), `map1_reference`.
- `two_map_calibration.json`: `prepare_manifest`, `command` (`argv`, `text`,
  record, backend), `ecalib_input` (file record of the input hash record with
  `header_sha256` and `payload_sha256` of the accepted `kspace_calib`),
  `map_count: 2`, `crop_value: 0.0`, `soft_sense: false`,
  `diagnostic_only: true`, `used_for_wave_reconstruction: false`, `maps`,
  `eigenvalues`, `accepted_kspace_calib`, `bart_version`, `bart_binary`
  (file record and text naming the canonical executable), `ecalib_log`,
  `environment`.
- `roi_template.json`: `geometry` (BART and stored shapes, stored and source
  affines, array flips, orientation transform, reference and prepare-manifest
  hashes), `orientation_round_trip`, `display_scale`,
  `accepted_fista_r0_nifti_geometry`, `label_definitions`,
  `label_descriptions`, `reference_nifti`, `label_template_nifti`,
  `instructions`, `reviewed_label_destination`,
  `automatic_roi_detection: false`.
- `csm_consistency_diagnostics.json`: `inputs` (upstream manifests and ROI
  provenance), `roi_identity_sha256`, `parameters` (every threshold above),
  `outputs` (every metric array, NIfTI view, figure, and CSV keyed by its
  output-root-relative path), `figures` (each also listed in `outputs` under
  its `relative_path`),
  `noise_model` (physical-coil `reference`, `candidate`, `comparison` or
  `comparison.error`, `compatibility`, and `sampling_null_reference` with
  `method`, `voxels`, `effective_samples`, `window_power_gain`, `repeats`,
  `seed`, `singular_draws`, per-criterion and trace-ratio `median` and `max`;
  `rnr_basis_check` with the same records in the accepted PCA basis;
  `rnr_status`; `rnr_failed_checks`; policy; held-out compatibility; dwell
  expectation; air voxel counts), `pca_basis_reproduction`,
  `map1_reproduction`, `eigenvalue_qc`, `support_voxels`,
  `native_grid_support_voxels`, `derived_masks`, `roi_summaries`,
  `native_grid_roi_summaries`, `alias_partner_test`, `support_mismatch`,
  `report`, `interpretation_policy`.

## Runtime, memory, and disk

For a 52-channel 256 x 256 x 220 acquisition (40-channel 256 x 256 x 192):

| Stage | Peak host RAM | Wall time | Disk |
| --- | --- | --- | --- |
| prepare | about 11-21 GiB (8-16 GiB), dense five-set refscan read | 4-10 min, including one TWIX hash and two header parses | 0.55 GB (0.43 GB) |
| calibrate | 6 GB or less | 2-15 min on CPU; measured by `/usr/bin/time -v` | 3.0 GB (2.6 GB) |
| roi-template | under 2 GB | under 1 min | about 0.1 GB |
| diagnose | 8-12 GB | 5-20 min, of which the two sampling nulls take under 20 s | about 1-1.5 GB |

No GPU is used. The output root needs at least 10 GB free.

## Limitations

- Calibration self-fit is optimistic: a residual failure indicates model
  insufficiency; a pass is not proof of generalization.
- The set-4 ACS is 32 x 32 in PE; low-resolution blurring and Gibbs ringing
  can mix sensitivities near bright edges, hence the Hann primary variant and
  the edge-matched control.
- TE, contrast, preparation, and excitation differences between FLASH and
  MPRAGE cannot be tested from FLASH data alone.
- Contrast and noncontrast acquisitions of one subject are not registered
  repeat measurements; they are not used to assess the PSF.
- Signal that was dephased or never acquired cannot be recovered by any
  calibration change.
