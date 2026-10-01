# Wave retrospective reconstruction handover

Updated: 2026-09-30, America/New_York

## Start here

Read all applicable workspace `AGENTS.md` files, then read:

- `README.md` for the supported MPRAGE and GRE workflows;
- `SETUP.md` for the host Python and BART environment;
- `TROUBLESHOOTING.md` before changing recovery or compatibility behavior;
- `docs/mprage_rovir_reconstruction.md` for the optional MPRAGE ROVir feature;
- `to-do/METAL_FRINGE_INVESTIGATION_PLAN.md` for the active metal-fringe
  investigation, its review gates, and its approved and excluded scope;
- `docs/mprage_csm_consistency_diagnostics.md` for the optional set-4
  coil-sensitivity consistency diagnostics built for that investigation;
- `../synthetic_wave_for_reg_baseline/docs/synthetic_mprage_regularization_pipeline.md`
  for the origin of the selected MPRAGE regularization parameters; and
- `../synthetic_wave_for_reg_baseline/docs/synthetic_gre_regularization_pipeline.md`
  for the corresponding GRE selection history.

Begin new tasks read-only: inspect Git status, relevant manifests and consumers
before editing. The metal-fringe investigation below is active and gated; do
not advance it past its current gate without the user's explicit review
result. The other work summarized here is complete; do not relaunch old
experiments merely to reproduce their accepted state.

## Active work: metal-fringe investigation

The investigation asks why central-FOV fringes appear near metal in two
Wave-MPRAGE subjects. Its state on 2026-09-30:

- Work happens on branch `investigation/metal-fringe-plan`, two local
  commits ahead of `main`: `bafd07f` (the plan) and the Stage 2 commit after
  it. Neither is pushed.
- Stage 1, the read-only evidence audit, was completed on 2026-09-29. Gate 1
  gave a conditional pass for dataset-independent Stage 2 code only.
- Stage 2 implemented the calibration-only set-4 coil-sensitivity consistency
  diagnostics. The package, this handover included, is in the local Stage 2
  commit; the files are listed below.
- Gate 2 had four reviews. Reviews v1 to v3 raised blockers, each fixed before
  the next review; v4 approved the Stage 2 code on 2026-09-30:
  - **v1** requested four changes, all fixed with negative tests. They bind
    the accepted ecalib record to the current accepted root by file identity,
    validate the accepted FISTA-r0 record (including its PSF, wave k-space,
    and output), validate reviewed ROI label values before any integer
    narrowing, and verify recorded outputs before any stage reuses existing
    results. An independent adversarial audit then found one more blocking
    gap (calibrate reused stale two-map maps whose command text matched) and
    five minor gaps (implementation identity scope, two provenance records
    not re-verified, manifest-trusted output lists, BART single-file name
    endings, and untyped-record errors). All are fixed with tests.
  - **v2** passed the 225-test suite and raised three blockers, all fixed.
    `//proc` and `//dev` spellings bypassed the process-dependent-path rule,
    so paths are now resolved one component at a time and any path through
    `/proc` or `/dev` is refused. Removing a summary, its output record, and
    its metric file together passed diagnostics reuse, so reuse now checks a
    fixed output contract taken from the implementation. Recorded environment
    logs were rewritten on reuse and never verified, so every shell invocation
    now writes a new log under `logs/environment`, manifests record their own
    log, and reuse verifies it.
  - **v3** passed the 228-test suite and raised two provenance blockers, both
    fixed with exact negative tests.
    Manifest file and CFL records and the `ecalib_input.sha256` names
    accepted relative paths whenever the working directory made them name the
    right file. Every recorded path must now be stable (absolute, not through
    `/proc` or `/dev`) and name the expected file, and derived path fields
    must equal records computed from stable paths. The sequence,
    accepted-manifest, and command records of the prepare manifest were
    checked by hash alone; they are now complete file records, so a
    nonexistent path or an identical copy elsewhere is refused.
  - A test now walks every recorded path in the four manifests (105 fields
    with ROI boxes, 106 with a label NIfTI) with five relative or
    process-dependent spellings, each checked from a working directory where
    it names the right file. Run against the reviewed v3 code, the new tests
    found 775 scenarios that v3 did not refuse, including 764 of the 1,055
    walk spellings; all are refused now. At 41 of those fields, v3 had also
    accepted `/proc` and `/dev/fd` spellings, contrary to the round-3 summary.
  - **v4** was approved. The reviewer independently replayed the four v3
    provenance reproductions (all rejected) and passed the 230-test suite.
    The approval authorizes neither the real-data stages nor a commit; each
    needs separate explicit authorization.
- No real-data stage has run, and nothing has been written to a real-data
  output root. The user confirmed `225_noncontrast` as the diagnostic pilot at
  the Gate 2 v3 review, and earlier confirmed an existing, empty output root
  for it. Both are recorded only in the ignored
  `configs/metal_fringe_investigation.local.json`, as `diagnostic_pilot_label`
  and `diagnostic_output_root`.
- On 2026-09-30 the user separately authorized two actions. One is the local
  Stage 2 commit, which includes this handover. The other is the Stage 3
  calibration-only pilot for `225_noncontrast` on that root: run
  `print-calibration-command`, `prepare`, `calibrate`, and `roi-template`,
  then stop for the user's manual ROI review. `diagnose` follows only after
  the reviewed labels are in place, and the pilot stops at Review Gate 3. The
  only BART computation is the diagnostic `bart ecalib -m 2 -c 0`. Wave
  reconstruction, Soft-SENSE, PSF recalibration, sets 0-3, PCA-24,
  convergence controls, and every Stage 4 experiment stay out of scope, and
  the two-map CSM is never used for reconstruction. On any failure, stop
  without overwriting or deleting outputs and report the exact error.
- **The Stage 3 pilot is stopped after a failed `prepare`.** It ran at commit
  `343cbad` on 2026-09-30 (20:08 America/New_York). `print-calibration-command`
  passed. `prepare` exited with code 2 after 2 min 43 s (peak RSS 11.5 GiB):
  `Error: FFT-scale entries occur under several header prefixes:
  [('sCoilSelectMeas', 'aRxCoilSelectData', '0'), ('sCoilSelectMeas',
  'aRxCoilSelectData', '1')].` The measurement-0 adjustment scan
  (`AdjCoilSens`) stores coil-select block 0 (52 head-array elements and
  FFT-scale entries) and block 1 (2 body-coil elements and entries).
  `twix_noise.fft_scale_factors()` is not block-aware, so it refuses. The
  MPRAGE measurement has block 0 only and parses. Before the error, the
  exporter wrote `inputs/physical_calibration/` and
  `manifests/physical_calibration.json`, plus one environment log (527 MB).
  No prepare manifest exists. These outputs were neither moved nor deleted.
  The export is complete and recorded by its own manifest, so a later
  `prepare` verifies and reuses it; nothing needs to be moved aside unless
  that verification fails. `calibrate` and `roi-template` have not run.
- **The fix is committed locally (the commit after `343cbad`, not pushed).**
  The user approved making it, then its commit and the pilot rerun, on
  2026-09-30. `twix_noise` now reads FFT scale per
  coil-select block, never merged: block 0 is reported as
  `fft_scale_factors`, and every block is recorded in `fft_scale_by_block`,
  with `fft_scale_block` naming the reported block. A header with a single
  prefix reads as before. Several prefixes must all be coil-select blocks
  and include block 0. The fix also parses hexadecimal `bValid` flags such as
  `0x1`, which the committed code rejected (`FFT-scale bValid 0 is not
  numeric: '0x1'.`) on both measured headers. The csm-consistency fixture's
  adjustment scan now carries the measured two-block layout and reproduces
  the pilot error on the committed code. The full suite passes 232/232, and
  synthetic outputs are unchanged (44 of 44 hashes). A read-only header
  parse of the pilot TWIX now succeeds for both measurements. The pilot
  rerun (`prepare`, `calibrate`, `roi-template`) follows this commit.
- **The Stage 3 pilot is waiting for the user's manual ROI review.** The
  rerun at commit `2b18541` (2026-09-30, 20:44-20:48 America/New_York)
  completed three stages:
  - `prepare` (16 s, peak RSS 8.8 GiB) reused the verified physical set-4
    export. Channel identity, the 32x32 set-4 lattice, and block-0 coil
    select all match. The held-out noise split is compatible, and the
    expected ACS/noise white-noise variance ratio is 0.8.
  - `calibrate` (2 min 23 s, peak RSS 3.7 GiB) ran `bart ecalib -m 2 -c 0`
    (1 min 46 s) on the prepared `kspace_calib`, and the input hash record
    matched it exactly.
  - `roi-template` (15 s, peak RSS 1.1 GiB) exported the RAS reference and
    the empty five-label template. The orientation round trip is exact, and
    shape and affine match the accepted FISTA-r0 magnitude NIfTI.

  The output root holds 3.4 GB. `diagnose` runs only after the user saves
  the reviewed labels at the recorded destination; the pilot then stops at
  Review Gate 3.
- **The ROI-based `diagnose` is paused (user, 2026-10-01).** The fringe is
  superimposed on anatomy, so mutually exclusive labels are not a sound
  primary endpoint. Do not create or infer a reviewed label map, and keep
  every Stage 3 output: the arms read them.
- **Intervention arms (approved 2026-10-01).** These are single-variable arms
  against the accepted baseline. They run one at a time in the order 1, 2a
  and 2b, 4, 3, with a visual review after each and a disk check before
  each run. There is no Soft-SENSE weighting. See
  `docs/mprage_metal_fringe_interventions.md`. The arm roots sit under
  `interventions/` in the subject directory, and their paths are recorded
  in the ignored config as `intervention_output_roots`. Only arm 1's name is
  final; the later i100/i300 names wait for its convergence result. Arm 1
  (`sample_mprage_fista_convergence.sh`) and the shared review figures
  (`mprage_intervention_qc.py`) passed code review on 2026-10-01 and are
  committed locally (not pushed). Review found two blockers, both fixed
  before approval: `verify_control` now checks a fixed required-output
  contract, and any entry left without a manifest counts as an
  interrupted run. The user authorized running Arm 1 once the shared GPU
  is below 100 % utilization, then stopping at visual review. No other
  arm is authorized.
- Do not push, and do not commit further changes, without the user's explicit
  authorization.

Terminology fixed at Gate 1:

- C in an acquisition name means contrast enhancement. The labels are
  `225_noncontrast`, `225_contrast`, `169_noncontrast`, and `169_contrast`.
- Contrast and noncontrast acquisitions are not registered repeats, so their
  PSFs are never compared.
- Physics wording: for an individual isochromat under a constant readout
  gradient, off-resonance phase is equivalent to a readout-direction shift,
  and Wave does not add EPI-like B0 distortion. Near metal, however,
  non-invertible pile-up, intravoxel dephasing, excitation differences, signal
  voids, and displaced or mixed coil sensitivities can still violate the
  single-map SENSE model.

Scope fixed by Gate 1 (the plan has the details):

- The only real-data BART command is the diagnostic `bart ecalib -m 2 -c 0` on
  the accepted PCA-12 calibration k-space. The accepted one-map `-c 0` CSM
  remains the reconstruction baseline.
- Out of scope: Soft-SENSE or `ecalib -S`, physical-coil ESPIRiT, PCA-24,
  refscan sets 0-3, and any change to normal defaults, `mprage.py`,
  launchers, converters, ROVir modules, NIfTI collection, or `SETUP.md`.

Stage 2 files, all under this directory:

| File | Role |
| --- | --- |
| `wave_retro_lr/twix_noise.py` | multi-raid TWIX walks, channel identity, coil select, measurement-0 noise covariance and compatibility |
| `wave_retro_lr/csm_consistency_metrics.py` | pure projection, eigenvalue, coherence, and local-rank metrics |
| `wave_retro_lr/csm_consistency_roi.py` | five-label ROI contract, RAS round trip, alias-partner test, edge control, summaries |
| `wave_retro_lr/csm_consistency.py` | stages, accepted-baseline binding, integrity checks, sampling nulls, figures, report |
| `scripts/mprage_csm_consistency.py` | Python CLI |
| `scripts/sample_mprage_csm_consistency.sh` | stage runner holding the only BART call |
| `tests/test_twix_noise.py`, `tests/test_csm_consistency_metrics.py`, `tests/test_csm_consistency_roi.py`, `tests/test_csm_consistency.py` | synthetic tests |
| `docs/mprage_csm_consistency_diagnostics.md` | guide, contracts, schema, resources |

Stage 2 also edited `README.md`, `TROUBLESHOOTING.md`, and the plan.

## Repository checkpoint

`main` and `origin/main` point to `91911b8`. The investigation branch adds
`bafd07f` (the plan) and the local Stage 2 commit after it, which also adds
this handover; neither is pushed. The working tree keeps one modification
outside that commit: the pre-existing `../synthetic_wave_for_reg_baseline/HANDOVER.md`
edit, which predates the investigation and must be preserved.

The most recent source checkpoints on `main` are:

- `91911b8`: retain standard and ROVir NIfTI collection branches together;
- `4fcdb88`: allow direct MPRAGE ROVir runs with explicit manual ROIs;
- `d605a3a`: complete the user-facing MPRAGE ROVir workflow;
- `46601d8`: use alias-free GRE coil calibration;
- `67fa343`: use alias-free MPRAGE coil calibration;
- `1895522`: add GRE native-R3x3 sweep and reconstruction workflows; and
- `295c82d`: add native-R3x3 MPRAGE Wave reconstruction workflows.

The latest complete `wave_retro_lr_recon` validation on the investigation
working tree passed all 230 tests; 129 of them predate Stage 2. No scientific
reconstruction was launched as part of these source changes. Do not commit,
push, reset, discard, or modify external submodules without the user's
explicit direction.

## Accepted MPRAGE workflow

The standard measured MPRAGE workflow is complete and documented in `README.md`.
It supports normal native R3x1, retrospective native R3x2, three retrospective
low-resolution R3x2 cases, and native R3x3. Each supported accelerated case
retains a FISTA lambda-zero control and its reviewed Wavelet branch:

| Case | Wavelet lambda |
| --- | ---: |
| native R3x1 | `3.5e-2` |
| native R3x2 | `3.5e-2` |
| LR-X R3x2 | `2.5e-2` |
| LR-Y R3x2 | `2.5e-2` |
| LR-XY R3x2 | `2.2e-2` |
| native R3x3 | `4.5e-2` |

R1 normal data remains FISTA-r0-only unless a separate R1 regularization study
selects another branch. Sampling masks contain only the Cartesian image
lattice; ACS remains separate. Retrospective reconstruction reuses validated
PSF and CSM artifacts and never silently recalibrates them.

Coil calibration no longer removes readout oversampling by direct k-space
striding. Both MPRAGE and GRE use a full-readout IFFT, central nominal-FOV
image crop, and centered FFT before ecalib. This avoids aliasing extended-FOV
body signal into the head. The pinned upstream submodules contain the accepted
implementations; treat them as read-only dependencies.

Legacy MPRAGE prepared inputs can be reused by retrospective entry points only
through the validated compatibility path. It binds source identities, finite
geometry-compatible arrays, the retained historical PSF and an explicit reuse
attestation. It does not rewrite old metadata or relabel an old PSF as the
current default. Existing CSM reuse still requires the same recorded ecalib
crop; pass an explicit override when the historical normal CSM used a value
other than the current default.

## Accepted MPRAGE ROVir workflow

ROVir is an optional MPRAGE-only recovery feature for coherent shoulder or
extended-body wrap. It is not part of normal reconstruction and is not exposed
for GRE. It uses native BART `rovir`; BART does not detect anatomy or select an
ROI automatically.

The public entry point is `scripts/sample_mprage_rovir_recon.sh`:

- `inspect` exports corrected physical-coil ACS RSS views with native indices
  and a conservative recommendation when possible;
- `run --use-recommended` requires a completed, reviewed `inspect`;
- `run --null-box ...` or `run --null-box-file ...` may be invoked directly;
  missing indexed ACS diagnostics are generated automatically for provenance;
- any number of manual boxes may be supplied and their binary union defines the
  immutable candidate identity;
- the selected virtual-coil count is always explicit;
- CPU is the default and `-g` applies to BART Wave reconstruction; and
- the standard normal reconstruction output is optional, but accepted normal
  prepared inputs, TWIX/sequence provenance and PSF are required.

An explicit manual ROI is authorization to proceed; there is no redundant
candidate-ID prompt. The overlay remains a troubleshooting artifact. One
representative feasibility study selected Ncc 24 and an RO nuisance interval
of `[0, 20]`, but that subject-specific choice is not a global default.

After normal ROVir completes, the standard retrospective launcher with
`--rovir` creates ROVir siblings for all five retrospective MPRAGE cases. It
reuses the normal ROVir transform, CSM and PSF and does not estimate a new ROI
or run the standard-coil retrospective workflow.

Interrupted atomic ROVir preparation can leave hidden
`.bart_inputs_rovir_ncc*-*` staging directories. Confirm no related process is
running and validate the canonical sibling state before proposing deletion.
Never remove such directories without the user's explicit approval.

## NIfTI collection contract

The MPRAGE collection is additive. It retains standard and ROVir results as
distinct branches even when case, resolution, acceleration and reconstruction
method match. In particular, `fista_r0` and `rovir_fista_r0` coexist; ROVir no
longer replaces the standard control. The same applies to regularized branches
when both exist.

Rerunning `scripts/sample_mprage_nifti_collection.sh` validates and atomically
synchronizes the tool-owned collection. Newly discovered branches are added;
source disappearance remains a hard error. The legacy manifest field
`synchronization.rovir_replacements` remains for compatibility but is empty.
The shared presentation mask may prefer a normal ROVir magnitude as its source,
but that preference affects only mask estimation and never removes a standard
reconstruction from the collection.

## Operational boundaries

- Use the host-compatible BART selected by `bart_startup.sh`; do not copy a
  BART executable path from another host.
- Ask the user to confirm an exact production output directory and name before
  creating one.
- By default, implement and test code, then provide the command for the user to
  run. Do not launch scientific jobs without an explicit request.
- Keep real paths only in ignored local launchers/configuration or private
  generated manifests. Tracked examples and documentation stay path-agnostic.
- Do not modify frozen private synthetic sweep or accepted reconstruction
  trees unless the user explicitly places them in scope.
- Shared storage has recently reached capacity. Prefer read-only size
  inventories and exact orphan-staging review; never perform broad cleanup or
  delete raw/scientific data without explicit approval.
