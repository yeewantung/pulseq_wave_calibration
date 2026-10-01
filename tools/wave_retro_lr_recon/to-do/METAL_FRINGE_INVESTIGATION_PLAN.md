# Metal-related Wave-MPRAGE fringe investigation plan

## Status and purpose

This document is the handoff plan for investigating coherent fringes observed
in two Wave-MPRAGE subjects with dental braces. It is deliberately written so
that a fresh agent can begin from the repository and the ignored local source
manifest without relying on prior chat context.

The leading physical hypothesis is:

```text
metal-induced B0 perturbation
    -> readout displacement, pile-up, and signal void
    -> a coil signature that is inconsistent with its apparent image position
    -> incomplete representation by a single CSM set and a B0-free Wave model
    -> coherent fringes after position-dependent Wave spreading
```

The investigation must distinguish solver convergence, coil compression,
noise correlation, sensitivity-map failure, and missing off-resonance encoding.
It is not a new regularization search.

No scientific processing or reconstruction is authorized by this plan alone.
At every review gate, stop and give the user the evidence, proposed change,
tests, and exact commands. The user intends to request a separate code-review
session before authorizing the next stage.

## Review status

- Stage 0: plan and scope confirmed.
- Stage 1: read-only evidence audit and diagnostic specification completed on
  2026-09-29; Gate 1 received a conditional pass for dataset-independent
  Stage 2 code only.
- Stage 2: the diagnostic code described in
  [`../docs/mprage_csm_consistency_diagnostics.md`](../docs/mprage_csm_consistency_diagnostics.md)
  is implemented on the investigation branch.
- The initial Gate 2 review (2026-09-30) requested changes in four areas:
  - binding the accepted ecalib record to the current accepted root;
  - validating the accepted FISTA-r0 record;
  - validating reviewed ROI label values before integer narrowing;
  - verifying recorded outputs before any stage reuses existing results.

  All four are fixed with negative tests, and the v2 review confirmed them.
  The sampling-null correction passed the initial review. An independent
  adversarial audit of the fixes found one further blocking gap and five
  minor gaps, all fixed with tests before resubmission. The blocking gap was
  that calibrate reused stale two-map outputs whose command text matched.
- The second Gate 2 review (v2) confirmed those fixes. It raised three
  blockers, fixed with exact negative tests and confirmed by the v3 review:
  - `//proc` and `//dev` paths bypassed the process-dependent-path rule;
  - a summary, its output record, and its metric file removed together passed
    diagnostics reuse;
  - recorded environment logs were rewritten on reuse and not verified.
- The third Gate 2 review (v3) raised two provenance blockers, fixed with
  exact negative tests:
  - manifest file and CFL records and the `ecalib_input.sha256` names
    accepted relative paths whenever the working directory made them name the
    right file; every recorded path must now be a stable absolute path, not
    through `/proc` or `/dev`, that names the expected file;
  - the sequence, accepted-manifest, and command records of the prepare
    manifest were checked by hash alone, so their paths could name
    nonexistent files; they are now verified as complete file records.
- Gate 2 approved the Stage 2 code at the v4 review on 2026-09-30. The approval
  authorizes neither the real-data stages nor a commit; each needs separate
  explicit authorization.
- On 2026-09-30 the user separately authorized a local commit of the Stage 2
  package and the Stage 3 calibration-only pilot for `225_noncontrast`. The
  pilot runs through `roi-template`, then stops for manual ROI review;
  `diagnose` follows after the reviewed labels, stopping at Review Gate 3.
- On 2026-10-01 the user paused the ROI-based `diagnose`. The fringe is
  superimposed on anatomy, so mutually exclusive fringe and preserved-tissue
  labels are not a sound primary endpoint. The user then approved
  single-variable intervention arms against the accepted baseline:
  1, a FISTA `-i 300` convergence control; 2a and 2b, map 1 alone and maps
  1 + 2 of the Stage 3 two-map calibration, without Soft-SENSE weighting;
  4, PCA-24; and 3, prewhitened PCA-12. They run one at a time in the order
  1, 2, 4, 3, with a visual review after each. This ordering supersedes
  Stages 4-6 below; see
  [`../docs/mprage_metal_fringe_interventions.md`](../docs/mprage_metal_fringe_interventions.md).
  Arm 1 and the shared review figures passed code review on 2026-10-01 and
  are committed locally. Arm 1 is authorized to run once the shared GPU is
  below 100 % utilization, and the work stops at visual review after it.
- The user confirmed `225_noncontrast` as the diagnostic pilot at the Gate 2 v3
  review, and earlier an existing, empty output root for it. Both are
  recorded only in the ignored local configuration. Nothing has been written
  there and no scientific processing has run.

Stage 1 corrected or added these facts, which the sections below include:

- The FLASH set-4 TE is sequence-specific (see acquisition facts).
- FLASH set 4 and MPRAGE share the readout gradient, polarity, sample count,
  and dwell.
- Measurement-0 noise comes from the adjustment scan at a different dwell.
- In BART v1.0, `ecalib -S` with `-c 0` keeps every map weight at or above
  0.5, and `|lambda| >= 1` produces weights near 2; Soft-SENSE is therefore
  not approved for Stage 2 or Stage 3 preparation.

## Required starting procedure for a fresh agent

1. Work in the repository root and read every applicable `AGENTS.md`.
2. Read, in full:
   - `tools/wave_retro_lr_recon/HANDOVER.md`;
   - `tools/wave_retro_lr_recon/README.md`;
   - `tools/wave_retro_lr_recon/SETUP.md`;
   - `tools/wave_retro_lr_recon/TROUBLESHOOTING.md`; and
   - this plan.
3. Treat the handover as potentially stale. Verify every material statement
   against the current branch, manifests, commands, and files.
4. Inspect `git status`, the current commit, relevant manifests, and ignored
   local configuration before editing.
5. Preserve all unrelated user changes. At plan creation, the repository was
   on `main` at `91911b8`, with a modified synthetic-experiment handover and an
   untracked Wave reconstruction handover. Recheck rather than assuming this
   remains true.
6. On `macha`, activate `cuda133py312-macha` and source the host BART startup
   script before Python or BART work.
7. Do not modify external BART or other submodules.
8. Do not create a scientific output root, launch reconstruction, commit,
   push, reset, or discard changes without explicit user authorization.
9. Real machine and data paths are stored only in the ignored local manifest
   `tools/wave_retro_lr_recon/configs/metal_fringe_investigation.local.json`.
   Do not copy those paths into tracked documentation, tests, or source.

The last recorded full Wave reconstruction test baseline was 129 passing
tests. It is historical evidence, not a substitute for rerunning relevant
tests after implementation.

## Existing contracts that must remain intact

- Normal and retrospective MPRAGE retain FISTA-r0 and the reviewed Wavelet
  branches.
- Native R3x3 remains in the default retrospective cases.
- Coil-calibration ACS readout oversampling is removed by full-readout IFFT,
  central nominal-FOV crop, and centered FFT. Direct readout k-space striding
  is forbidden.
- Image sampling k-space and calibration ACS remain separate.
- The identical coil-domain transform must be applied to image data and ACS;
  CSMs must then be recalibrated in that transformed basis.
- The accepted measured Wave PSF is reused byte-for-byte for coil-domain
  experiments unless a later review gate explicitly reopens PSF calibration.
- ROVir remains optional and MPRAGE-only. Standard and ROVir NIfTI collection
  branches are additive.
- Production artifacts must have source, command, version, geometry, and hash
  provenance sufficient to reject incompatible reuse.

## Current reconstruction and acquisition facts

The ignored local manifest identifies two subjects, each with a noncontrast
and a contrast-enhanced acquisition. The letter C in a `+C` or `-C` filename
denotes contrast enhancement; the sign indicates no reversed readout, gradient
polarity, or other reconstruction condition. Use these labels everywhere:
`225_noncontrast`, `225_contrast`, `169_noncontrast`, and `169_contrast`.

The two acquisitions of one subject are not registered repeat measurements.
Motion, slab position and orientation, centre position, and adjustments
differ, and contrast enhancement changes the measured signal. Their PSFs are
therefore not compared as matched repeats, within-subject PSF differences are
not used to assess PSF correctness, and those differences do not reopen
refscan sets 0 through 3. If PSF fitting must be mentioned, the neutral
wording is: acquisition-specific PSF-fit diagnostic difference with unresolved
geometry, motion, contrast, and adjustment confounds.

The accepted `c=0` branches use:

```text
bart ecalib -m 1 -c 0 ...
bart wave -g -w -f -r 0 -i 100 -t 1e-6 ...
```

The current virtual-coil transform is a global covariance PCA/SVD estimated
from integrated set-4 ACS. It retains 12 virtual coils:

| Dataset arm | Physical coils | Ncc | Recorded retained ACS energy |
| --- | ---: | ---: | ---: |
| `225_noncontrast` | 52 | 12 | 0.91087 |
| `225_contrast` | 52 | 12 | 0.89157 |
| `169_noncontrast` | 40 | 12 | 0.89705 |
| `169_contrast` | 40 | 12 | 0.86135 |

The inspected sequence definitions describe a five-set integrated FLASH
refscan, but only zero-based set 4 is used for sensitivity-map calibration:

| Set | Sequence role | Used for CSM? |
| ---: | --- | --- |
| 0 | no-Wave, wide first projection / narrow second projection | No |
| 1 | sine-Wave partner of set 0 | No |
| 2 | no-Wave, narrow first projection / wide second projection | No |
| 3 | cosine-Wave partner of set 2 | No |
| 4 | central 32-by-32 no-Wave ACS | Yes |

For the inspected MPRAGE sequence:

- the refscan is a 3D slab-selective FLASH acquisition;
- set 4 has a 32-by-32 central PE region and full oversampled readout;
- calibration readout duration is 5.12 ms with 1024 samples, or 5 microseconds
  per sample;
- calibration TE is 4.3625 ms in the FOV-220 sequence used for subject 225
  and 4.3825 ms in the FOV-192 sequence used for subject 169;
- MPRAGE TE is 3.6325 ms in both sequences;
- measured MPRAGE dwell time is also 5 microseconds;
- every ADC of MPRAGE and of refscan sets 0 through 4 uses the same readout
  gradient event, a constant +195,312.5 Hz/m with the same polarity, so FLASH
  set 4 and MPRAGE share the readout gradient, polarity, 1024 samples, and
  5 us dwell, and their off-resonance readout displacement is the same:
  5.12 mm/kHz along readout; and
- MPRAGE contains one acquired contrast/echo, so it cannot directly provide a
  conventional phase-difference B0 map.

The production CSM path below selects the last refscan set under a
"at least five sets" guard. That equals set 4 for these files, which contain
exactly five sets with a complete, duplicate-free 32 x 32 set-4 lattice; the
diagnostic path requires set index exactly 4.

The current CSM preparation path is:

1. load the integrated refscan;
2. select `reference[:, :32, :32, 4, :]`;
3. remove fourfold readout oversampling with the validated alias-free
   IFFT-crop-FFT operation;
4. estimate the global physical-to-12-coil PCA basis from that ACS;
5. apply the same basis to measured Wave image data and ACS;
6. center-embed the compressed 32-by-32 ACS on the calibration grid; and
7. run one-map ESPIRiT calibration.

The ACS cropping operation is an accepted correction and is not a regression
hypothesis.

## Scope decision for refscan sets 0 through 3

Do not investigate or refit sets 0 through 3 in the initial stages.

A materially wrong Wave PSF usually produces a more global, directionally
structured spreading error across the object. The observed fringes are instead
anchored near metal-related signal null/displacement in the central FOV. That
pattern makes set-4 CSM inconsistency and missing B0 encoding more plausible
than a primary PSF-fit failure.

This is a prioritization decision, not proof that the PSF is exact. Reopen
sets 0 through 3 only if at least one reviewed trigger is met:

- set-4, convergence, PCA-24, whitening, and multi-map diagnostics fail to
  explain or alter the artifact;
- accepted PSF diagnostic residuals or coefficient curves are abnormal within
  one acquisition; or
- the artifact is also global and present away from metal-related regions.

Contrast and noncontrast acquisitions are not registered repeats, so
within-subject PSF differences, such as the observed 12.9% and 16.0% complex
PSF differences, are not evidence of a PSF error and are not a trigger.

Until a trigger is reviewed, reuse the accepted PSF exactly and do not add a
PSF parameter sweep. The fringe remains more consistent with metal-related
readout displacement and pile-up, set-4 CSM inconsistency, R = 3 unfolding
leakage, and Wave spreading than with a primary global PSF error.

## Why FLASH set 4 can be B0-sensitive

Coil sensitivity ratios ideally cancel object magnitude and common phase.
Simple FLASH contrast or a spatially smooth common B0 phase therefore need not
corrupt a CSM by themselves. The metal case is more difficult:

1. The low receiver bandwidth permits readout displacement.
2. Rapid subvoxel frequency variation causes dephasing and signal void.
3. Multiple source locations can pile up at one apparent image location.
4. Their received coil vectors are mixtures of sensitivities from different
   physical locations and need not be rank one.
5. Single-map ESPIRiT can then estimate a displaced, unstable, or incomplete
   CSM near the metal-affected region.
6. The Wave reconstruction subsequently combines that CSM with a B0-free,
   position-dependent PSF, allowing the local mismatch to spread coherently.

FLASH set 4 and MPRAGE use the same readout gradient, polarity, 1024 samples,
and 5 us dwell, so their off-resonance readout displacement is the same:
5.12 mm/kHz along readout. For an individual isochromat under a constant
readout gradient, off-resonance phase is equivalent to a readout-direction
shift, and Wave does not add EPI-like B0 distortion. Near metal, however,
non-invertible pile-up, intravoxel dephasing, excitation differences, signal
voids, and displaced or mixed coil sensitivities can still violate the
single-map SENSE model. The shared displacement can make the calibration
partly self-consistent, but it is not a physical B0 correction: the TE
(0.730 ms or 0.750 ms longer for FLASH), contrast, inversion preparation,
excitation pulse, temporal ordering, and pile-up mixtures differ.

The R = 3 LIN alias-partner mechanism, in which a locally inconsistent metal
region leaks through unfolding to positions about `round(N_LIN / 3)` LIN away
(about +-85 LIN for 256), is a testable hypothesis, not a conclusion. Stage 3
pre-registers it as an ROI test together with an edge-matched control.

The most informative initial question is therefore not merely whether the CSM
phase looks unusual. It is whether set-4 coil data require more than one local
coil-sensitivity component and whether that failure localizes to the metal and
fringe regions.

## Ranked hypotheses

### H1: solver convergence

The existing logs record `-i 100 -t 1e-6`, not the actual stopping iteration.
BART emits the final FISTA counter at `BART_DEBUG_LEVEL=4` and per-iteration
internal normalized residuals at level 5. Old logs cannot establish whether
the solver stopped early or hit the limit.

### H2: PCA-12 discards useful encoding information

PCA-12 retains only 86% to 91% of set-4 ACS energy in these acquisitions.
Low-energy physical-coil modes may still help distinguish displaced signal
from signal genuinely originating at the apparent location. PCA-24 is the
maximum planned control; no higher channel count is in scope.

### H3: correlated noise biases compression and conditioning

Every TWIX file contains explicit noise data in measurement 0, which is the
AdjCoilSens adjustment scan: 256 noise lines of 512 samples at 4 us dwell
with twofold oversampling, whereas MPRAGE and set 4 use 5 us. Both subject-225
files embed the same adjustment measurement, so `225_contrast` has no
independent noise scan; `169_contrast` has its own. MDH channel IDs are
identical and sequential in noise, image, and all refscan sets, and the
block-0 coil-element-to-ADC maps are identical between the adjustment and
acquisition headers. Raw-data correction is flagged on no MPRAGE or refscan
line, and FFT-scale metadata differ between measurements and are not applied.
Estimated noise covariance is materially non-diagonal:

| Acquisition | Median absolute off-diagonal correlation | 95th percentile | Covariance condition number |
| --- | ---: | ---: | ---: |
| `225_noncontrast`, `225_contrast` (shared scan) | 0.056 | 0.285 | 17.5 |
| `169_noncontrast` | 0.066 | 0.316 | 26.4 |
| `169_contrast` | 0.057 | 0.305 | 21.5 |

Prewhitening before PCA is therefore justified, subject to explicit channel,
scaling, and held-out-noise validation. Scaling the noise covariance by the
dwell ratio (0.8) is only a white-noise approximation and does not establish
absolute ACS noise calibration.

### H4: FLASH set-4 single-map CSM is locally inconsistent

Metal displacement and pile-up can make the set-4 coil vector locally
multi-component. ESPIRiT second-eigenvalue evidence and one-map versus two-map
coil-space projection residuals should be examined before reconstruction. The
phase-free projection residual `rho` is the primary diagnostic; the
noise-normalized RNR is a conditional secondary diagnostic that is interpreted
only after a matrix-level comparison of the noise-scan and empirical-air
covariances.

### H5: the forward model is missing B0 evolution

Even a perfect CSM cannot unshift or rephase data when the encoding operator
omits `exp(-i 2*pi*DeltaF(r)*t)`. This is the most physically complete
explanation if the earlier controls only partially help.

### H6: ROVir can suppress a separable nuisance subspace

ROVir is lower priority. The fringe and desired anatomy can overlap and share
coil modes, so nuisance suppression may remove anatomy without correcting
displacement. Reconsider it only with a reviewed, spatially disjoint nuisance
ROI and explicit target-retention assessment.

## Review stage 0: plan and scope review

Deliverables:

- this document;
- the ignored local source manifest with no output root selected; and
- a concise statement of the sets 0 through 3 deferral rule.

Review gate 0:

- The user confirms scope and priority.
- Stop after any requested plan edits. Do not begin implementation merely
  because this plan is approved.

## Review stage 1: evidence audit and diagnostic specification

This stage is read-only and produces no scientific output directory. It was
completed on 2026-09-29; the corrections it produced are listed under Review
status.

### Required checks

1. Verify every source TWIX, sequence, existing reconstruction root, command
   record, and source hash in the ignored local manifest.
2. Confirm sequence definitions for set identity, TE, dwell, readout axis,
   FOV, ACS width, and readout oversampling.
3. Confirm that noise, image, and refscan physical-channel IDs match exactly,
   not only in count.
4. Record raw-data-correction and channel-scaling flags. Do not silently apply
   scanner adjustment factors from another measurement.
5. Verify the current set-4 extraction and IFFT-crop-FFT implementation.
6. Verify the current BART build and its `ecalib`, `wave`, and debug-level
   behavior on the active host.
7. Inspect the historical two-map commit `b5f1eb5` only as a design reference.
   It predates current main and must not be merged or cherry-picked wholesale.

### Diagnostic specification to propose

The agent must propose exact definitions for:

- set-4 physical-coil RSS and signal-support masks;
- normalized local coil vectors, excluding low-signal voxels;
- ESPIRiT eigenvalue-map summaries for map sets 1 and 2;
- one-map and two-map coil-space projection residuals;
- local coil-vector rank or singular-value ratios in reviewed neighborhoods;
- CSM magnitude/phase smoothness diagnostics that avoid phase interpretation
  in low-signal voxels;
- geometry-bound metal/null, fringe, preserved-anatomy, and background ROIs;
- fixed display normalization and image comparison rules; and
- all manifest fields, hashes, commands, and version records.

For coil data `d(r)` and sensitivity matrix `S(r)`, the proposed projection
metric should have the form

```text
rho_M(r) = ||d(r) - S_M(r) S_M(r)^dagger d(r)||_2 / ||d(r)||_2,
```

where `M` is the number of map sets and `dagger` is a stable local
pseudoinverse. The implementation must define low-signal exclusion and
conditioning guards. A reduction from `rho_1` to `rho_2`, co-localized with a
second ESPIRiT eigenvalue near one and with the visible metal/fringe region,
would support H4. Calibration self-fit is optimistic, so it is evidence of
model insufficiency when it fails, not proof of generalization when it passes.

### Controls

- Contrast and noncontrast acquisitions of one subject are not registered
  repeat measurements. They are not reproducibility checks for calibration
  metrics or PSFs; report each acquisition separately.
- A non-metal control is desirable for false-positive calibration. The agent
  must propose a candidate and obtain user confirmation rather than selecting
  one from private data by assumption.
- If an independent MPRAGE coil-space consistency diagnostic is proposed, it
  must account for Wave encoding and undersampling. A naive zero-filled image
  must not be treated as ground-truth coil sensitivity.

Review gate 1:

- Submit the evidence table, diagnostic formulas, proposed CLI, output tree,
  estimated memory/runtime, and file-level implementation plan.
- Explicitly list unresolved scientific choices.
- Stop for code review. No implementation or scientific run follows
  automatically.

## Review stage 2: dataset-independent diagnostic implementation

Implement only after review gate 1 approval. Gate 1 approved Stage 2 code with
these scientific decisions:

- Approved: accepted PCA-12 data for ESPIRiT, eigenvalue, and projection
  diagnostics; physical-coil model-free local-rank diagnostics; Hann-apodized
  coil images as the primary variant and unapodized images as a secondary
  sensitivity analysis; the five-label geometry-bound ROI contract (metal
  void/null, metal pile-up/bright displacement, fringe, preserved anatomy,
  background air); the pre-registered +-85 LIN partner test and edge-matched
  control; `225_noncontrast` as the preferred future pilot only if the user
  confirms that its fringe is representative (the user confirmed it as the
  pilot at the Gate 2 v3 review); hashing each TWIX once during
  an authorized prepare stage; and reuse of the existing physical set-4
  exporter without changing ROVir code, documenting its inherited ROVir-named
  manifest status.
- Not approved: any Soft-SENSE run or `ecalib -S`; physical-coil ESPIRiT;
  PCA-24 before its reviewed stage; changed normal reconstruction defaults;
  edits to `mprage.py`, production launchers, converters, ROVir modules, or
  NIfTI collection without a reviewed blocker; investigation of refscan sets
  0 through 3; and any real-data BART `ecalib`, `wave`, `fft`, or `rss`.
- Physical-coil ESPIRiT is not implemented. With the default 24^3 calibration
  region, 40 or 52 coils give a wide, poorly constrained calibration matrix
  (8640 or 11,232 columns against 6859 rows), which makes estimator choice
  and conditioning questionable.
- Thresholds are pre-registered descriptive defaults, recorded in manifests
  and reports. They do not select a winner or prove a mechanism.

### Intended code boundary

- Add focused set-4 CSM diagnostic functions under `wave_retro_lr`.
- Add a thin explicit CLI under `scripts/`.
- Reuse the existing physical set-4 export and BART CFL utilities.
- Keep Python responsible for validation, provenance, metrics, and figures.
- Keep every BART invocation explicit in a reviewed shell entry point or
  printed command plan; library functions must not launch hidden BART jobs.
- Do not modify normal reconstruction defaults.

### Required tests

- exact set-4 selection and rejection of another set;
- IFFT-crop-FFT provenance and rejection of direct striding;
- complex coil-vector normalization with low-signal masking;
- one-map and two-map projection residuals on synthetic rank-one and rank-two
  data;
- stable pseudoinverse and conditioning failure cases;
- eigenvalue-map shape and map-dimension validation;
- geometry-bound ROI validation;
- fixed display scaling and deterministic metrics;
- source/hash/command/BART-version provenance;
- reuse rejection for incomplete or mismatched artifacts; and
- proof that no BART reconstruction or production output is launched by unit
  tests.

Every new function and method requires an English docstring describing inputs,
outputs, validation errors, and externally visible side effects.

Review gate 2:

- Run focused tests, then the full applicable suite.
- Present the complete diff, test output, sample synthetic diagnostic report,
  and exact proposed real-data command.
- Stop for code review before selecting or creating a real-data output root.

## Review stage 3: one-subject set-4 diagnostic pilot

Begin only after the user confirms the exact diagnostic output directory and
authorizes the run. Use one subject and one acquisition first; the user chooses
which arm has the clearest representative fringe. The user chose
`225_noncontrast`, confirmed its output root, and authorized the pilot on
2026-09-30 after Gate 2 approval.

The pilot is calibration-only. It must not run Wave reconstruction.

Required products:

```text
OUTPUT_ROOT/
  inputs/
    physical_calibration/
    noise/
  csm/
    map1/
    map2_uncropped/
    eigenvalues/
  rois/
    template/
    reviewed/
  diagnostics/
    calibration_views/
    coil_projection_residuals/
    local_rank/
    csm_coherence/
    roi_overlays/
  manifests/
  reports/
  logs/
```

The only proposed real-data calibration command is
`bart ecalib -m 2 -c 0 ...` on the accepted PCA-12 `kspace_calib`. It is
diagnostic only and produces two uncropped map sets and their eigenvalue
maps; it must not be used for Wave reconstruction. The accepted one-map CSM
remains the reconstruction baseline. Map components are inspected separately,
and RSS is never the only presentation.

Soft-SENSE is not part of Stage 2 or Stage 3 preparation. In BART v1.0,
`-S` weights each map by `s((sqrt(|lambda|) - c) / (1 - c))`; with the
accepted `c = 0` every weight stays at or above 0.5, and the upper branch of
`s` makes weights near 2 for `|lambda| >= 1`. Any later Soft-SENSE proposal
needs a separately reviewed crop value.

Review gate 3:

- Present map components separately, never only their RSS combination.
- Present eigenvalue maps, projection residuals, local-rank diagnostics,
  fixed-window figures, and ROI summaries.
- State whether evidence supports a locally multi-component FLASH ACS.
- Stop. Do not launch a two-map Wave reconstruction from calibration evidence
  alone.

## Review stage 4: instrumented convergence control

This is the first reconstruction stage and requires a separately confirmed
output root and command.

Use the existing accepted set-4/PCA-12/one-map/`c=0` inputs and FISTA-r0 only.
Do not run Wavelet or LLR. Capture `BART_DEBUG_LEVEL=5` and raise the maximum to
300 while retaining `-t 1e-6`. Compare the result against the existing
100-iteration FISTA-r0 image using fixed normalization and reviewed ROIs.

Required evidence:

- final FISTA counter;
- complete internal residual trace;
- command, BART version, runtime, and peak memory if available;
- relative image difference against iteration 100;
- fringe, preserved-anatomy, and background metrics; and
- fixed-window montages.

Decision rule:

- If the solver stopped before 100, or the residual and image are stable while
  the fringe persists, deprioritize convergence.
- If it runs materially beyond 100 and the fringe changes, review a convergence
  policy before any other reconstruction comparison.

Review gate 4:

- Present the convergence report and no unrelated algorithm changes.
- Stop for result and code review.

## Review stage 5: PCA-24 and prewhitening controls

No virtual-coil count above 24 is permitted in this investigation.

### PCA-24 control

Reuse the accepted PSF and set-4 ACS construction. Estimate one global PCA-24
basis and apply it identically to image and ACS. Re-estimate the CSM in the new
basis. Run only FISTA-r0 with the reviewed convergence policy.

### Prewhitened PCA-24 control

Estimate complex noise covariance from measurement-0 noise after demeaning.
Split samples into estimation and held-out validation subsets. Construct a
stable Hermitian whitening operator and verify on held-out noise that:

- the covariance is finite and positive definite after the reviewed numerical
  floor;
- channel IDs and ordering match the image and refscan;
- diagonal variance is close to one;
- off-diagonal correlation is substantially reduced; and
- the identical physical-channel whitening convention is applied to image and
  set-4 ACS before PCA estimation.

Record the covariance, whitening matrix, eigenvalue floor, split rule,
convention, hashes, and held-out validation statistics. Re-estimate PCA-24 and
CSMs after whitening.

If resources allow only one new PCA-24 reconstruction, the user must choose
between unwhitened PCA-24, which isolates coil count, and whitened PCA-24,
which is more principled but confounds whitening with coil count. Do not make
that tradeoff silently.

Review gate 5A:

- Review prewhitening/PCA implementation and synthetic tests before real data.

Review gate 5B:

- Review PCA spectrum, whitening validation, commands, resource estimate, and
  proposed output root before reconstruction.

Review gate 5C:

- Review fixed-window images, ROI metrics, data consistency, runtime, and
  memory after each authorized reconstruction. Do not batch additional arms
  before this review.

## Review stage 6: two-map Soft-SENSE Wave pilot

Proceed only if stage 3 shows credible second-map evidence or earlier controls
leave a clear single-map inconsistency.

Port the useful design from historical commit `b5f1eb5` onto current main.
Do not merge or cherry-pick the branch wholesale. Preserve all current ACS,
PSF, retrospective, ROVir, provenance, and NIfTI collection contracts.

Requirements:

- explicit opt-in map count and Soft-SENSE mode;
- isolated experimental output tree;
- exact `ecalib` command and eigenvalue-map provenance;
- converter support for the BART map dimension;
- separate magnitude and phase exports for component 1 and component 2;
- RSS only as an additional display, not the sole result;
- FISTA-r0 only for the first pilot; and
- tests proving the one-map default is unchanged.

Decision rule:

- Improvement means reduced reviewed fringe without material loss or
  redistribution of desired anatomy. Moving the fringe into component 2 and
  then restoring it in RSS is diagnostic, not a successful correction.

Review gate 6A:

- Review code and tests before any real-data command.

Review gate 6B:

- Review each map component, RSS, eigenvalue evidence, ROI metrics, residuals,
  and resource use before considering a second subject.

## Review stage 7: B0-aware reconstruction feasibility

Begin only after the lower-cost controls are reviewed. This stage starts with
a design report, not code.

### Known B0 map path

If a registered field map becomes available, add the sample-time phase term

```text
exp(-i * 2*pi * DeltaF(r) * t_n)
```

to the Wave forward and adjoint operators. Time or frequency segmentation will
likely be needed. The current `bart wave` interface has no B0-map input, so
this requires a repository-owned operator or another reviewed backend. Do not
modify the external BART submodule.

### B0 as a free variable

Joint image/B0 estimation from the current single-echo acquisition is possible
as a nonlinear inverse problem but is not the first implementation target.
Readout off-resonance is strongly confounded with spatial displacement;
unknown object phase absorbs constant-TE phase; and a signal void contains no
field information. Strong smoothness priors can also be wrong next to metal.

The feasibility report should evaluate, in order:

1. a separately acquired dual- or multi-echo field map;
2. acquisitions repeated with reversed readout polarity or a different bandwidth;
3. whether any existing auxiliary acquisition truly supplies compatible B0
   information; and
4. only then a regularized joint image/B0 model.

Do not infer that acquisition suffixes indicate reversed polarity. Verify
sequence and raw-header provenance.

Review gate 7:

- Present identifiability assumptions, acquisition requirements, operator
  equations, approximation error, synthetic validation design, expected
  memory/runtime, and failure modes.
- Stop for scientific design review before implementation.

## Comparison and reporting contract

Every reconstruction comparison must hold constant all factors not named by
the experiment, including PSF, sampling, reconstruction regularization,
conversion, orientation, intensity normalization, display window, and ROI
geometry.

The initial matrix is sequential rather than a batch:

| Order | Experiment | Changed factor | Stop condition |
| ---: | --- | --- | --- |
| 1 | Set-4 calibration pilot | CSM diagnostics only | Review calibration evidence |
| 2 | PCA-12 convergence | iteration ceiling and logging | Rule out or address convergence |
| 3 | PCA-24 | coil count | Review before whitening run |
| 4 | Prewhitened PCA-24 | whitening plus, if necessary, coil count | Review attribution limits |
| 5 | Two-map Soft-SENSE | CSM model rank | Run only with map-2 evidence |
| 6 | B0-aware design | forward model | Separate research decision |

At minimum, reports must include:

- fixed-window slice montages covering every affected slice range;
- fringe-region energy and robust intensity summaries;
- preserved-anatomy signal and edge comparisons;
- outside-head energy, reported separately from in-head fringe;
- solver/data-consistency evidence appropriate to the method;
- map-component and eigenvalue diagnostics where applicable;
- commands, logs, environment, versions, runtimes, and memory;
- source and output hashes; and
- a limitations section that distinguishes displacement correction from
  unrecoverable dephasing or unacquired signal.

Metrics do not select a winner automatically. Final acceptance requires user
visual review.

## Explicit exclusions

- No further hard `ecalib -c` sweep.
- No reversal of the accepted alias-free ACS readout crop.
- No new Wavelet or LLR tuning.
- No virtual-coil count above 24.
- No default ROVir experiment without a separable nuisance ROI.
- No sets 0 through 3 PSF refit unless a documented trigger is reviewed.
- No claim that multi-map ESPIRiT or ROVir is a physical B0 correction.
- No claim that reconstruction can recover signal destroyed by intravoxel
  dephasing or shifted outside excitation/receiver bandwidth.

## Completion criteria

The investigation is complete only when the reviewed evidence supports one of
these outcomes:

1. a validated, bounded pipeline change materially reduces fringes without
   unacceptable anatomy loss and is documented with reproducible provenance;
2. the artifact is localized to missing B0 physics and a separately reviewed
   B0-aware implementation or acquisition plan is accepted; or
3. the available single-echo acquisition is shown insufficient for further
   reliable correction, with negative results and limitations documented.

Completion does not authorize changing defaults, deleting earlier outputs, or
promoting an experimental result. Those require separate user review.

## Scientific references

- Bilgic et al., Wave-CAIPI for highly accelerated 3D imaging:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4281518/>
- Wave off-resonance behavior in a Wave-CAIPI application:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4691433/>
- Uecker et al., ESPIRiT and multiple sensitivity-map sets:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/>
- Kim et al., region-optimized virtual coils:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8248187/>
- Metal-induced displacement, pile-up, and signal-loss review:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5562503/>
- Off-resonance correction review and model-based reconstruction context:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10284460/>
