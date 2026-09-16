# Native-grid R3x3 synthetic-Wave GRE sweep

Updated: 2026-09-15

This extension adds only `native_r3x3`; it does not add a low-resolution R3x3
case and does not modify the completed R3x1/R3x2 run. Both GRE echoes remain
separate reconstruction jobs, while review uses identical candidate settings
and pooled two-echo metrics for a shared-lambda decision.

The pure image-lattice mask is defined on the 250-by-72 LIN/PAR grid with
acceleration `(3, 3)` and GRE-derived residue `(2, 0)`. Its exact acquired
coordinate count is 1992 and its logical SHA-256 is
`e57069cd4f3cc9af4a78e70cb10f66b79ad1ceb35efac966c871df3822febefa`.
It is an exact subset of the completed native-R3x1 mask. ACS/refscan samples
remain separate and are never unioned into reconstruction k-space.

## End-to-end pipeline

This document is the canonical procedural record for the GRE path from
synthetic data through parameter selection and presentation. The downstream
measured-data implementation is deliberately a separate stage:

```text
fully sampled two-echo no-Wave GRE
        │
        ├── validate/crop to native 250 x 250 x 72
        ├── preserve echo times 10 ms and 20 ms and their relative scaling
        ├── synthesize Wave encoding with the 20260821 sequence-theoretical PSF
        └── retain CSM, direct-FFT references, and approved BET mask separately
                │
                ▼
native-R3x3 pure Cartesian sampling case
        │
        ├── acceleration (3, 3), GRE-derived residue (2, 0)
        ├── exact mask count/hash and native-R3x1 subset proof
        ├── bitwise acquired-sample equality
        └── exact zeros outside the image-lattice mask; no ACS/refscan union
                │
                ▼
GPU coarse sweep, both echoes reconstructed separately
        │
        ├── FISTA lambda-zero control
        ├── Wavelet coarse grid
        └── corrected LLR block 4/8/16 coarse grids
                │
                ▼
per-echo metrics + cross-echo metrics + fixed-window figures
                │
                └── explicit human review chooses the fine family/grid
                                │
                                ▼
Wavelet-only fine sweep with echo-matched lambdas
        │
        ├── per-echo magnitude/phase evaluation
        ├── pooled two-echo global metrics
        └── magnitude-ratio and delta-B0 preservation metrics
                │
                └── explicit shared-lambda decision: Wavelet lambda=0.015
                                │
                                ├── manifested presentation package
                                │   FISTA + selected Wavelet, both echoes
                                │
                                └── parameter/provenance handoff only
                                                │
                                                ▼
wave_retro_lr_recon measured GRE native-R3x3 implementation
```

The coarse and fine evaluators never select a winner. The transition from
coarse to fine and the final shared-lambda choice are manual gates. A metric
minimum becomes a downstream default only after explicit review; the
presentation package remains a derived deliverable and does not replace the
sweep/evaluation provenance.

## Reuse and output layout

The ignored local configuration pins completed manifests for the no-Wave
source, native CSM, 20260821 theoretical PSF, both direct-FFT references,
approved BET mask, and prepared cases. `validate-reused-inputs` is read-only
and verifies those bindings, payload hashes, geometry, finite values, and the
R3x3 subset contract. `prepare-reused-r3x3` then creates R3x3 k-space by
bitwise-exact subsetting of native-R3x1 synthetic-Wave k-space; it does not run
TWIX import, ecalib, PSF generation, direct FFT, or BET.

The confirmed output root is organized as:

```text
gre_sweep_native_r3x3/
  run_manifest.json
  preparation/
    reused_inputs/manifest.json
    cases/
      manifest.json
      native_r3x3/
        sampling_mask.npy
        echo-01/{wave_kspace.hdr,wave_kspace.cfl,manifest.json}
        echo-02/{wave_kspace.hdr,wave_kspace.cfl,manifest.json}
  reconstructions/{coarse,fine}/native_r3x3/echo-XX/<candidate>/
  evaluation/{coarse,fine}/
  evaluation/{coarse,fine}_shared_lambda/
  figures/{coarse,fine}/{fixed_window,metric_curves}/
  figures/{coarse,fine}_shared_lambda/
  selections/refinement_request.json
  presentation/
    manifest.json
    niftis/
    tiff_images/
    metric_curves/
    metrics/
```

## Lambda policy and tmux stages

Coarse uses one FISTA lambda-zero control, Wavelet at
`[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`, and LLR block sizes 4/8/16 at the same five
lambdas: 21 jobs per echo and 42 total. Manual coarse review retained only
Wavelet for fine evaluation. The completed echo-matched fine grid was:

```text
0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050
```

This gives eight fine jobs per echo and 16 total. No fine LLR job was run, and
no candidate was selected automatically.

In one existing tmux session:

```bash
RUNNER=tools/synthetic_wave_for_reg_baseline/scripts/run_gre_synthetic_wave_r3x3.local.sh
$RUNNER validate
$RUNNER validate-reused-inputs
$RUNNER prepare-case
$RUNNER coarse-check
$RUNNER coarse
$RUNNER coarse-nifti
$RUNNER coarse-evaluate
$RUNNER coarse-plot
$RUNNER coarse-shared-evaluate
$RUNNER coarse-shared-plot
$RUNNER record-refinement "$USER"
$RUNNER fine-check
$RUNNER fine
$RUNNER fine-nifti
$RUNNER fine-evaluate
$RUNNER fine-plot
$RUNNER fine-shared-evaluate
$RUNNER fine-shared-plot
```

Every BART reconstruction uses the host-selected `bart` and GPU flag. The
ordinary evaluation retains per-echo magnitude/phase metrics and same-setting
inter-echo scaling and delta-B0 metrics. Shared-lambda evaluation pools both
echoes independently for each Wavelet or LLR block curve. All figures use
candidate-independent fixed windows and contain no winner marker.

## Reviewed shared-lambda decision

The explicit native-R3x3 GRE choice is Wavelet `lambda=0.015`, shared by both
echoes while the echo images remain separate reconstructions. It is GRE-
specific and must not be replaced by the MPRAGE native-R3x3 value.

At `0.015`, the pooled two-echo evaluation reached its best global magnitude
NRMSE (`0.0331401`), best global magnitude NCC (`0.996136`), best pooled phase
p95 error (`0.118964 rad`), and best circular dispersion (`0.00274572`). Echo 1
clearly favored `0.015`; echo 2 was nearly flat between `0.015` and `0.020`.
The latter slightly improved SSIM, gradient, and delta-B0 metrics, but manual
review chose `0.015` as the better overall image-fidelity tradeoff.

## Presentation contract

`build_gre_r3x3_presentation.py` builds or safely refreshes the run-relative
`presentation/` folder. The completed package contains only native-R3x3 GRE
artifacts:

- eight canonical-RAS NIfTIs: magnitude and wrapped phase for FISTA and
  Wavelet `0.015`, for both echoes;
- twelve uint16 center-slice TIFFs: sagittal, coronal, and axial for both
  reconstructions and both echoes;
- three metric curves: two per-echo fine curves and one pooled shared-lambda
  curve;
- two CSVs: per-echo fine metrics and pooled shared-echo metrics; and
- one presentation manifest binding the manual selection, sources, copies,
  TIFF display parameters, and SHA-256 identities.

NIfTIs are byte-identical copies of the validated exports and are not
resampled. TIFFs are display derivatives only. The presentation manifest
records FISTA as the control and Wavelet `0.015` as the manual shared-lambda
selection; it does not authorize any other case or sequence family.

## Handoff to `wave_retro_lr_recon`

Only the following synthetic-study conclusions transfer into the measured-Wave
GRE native-R3x3 implementation:

- reconstruction family: Wavelet;
- shared lambda across echoes: `0.015`;
- retain a FISTA lambda-zero control branch;
- native target geometry: `250 x 250 x 72`, nominal FOV
  `220 x 220 x 180 mm`, and extended Wave readout 1000;
- reconstruct 10-ms and 20-ms echoes separately while preserving relative
  magnitude, phase, magnitude ratio, and delta-B0 behavior; and
- pure Cartesian native-R3x3 sampling with a residue derived from the measured
  GRE acquisition/grid contract, not copied from MPRAGE.

Synthetic k-space, its theoretical PSF, synthetic CSM, direct-FFT references,
BET mask, normalization factors, and presentation files do not become
measured-data reconstruction inputs. `wave_retro_lr_recon` must use the
measured GRE image stream, keep ACS/refscan calibration separate, construct or
reuse a measured-data CSM under its own provenance checks, and use the
appropriate calibrated measured-Wave PSF. Geometry, finite-value, mask,
acquired-sample, zero-outside-mask, echo, and provenance checks remain required
on that side of the boundary.

The synthetic tool may reuse shared utilities from `wave_retro_lr_recon`; the
measured reconstruction tool must not depend at runtime on synthetic output
directories or import the experiment workflow. The reviewed method/lambda and
its manifest provenance are configuration inputs, not a reverse code or data
dependency.

The implemented measured-data entry points are
`sample_gre_retro_r3x3_recon.sh` for focused resumable R3x3 work and
`sample_gre_retro_lr_recon.sh` for the combined R3x2/LR/R3x3 workflow. The
focused path supports source-bound legacy normal-artifact reuse without
rewriting the historical manifest or recalibrating per-echo PSFs. The GRE
NIfTI collector discovers `native_r3x3` and atomically appends it to an intact
older collection. These code contracts remain unit-tested rather than
real-data validated until the user runs and visually reviews both echoes and
both reconstruction branches.

## Artifact authority and future re-entry

| Decision | Authoritative artifact |
| --- | --- |
| Reused source/CSM/PSF/reference/mask identity | `preparation/reused_inputs/manifest.json` |
| R3x3 mask, geometry, echo, and BART inputs | `preparation/cases/native_r3x3/echo-XX/manifest.json` |
| Executed candidates | `reconstructions/{coarse,fine}/sweep_manifest.json` and candidate manifests |
| Per-echo quantitative evidence | `evaluation/{coarse,fine}/evaluation_manifest.json` |
| Pooled echo-matched evidence | `evaluation/{coarse,fine}_shared_lambda/evaluation_manifest.json` |
| Fine-grid authorization | `selections/refinement_request.json` |
| Reviewed choice and presentation copies | `presentation/manifest.json` |

For a later session:

1. Read the applicable `AGENTS.md`, this document, and the current handover.
2. Validate manifests and hashes; do not infer completion from filenames.
3. Treat Wavelet `0.015` as a GRE native-R3x3, two-echo shared-lambda choice
   only.
4. Keep the completed R3x1/R3x2 and R3x3 synthetic outputs immutable.
5. Use the measured GRE entry points with their own source-bound inputs and
   provenance; retain FISTA and review both echoes before calling real data
   validated.
6. Run both tool test suites, upstream tests where applicable, privacy checks,
   and `git diff --check` before commit or push.
