# Synthetic MPRAGE regularization-to-reconstruction pipeline

Updated: 2026-09-15

This document is the canonical procedural reference for the MPRAGE workflow
that starts with fully sampled no-Wave data, selects Wave reconstruction
regularization with a coarse-to-fine synthetic experiment, and transfers only
the reviewed parameter choice into `wave_retro_lr_recon` for measured-Wave
reconstruction.

The two tools have deliberately different responsibilities:

```text
wave_retro_lr_recon reusable geometry, sampling, BART I/O, and conversion code
                              ↑
synthetic_wave_for_reg_baseline experiment preparation, sweeps, metrics, review
                              │
                              └── reviewed method/lambda decision
                                      ↓
wave_retro_lr_recon measured-Wave user-facing reconstruction scripts
```

`synthetic_wave_for_reg_baseline` may import `wave_retro_lr_recon`. The reverse
dependency is prohibited. A completed synthetic experiment supplies scientific
parameter provenance, not a runtime dependency for measured reconstruction.

## Scientific boundary

The synthetic experiment answers one question: for a specified MPRAGE image
grid and pure Cartesian undersampling lattice, which BART Wave regularization
setting provides the preferred fidelity-versus-artifact tradeoff?

The following invariants apply throughout the experiment:

- Wave k-space is synthesized from fully sampled no-Wave image k-space with
  the theoretical sequence trajectory. A calibrated PSF is appropriate for
  measured-Wave data, not for this synthetic forward model.
- Sampling masks contain only the Cartesian image lattice. Refscan or image
  ACS remains a separate calibration input and is never unioned into Wave
  reconstruction k-space. Historical ACS-union masks are rejected.
- Existing accepted, case-compatible CSMs, theoretical PSFs, direct-FFT
  references, and the approved BET brain mask are reused only after strict
  hash, provenance, FOV, dimension, and finite-value validation. The sweep
  does not rerun ecalib, PSF calibration, direct FFT, or BET.
- Acquired Wave samples must be bitwise equal to the full synthetic-Wave
  source at mask locations, and every sample outside the mask must be exactly
  zero.
- Every candidate for one case changes only the regularization method,
  lambda, and, for LLR, block size. Solver iterations, tolerance, CSM, PSF,
  k-space, geometry, and BART build remain fixed and are recorded.
- Metrics and figures never choose a winner automatically. Fine settings,
  shortlist membership, and the final selection are explicit user decisions.
- Real paths and dataset names belong only in ignored `.local.json` and
  `.local.sh` files or private output manifests. Tracked examples use
  placeholders.

For LR cases, the primary fidelity reference is the direct FFT on the same
cropped, lower-resolution grid. A matched-1-mm comparison is a secondary
cross-resolution diagnostic. This avoids asking an LR reconstruction to
recover spatial frequencies that were intentionally removed, which is the
appropriate contract when the reconstructed LR volume will become input to a
super-resolution model.

Native cases on the same grid may reuse hash-identical full synthetic-Wave
k-space and direct-FFT content before masking. LR crops change those arrays and
therefore require case-specific, resolution-matched artifacts.

## Configuration and output ownership

Start from one of these tracked contracts:

- `configs/pure_mask_regularization_rerun.example.json` for native R3x1,
  native R3x2, LR-X R3x2, LR-Y R3x2, and LR-XY R3x2;
- `configs/native_r3x3_pure_mask_sweep.example.json` for the independent
  native-grid R3x3 experiment.

Copy the selected example to its ignored `.local.json` counterpart and fill in
the exact paths, file hashes, manifest hashes, and provenance assertions. The
output root must be empty, explicitly reviewed, and confirmed before any
production-writing action. Do not point a new run at frozen or previously
accepted output trees.

The R3x3 launcher follows the same pattern:

```bash
cp tools/synthetic_wave_for_reg_baseline/scripts/run_native_r3x3_pure_mask_sweep.example.sh \
   tools/synthetic_wave_for_reg_baseline/scripts/run_native_r3x3_pure_mask_sweep.local.sh
```

Set only the private environment paths in the ignored local copy. The launcher
does not create tmux; run each action sequentially in one user-managed tmux
session.

The owned output layout is:

```text
OUTPUT_ROOT/
├── preparation_manifest.json
├── cases/<case>/
│   ├── case_manifest.json
│   ├── sampling_mask.npy
│   └── bart_inputs/
│       ├── manifest.json
│       ├── wave_kspace.{hdr,cfl}
│       ├── coil_sens.{hdr,cfl}
│       └── psf.{hdr,cfl}
├── sweeps/
│   ├── coarse/<case>/<candidate>/
│   ├── coarse/sweep_manifest.json
│   ├── fine/<case>/<new-candidate>/
│   └── fine/sweep_manifest.json
├── evaluation/
│   ├── coarse/{metrics.csv,evaluation_manifest.json,<case>/...}
│   ├── fine/{metrics.csv,evaluation_manifest.json,<case>/...}
│   └── review/{shortlist_manifest.json,selection_manifest.json,...}
└── presentation/
    ├── niftis/
    ├── center_slices/
    ├── metrics.csv or metrics/
    └── presentation_manifest.json
```

Accepted CSM, PSF, and reference files may appear as links, but their payloads
and resolved provenance remain hash-bound. Candidate directories contain the
exact BART command, log, complex CFL output, restored magnitude and phase NPY
files, and a completion manifest. Resume accepts a candidate only when its
command, BART identity, prepared-case manifest, and output hashes still match.

## Coarse-to-fine execution

Use the local launcher as a short alias:

```bash
RUNNER=tools/synthetic_wave_for_reg_baseline/scripts/run_native_r3x3_pure_mask_sweep.local.sh
```

For the five-case run, use the corresponding ignored
`run_pure_mask_rerun.local.sh`. The dispatcher and action semantics are the
same.

### Prepare and validate

If deleted-but-rebuildable full source arrays must be restored, run the
dispatcher’s `validate-sources` and `materialize-sources` actions first. The
materialized arrays must reproduce their accepted hashes exactly. When those
sources already exist, begin with:

```bash
$RUNNER validate-inputs
$RUNNER prepare
```

`validate-inputs` is read-only. `prepare` revalidates the immutable contract,
creates the pure image-lattice mask and masked BART k-space, and records exact
mask count/hash, acquired-sample equality, zero-outside-mask, and CSM/PSF
geometry. It does not reconstruct an image.

### Run the fixed coarse grid

```bash
$RUNNER validate-coarse
$RUNNER run-coarse
$RUNNER evaluate-coarse
```

The fixed coarse pool contains 23 candidates per case:

| Family | Settings |
| --- | --- |
| FISTA control | Wavelet solver, `lambda=0` |
| Wavelet | `0.002, 0.005, 0.01, 0.015, 0.022, 0.03, 0.05` |
| LLR block 4 | `0.002, 0.005, 0.01, 0.02, 0.04` |
| LLR block 8 | `0.002, 0.005, 0.01, 0.02, 0.04` |
| LLR block 16 | `0.002, 0.005, 0.01, 0.02, 0.04` |

The exact GPU BART forms are:

```text
bart wave -g -w -f -r 0          -i 100 -t 1e-6 CSM PSF KSPACE OUTPUT
bart wave -g -w -f -r LAMBDA     -i 100 -t 1e-6 CSM PSF KSPACE OUTPUT
bart wave -g -l -v -b BLOCK -f -r LAMBDA -i 100 -t 1e-6 CSM PSF KSPACE OUTPUT
```

The synthetic sweep always uses `-g`. LLR uses BART split-complex output and
the shared utility recombines it into the native complex representation before
metric evaluation.

### Review and declare the fine grid

Review each family’s metric curve and fixed-window orthogonal figure. Do not
choose a family from a single scalar metric. Consider brain NRMSE, 3D SSIM,
NCC, gradient NCC, edge-gradient preservation, background QC, and visible
ringing, blur, or residual aliasing together.

Then add `fine_sweep` to the ignored local JSON. It must bind the completed
coarse sweep and coarse evaluation manifests and explicitly list every new
candidate per case. The fine runner neither interpolates a grid nor infers a
winner.

```bash
$RUNNER validate-fine
$RUNNER run-fine
$RUNNER evaluate-fine
```

The fine sweep manifest aggregates the hash-bound coarse candidates with the
new fine candidates for evaluation. Existing coarse candidates are reused;
they are not reconstructed again.

The completed five-case experiment refined only Wavelet:

| Cases | Additional fine Wavelet lambdas |
| --- | --- |
| native R3x1, native R3x2 | `0.025, 0.0275, 0.0325, 0.035, 0.04` |
| LR-X, LR-Y, LR-XY R3x2 | `0.0175, 0.02, 0.025, 0.0275, 0.0325, 0.035` |

The native R3x3 coarse optimum reached the upper explored region, so its
additional Wavelet grid extended upward:

```text
0.04, 0.045, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09, 0.10
```

The coarse `0.05` candidate remained available through aggregation. LLR was
not fine-swept after review showed that Wavelet was the relevant family.

### Shortlist and record the user decision

After reviewing fine metrics and figures, add `manual_shortlist` and
`manual_final_selections` to the ignored local JSON, then run:

```bash
$RUNNER validate-shortlist
$RUNNER render-shortlist
# Review the rendered fixed-window shortlist before continuing.
$RUNNER record-selections "Concise description of the completed visual review"
```

`record-selections` requires the explicit visual-review acknowledgement and a
nonempty reviewer note. Its selection manifest is the sole authority for the
chosen method/lambda. Metric leaders are descriptive and no composite score is
used.

## Evaluation and presentation conventions

All quantitative fidelity metrics use the approved BET brain support. Each
candidate receives one BET-restricted least-squares intensity scale to its
resolution-matched direct-FFT reference. Figures use that same scale and a
shared direct-FFT positive-value p99.5 window, so apparent brightness changes
do not masquerade as regularization quality.

The evaluator records brain NRMSE, RMSE, MAE, NCC, 3D SSIM, fixed-edge
gradient NCC, edge-gradient preservation ratio, background mean/SD QC, and a
missed-anatomy fraction. Background measures are QC only. There is no
candidate-specific registration, DICOM intensity ranking, or automatic
composite selection.

After the selection manifest is complete, validate and build the standard
presentation package:

```bash
$RUNNER validate-presentation
$RUNNER build-presentation
```

`build_pure_mask_presentation.py` exports the FISTA control and approved
regularized magnitude for every configured case as canonical-RAS float32
NIfTIs, three center-slice uint16 TIFFs per reconstruction, selected/control
metric rows, and hash-backed manifests. NIfTIs retain raw reconstructed
intensity and undergo no spatial resampling. TIFFs use the evaluation scale and
shared reference window.

A reduced slide-only package may intentionally omit controls and references.
In that case it must remain a derived deliverable: retain only manifest-bound
selected NIfTIs/TIFFs, include the relevant positive-lambda family metrics and
curve, identify the selected point, state `baselines_exported=false`, and
record hashes in its own presentation manifest. It must not replace the sweep,
evaluation, shortlist, or selection manifests.

The completed native-R3x3 reduced package follows this convention: one
Wavelet-`0.045` NIfTI, three orthogonal center-slice TIFFs, all 16 positive-
lambda Wavelet rows from the aggregated fine evaluation, and one metric curve
with `0.045` marked. It contains no FISTA, direct-FFT, or LLR deliverable.

## Reviewed MPRAGE decisions

The current explicit selections are:

| Synthetic case | Reviewed method | Lambda | Selection-manifest SHA-256 |
| --- | --- | ---: | --- |
| native R3x1 | Wavelet | `0.035` | `07cd8fe9...ea90f6` |
| native R3x2 | Wavelet | `0.035` | `07cd8fe9...ea90f6` |
| LR-X R3x2 | Wavelet | `0.025` | `07cd8fe9...ea90f6` |
| LR-Y R3x2 | Wavelet | `0.025` | `07cd8fe9...ea90f6` |
| LR-XY R3x2 | Wavelet | `0.022` | `07cd8fe9...ea90f6` |
| native R3x3 | Wavelet | `0.045` | `07fec187...b6002d` |

The full five-case selection hash is
`07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`.
The independent native-R3x3 selection hash is
`07fec1879821dcef6cd177766224f23930a0c556c96a28055a339c6530b6002d`.
Native R3x3 was evaluated only on the native grid; no R3x3 LR case was run.

These values are MPRAGE-specific. They must not be transferred to GRE, to an
unreviewed acquisition/contrast, or to fully sampled R1 normal reconstruction.

## Transfer into measured-Wave reconstruction

Only the reviewed method/lambda decision transfers from the synthetic study.
Synthetic k-space, theoretical PSF, CSM, direct-FFT reference, BET mask,
intensity scale, and presentation files do not transfer.

Measured-Wave reconstruction uses the measured dataset’s own validated image
k-space, separate calibration k-space, native CSM, and calibrated PSF. The
user-facing scripts preserve an unregularized FISTA-r0 branch beside the
selected Wavelet branch:

MPRAGE PSF coefficient processing defaults to automatic `sine-line`. Existing
normal inputs are reused only when their source and coefficient-processing
metadata match exactly; changing the mode or reviewed fit bounds requires a
new compatible preparation. This measured-data calibration policy is
independent of the theoretical PSF used by the synthetic sweep.

| Measured case | Entry point | Wavelet lambda |
| --- | --- | ---: |
| native measured R3x1 | `sample_mprage_normal_recon.sh` | `0.035` |
| retrospective native R3x2 | `sample_mprage_retro_lr_recon.sh` | `0.035` |
| retrospective LR-X R3x2 | `sample_mprage_retro_lr_recon.sh` | `0.025` |
| retrospective LR-Y R3x2 | `sample_mprage_retro_lr_recon.sh` | `0.025` |
| retrospective LR-XY R3x2 | `sample_mprage_retro_lr_recon.sh` | `0.022` |
| independent retrospective native R3x3 | `sample_mprage_retro_r3x3_recon.sh` | `0.045` |

For native R3x3:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_retro_r3x3_recon.sh \
    /path/to/measured_r1_or_r3x1_wave_mprage.dat \
    /path/to/output_root \
    /path/to/matching_wave_mprage.seq \
    -g
```

The R3x3 script is independent of the standard R3x2/LR script. It accepts a
fully sampled measured-Wave R1 source or a compatible regular R3x1 source. R1
uses reviewed LIN residue 1; R3x1 inherits its measured LIN residue. PAR uses
the center-aligned residue `(Npar // 2) mod 3`. It prepares only
`retro/native_r3x3`. Compatible normal preparation, native CSM, and calibrated
PSF are reused; if normal inputs are absent, the script prepares them and runs
ecalib once before the R3x3 branches. It then runs exactly FISTA-r0 plus
Wavelet `0.045`.

The measured scripts default to CPU for portability and accept `-g` for BART
GPU execution. This differs intentionally from the synthetic sweep, whose
local tool contract requires GPU BART.

After any desired measured reconstructions finish, synchronize the NIfTI
collection:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_nifti_collection.sh \
    /path/to/output_root
```

The collection discovers every populated `normal/nifti/<branch>` and
`retro/<case>/nifti/<branch>`, including `native_r3x3`. Re-running it validates
the existing collection and adds newly discovered branches without rerunning
preparation, ecalib, PSF calibration, or Wave reconstruction.

## Verification checklist

Before accepting a new sweep or transfer, verify:

1. the local config and launcher are ignored by Git;
2. the output root is new and explicitly confirmed;
3. masks are pure Cartesian lattices with exact count/hash and no ACS union;
4. acquired samples equal their source and all outside-mask samples are zero;
5. CSM/PSF shapes, FOV, finite values, normalization, and provenance pass;
6. every BART candidate has an exact command/log and complete hash-bound
   manifest;
7. coarse and fine metrics use the approved BET mask and resolution-matched
   direct FFT;
8. shortlist and final selection were explicitly reviewed by the user;
9. presentation files are fully owned by their manifest; and
10. the measured implementation transfers only the selected lambda and keeps
    FISTA-r0 available.

Run both tool suites after implementation changes:

```bash
python -m unittest discover \
    -s tools/synthetic_wave_for_reg_baseline/tests \
    -p 'test_*.py'
python -m unittest discover \
    -s tools/wave_retro_lr_recon/tests \
    -p 'test_*.py'
```

Also inspect `git diff --check`, repository status, and tracked text for private
paths before review or commit. Scientific reconstruction jobs remain
user-launched, one named action at a time in tmux.
