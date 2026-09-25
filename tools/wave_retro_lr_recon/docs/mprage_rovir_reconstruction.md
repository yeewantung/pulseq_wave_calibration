# Optional MPRAGE ROVir reconstruction

## Scope

ROVir is an optional troubleshooting and recovery feature for Wave-MPRAGE data
with coherent shoulder or extended-body signal wrapped into the head. It is not
part of the normal MPRAGE workflow and does not require the standard normal
reconstruction to have completed.

ROVir is currently supported only for MPRAGE. It is not exposed for GRE. The
GRE acquisition FOV normally avoids the extended-body shoulder-wrap failure
mode that motivated this implementation, and no GRE ROVir workflow has been
validated.

The implementation uses native BART `rovir`. BART does not detect an ROI: the
user reads native array indices from the corrected physical-coil ACS RSS and
explicitly supplies a nuisance region, or explicitly chooses the conservative
recommendation. For the scientific rationale, backend decision, mask contract,
and development history, see
[`mprage_rovir_diagnostic_plan.md`](mprage_rovir_diagnostic_plan.md).

## Prerequisites

Start from an existing reconstruction root with accepted prepared MPRAGE
inputs, including:

- `normal/bart_inputs/manifest.json`;
- the accepted case-matched PSF recorded by that manifest; and
- the original TWIX and sequence at the paths recorded by the normal manifest.

A standard normal CSM, ecalib command, FISTA output, and NIfTI are optional.
When exactly one standard FISTA-r0 magnitude exists, including below a nested
`sub-normal/` directory, final QC compares it with ROVir. Otherwise ROVir writes
standalone fixed-window QC and records why the standard comparison was skipped.

Activate the tool's Python environment and host-compatible BART build before
running the commands. The public ROVir interface validates the explicit source
paths against the prepared-input manifest. It has no separate public
configuration file.

ROVir writes only below `normal/rovir/` and leaves the standard normal branch
unchanged.

## Optional step: inspect the corrected ACS

Use the same positional argument order as the normal and retrospective MPRAGE
launchers. The command validates the supplied TWIX and sequence against the
prepared-input manifest before reading or writing ROVir artifacts.

```bash
scripts/sample_mprage_rovir_recon.sh inspect \
    /path/to/measured_wave_mprage.dat \
    /path/to/existing_reconstruction_root \
    /path/to/matching_wave_mprage.seq
```

Run `inspect` when the ROI location is not already known or when using
`--use-recommended`. It:

1. validates accepted prepared inputs and their source contract;
2. exports or strictly reuses corrected, alias-free physical-coil set-4 ACS;
3. runs explicit BART IFFT and RSS commands;
4. writes RO, LIN, and PAR views with native array-index ticks; and
5. writes an auditable conservative null-ROI recommendation when separation
   is sufficiently clear.

The recommendation is not an anatomical detector and is never approved
automatically. It may report `no_safe_automatic_recommendation`. Review:

```text
normal/rovir/feasibility/diagnostics/calibration_views/
normal/rovir/feasibility/diagnostics/roi_recommendation/
```

## Specify one null-region union and reconstruct

If the recommendation is appropriate:

```bash
scripts/sample_mprage_rovir_recon.sh run \
    /path/to/measured_wave_mprage.dat \
    /path/to/existing_reconstruction_root \
    /path/to/matching_wave_mprage.seq \
    --use-recommended \
    --virtual-coils 24
```

Alternatively, supply any number of inclusive native-grid boxes:

```bash
scripts/sample_mprage_rovir_recon.sh run \
    /path/to/measured_wave_mprage.dat \
    /path/to/existing_reconstruction_root \
    /path/to/matching_wave_mprage.seq \
    --null-box "ro=0:20,lin=0:25,par=all" \
    --null-box "ro=0:20,lin=48:71,par=all" \
    --virtual-coils 24
```

An explicit `--null-box` or `--null-box-file` may be run directly without a
prior `inspect` command. When the indexed ACS diagnostics are absent, `run`
creates them automatically for provenance and troubleshooting, then continues
without another review prompt. `--use-recommended` still requires a completed
`inspect`, because the generated recommendation must be reviewed before use.

The same records may be provided as a path-free JSON list with
`--null-box-file`. Inline boxes and a box file are mutually exclusive. Boxes
may overlap. Exact duplicates are removed, the final binary union determines
the candidate ID, and argument order does not change that identity.

An explicit `--null-box`, `--null-box-file`, or `--use-recommended` option is
the user's authorization to use that ROI. There is no second interactive
candidate-ID prompt. The command still writes the hash-derived candidate ID
and exact red-outline overlay for provenance and later troubleshooting. Once
installed, the ROI provenance is immutable within that output root.

The virtual-coil count is always explicit and is never selected from the QC
curve automatically. BART Wave uses CPU by default; append `-g` to use its GPU
branch. If a standard normal ecalib command exists, ROVir inherits its crop by
default. Otherwise it uses the normal MPRAGE launcher default `0.6`.
`--ecalib-crop` records a deliberate ROVir-specific override.

The command is resumable. If GPU Wave reconstruction stops for insufficient
memory, rerun the same command without `-g` to continue on CPU. Validated ACS,
ROI, transform, projected inputs, and CSM are reused rather than regenerated.

After the explicit ROI selection, the command automatically:

- prepares the positive and negative physical-coil images;
- runs BART ROVir and validates the full transform;
- applies the same selected basis to image k-space and separate ACS;
- reuses the accepted PSF without recalibration;
- estimates a CSM in the ROVir coil basis;
- reconstructs the mandatory FISTA lambda-zero control;
- exports magnitude and phase NIfTIs; and
- writes shared-window standard-versus-ROVir QC.

The completed canonical contract is:

```text
normal/rovir/manifest.json
```

The complete output layout is:

```text
normal/rovir/
  manifest.json
  feasibility/
    inputs/
    masks/
    transforms/
    diagnostics/
    manifests/
    logs/
  bart_inputs/
  bart_output/
  nifti/
  qc/
```

## Optional retrospective ROVir branches

After the canonical normal ROVir branch is complete, add ROVir siblings to the
five default MPRAGE retrospective cases with:

```bash
scripts/sample_mprage_retro_lr_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/existing_reconstruction_root \
    /path/to/matching_wave_mprage.seq \
    --rovir
```

The command consumes only the completed normal ROVir branch; it never estimates
a new ROI or transform and does not run the standard-coil retro workflow.
When an older interrupted run has complete normal ROVir scientific artifacts
but lacks the final canonical manifest, the command validates those artifacts
and creates only the missing QC/wrapper contract. Standard retro outputs remain
unchanged and are not prerequisites. ROVir results are written under:

```text
retro/<case>/rovir/{bart_inputs,bart_output,nifti}/
```

FISTA-r0 remains the mandatory control. The existing standard-coil Wavelet
lambdas are available for comparison but are explicitly labeled as reused,
not ROVir-optimized.

Each case is resolved directly from the canonical normal FOV and matrix. Its
sampling mask is a pure Cartesian image lattice with exact count and logical
hash; ACS remains separate. Native-grid PSFs are exact links to the accepted
normal ROVir PSF. LR PSFs are resolution-matched evaluations of its validated
phase-plane representation, so no PSF calibration or ecalib is rerun.

The MPRAGE NIfTI collection synchronizes available standard and ROVir outputs
idempotently. For one case and reconstruction method, a complete ROVir result
replaces its standard counterpart in the collection; the standard result is a
fallback when ROVir is absent. A partially populated ROVir set therefore
replaces only its exact method-matched cases and does not become a global
`--require-retro` requirement.

## Failure handling

The workflow fails closed for partial CFL pairs, changed commands, source or
manifest hash mismatches, altered ROI or transform provenance, incompatible
Ncc, nonfinite values, PSF mismatch, corrected-ACS mismatch, or CSM/ecalib
mismatch. It does not provide a destructive overwrite option.

For specific recovery guidance, see the ROVir sections in
[`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md).
