# Optional MPRAGE ROVir reconstruction

## Scope

ROVir is an optional troubleshooting and recovery feature for a completed
Wave-MPRAGE reconstruction with coherent shoulder or extended-body signal
wrapped into the head. It is not part of the normal MPRAGE workflow and does
not replace the standard reconstruction.

ROVir is currently supported only for MPRAGE. It is not exposed for GRE. The
GRE acquisition FOV normally avoids the extended-body shoulder-wrap failure
mode that motivated this implementation, and no GRE ROVir workflow has been
validated.

The implementation uses native BART `rovir`. BART does not detect an ROI: the
user reviews the corrected physical-coil ACS RSS and approves the region used
to estimate nuisance signal. For the scientific rationale, backend decision,
mask contract, and development history, see
[`mprage_rovir_diagnostic_plan.md`](mprage_rovir_diagnostic_plan.md).

## Prerequisites

Start from an existing reconstruction root with a completed normal MPRAGE
branch, including:

- `normal/bart_inputs/manifest.json`;
- the recorded normal CSM and ecalib command;
- a normal FISTA-r0 magnitude NIfTI; and
- the original TWIX and sequence at the paths recorded by the normal manifest.

Activate the tool's Python environment and host-compatible BART build before
running the commands. The public ROVir interface derives the source paths and
normal reconstruction settings from the manifest. It has no separate public
configuration file.

ROVir writes only below `normal/rovir/` and leaves the standard normal branch
unchanged.

## Step 1: inspect the corrected ACS

The reconstruction root defaults to the current directory. It can instead be
supplied as the single positional argument.

```bash
cd /path/to/existing_reconstruction_root
scripts/sample_mprage_rovir_recon.sh inspect
```

`inspect`:

1. validates the completed normal reconstruction and its source contract;
2. exports or strictly reuses corrected, alias-free physical-coil set-4 ACS;
3. runs explicit BART IFFT and RSS commands;
4. writes indexed RO, LIN, and PAR views; and
5. writes an auditable conservative null-ROI recommendation when separation
   is sufficiently clear.

The recommendation is not an anatomical detector and is never approved
automatically. It may report `no_safe_automatic_recommendation`. Review:

```text
normal/rovir/feasibility/diagnostics/calibration_views/
normal/rovir/feasibility/diagnostics/roi_recommendation/
```

## Step 2: review one null-region union and reconstruct

If the recommendation is appropriate:

```bash
scripts/sample_mprage_rovir_recon.sh run \
    --use-recommended \
    --virtual-coils 24
```

Alternatively, supply any number of inclusive native-grid boxes:

```bash
scripts/sample_mprage_rovir_recon.sh run \
    --null-box "ro=0:20,lin=0:25,par=all" \
    --null-box "ro=0:20,lin=48:71,par=all" \
    --virtual-coils 24
```

The same records may be provided as a path-free JSON list with
`--null-box-file`. Inline boxes and a box file are mutually exclusive. Boxes
may overlap. Exact duplicates are removed, the final binary union determines
the candidate ID, and argument order does not change that identity.

The command writes an exact red-outline review figure and pauses once. It
continues only after the user types the exact candidate ID. If the outline is
not satisfactory, stop before approval and rerun with revised boxes. Distinct
pre-approval candidates are retained by ID; after one candidate is approved,
the ROI provenance is immutable.

After reviewing the figure, a resumed noninteractive invocation may use:

```bash
scripts/sample_mprage_rovir_recon.sh run \
    --null-box "ro=0:20,lin=all,par=all" \
    --virtual-coils 24 \
    --confirm-roi-id EXACT_CANDIDATE_ID
```

The virtual-coil count is always explicit and is never selected from the QC
curve automatically. BART Wave uses CPU by default; append `-g` to use its GPU
branch. The ROVir ecalib crop is inherited from the normal reconstruction by
default. `--ecalib-crop` records a deliberate ROVir-specific override.

After approval, the command automatically:

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

The command consumes only `normal/rovir/manifest.json`; it never estimates a
new ROI or transform. Standard retro outputs remain unchanged. ROVir results
are written under:

```text
retro/<case>/rovir/{bart_inputs,bart_output,nifti}/
```

FISTA-r0 remains the mandatory control. The existing standard-coil Wavelet
lambdas are available for comparison but are explicitly labeled as reused,
not ROVir-optimized.

The MPRAGE NIfTI collection discovers available standard and ROVir normal and
retro branches idempotently. A partially populated optional ROVir set does not
become a `--require-retro` requirement for standard branches.

## Failure handling

The workflow fails closed for partial CFL pairs, changed commands, source or
manifest hash mismatches, altered ROI or transform provenance, incompatible
Ncc, nonfinite values, PSF mismatch, corrected-ACS mismatch, or CSM/ecalib
mismatch. It does not provide a destructive overwrite option.

For specific recovery guidance, see the ROVir sections in
[`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md).
