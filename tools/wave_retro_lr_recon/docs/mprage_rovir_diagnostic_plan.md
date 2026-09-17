# Wave-MPRAGE ROVir diagnostic plan

## Purpose

This plan addresses a measured Wave-MPRAGE case with severe coherent
background and shoulder-wrap artifacts. Read-only inspection found that the
artifact is already present before coil compression. Standard 64-to-12 coil
compression can preserve and redistribute that signal, but exchanging the
compression basis with one from a clean matched acquisition does not remove
it. The diagnostic therefore separates two questions:

1. does standard compression discard spatial encoding information needed to
   separate the desired head and shoulder signal; and
2. can region-optimized virtual (ROVir) coils retain the complete desired head,
   including skull, scalp, and face, while suppressing the shoulder
   contribution?

This is a diagnostic workflow, not a change to the normal reconstruction
defaults. No ROVir result becomes a production default without explicit visual
review and approval.

## Evidence motivating the experiment

The affected and clean comparison acquisitions use the same receive-coil
selection and channel ordering. The affected physical-coil image stream has
higher readout-edge relative to center energy in both head and neck elements.
Its edge covariance is strongly low rank and broadly distributed across the
array, which is more consistent with coherent out-of-FOV signal or acquisition
inconsistency than independent thermal noise. No isolated high-energy k-space
line was identified.

The integrated-ACS singular-value curves are nevertheless well behaved. For
the affected acquisition, standard compression retains approximately 86%,
92%, 95%, and 97% of ACS energy at 12, 16, 20, and 24 virtual coils,
respectively. More virtual coils are therefore a useful control, but energy
retention alone does not target shoulder suppression.

## Backend decision

The supplied MATLAB reference implements the original ROVir procedure:

1. multiply physical-coil calibration images by signal and interference
   masks;
2. form the two intercoil correlation matrices;
3. solve their generalized eigenvalue problem; and
4. orthonormalize the ordered eigenvectors.

Its demonstration script is specific to HASTE and GRAPPA and is not suitable
as a Wave-MPRAGE runner. The reference remains read-only and is used to check
the scientific operation, not as a production dependency.

BART v1.0 provides a native `bart rovir` command implementing the same
generalized-eigenvalue and Gram-Schmidt operation, together with Cartesian and
non-Cartesian regression tests. This is the selected numerical backend. The
repository will not maintain a second Python eigensolver implementation.
Every feasibility or reconstruction ROVir transform must be emitted by
`bart rovir`; MATLAB and Python implementations are not permitted fallback
backends. The supplied MATLAB code remains a read-only scientific reference.
Python owns validation, explicit command construction, provenance, diagnostic
metrics, and consistent application of one transform to every coil-domain
input. Production manifests must record the resolved BART executable and full
`bart version` output because host builds can differ.

No mature, released general-purpose Python ROVir package was identified during
the implementation audit. A public MRI-NUFFT request to add ROVir remains an
open feature request. Depending on that prospective API would add uncertainty
without reducing the local integration work.

References:

- Kim et al., *Region-optimized virtual (ROVir) coils*, Magnetic Resonance in
  Medicine 2021, [doi:10.1002/mrm.28706](https://doi.org/10.1002/mrm.28706).
- [BART](https://mrirecon.codeberg.page/), whose v1.0 release adds the native
  `rovir` command.

## Implementation boundary

The initial code-review frame contains only dataset-independent operations:

- strict validation of finite, nonnegative, shape-matched, nonoverlapping
  signal and interference masks;
- construction of masked physical-coil images;
- interference-correlation rank and conditioning gates before the
  unregularized generalized eigensolve;
- validation of a finite square orthonormal ROVir transform;
- cumulative desired-region retention and interference-remaining diagnostics;
- explicit `bart rovir` and `bart ccapply` command plans; and
- focused synthetic unit tests.

The diagnostic projection follows BART forward `ccapply` exactly: BART stores
physical coils by virtual coils in dimensions 3 and 4, then applies the
complex conjugate of that stored matrix. A complex-unitary interoperability
test guards this convention; a real-valued transform would not expose a
conjugation error.

The Python modules must not launch BART. Reviewed shell entry points show every
BART command explicitly, consistent with the existing measured reconstruction
workflow.

The frame deliberately excludes subject-specific masks, TWIX loading,
production output creation, ecalib, Wave reconstruction, and automatic method
selection. These operations begin only after code review.

## Scientific invariants

- Estimate one transform from physical-coil calibration images.
- Apply the exact same transform to measured Wave image k-space and integrated
  set-4 ACS.
- Re-estimate CSMs from the transformed ACS. A CSM from a different coil basis
  is incompatible and must be rejected.
- Reuse the already accepted calibrated PSF. A coil-domain linear transform
  does not require PSF recalibration.
- Keep image sampling k-space and ACS separate.
- Do not remove neck elements by label alone. The coherent artifact is observed
  by both head and neck elements.
- Do not automatically select a virtual-coil count or reconstruction winner.
- Preserve complex data and validate finite values, dimensions, hashes, source
  provenance, transform orthogonality, and acquired-sample equality.
- Reject a singular interference correlation matrix. Do not add diagonal
  loading or another numerical regularizer without a separately reviewed
  scientific decision.

## Four-region mask contract

The desired reconstruction retains the complete head, not merely the brain.
The reviewed masks therefore distinguish the final preservation objective from
the spatial samples that are sufficiently pure to estimate the ROVir basis:

1. `preservation_mask` covers the desired head, including brain, skull, scalp,
   and face. It is evaluation-only and is never passed to `bart rovir`.
2. `positive_estimation_mask` samples clean portions of the desired head. It
   should include representative brain and extracranial head signal while
   excluding visible shoulder contamination. This is the positive BART input.
3. `negative_estimation_mask` samples only spatially pure external shoulder
   signal and must lie completely outside the preservation mask. This is the
   negative BART input.
4. `contaminated_holdout_mask` covers desired-head locations in which wrapped
   shoulder signal overlaps anatomy. It is excluded from both BART inputs and
   used only for review.

The positive and holdout masks are disjoint subsets of preservation; the
negative mask is outside preservation and disjoint from the positive mask. A
safety gap is enforced between the positive and negative estimation masks. The
entire nonhead complement must not be used automatically because it can include
unrelated nuisance sources and can penalize desired inferior-head signal.

The user-marked presentation PNG records the intended preservation boundary in
selected LIN and PAR center slices. It is incomplete in RO, is lossy, and was
drawn on a rendered image, so its red pixels are not a computational mask.
Candidate masks must instead be generated in the native calibration geometry
and reviewed against indexed slices in all three orientations.

The first subject-specific ellipsoid candidates failed visual review because
their shoulder-negative and contaminated-holdout regions were misplaced. They
are rejected and must not be approved. BART does not detect ROIs: `bart rovir`
only forms correlation matrices from already separated positive and negative
coil images. The supplied MATLAB reference similarly creates masks explicitly
before invoking its ROVir solver.

For this case, reviewed manual sparse seeds replace parametric ellipsoids. A
geometry-bound calibration RSS and empty uint8 label template are exported as
canonical-RAS NIfTI files using the DICOM-validated MPRAGE orientation policy.
Label 1 denotes unequivocal clean desired signal, label 2 denotes unequivocal
pure shoulder interference, and label 0 retains every mixed, uncertain, or
unassigned voxel. Complete anatomical segmentation is neither required nor
preferred. The reviewed label map must retain the exact exported shape and
affine; validation inverts only axis permutations and reversals, with no
interpolation, before any BART input can be prepared.

The bright shoulder wrap visible over the vertex and extending inside the
desired head boundary cannot be labeled as pure shoulder from spatial position
alone. Those mixed voxels belong in the contaminated holdout, not in either
estimation mask. ROVir is feasible only if the coil signature learned from the
pure external shoulder remains separable from the clean desired-head coil
subspace. Energy in the holdout is a mixture of anatomy and artifact, so a
reduction there is not independently evidence of successful suppression;
fixed-window visual assessment remains mandatory.

Mask generation and mask review are separate from transform estimation. Every
mask must record its geometry, construction, source image, and hash. Reusing a
mask on different geometry is forbidden.

## Proposed diagnostic matrix

The first reconstruction comparison uses FISTA-r0 only:

| Basis | Virtual coils | Purpose |
| --- | ---: | --- |
| Standard ACS PCA | 12 | Matched baseline |
| Standard ACS PCA | 20 | Approximately 95% energy control |
| Standard ACS PCA | 24 | Approximately 97% energy control |
| ROVir | 12 | Strong compression and suppression |
| ROVir | 16 | Intermediate ROVir subspace |
| ROVir | 20 | Higher-retention ROVir subspace |

Increasing the number of ROVir coils is not assumed to improve suppression.
Lower-ranked generalized eigenvectors can reintroduce interference. The review
material must therefore show the complete clean-positive, whole-head,
contaminated-holdout, and pure-shoulder curves before reconstruction is
considered. Whole-head and holdout curves contain mixed energy and are
diagnostic, not automatic optimization objectives.

All reconstruction comparisons must keep the PSF, ecalib crop, FISTA options,
iterations, tolerance, conversion, and display windows fixed. A standard
PCA-12 branch must be produced under the same diagnostic preparation whenever
an existing reconstruction is not exactly matched.

## Standard PCA Ncc=24 control

The first reconstruction control retains 24 standard ACS-PCA coils and runs
only FISTA-r0. This isolates coil-subspace truncation from ROVir masking and
regularization. The implementation is separate from the normal reconstruction
defaults and uses `scripts/sample_mprage_pca_control.sh` with three explicit
stages: `prepare`, `reconstruct`, and `qc`.

Preparation estimates the PCA basis from the already exported physical-coil
set-4 ACS. It requires the resulting leading spectrum to reproduce the values
recorded by the accepted Ncc=12 manifest before extending the same basis to 24
columns. For the inspected acquisition, the reproduced retained energies are
approximately 86.36% at Ncc=12 and 97.09% at Ncc=24. The exact same 64-to-24
basis is then applied to the measured image stream and set-4 ACS. ACS remains
separate from image k-space, and samples outside the measured image lattice
remain zero.

The accepted calibrated PSF is copied byte-for-byte after geometry,
finite-value, source, and SHA-256 validation; PSF calibration is never rerun.
Because the coil basis changes, the control must estimate a new CSM from its
24-channel ACS. The matched reconstruction contract is:

```bash
bart ecalib -m 1 -c 0.1 kspace_calib coil_sens
bart wave -g -w -f -r 0 -i 100 -t 1e-6 \
  coil_sens psf wave_kspace image_wave
```

The output uses an independent root with `normal/bart_inputs`,
`normal/bart_output`, `normal/nifti/fista_r0`, `qc`, and `logs` subdirectories.
The ignored local launcher stores the confirmed server paths. Its `qc` stage
compares the accepted Ncc=12 and new Ncc=24 FISTA-r0 magnitude images at the
same center slices using one Ncc=12-anchored display window. It does not select
a preferred coil count.

## Review gates

### Code-review gate

Before accessing subject data, review:

- array axes and complex-conjugation conventions;
- BART coil and maps dimensions of the transform;
- mask validation and disjointness policy;
- orthogonality and finite-value tolerances;
- command construction and the invariant that image and ACS share one
  transform; and
- synthetic unit-test coverage.

### Feasibility gate

After code approval, but before BART reconstruction:

- generate low-resolution physical-coil calibration images;
- review desired-head and pure-shoulder separability;
- review all manual positive and negative seed labels in all three orientations;
- compute generalized-eigenvalue ordering and cumulative region metrics;
- compare ROVir with standard PCA at matched output dimensions;
- verify transform orthogonality and projected noise covariance; and
- stop if desired-head retention and shoulder suppression do not have a safe
  tradeoff.

### Reconstruction gate

Only after feasibility review and confirmation of a new output directory:

- prepare immutable, manifest-backed PCA and ROVir inputs;
- run matched ecalib and FISTA-r0 branches;
- compare fixed-window three-orientation figures and quantitative region
  metrics; and
- request explicit user selection. Do not overwrite the existing normal tree
  or promote any branch automatically.

## Confirmed feasibility output layout

The exact server root was confirmed and is stored only in the ignored local
launcher. The tracked structure is:

```text
ROVIR_DIAGNOSTIC_ROOT/
  inputs/
    physical_calibration/
    rovir/
  masks/
    candidates/
    manual_annotation/
      reference/
      template/
      reviewed/
    approved/
  transforms/
    rovir_full/
  diagnostics/
    calibration_views/
    region_curves/
  manifests/
  logs/
```

The PCA/ROVir reconstruction comparison remains a later review gate and is not
mixed into this feasibility root. No output directory is created merely by
running source tests or preflight.

## Implemented feasibility stages

The reviewed implementation uses
`scripts/mprage_rovir_feasibility.py` for bounded Python preparation and an
ignored `scripts/run_mprage_rovir_feasibility.local.sh` launcher for exact
server paths and explicit BART commands. The tracked ellipsoid schema remains
only as a dataset-independent example. Manual geometry-bound NIfTI annotation
remains available when an anatomy-shaped region is required.

For the inspected MPRAGE acquisition, the corrected alias-free RSS supports a
simpler explicitly reviewed two-region partition. The nuisance region is the
complete LIN/PAR slab at inclusive array indices `RO=0..30`; the desired-signal
region is its exact complement `RO=31..255`. This retains brain, skull, scalp,
and face without requiring an anatomical mask. The unused contaminated
holdout mask is explicitly empty, and QC writes two-region curves. The index
contract is generated by `derive-ro-partition`, remains unapproved until the
user supplies its exact candidate ID, and is never inferred automatically.

Run the local launcher in one existing tmux shell, one stage at a time:

```bash
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh preflight
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh prepare
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh images
# Create the reviewed exact RO partition; this does not approve it.
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh ro-partition
# Install only the explicitly approved candidate.
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh approve negative_ro000_030
# Export the geometry-bound manual annotation reference and empty label map.
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh export-manual-roi
# Draw sparse labels 1 and 2, save to the documented reviewed path, then validate.
tools/wave_retro_lr_recon/scripts/run_mprage_rovir_feasibility.local.sh validate-manual-roi
```

The `prepare` stage may appear quiet while it hashes the large TWIX payload and
loads the integrated refscan. It exports only zero-based set 4, removes readout
oversampling by a full-readout centered IFFT, central nominal-FOV image crop,
and centered FFT, then embeds the packed ACS at the center of the 72-by-72
calibration grid. Direct `::4` k-space striding is forbidden because it aliases
outside-FOV body signal into the head. The manifest records the versioned crop,
exact acquired-sample equality before and after embedding, and zeros outside
the centered ACS. Legacy stride-derived ROVir calibration exports are rejected
rather than silently reused.

The `images` stage contains the explicit commands:

```bash
bart fft -iu 7 physical_set4_kspace physical_set4_coil_images
bart rss 8 physical_set4_coil_images physical_set4_rss
```

It produces a fixed-window six-panel center-slice/maximum-projection RSS figure
and indexed RO, LIN, and PAR slice montages. The manual annotation export
normalizes RSS only for display, creates a matching empty label template, and
records the source CFL, source TWIX orientation, affine, spacing, hashes, axis
operations, and the no-resampling contract. Neither export nor validation
creates an approved ROVir mask or launches BART.

After separate visual review and explicit approval, a later import stage may
write bounded-memory masked physical-coil images, reject an inadequately
conditioned interference covariance, and execute the only supported solver:

```bash
bart rovir positive_signal_images negative_interference_images transform
```

Final QC validates the full transform, records the BART version, and writes
all-channel CSV/PNG curves for clean-positive retention, whole-head mixed
energy, contaminated-holdout mixed energy, and pure-shoulder energy remaining.
It does not select a virtual-coil count, run ecalib, or launch Wave
reconstruction.

## Planned user-facing box-ROI workflow

The feasibility implementation will ultimately become a dataset-independent,
staged command-line workflow in `wave_retro_lr_recon`. A user should be able to
start from compatible prepared MPRAGE inputs without editing Python or copying
a subject-specific launcher. The intended interaction is:

1. generate a geometry-bound, indexed physical-coil ACS RSS image from the
   corrected alias-free set-4 calibration;
2. inspect that image and specify an inclusive rectangular box, including a
   square box when appropriate, in native `RO, LIN, PAR` array coordinates;
3. generate mask CFLs and review overlays for that exact box without running
   ROVir;
4. explicitly approve the candidate ID after visual assessment;
5. run native `bart rovir`, inspect its conditioning and all-channel retention
   curves, and explicitly choose the retained coil count; and
6. project image k-space and ACS with the same transform, run matched ecalib
   and reconstruction, and produce fixed-window NIfTI QC.

The command interface should expose separate resumable stages rather than hide
review gates in one opaque command. A future dataset-independent interface may
take this form; exact names remain subject to code review:

```bash
sample_mprage_rovir_recon.sh render-acs \
  TWIX SEQUENCE ACCEPTED_NORMAL_ROOT FEASIBILITY_ROOT
sample_mprage_rovir_recon.sh propose-box \
  TWIX SEQUENCE ACCEPTED_NORMAL_ROOT FEASIBILITY_ROOT \
  --ro 0:20 --lin 0:71 --par 0:71
sample_mprage_rovir_recon.sh approve-box \
  FEASIBILITY_ROOT negative_ro000_020
sample_mprage_rovir_recon.sh estimate-transform FEASIBILITY_ROOT
sample_mprage_rovir_recon.sh reconstruct \
  TWIX SEQUENCE ACCEPTED_NORMAL_ROOT FEASIBILITY_ROOT OUTPUT_ROOT \
  --virtual-coils 24 --ecalib-crop 0.1 -g
```

All public bounds are human-readable and inclusive; manifests also record the
corresponding half-open Python slices. Every stage must validate source hashes,
geometry, finite values, transform orthogonality, and exact artifact reuse.
Existing outputs are reused only after those checks. The interface must never
infer or automatically approve an ROI, choose a coil count, rerun PSF
calibration, merge ACS into image k-space, or overwrite another reviewed ROI
experiment. Real paths remain confined to ignored local configuration or
launcher files.

The box is an estimation region, not an anatomical preservation mask. Its
review overlay must therefore make clear which side of the box is treated as
negative interference and which exact complement is treated as positive
signal. Because a box can suppress desired inferior-head or neck signal, the
final QC must reverse any display-only NIfTI normalization and compare
candidate reconstructions with one shared absolute window before a setting is
adopted.

## FMP220 feasibility conclusion

After fixed-window review of ROVir-24 reconstructions using inclusive negative
RO slabs 0--30, 0--20, and 0--10, the user selected RO 0--20 for FMP220. The
selected reconstruction uses 24 retained ROVir coils, ecalib crop 0.1, and the
FISTA lambda-zero control. RO 0--30 and RO 0--10 remain comparison-only
artifacts and must not be deleted or presented as selected results.

This is a dataset-specific feasibility conclusion, not a default ROI or an
assumption for another subject. The immutable selection manifest in the
reviewed output tree binds the decision to the approved mask, BART transform,
prepared inputs, exact reconstruction commands, magnitude and phase NIfTIs,
and comparison QC. The generic
`scripts/record_mprage_rovir_selection.py` command validates those bindings and
requires explicit user confirmation; it does not run BART or select a winner.
