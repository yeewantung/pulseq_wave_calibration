# Experimental metal CSM support extrapolation (MPRAGE only)

This feature is **disabled by default**, intended only for explicitly reviewed
metal-artifact cases. It does not detect metal automatically. Without the
explicit switch, standard reconstruction behavior is unchanged.

**Support scope: standard-PCA MPRAGE only.** GRE normal reconstruction,
retrospective reconstruction and NIfTI collection do not support this feature.
Do not pass `--metal-csm-extrapolation` to GRE launchers. Extending it to GRE is
deferred: MPRAGE pilot results do not establish suitability for GRE, which would
require separate validation of magnitude, phase, inter-echo scaling and delta-B0
behavior. This feature extends CSM support; it is not a general metal-artifact
correction method.

The normal, retro LR, and independent R3x3 MPRAGE launchers accept
`--metal-csm-extrapolation`. They support standard-PCA native R3x1, native
R3x2/R3x3 and LR-Y R3x2, using one shared native CSM continuation. The original
standalone native-R3x1 diagnostic launcher is also retained. Both interfaces
accept any validated physical/PCA coil count, not just VCC24. GRE, ROVir and
multi-map/soft-cropped ESPIRiT inputs remain unsupported.
Do not substitute these candidate maps into another workflow by renaming files.

## Use the existing normal and retro launchers

Append the metal switch to the existing command. No previous diagnostics,
completed baseline reconstruction, reference NIfTI or JSON configuration is
required for automatic mode. CPU BART remains
the default; append `-g` to either command for GPU BART:

```bash
bash scripts/sample_mprage_normal_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --virtual-coils 24 \
  --metal-csm-extrapolation

bash scripts/sample_mprage_retro_lr_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --virtual-coils 24 \
  --metal-csm-extrapolation
```

Without the explicit metal switch, none of the automatic completion runs.
Without a local manual config, the enabled pipeline follows this order:

1. Validate and reuse the requested `vccN/normal/bart_inputs` if present. Never
   replace incompatible, partial or hash-mismatched inputs.
2. If no matching count exists, reuse a compatible hash-bound standard/ROVir
   source-anchor PSF and trajectory when available, preparing a new standard
   PCA basis and ACS/image k-space from raw data. ROVir CSMs and transforms are
   never relabeled as standard PCA. The first advertised donor must validate;
   an incompatible/corrupt donor is not silently ignored.
3. If no source exists, prepare from TWIX + sequence using the existing strict
   geometry, source, physical-coil, finite-value and PSF checks.
4. Reuse an original CSM only when a completed standard branch hash-binds it to
   the selected inputs and crop; otherwise, when no original CSM exists, run
   CPU ecalib in the isolated metal tree. No ordinary Wave image is reconstructed.
5. Automatically propose/gate support additions from unmasked calibration RSS
   and native CSM, then run the requested normal/retro branches.

The normal, retro LR and independent R3x3 launchers all support this `-g` option.
CSM continuation and LR resampling remain on CPU; only `bart wave` receives
`-g`. The caller's `CUDA_VISIBLE_DEVICES` is preserved in GPU mode (including an
unset or deliberately empty value); no GPU is selected automatically and no
failed GPU run silently falls back to CPU. Use a CUDA-enabled host BART build.
The original standalone diagnostic launcher remains CPU-only.

Backend selection is recorded in each branch's request, BART command, execution
record and NIfTI sidecars, and validated during collection. Changing CPU/GPU
backend or the recorded CUDA visibility on an existing reconstruction rejects
resume rather than relabeling outputs. A command rejected by the old CPU-only
argument check created no metal output and can be retried with `-g` after this
update. Previously prepared trees remain subject to strict software-hash checks.

An existing `OUTPUT_ROOT/metal_csm.local.json` selects the optional reviewed
manual-ROI path; `--metal-csm-config /path/to/CASE.local.json` selects it explicitly.
An explicitly supplied missing/bad config is an error, never an automatic fallback.
`--metal-prepare-only` prepares and validates maps/case manifests without
launching BART; if CSMs do not yet exist it stops before ecalib and requests a
normal invocation that permits calibration. `--cpu-threads` accepts 1--8,
default 8. Source-related
flags (Ncc, prewhitening, ecalib crop and PSF processing) must match the reviewed
normal preparation when reused. No new PCA/PSF fit is permitted by mismatch fallback.
When compatible historical PSF artifacts have a strict hash-bound accepted
source, the existing artifact-preserving reuse policy is supported without a
manual fitting override. Metadata records both the requested settings and the
historical PSF contract; the latter is never relabeled as a new fit.

### Moved source datasets

Moving an existing dataset between disks does not authorize rewriting its
historical manifests. For an explicitly confirmed relocation, the maintenance
utility `scripts/register_mprage_relocation.py` audits the moved `vccN` inputs
and writes `OUTPUT_ROOT/metadata/path_relocation.local.json` exclusively.
Pass each exact historical **dataset** prefix with `--source-prefix` (including
old symlink spellings used by command records); the destination dataset is the
parent of the confirmed output root. This utility does not move data or run BART.

The receipt lists original document hashes and verified array records. Readers
apply the mapping only to listed, unchanged documents, in memory; original JSON,
command records and all scientific hashes remain unchanged. Normal/retro input
reuse, automatic metal source validation, conversion and collection consume this
view. New metal requests/sources bind the receipt hash. Missing or changed data
still fail the existing geometry, finite-value, PCA, source and hash checks.

Legacy normal and retrospective manifests may both predate the
`noise_prewhitening` field. These can be reused only with the exact normal
manifest SHA binding and the other case/artifact checks intact; an absent case
field never matches an explicit normal whitening record. No legacy manifest is
edited to add this field. Receipt-bound retrospective aggregate manifests are
also preserved; individual case manifests remain authoritative for the requested
case selection. Reusing measured retro inputs does not reuse ordinary images:
unfinished metal branches still reconstruct with the extended CSM.

Historical TWIX identities that contain only size/mtime are explicitly recorded
as **not historically SHA-256 verified**. The user-confirmed relocation must
match those historical fields; a new current SHA-256 is recorded and checked on
subsequent reads. This is not proof that raw bytes matched an unavailable old
hash. No generic search-by-filename or automatic trust of another disk is used.
Registration is currently for standard-PCA MPRAGE source trees; it does not
rewrite or migrate ROVir transforms, GRE trees or prior metal experiments.

Normal remains Wavelet-only by default (`0.03` for VCC24; the existing transferred
value for other counts); `--r3-lambda` explicitly overrides the normal value.
Retro remains FISTA-r0-only for native R3x2, LR-Y R3x2 and native R3x3. Both accept
the existing mutually exclusive `--reg-full`, `--wavelet-only`, `--fista-only`
profiles. Positive lambdas are labeled `wavelet_metal_unvalidated`, not optimal
for the modified forward model. The independent R3x3 launcher accepts the same
switch and configuration. `--rovir`, `--include-legacy-lr` and
`--n4-bias-correction` combinations are explicitly rejected in metal mode.

The user-selected flat variant layout is:

```text
OUTPUT_ROOT/
  vccN/                         # original normal/retro products, not replaced
  vccN_metal_csm/
    metadata/                   # immutable request, automatic source identity
    source_calibration/         # isolated ecalib metadata/reconstruction/logs, if needed
    source_reference/           # manual-ROI only: original magnitude + JSON copy
    shared_csm/                 # candidate, support/confidence NPZ, metadata and QC figure
    normal/bart_output/BRANCH/
      metadata/                 # preparation, execution, export provenance
      reconstruction/           # complex image + wave command
      nifti/                    # magnitude/phase + JSON sidecars
      logs/                     # BART + memory monitor
    retro/CASE/
      maps/                     # LR-Y only: Fourier-resampled candidate CSM
      bart_output/BRANCH/       # same per-branch layout as normal
```

Native R3x1, R3x2 and R3x3 use the exact same candidate CFL pair. LR-Y applies
the existing same-FOV centered Fourier PE resampling and RSS renormalization
to that **candidate**, never to the original hard-cropped CSM. LR geometry is
prepared from the measured source with existing PSF validation. Canonical
retro k-space inputs may be prepared when absent, but canonical CSMs, completed
reconstructions and NIfTIs are not overwritten.

The variant holds an exclusive lock across preparation/reconstruction/export.
Changed config, source, geometry, software, maps or solver request rejects
reuse, including changes between normal and retro invocations. A new ROI cannot
silently replace a completed `vccN_metal_csm` tree; preserve/archive the prior
experiment and explicitly arrange a new run before retrying. Failed/partial
outputs are retained, never automatically cleared. Normal and retro branches
may be asymmetric. Integration uses the standard converter's per-image
normalization, not the standalone diagnostic's shared-baseline intensity scale.

### Collection preference

The standard converter writes paired images/sidecars below `nifti/sub-CASE/`.
Metal export registration and collection validate these nested files (historical
flat pairs are also supported), rejecting extra files, symlinks and bad pairs.
An export-registration journal binds raw images and sidecars before adding metal
provenance; a completed registration can be resumed without conversion.

The early automatic implementation incorrectly treated `sub-CASE/` as a file.
For that exact reviewed implementation hash only, the layout fix permits reuse
of unchanged requests/preparations and completed BART execution. If four
unregistered nested exports remain, retry independently converts the validated
complex image to temporary files and checks exact image hashes and sidecar
content before registration. It does **not** rerun BART, rewrite historical
request hashes, or silently adopt unmatched partial exports. A mismatch remains
an error with existing files preserved. Recovery provenance is recorded in
`metadata/export_registration.json` and bound by `metadata/export.json`.

Run the existing collection command without an additional switch:

```bash
bash scripts/sample_mprage_nifti_collection.sh "$OUTPUT_ROOT" --require-retro
```

Complete `vccN_metal_csm` branches are preferred inside the existing
`nifti_collection/vccN/` subtree, not a second metal collection. A unique
same-case/method counterpart keeps its existing branch label and filenames;
magnitude **and phase** are copied from the matched metal reconstruction. If
multiple ordinary Wavelet branches exist, a unique lambda match is required
instead of selecting a winner. Missing ordinary cases can be supplied entirely
by completed metal branches, including the three-case retrospective gate.
Incomplete metal branches without an export-completion manifest fall back to
available ordinary results. A corrupt advertised completion is a hard error.

Metal preference takes precedence over ordinary N4: no bias field or phase from
the replaced ordinary reconstruction is applied to metal images. Unmatched
ordinary branches keep their existing N4/canonical preference. Other coil
counts, legacy root-level products and canonical ROVir remain independent.
The collection manifest and copied sidecars preserve exact selected config,
native/LR CSM lineage, execution/export hashes, lambda, and the replaced ordinary
branch's provenance. A retained filename containing an old selection label is
**not** evidence that its lambda was re-optimized for metal CSM.

In manual-ROI mode, the pipeline preserves the original reference
magnitude/sidecar under `source_reference/`, with exact hashes. This is essential
when configuration initially references a collection copy: replacement must not
destroy the reference needed for later strict resume. Existing source NIfTI
trees can be pruned only under the separate reviewed archival policy; when a
prior collection supplies established names and metal exports remain complete,
those names are retained. If a previously collected metal branch disappears or
becomes incomplete, synchronization refuses a silent downgrade. The previous
tool-owned collection is replaced atomically only after all validation/copying
succeeds. Original reconstruction trees are never overwritten by collection.

## Automatic region policy (experimental v1)

This policy is only invoked after the user enables metal mode. It is not an
automatic metal detector, a tissue segmentation or an established SNR estimator.
All geometry uses logical RO/LIN/PAR coordinates and physical millimetres;
no subject-specific voxel ROI or diagnostic result is transplanted.

- Propose holes/concavities inside largest-component planar convex envelopes,
  within 8 mm of original support. Adjacent RO planes must also have evidence.
- Compute ACS RSS by centered IFFT, without applying the cropped CSM. Gate by
  the larger of an outside-envelope median + 4 scaled MAD and 1% of the median
  in native support. If outside-envelope evidence is insufficient, use no-op.
- Phase-align donors through a strong reference coil, excluding weak reference
  samples. Compare normalized 2-mm and 4-mm Gaussian predictions using
  phase-insensitive inner products (minimum agreement 0.98), donor weight and
  vector coherence; restore nearby native phase only for added vectors.
- Keep components of 8--20000 cubic mm; if additions exceed 3% of native
  support, reject all automatic additions. Preserve original vectors exactly.
- Missing reliable additions or a safety-cap rejection is a recorded no-op:
  continue reconstruction with unchanged maps, not an error or silent loosening
  of thresholds. Invalid/nonfinite scientific inputs still fail hard.

These are conservative experimental defaults, not population-optimized values.
Weak calibration near metal can legitimately cause under-recovery. Save and
review `shared_csm/intermediate/support.npz`, `figures/automatic_support.png`
and `metadata/preparation.json`; masks are in logical coordinates, not NIfTI
voxel order. They are diagnostic artifacts, not collection head masks.

## Optional manual-ROI mode

Copy `configs/mprage_metal_pipeline.example.json` into an ignored local JSON
and fill `source_manifest`, `reference_nifti` and all `parameters` after review.
This legacy diagnostic-compatible path requires a completed original normal
baseline, its complex image and exact uncorrected magnitude/sidecar. Relative
paths resolve against the config directory. Use `git check-ignore` for private
repository-local configs; never commit actual source paths or subject ROIs.
Automatic and manual policies cannot overwrite each other's existing variant.

## When this can help

Native `ecalib` hard cropping can exclude weak-signal regions near metal. A
bounded continuation of nearby complex coil vectors can enlarge the forward
model's support there. The metal pilot showed visually improved local signal;
this is not proof of anatomical accuracy. The non-metal facial pilot changed
little: most of its dark facial region was already inside native CSM support.
This is therefore not a general intensity/bias correction or an SNR recovery
method. It is also not ESPIRiT eigenvalue soft cropping or multi-map soft-SENSE.

All original supported CSM vectors remain bitwise unchanged. Reconstruction
intensities can nevertheless change outside the added support because the
inverse problem is globally coupled. Monitor brain-interior changes and remote
background as well as the visibly recovered region.

## Standalone native-R3x1 diagnostic interface

Activate the Python environment and source the compatible **host** BART startup
as described in [SETUP.md](../SETUP.md). Copy
`configs/mprage_metal_csm.example.json` into an ignored `CASE.local.json`, then
replace every `null` with reviewed case-specific values. The incomplete template
deliberately cannot run. Keep real dataset paths only in ignored local launchers
or local JSON configuration, never in tracked examples.

| Setting | Meaning |
| --- | --- |
| `roi_voxels` | Three half-open integer ranges `[[x0,x1],[y0,y1],[z0,z1]]` on the exact reference NIfTI voxel grid, **not** logical BART coordinates or world-mm coordinates. |
| `envelope_axis` | Axis normal to each 2-D convex-envelope plane: 0, 1 or 2. Confirm the reference orientation; 2 is axial only for the appropriate export. |
| `base_distance_mm`, `max_distance_mm` | Near-support and maximum local extension lengths; maximum must be at least base. |
| `near_sigma_mm`, `far_sigma_mm` | Physical Gaussian widths for normalized complex coil-vector continuation. |
| `edge_ramp_mm` | Separable smooth taper at the ROI boundaries. |
| `global_base_extension` | Default `false`: only add inside the ROI. Explicit `true`: additionally permit bounded near-scale support completion outside the ROI. |
| `minimum_weight`, `minimum_coherence` | Positive gates in `(0,1]`; not validated tissue/noise classifiers. |

One reviewed metal pilot used base/maximum distances 8/16 mm, Gaussian widths
4/6 mm and an 8-mm ROI ramp. Those values are **not universal defaults**. That
pilot also started from a global axial hull8 extension; reproducing its policy
requires explicitly reviewing `global_base_extension: true`. A pure local
extension is not identical to that historical experiment. No subject-specific
ROI is supplied by the repository.

Run the single opt-in launcher, specifying an independent output directory you
have chosen and confirmed:

```bash
bash scripts/sample_mprage_metal_recon.sh \
  --metal-csm-extrapolation \
  --source-manifest /path/to/vccN/normal/bart_output/BRANCH/reconstruction_manifest.json \
  --reference-nifti /path/to/original_collected_part-mag_image.nii.gz \
  --config /path/to/CASE.local.json \
  --output /path/to/approved_independent_metal_derivative \
  --cpu-threads 8
```

The reference must be an **uncorrected original magnitude export**, with its
matching JSON sidecar. An exact NIfTI collection copy is valid even when the
source NIfTI tree was pruned. N4 derivatives, resliced images and arbitrary
look-alike references are rejected. The original complex baseline, native CSM,
prepared artifacts and PCA basis must remain available and hash matched.

`--prepare-only` performs map preparation without locating or launching BART.
Omitting `--metal-csm-extrapolation` rejects the request before input reads or
output creation. There is no GPU option. Run a long reconstruction in your own
tmux session; the launcher is foreground, not an automatic job scheduler.

## Scientific and execution contract

- Native support must be binary unit-RSS, with one sensitivity map per coil.
  A largest-component planar convex envelope bounds eligible additions. This
  is a geometric restriction, not an anatomical segmentation.
- Gaussian widths, distance transforms and ROI tapers use verified voxel sizes
  in mm. Normalized complex vectors blend from the near to the far scale;
  unstable cancellation or nonfinite vectors are rejected.
- With `global_base_extension: false`, the local distance cap tapers from zero
  to the requested maximum, and every vector outside the ROI is unchanged.
  With the explicit global base enabled, the cap is base plus the tapered
  base-to-maximum increment. The manifest reports additions outside the ROI.
- The standalone diagnostic preserves the source lambda. All interfaces use
  `wave -w -f`, 100 iterations and tolerance
  `1e-6`. Manual baseline sources using other solver settings are rejected.
  No lambda sweep or automatic winner selection is performed. Existing
  compatible inputs are not refitted or re-whitened; fresh automatic preparation
  may compute the requested PCA/PSF/calibration when no compatible source exists.
  Previously selected lambdas are not declared optimal for the
  modified forward model.
- At most 8 OpenMP threads, one BLAS thread, Linux idle scheduling and nice 19.
  CPU BART hides CUDA devices; explicitly requested GPU BART preserves caller
  visibility. The standalone diagnostic shell uses idle I/O priority where
  available. These mechanisms yield CPU priority but cannot guarantee zero
  shared-cache/network-storage contention.
- Require 112 GiB available RAM before launching; stop **only this job's BART
  child** if its RSS exceeds 64 GiB or host available RAM falls below 64 GiB.
  CPU address space is capped at 96 GiB. GPU mode does not impose that virtual
  address-space cap because CUDA mappings can exceed physical RAM; inherited
  system limits remain in effect. Host RAM admission and RSS monitoring still
  apply to GPU runs. VRAM is not reserved or guaranteed by these guards; a
  CUDA allocation failure is reported without automatic retry or CPU fallback.
  These are conservative pilot-based limits, not a claim that every geometry
  fits; larger datasets may be refused/stopped.
- Before-launch resource refusal allows a later retry with the same prepared
  maps. An interrupted/failed BART attempt is retained and never silently
  restarted over its partial output. Use a newly approved output directory.

## Standalone diagnostic layout, provenance and reuse

```text
approved_independent_metal_derivative/
  intermediate/      # candidate coil_sens and native/added support archive
  reconstruction/    # complex BART image
  nifti/             # metal_csm_magnitude, phase, added_support + JSON sidecars
  figures/           # shared-scale baseline/candidate comparison
  metadata/          # immutable request, preparation, execution and comparison
  logs/              # BART output and memory monitoring
```

Neither historical root-level products nor count-specific normal/retro products
are replaced. Canonical collection builders do not discover these **standalone diagnostic** derivatives,
and the runner refuses output inside a source normal tree, retro tree or
`nifti_collection`. There is no automatic N4 step. Only the integrated
`vccN_metal_csm` interface participates in collection preference.

Reuse requires exact source/configuration/software identity: source manifests,
PCA basis, coil count, geometry, source CSM, PSF, measured k-space, baseline,
reference NIfTI/sidecar, candidate CSM, support archive and output hashes are
checked. Changing Ncc, ROI, parameters or inputs requires a new independent
directory. Concurrent writers are excluded by an output-local advisory lock.
Incomplete or unbound artifacts are rejected, not adopted or overwritten.

Completed reconstruction can be exported on a later invocation if export never
started. Partial exports are preserved and rejected. Fully completed runs are
hash-validated and reused without launching BART again. Magnitude uses the
**same baseline normalization**, and output affine/qform/sform match the
reference. The comparison reports added-support signal, native-interior changes
and remote background. A historical GPU baseline is not a matched CPU control;
do not attribute every small difference solely to the support change.

## Validation scope

The repository implementation has synthetic tests for spatial bounds, physical
spacing, exact native-vector preservation, unit RSS, invalid/soft-map inputs,
default-off behavior, mismatched reuse, retained collection copies, output
isolation, CPU command construction, memory admission/child-only stopping,
and NIfTI geometry/provenance. BART is mocked in execution tests. The portable
implementation still needs user review on real metal data before wider use;
the earlier real-data pilot scripts are not relabeled as validation of this
new implementation.

Launcher integration tests additionally cover flat variant isolation, default
profiles/cases, native-map identity across accelerations, LR-Y lineage and RSS,
cross-count/config mismatch rejection, explicit GPU dispatch, prepare-only
execution, and completed normal/retro/R3x3 resume without rewriting outputs.
Collection tests cover paired preference under unchanged names, metal-only
retro cases, count isolation, partial fallback, corrupt-completion rejection,
atomic synchronization, no silent downgrade, optional preferred-source masking,
and baseline-reference survival across replacement. GPU tests mock BART and
verify command construction, CUDA visibility, CPU/GPU resume mismatches,
GPU-compatible address-space policy, failure handling and collection provenance;
they do not establish real-GPU numerical equivalence or VRAM requirements.
Automatic tests additionally cover known synthetic support holes, gauge-invariant
coil-direction prediction, physical spacing, signal/volume no-op gates, fresh
preparation without a baseline image, pruned baseline reuse, source/crop mismatch,
ROVir-anchor PSF hashes and exact automatic collection provenance.
