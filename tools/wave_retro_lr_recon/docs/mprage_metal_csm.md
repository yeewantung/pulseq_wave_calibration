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
R3x2/R3x3 and LR-Y R3x2, using one reviewed native CSM continuation. The original
standalone native-R3x1 diagnostic launcher is also retained. Both interfaces
accept any validated physical/PCA coil count, not just VCC24. GRE, ROVir and
multi-map/soft-cropped ESPIRiT inputs remain unsupported.
Do not substitute these candidate maps into another workflow by renaming files.

## Use the existing normal and retro launchers

One-time setup: copy `configs/mprage_metal_pipeline.example.json` into
`OUTPUT_ROOT/metal_csm.local.json`, or an explicitly ignored repository-local
JSON file. Fill all nulls after reviewing the case. This is a **nested** config:

- `source_manifest`: completed original normal branch's
  `reconstruction_manifest.json`;
- `reference_nifti`: its uncorrected original magnitude export, with matching
  JSON sidecar (an exact collection copy is accepted); and
- `parameters`: the spatial settings described below, including the reviewed
  ROI on that exact reference grid.

Relative paths resolve against the configuration file's directory. Verify
repository-local private configs with `git check-ignore`; never put actual
dataset paths or subject ROIs in the tracked example. A completed original
normal baseline, its complex image, CSM, PCA basis and prepared inputs are
required. The launcher does not silently reconstruct a missing baseline.

After setup, append one switch to the existing command, **without `-g`**:

```bash
bash scripts/sample_mprage_normal_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --virtual-coils 24 \
  --metal-csm-extrapolation

bash scripts/sample_mprage_retro_lr_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --virtual-coils 24 \
  --metal-csm-extrapolation
```

Use `--metal-csm-config /path/to/CASE.local.json` to override the default config
location. `--metal-prepare-only` prepares and validates maps/case manifests
without launching BART. `--cpu-threads` accepts 1--8, default 8. Source-related
flags (Ncc, prewhitening, ecalib crop and PSF processing) must match the reviewed
normal preparation. No new PCA/PSF fit is permitted by mismatch fallback.
When compatible historical PSF artifacts have a strict hash-bound accepted
source, the existing artifact-preserving reuse policy is supported without a
manual fitting override. Metadata records both the requested settings and the
historical PSF contract; the latter is never relabeled as a new fit.

Normal remains Wavelet-only by default (`0.03` for VCC24; the existing transferred
value for other counts); `--r3-lambda` explicitly overrides the normal value.
Retro remains FISTA-r0-only for native R3x2, LR-Y R3x2 and native R3x3. Both accept
the existing mutually exclusive `--reg-full`, `--wavelet-only`, `--fista-only`
profiles. Positive lambdas are labeled `wavelet_metal_unvalidated`, not optimal
for the modified forward model. The independent R3x3 launcher accepts the same
switch and configuration. `-g`, `--rovir`, `--include-legacy-lr` and
`--n4-bias-correction` combinations are explicitly rejected in metal mode.

The user-selected flat variant layout is:

```text
OUTPUT_ROOT/
  vccN/                         # original normal/retro products, not replaced
  vccN_metal_csm/
    metadata/                   # immutable config/source/software identity
    source_reference/           # hash-bound original magnitude + JSON copy
    shared_csm/                 # native candidate + added-support archive
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

Before reconstruction, the integrated pipeline preserves the original reference
magnitude/sidecar under `source_reference/`, with exact hashes. This is essential
when configuration initially references a collection copy: replacement must not
destroy the reference needed for later strict resume. Existing source NIfTI
trees can be pruned only under the separate reviewed archival policy; when a
prior collection supplies established names and metal exports remain complete,
those names are retained. If a previously collected metal branch disappears or
becomes incomplete, synchronization refuses a silent downgrade. The previous
tool-owned collection is replaced atomically only after all validation/copying
succeeds. Original reconstruction trees are never overwritten by collection.

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
  `1e-6`. Sources using other solver settings are rejected. No lambda sweep,
  automatic winner selection, PCA refit, PSF refit or noise-prewhitening change
  is performed. Previously selected lambdas are not declared optimal for the
  modified forward model.
- CPU only: at most 8 OpenMP threads, one BLAS thread, hidden CUDA devices,
  Linux idle scheduling and nice 19. The shell uses idle I/O priority where
  available. These mechanisms yield CPU priority but cannot guarantee zero
  shared-cache/network-storage contention.
- Require 112 GiB available RAM before launching; stop **only this job's BART
  child** if its RSS exceeds 64 GiB or host available RAM falls below 64 GiB.
  Address space is capped at 96 GiB. These are conservative pilot-based limits,
  not a claim that every geometry fits; larger datasets may be refused/stopped.
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
cross-count/config mismatch rejection, CPU-only argument guards, prepare-only
execution, and completed normal/retro/R3x3 resume without rewriting outputs.
Collection tests cover paired preference under unchanged names, metal-only
retro cases, count isolation, partial fallback, corrupt-completion rejection,
atomic synchronization, no silent downgrade, optional preferred-source masking,
and baseline-reference survival across replacement.
