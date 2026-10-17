# Experimental metal CSM support extrapolation

This feature is **disabled by default**, intended only for explicitly reviewed
metal-artifact cases. It does not detect metal automatically. Standard normal,
retrospective, ROVir, GRE, N4, and NIfTI collection commands are unchanged.

The initial supported workflow is a separate **native R3x1, single-echo,
standard-PCA MPRAGE derivative** of a completed normal reconstruction. It
accepts any validated physical/PCA coil count, not just VCC24. GRE, ROVir,
retrospective maps and multi-map/soft-cropped ESPIRiT inputs are rejected.
Do not substitute these candidate maps into another workflow by renaming files.

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

## Explicit configuration and execution

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
- Preserve the source lambda; use `wave -w -f`, 100 iterations and tolerance
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

## Layout, provenance and reuse

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
are replaced. Canonical collection builders do not discover these derivatives,
and the runner refuses output inside a source normal tree, retro tree or
`nifti_collection`. There is no automatic N4 step or collection preference change.

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
