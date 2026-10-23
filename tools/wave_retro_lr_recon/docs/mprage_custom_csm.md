# Explicit custom CSM: MPRAGE standard-PCA and ROVir

This experimental opt-in path accepts a **native single-map CSM with a completed
provenance JSON**, not a bare CFL file. It does not infer coil ordering from Ncc,
relabel ROVir as PCA, fit another transform, rerun ecalib, or automatically choose
a crop/lambda. Existing commands without `--custom-csm` are unchanged. GRE,
combined metal mode, N4 and new ROVir ROI-selection flags are not supported on
this path; incompatible options fail before execution.

## Local mouth-only candidate

`prepare_local_custom_csm.py` consumes an existing uncropped calibration cache,
its exactly matching hard-0.6 baseline, and a reviewed local JSON containing
`diagnostic_root`, `baseline_csm` and `parameters`. Keep real paths and
subject-specific ROIs in ignored `.local.json` files. Confirm the output location
before running:

```bash
python tools/wave_retro_lr_recon/scripts/prepare_local_custom_csm.py \
  --config "$REVIEWED_LOCAL_CONFIG" --output "$CONFIRMED_DIAGNOSTIC_DIR"
```

The numerical parameters follow `MetalCSMParameters`, but this interface's
`roi_voxels` and `envelope_axis` are in **logical RO/LIN/PAR order**, not NIfTI
order. Bounds are zero-based, half-open. Distances, sigmas and edge ramp are mm.
`global_base_extension` must be false. Near/far Gaussian predictions of complex
coil vectors are confidence-gated within a bounded convex envelope and distance
cap, with an ROI-edge taper. This is local complex-vector continuation, **not**
global soft cropping or guaranteed tissue segmentation. The existing vectors
and every non-added vector remain bitwise unchanged; added vectors have unit
coil RSS. A changed inverse problem can still change the reconstructed image
outside the ROI. Neither image quality nor recovered signal is guaranteed.

The tool verifies that the original CSM equals the cache's hard-0.6 maps exactly,
thereby checking basis and complex phase, not just shape/count. It preserves
calibration data and writes only:

```text
CONFIRMED_DIAGNOSTIC_DIR/
  csm/coil_sens.{hdr,cfl}
  masks/support.npz                  # native, added and ROI; logical axes
  figures/mouth_support.png           # same-basis ACS RSS; no CSM masking
  metadata/request.json
  metadata/custom_csm.json            # pass this completion to --custom-csm
```

Generation is CPU-only, single-threaded and low priority with the existing RAM
guard. It creates one full CSM (about 2.25 GiB at 256x256x192x24), not another
image reconstruction. An empty proposal, invalid geometry, unstable prediction,
source mismatch or partial output fails without overwriting previous artifacts.
Review the green additions against ACS anatomy before reconstruction.

### Partial hand annotations

`prepare_annotated_custom_csm.py --config "$REVIEWED_LOCAL_CONFIG" --output
"$CONFIRMED_DIAGNOSTIC_DIR"` supports a separately reviewed local candidate from
three **open** red traces drawn on an ACS figure. It does not infer a closed
head/tissue segmentation. Configuration explicitly records each panel's pixel
bounds, logical shape, slice index and one-sided boundary policy, a local ROI,
physical outward margin, Gaussian scales and donor-distance cap. Real paths
and anatomical settings belong only in ignored `.local.json` files.

The traces are not joined at their endpoints or extrapolated beyond their
sampled coordinate spans. Their local one-sided constraints are intersected
inside the ROI. This interpolation between three views is only a reviewable
approximation, not a measured 3D tissue surface. Original supported vectors
remain exactly unchanged; reliable unit-RSS complex predictions are added only
within the allowed region. Multiscale continuation can reach beyond the old
support's convex envelope, but longer-distance extrapolation is less certain.
Metadata reports additions beyond 16 mm and confidence rejection separately.
No ACS intensity threshold is claimed to prove tissue or recoverable signal.

The same tidy output layout additionally contains `metadata/open_traces.json`
and an `annotation_domain` mask in `masks/support.npz`. QC overlays the open red
traces, original support, actual additions, ROI and local constraint domain on
unmasked same-basis ACS RSS. Both configuration and the original annotated PNG
are hash-bound inputs. Do not edit provenance-bound figures after completion;
keep annotations as a separate file for future rounds. If an older QC figure
was edited, its old completion intentionally fails hash validation; do not
rewrite historical hashes to conceal that change. New derivatives bind the
annotated file independently and leave old arrays/metadata untouched.

### Slice-local corrections to an accepted candidate

Partial traces drawn on different slices are not global exclusion surfaces.
Intersecting their extrusions can wrongly remove a nose visible on a sagittal
slice or a lateral region visible on an axial slice. For a reviewed correction,
`prepare_slice_local_custom_csm.py --config "$REVIEWED_LOCAL_CONFIG" --output
"$CONFIRMED_DIAGNOSTIC_DIR"` accepts an annotation-bounded parent completion,
an outward margin and explicit local patches. Each patch selects one trace and
records a bounded ROI, slab core/outer half-width, ROI-edge taper and distance
cap measured from parent support. Other slices' traces do not veto this patch.
Extension shrinks smoothly toward the slab/ROI boundaries; this is geometric
tapering of binary support, not soft weighting of CSM amplitude.

The parent maps remain bitwise unchanged everywhere except newly added voxels.
Donor vectors still come exclusively from the original calibrated maps, never
from earlier extrapolated additions. No global union of unbounded halfspaces is
used. This explicit local approximation is **not** a measured 3D segmentation;
its slab thickness and out-of-plane behavior require visual review. Masks and
metadata distinguish parent additions, new additions and individual patch
domains. `figures/mouth_support.png` compares ACS, parent and corrected maps;
`figures/adjacent_slices.png` shows neighboring slices through each patch.
Parent/source/map/annotation provenance is retained and hash-checked. Candidate
generation remains CPU-only, single-threaded, default-off and reconstruction-free.

For an inexpensive geometry review before writing another full CSM, use
`preview_custom_csm_support.py --config "$REVIEWED_LOCAL_CONFIG" --output
"$CONFIRMED_PREVIEW_DIR"`. This writes only `masks/`, `figures/` and `metadata/`.
The parent completion must bind the selected support masks and annotation file.
Existing support is retained exactly; candidate additions are geometrically
bounded but **have not undergone coil-vector confidence filtering**. The
`metadata/support_preview.json` schema is deliberately incompatible with
`--custom-csm`: a preview is not a reconstruction-ready calibration.

Preview figures show **full existing support** (native plus all additions), not
only the additions. An uncolored native island inside an addition-only overlay
must not be mistaken for a hole in the CSM. Full-support previews and neighboring
slices should be reviewed before commissioning another map derivative. A trace
applied away from its original slice is only an explicit local approximation,
not an anatomical measurement at that new slice.

`compare_custom_csm_support.py --config "$REVIEWED_LOCAL_CONFIG" --output
"$CONFIRMED_PREVIEW_DIR"` compares conservative distance caps within a previously
bound preview. A common narrowed ROI/slab is used for all variants, and every
proposal must be a nested subset of the prior proposal. Caps measure distance
from existing support, not from a known tissue boundary. Outputs are compressed
mask files per cap, shared-window comparison figures and
`metadata/support_comparison.json`; no new CSM is calculated and no winner is
selected. The existing custom CSM remains unchanged and the comparison metadata
cannot be used as a reconstruction manifest.

### Recorded user selection: conservative lower-RO preview

The user selected the **4 mm distance-cap variant** from the reviewed 4/6 mm
comparison. The 6 mm variant remains comparison-only. This choice applies only
to that explicitly reviewed local patch; it does not change a pipeline default,
global ecalib crop, tissue margin or regularization parameter.

The exact selected artifacts are recorded without private dataset paths:

- Comparison completion: `metadata/support_comparison.json`, SHA-256
  `654938b43273e846f845c3de315e2157a7678a761b45b1fcddac483da9ba8b3c`.
- Selected mask archive: `masks/cap_4mm.npz`, SHA-256
  `23375412bb5faf7288263403576c8dce32bebf73c52c3b73da69711063cd9951`.
- Proposed additions: 6,178 native-grid voxels; 428 pixels on the reviewed
  RO=43 slice. Existing parent support is retained; RO=59/67 are unchanged.

Status: **geometry approved by the user; final coil vectors and reconstruction
remain pending**. The immutable comparison metadata still records that its
generator selected no winner; this later user decision is recorded here instead
of rewriting historical results. Future CSM generation must bind this exact mask
and its recorded parent/source provenance, validate the new vectors and publish
a separate completed custom-CSM manifest. Do not pass the preview completion to
`--custom-csm` or silently substitute the earlier parent maps. Real paths and
case-specific launch settings remain in ignored local files.

## Reconstruction entry points

Prepare normal native inputs first. Standard PCA uses its existing `vccN` inputs;
ROVir requires its existing complete canonical normal contract and the **same
ROVir projection basis**. Custom mode does not bootstrap a new raw preparation.
Initialize the documented Python environment and host BART build when running
reconstruction. Source validation includes current TWIX/sequence hashes and may
take time on network storage.

```bash
# Standard PCA (Ncc is derived from the bound manifest unless explicitly supplied).
bash tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --custom-csm "$CUSTOM_MANIFEST" -g
bash tools/wave_retro_lr_recon/scripts/sample_mprage_retro_lr_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --custom-csm "$CUSTOM_MANIFEST" -g

# ROVir: reuse the existing transform, without new --null-box/ROI options.
bash tools/wave_retro_lr_recon/scripts/sample_mprage_rovir_recon.sh run \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --custom-csm "$CUSTOM_MANIFEST" -g
bash tools/wave_retro_lr_recon/scripts/sample_mprage_retro_lr_recon.sh \
  "$TWIX" "$OUTPUT_ROOT" "$SEQ" --rovir --custom-csm "$CUSTOM_MANIFEST" -g
```

The independent native-R3x3 launcher and root-only ROVir retro launcher also
accept the switch. Add `--custom-check-only` to validate read-only without
creating directories or invoking BART. `--custom-prepare-only` instead creates
isolated case inputs/maps and request records, but still never runs BART.
`-g` selects GPU BART; map derivation stays on CPU. CPU is the default. Existing
memory admission/monitoring stops only this job, not other workloads.

Profiles remain mutually exclusive. Standard normal defaults to Wavelet-only;
ROVir normal defaults to FISTA-r0 plus Wavelet; retro defaults to FISTA-r0 for
native R3x2, LR-Y R3x2 and native R3x3. `--fista-only`, `--wavelet-only` and
`--reg-full` override those defaults. Normal Wavelet defaults are transferred
references: 0.03 for standard VCC24, 0.035 for other PCA counts and ROVir.
Retro Wavelet references are 0.035/0.025/0.045 respectively, when requested.
Normal accepts `--r3-lambda VALUE`. All positive branches are labelled
`wavelet_custom_unvalidated`; no old optimum is asserted for this new operator.
R1 inputs require `--fista-only`.

## Isolation, LR support and provenance

```text
OUTPUT_ROOT/
  vccN_custom_csm/<manifest-sha256-prefix>/
  rovir_custom_csm/<manifest-sha256-prefix>/
    metadata/request.json
    normal/bart_output/BRANCH/{metadata,reconstruction,logs,nifti}
    retro/CASE/
      metadata/                     # case request/completion
      bart_inputs/                  # measured subset, optional LR PSF/maps
      bart_output/BRANCH/{metadata,reconstruction,logs,nifti}
```

Native cases use the exact supplied native CSM. LR-Y uses the shared
support/gain-preserving resampler on **that custom CSM**, never on the canonical
old CSM. The native support is mapped separately from Fourier coil direction;
scalar gain is reapplied after direction normalization. See
[support/gain transfer](csm_calibration_cache.md#retrospective-csm-support-and-gain-transfer).

Retro cases are derived independently from validated native measured k-space,
using the existing pure-mask/crop and calibrated PSF phase-plane implementations.
This writes new retrospective k-space and may require substantial disk space;
it does not touch or depend on historical retro CSMs. Old canonical/metal trees
are never overwritten. Custom outputs are **not automatically collected or
preferred by the canonical NIfTI collection** in this phase.

Every run binds the exact custom manifest, maps, source, basis, PSF, k-space,
geometry, software, solver/backend and lambda. Sidecars record `CustomCSM`,
the native and case maps, method/lambda and basis. The original ecalib command
is labelled `CSMDonorEcalibCommand`, not presented as calibration of the custom
maps. Completion records bind both magnitude and phase exports. Resume requires
exact hashes; changed maps/configuration use another fingerprint. Partial maps,
failed BART attempts and interrupted exports are preserved and rejected rather
than overwritten. Switching CPU/GPU on a completed branch is a mismatch.

Tests use synthetic arrays and mocked BART for locality, basis/source isolation,
ROVir vs PCA, native identity, LR support transfer, three-case defaults, explicit
GPU routing, prepare-only, exact exports and immutable reuse. This is software
validation, not real-data validation of the new custom reconstruction.
