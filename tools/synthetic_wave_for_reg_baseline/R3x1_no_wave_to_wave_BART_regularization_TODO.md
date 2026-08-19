# R3×1 No-Wave → Synthetic Wave → BART Regularization Tuning
## Implementation To-Do and Progress Tracker

**Purpose:** Build a reproducible offline experiment for tuning BART Wave reconstruction regularization using an acquired **R3×1 no-wave** dataset and its **online scanner DICOM** as the practical reference.

**Original strategy:** Reconstruct the missing no-wave k-space with **2D GRAPPA**, preserving coil-wise k-space, then apply the existing Wave forward model and retrospectively re-apply the same R3×1 sampling mask.

**Selected baseline:** Use **joint-coil 5×5×5 GRAPPA at Ncc=12** for downstream synthetic-Wave generation. The user visually confirmed that it removes the aliasing retained by 5×5×1 and 5×5×3, preserves the full nose/mouth anatomy, and appears to have less parallel-imaging noise amplification than the tested SENSE variants. It also directly provides completed multi-coil k-space; SENSE would require the additional model-dependent back-projection `F(Sx)`. Treat the g-factor assessment as qualitative until it is measured explicitly.

---

## 0. Project objective

We want to answer:

> Which BART `wave` regularization choice and parameter value gives the best reconstruction quality for an R3×1 Wave acquisition?

Use:

1. Acquired **R3×1 no-wave raw data**.
2. Corresponding **online scanner DICOM**.
3. Raw no-wave data to synthesize **R3×1 Wave data**.
4. BART `wave` reconstructions across regularization choices/weights.
5. Quantitative and qualitative comparison against the reference.

The correct processing order is:

$$
\text{recover/estimate full no-wave coil data}
\rightarrow
\text{apply Wave encoding}
\rightarrow
\text{apply R3×1 sampling mask}.
$$

---

# 1. Core design decisions

## 1.1 Do not Wave-encode already-aliased hybrid-space data

For one coil, the desired Wave acquisition is

$$
d_{\mathrm{Wave}}
=
M F_{yz}\left[P(k_x,y,z)h(k_x,y,z)\right],
$$

where:

- `M` = R3×1 sampling mask,
- `F_yz` = Fourier transform over PE1/PE2,
- `P` = Wave PSF in `[kx,y,z]` hybrid space,
- `h` = fully resolved no-wave hybrid-space signal.

If the no-wave data are undersampled first,

$$
\hat h_{\mathrm{alias}}
=
F_{yz}^{-1} M F_{yz} h,
$$

then multiplying `P * h_alias` is not equivalent to the desired Wave encoding.

**Decision:** reconstruct/estimate the missing no-wave information first.

- [ ] Document this ordering in the implementation comments.
- [ ] Ensure no code path multiplies the Wave PSF into already PE-aliased hybrid-space data.

## 1.2 GRAPPA first; SENSE emergency continuation

GRAPPA directly estimates missing k-space while preserving coil channels:

$$
d_{\mathrm{NW,R3}}
\rightarrow
\hat k_{\mathrm{NW,full},c}.
$$

SENSE instead reconstructs a common image and then regenerates coil data:

$$
d_{\mathrm{NW,R3},c}
\rightarrow
\hat x
\rightarrow
S_c\hat x
\rightarrow
\hat k_{\mathrm{NW,full},c}.
$$

**Decision:**

- [x] Implement GRAPPA branch first.
- [ ] Implement the explicitly authorized SENSE/ESPIRiT emergency continuation.
- [ ] Compare the two synthetic Wave datasets and their preferred BART regularization. (Deferred unless otherwise stated)

## 1.3 GRAPPA kernel progression and accepted 5×5×5 result

Acquisition dimensions:

```text
RO  = kx
PE1 = ky   ← accelerated, R = 3
PE2 = kz   ← fully sampled, R = 1
coil
```

Original baseline GRAPPA kernel:

```text
5 × 5 × 1
RO × PE1 × PE2
```

The original baseline used no PE2 neighbors. Subsequent 5×5×3 GRAPPA retained
the visual alias, while the final 5×5×5 reconstruction removed it.

Conceptually:

$$
\hat k(k_x,k_y,k_z,c)
=
\sum_{\Delta k_x,\Delta k_y,c'}
W_{\Delta k_x,\Delta k_y,c',c}
\,k(k_x+\Delta k_x,k_y+\Delta k_y,k_z,c').
$$

- [x] Baseline uses a 5×5 RO×PE1 kernel.
- [x] PE2 kernel extent is 1.
- [x] Generalize true 3D GRAPPA to a configurable positive odd PE2 extent.
- [x] Run and visually approve the joint-coil 5×5×5, Ncc=12 reconstruction.
- [x] Select joint-coil 5×5×5 GRAPPA as the final no-wave completion baseline.

## 1.4 Train one shared GRAPPA model from all available ACS PE2 partitions

Phase A established that the product TWIX stores its PAT calibration in the selected measurement's separate `refscan` stream. After readout oversampling removal, its support is:

```text
RO × PE1 × PE2 = 256 × 24 × 256
PE1 indices     = 115...138
PE2 indices     = 0...255
```

Do not train 256 independent kernels.

Instead:

```text
all 256 available refscan PE2 partitions
            │
            ├─ contribute 2D RO×PE1 training examples
            ▼
      one shared R3×1
      GRAPPA model
            │
            ├─ apply to PE2 = 0
            ├─ apply to PE2 = 1
            ├─ ...
            └─ apply to PE2 = 255
```

- [x] Verify product refscan support is 256×24×256 after readout oversampling removal.
- [x] Pool training examples across the available 256 ACS PE2 partitions.
- [x] Accumulate calibration normal equations in chunks so the pooled design matrix is not retained in memory.
- [x] Train GRAPPA weights once.
- [x] Apply the same weights to all 256 PE2 partitions.

## 1.5 Code clarity and comments

Keep the implementation interpretable without burying the algorithm in commentary.

- [x] Add concise comments for non-obvious MRI conventions, GRAPPA source/target geometry, axis changes, FFT conventions, and preservation of acquired samples.
- [x] Add docstrings for public helpers and state their expected layouts.
- [x] Do not comment obvious assignments or repeat the code line-by-line.
- [x] Prefer named layout-conversion helpers and assertions over comments that compensate for ambiguous code.

## 1.6 Implement GRAPPA locally; reuse established Wave/BART utilities

The GRAPPA calibration and application code belongs to this experiment and will be written locally. Use `pygrappa` as an algorithmic reference, particularly for source/target geometry, but do not make the production path a thin call to `pygrappa.grappa()` and do not recalibrate once per PE2 partition.

For Wave reconstruction and TWIX-loading conventions, inspect and reuse the useful pieces from:

```text
external/wave-mprage/recon
sources/published_code/wave-gre-flow-comp/recon
```

Wave-MPRAGE is a pinned git submodule because this experiment imports its BART CFL and TWIX-to-NIfTI utilities at runtime. The current code does not import Wave-GRE utilities, so adding a second submodule would create an unused dependency; consult its sibling checkout only as a development reference. Add Wave-GRE as a submodule later if a unique utility is actually incorporated.

- [x] Implement explicit local R=3 GRAPPA calibration and application routines.
- [x] Record the relevant `pygrappa` behavior or source version used as a reference.
- [ ] Reuse/adapt the existing BART CFL export, ESPIRiT calibration, Wave reconstruction wrapper, and NIfTI conversion instead of reimplementing them.
- [ ] Infer ESPIRiT maps from the fully sampled no-wave ACS after applying the same coil-compression matrix used for the synthetic Wave data.
- [x] Pin Wave-MPRAGE as a git submodule at the verified utility revision used by the runtime path.
- [x] Leave Wave-GRE as a non-runtime reference unless a unique utility is incorporated later.
- [x] Do not attempt a broader utility unification in this experiment.

---

# 2. Coil compression

## 2.1 Baseline: 64 physical coils → 12 virtual coils

Use one compression matrix `C`:

$$
k_{\mathrm{cc}} = k C.
$$

The same `C` must be applied to:

- accelerated imaging k-space,
- ACS/calibration data,
- data later used for Wave synthesis,
- ESPIRiT calibration data used with the same synthetic Wave dataset.

Do not calculate separate compression matrices for ACS and imaging data.

- [x] Compute one compression basis from the product refscan using the verified Wave-MPRAGE covariance/eigendecomposition convention.
- [x] Record saved compression basis shape: `[64, 24]`; use leading columns for Ncc=12/16/24.
- [x] Implement one coil-last application helper and verify the identical basis on image and refscan payload probes.
- [x] Verify virtual-coil ordering is nested and consistent: Ncc=12/16 are the leading columns of the saved Ncc=24 basis.
- [x] Apply the selected basis to the complete imaging/refscan data as part of the chunked Phase C pipeline; only the GRAPPA-completed selected-Ncc volume is saved.

## 2.2 Validate that 12 coils are enough for GRAPPA

Signal-energy retention alone is not sufficient; weak coil modes may still contribute parallel-imaging encoding.

Test at least:

```text
Ncc = 12
Ncc = 16
Ncc = 24
```

Optional reference:

```text
Ncc = 64
```

Use held-out ACS reconstruction error:

1. Start from fully sampled ACS.
2. Artificially remove the R=3 PE1 lines.
3. Train/apply GRAPPA.
4. Compare predicted samples with the known held-out ACS samples.

Metric:

$$
\mathrm{NRMSE}_{ACS}
=
\frac{\|\hat k_{\mathrm{heldout}}-k_{\mathrm{true}}\|_2}
{\|k_{\mathrm{true}}\|_2}.
$$

- [x] Ncc=12 held-out ACS NRMSE measured: `0.17240017`.
- [x] Ncc=16 held-out ACS NRMSE measured: `0.14574042`.
- [x] Ncc=24 held-out ACS NRMSE measured: `0.12384302`.
- [ ] Optional Ncc=64 reference measured.
- [x] Final coil count selected.

**Active coil count:** `12` (explicit user choice after per-coil image review)

**Reason:** Held-out ACS NRMSE decreased monotonically from `0.17240017` (12) to `0.14574042` (16) to `0.12384302` (24), so the original aggregate-NRMSE rule preferred 24. Per-coil image review showed conspicuous residual structure in several higher-order virtual coils (including 17, 19, and 24). The user therefore explicitly selected the leading 12 nested virtual coils as the active baseline. Preserve the 24-coil result as a comparison rather than presenting the 12-coil choice as NRMSE-optimal.

---

# 3. GRAPPA implementation

## 3.1 Use pygrappa as a reference/start point, but avoid 256 independent calibrations

`pygrappa.grappa()` is suitable as a 2D algorithmic reference and test oracle for small arrays, but the production GRAPPA implementation will be written in this project. Its workflow is:

```text
calibrate once
     ↓
save/reuse R=3 weights
     ↓
apply across all PE2 partitions
```

For regular R=3, there are two missing-line target geometries:

```text
X . . X . . X . . X

X 1 . X 1 . X 1 . X
X . 2 X . 2 X . 2 X
```

Prefer explicit weights for the two target offsets.

- [x] Inspect pygrappa's source/target geometry for R=3.
- [x] Implement the calibration and application math locally, with comments around the non-obvious source/target indexing.
- [x] Compare the local implementation with pygrappa on a small controlled case where their conventions match.
- [x] Separate calibration from application.
- [x] Represent target type 1 and target type 2 explicitly.
- [x] Do not recalibrate for each PE2 plane.

## 3.2 Preserve measured samples exactly

Final completed k-space must obey:

$$
\hat k(k)=
\begin{cases}
k_{\mathrm{measured}}(k), & k\in\Omega,\\
k_{\mathrm{GRAPPA}}(k), & k\notin\Omega.
\end{cases}
$$

- [x] Build/verify acquisition sampling mask.
- [x] Copy measured locations unchanged.
- [x] Fill only missing samples.
- [x] Unit test and full-volume audit verify acquired samples are bitwise identical.

## 3.3 Array conventions

Recommended internal Python convention:

```text
[coil, RO, PE1, PE2]
```

2D GRAPPA work plane:

```text
[RO, PE1, coil]
```

BART convention:

```text
[RO, PE1, PE2, coil]
```

- [x] Select and document canonical `[RO, PE1, PE2, coil]` layout (with `[RO, PE1, coil]` planes).
- [x] Add shape assertions before/after every major operation.
- [ ] Add explicit transpose helpers rather than ad-hoc `np.transpose` calls.
- [ ] Verify BART layout separately.

Expected matrix:

```text
RO  = 256
PE1 = 256
PE2 = 256
physical coils = 64
virtual coils  = selected Ncc
```

---

# 4. GRAPPA compute strategy

## 4.1 Do not use generic whole-volume 3D mdgrappa

Avoid starting with:

```python
mdgrappa(full_256x256x256_volume, ...)
```

Reasons:

- no PE2 acceleration,
- unnecessarily large temporary arrays,
- generic missing-point enumeration,
- poor scaling for 256³ data.

- [x] The original 2D baseline did not use PE2 neighbors; the accepted final
  method uses the custom resumable joint-coil 5×5×5 implementation.
- [x] Baseline does not run full-volume generic 3D GRAPPA.

## 4.2 Apply shared 2D weights PE2-by-PE2 or in chunks

Correctness-first implementation:

```text
train once
   ↓
PE2 = 0
PE2 = 1
...
PE2 = 255
```

Optimization later:

```text
chunk size = 4–16 PE2 partitions
```

- [ ] Start with one PE2 plane at a time.
- [ ] Benchmark one plane.
- [ ] Benchmark 8 planes.
- [x] Add chunked/vectorized application after verifying it against plane-wise application.
- [x] Record total application runtime (`904.54 s`, including final shared-filesystem writeback); peak memory was not instrumented.

## 4.3 Runtime expectations

Planning range for a good 2D implementation:

```text
12 coils: a few minutes to ~10 minutes
64 coils: several minutes to tens of minutes
```

These are engineering expectations, not guaranteed benchmarks.

If runtime reaches multiple hours, check for:

- repeated calibration,
- Python loops over all 3D holes,
- accidental 3D kernels,
- unnecessary complex128 application arrays,
- unnecessary copies,
- slow memmap/temp storage,
- unnecessary use of all 64 coils.

**Runtime Ncc=12:** `TBD`

**Runtime Ncc=64:** `TBD`

---

# 5. Build the full coil-wise no-wave k-space

Target:

```text
full no-wave k-space
[RO, PE1, PE2, Ncc]
```

Validation:

- [x] No missing samples remain in the nominal Cartesian matrix (`243,793,920/243,793,920` expected missing coil samples populated).
- [x] Measured locations are unchanged (bitwise audit passed).
- [ ] Filled lines have plausible magnitude/phase continuity.
- [x] Coil-combined central-partition IFFT is anatomically plausible.
- [x] Central RSS diagnostic shows no strong residual R=3 aliasing.
- [x] Central PE2 partition inspected.
- [ ] Edge PE2 partitions inspected.
- [ ] PE1/PE2 orientation verified.

Suggested intermediate outputs:

```text
grappa_full_kspace.npy or CFL
grappa_coil_images.npy          optional
grappa_rss_reference.nii.gz     optional
grappa_metadata.json
```

- [ ] Save completed k-space.
- [ ] Save metadata.
- [ ] Save diagnostic images.

---

# 6. Wave synthesis from the GRAPPA-completed data

Reuse the existing fully-sampled-no-wave → Wave forward model, but use the theoretical PSF generated from the matching sequence rather than a measured/calibrated PSF.

## 6.1 Extend the readout image FOV before Wave encoding

The active GRAPPA-completed input uses canonical layout:

```text
[RO, PE1, PE2, Ncc] = [256, 256, 256, 12]
```

Phase A and the mapVBVD stream layout establish that axis 0 is readout. Do not infer the readout axis from equal matrix sizes at runtime; carry this named layout forward and assert it at every conversion.

The matching Wave sequence uses readout oversampling factor 4. Keep the 1 mm voxel size and extend the represented readout FOV from 256 mm to 1024 mm:

1. Apply a centered orthonormal 3D inverse FFT to each no-wave virtual coil.
2. Allocate exact-zero coil images with shape `[1024, 256, 256, 12]`.
3. Center the original 256 readout voxels at `ROext[384:640]`.
4. Assert that the center crop exactly recovers the original image and that both outer readout regions remain zero.

This is equivalent to Fourier interpolation onto the sequence's 1024-sample readout grid. Implement the Wave forward operator directly from the extended coil image to avoid a redundant full FFT/IFFT pair:

$$
h_{\mathrm{NW},c}=F_{RO}x_{\mathrm{NW,ext},c},
$$

$$
h_{\mathrm{Wave},c}=P_{\mathrm{theory}}h_{\mathrm{NW},c},
$$

$$
k_{\mathrm{Wave,full},c}=F_{PE1,PE2}h_{\mathrm{Wave},c}.
$$

The final full synthetic Wave k-space must have shape `[1024, 256, 256, 12]`. Use centered orthonormal FFTs throughout and record the convention in metadata.

- [x] Assert input layout `[RO, PE1, PE2, Ncc] = [256,256,256,12]` and identify RO as axis 0 from the loader convention.
- [x] Centered-IFFT the completed no-wave k-space to coil images.
- [x] Embed the image at readout indices `384:640` of an exact-zero `[1024,256,256,12]` array.
- [x] Verify center-crop recovery, zero exterior, 1 mm readout voxel size, and 1024 mm extended readout FOV.
- [x] Apply the forward operator as `F_RO → theoretical PSF → F_PE1,PE2` without redundant transforms.

## 6.2 Generate the theoretical PSF from the matching sequence

The exact sequence path for the current dataset is recorded in the ignored `LOCAL_DATASETS.md`. Its relevant definitions are:

```text
ReadoutOversamplingFactor = 4
Calibration_ReadoutSamples = 1024
Calibration_Ncalib1 = 72
Calibration_Nacs = 32
FOV = 0.256 × 0.256 × 0.256 m
OrientationMapping = SAG
MPRAGE_UseWaveSin/Cos = 1/1
Wave amplitude/cycles = 8 mT/m / 10
```

Reuse/adapt `generate_theoretical_wave_trajectory()` from the Wave-MPRAGE reconstruction. Exclude the integrated calibration/ACS tail exactly as the reference code does:

```text
Nacs_total = 1024 × (4 × 72 + 32²) = 1,343,488 ADC samples
```

Generate `delta_ky_idx` and `delta_kz_idx` from the sequence trajectory, then form the theoretical hybrid-space PSF on the final `[1024,256,256]` grid. Verify the sagittal `yflip/zflip` convention against the reference code rather than silently assuming it.

Use this same theoretical PSF—without measured calibration correction—for both synthetic Wave encoding and BART reconstruction. Export one canonical PSF and derive both consumers from it; record its sequence path/hash, dimensions, flips, trajectory excursion, dtype, and array/BART layouts. Do not use the pre-existing calibrated PSF under `mprage_bart/bart_inputs` for this experiment.

- [x] Generate trajectory offsets from the supplied sequence using the verified integrated-tail exclusion.
- [x] Build a unit-magnitude theoretical PSF with shape `[1024,256,256]` complex64.
- [x] Verify PSF phase/orientation and sagittal flip conventions.
- [x] Export BART PSF shape `[1024,256,256,1,1]`.
- [x] Prove the synthesis and BART PSF payloads are identical, for example with a SHA-256 hash.
- [x] Record sequence and PSF provenance in machine-readable metadata.

## 6.3 Pre-BART full-Wave sanity images

Before applying the R3×1 sampling mask or running BART, directly centered-IFFT the full synthetic Wave k-space over all three spatial axes. Interpret the requested extended-FOV image size as `1024×256×256` per channel.

For a configurable first few active virtual coils (default: coils 1–4):

- save magnitude and phase NIfTIs with shape `[1024,256,256]`, 1 mm isotropic;
- save compact central-slice montage PNGs for fast review;
- optionally save an RSS magnitude diagnostic across all 12 coils without retaining a redundant full complex image volume;
- record that these are direct-IFFT Wave-encoded diagnostics, not de-Waved/BART reconstructions.

Pause before BART until these diagnostics have been visually reviewed.

- [x] Save full-Wave direct-IFFT magnitude/phase NIfTIs for the first few channels.
- [x] Save montage/RSS quick-look diagnostics.
- [x] Verify all diagnostic arrays are finite and have the expected extended-FOV geometry.
- [x] Obtain visual approval before starting BART reconstruction.

## 6.4 Apply Wave encoding, then the acquisition mask

The extended-readout forward operator is defined in Section 6.1. Keep its full `[1024,256,256,12]` output unchanged for the Section 6.3 diagnostics. Only after those checks pass, apply the authoritative product acquisition mask:

$$
d_{\mathrm{Wave,R3},c}
=
M_{R3\times1}k_{\mathrm{Wave,full},c}.
$$

- [x] Verify PSF orientation/axis convention.
- [x] Reuse existing Wave synthesis code where possible.
- [x] Assert PSF dimensions.
- [x] Document FFT normalization convention.
- [x] Generate full synthetic Wave k-space.
- [x] Apply R3×1 mask only after Wave encoding.
- [x] Set unacquired BART samples to exact complex zero.

---

# 7. Re-apply the exact R3×1 sampling pattern

Do not assume simply:

```python
mask[::3] = 1
```

unless confirmed by the actual sequence/raw-data indices.

Preserve:

- ACS lines,
- acceleration offset,
- exact acquired line set,
- any sequence-specific edge behavior.

- [x] Extract/reconstruct the actual PE1 sampling mask.
- [x] Derive the mask from TWIX acquisition indices/metadata rather than treating nonzero sample magnitude as the authoritative acquisition indicator.
- [x] Verify ACS fully sampled region.
- [x] Verify R=3 offset.
- [x] Verify number of acquired lines matches the source scan.
- [x] Verify synthetic Wave unacquired positions are exact zero.

---

# 8. BART Wave reconstruction

Expected BART dimensions:

```text
coil maps:
[RO, PE1, PE2, Ncc, Nmaps]

Wave PSF:
[ROext, PE1, PE2, 1, 1]

Wave k-space:
[ROext, PE1, PE2, Ncc, 1]
```

Baseline ESPIRiT:

```text
-m1
```

Optional later:

```text
-m2   # soft-SENSE / multi-map comparison
```

Generate the ESPIRiT maps strictly from the measured **no-wave product refscan ACS**, not from the GRAPPA-completed volume and not from the synthetic Wave k-space:

The GRAPPA selection governs the completed multi-coil source used for Wave
synthesis. It does not eliminate BART Wave reconstruction's separate need for
coil maps. Validate that reconstruction-map support does not mask anatomy,
without reopening SENSE as the no-wave completion method.

1. Load the `refscan` stream from the same selected product TWIX measurement, with readout oversampling removed using the established loader convention.
2. Apply the exact leading 12 columns of the saved 64→24 compression basis used for GRAPPA and Wave synthesis. Do not estimate a new compression basis for map calibration.
3. Preserve the measured ACS support: 256 readout samples, raw PE1 lines `115:139` (24 lines), and all 256 PE2 partitions.
4. Insert that compressed ACS into an exact-zero logical no-wave grid `[256,256,256,12]`, keeping PE1 lines at their raw coordinates, and export it as BART `kspace_calib.{hdr,cfl}` in `[READ,PHS1,PHS2,COIL]` order.
5. Verify the CFL payload against the compressed refscan: measured ACS values must match bitwise, samples outside the ACS support must be exact complex zero, and all values must be finite.
6. Run `bart ecalib -m 1` on `kspace_calib`, following the pinned Wave-MPRAGE wrapper and consulting Wave-GRE only if it contains a needed newer behavior. Record the BART version, full command, calibration options, runtime, and output hash.
7. Require the calibrated map output to have logical shape `[256,256,256,12,1]`, matching the active virtual-coil ordering. Save magnitude/phase quick looks and check finiteness, spatial support, and absence of coil-axis or PE-axis swaps before Wave reconstruction.

- [x] ESPIRiT maps generated in the same virtual-coil basis.
- [x] Reuse/adapt `recon/bart/bart_utils/bart_io.py` and `recon/bart/run_wave_recon.sh` from the existing Wave repositories.
- [x] Load the measured no-wave product refscan ACS and compress it with the exact active 64→12 basis.
- [x] Build and validate BART `kspace_calib` from the measured no-wave ACS in that same virtual-coil basis.
- [x] Run `bart ecalib -m 1` and record its version, command, options, runtime, and output hash.
- [x] Validate `[256,256,256,12,1]` ESPIRiT-map dimensions and magnitude/phase diagnostics.
- [ ] Before the final BART Wave sweep, verify that the chosen reconstruction
  maps preserve full anatomy; treat this separately from the selected 5×5×5
  GRAPPA multi-coil source.
- [x] BART dimensions verified.
- [x] Unregularized Wave reconstruction completed.
- [ ] Wavelet sweep completed.
- [ ] Coarse LLR pilot completed.
- [ ] Optional multi-map ESPIRiT comparison completed.

---

# 9. Initial BART regularization sweep

## 9.1 Prerequisites and one-time CSM gate

The existing synthetic-Wave manifest points to the older GRAPPA completion.
Before any sweep:

1. Start a new output tree; do not overwrite the historical synthetic-Wave run.
2. Regenerate the full and R3×1-masked synthetic Wave k-space from the accepted
   joint-coil 5×5×5, Ncc=12 no-wave k-space.
3. Reuse the identical saved 64→12 coil basis, theoretical PSF, sequence file,
   readout extension/crop convention, and measured sampling mask.
4. Verify source/output hashes, shapes, finiteness, acquired-mask counts, and
   direct-IFFT coil diagnostics before reconstruction.
5. Run one hard `bart ecalib -m 1 -c 0.5` candidate from the measured compressed
   no-wave ACS, saving the eigenvalue map, command, BART version, runtime, and
   map hash. Do not use `-S`.
6. Run λ=0 BART Wave once with the new 5×5×5-derived inputs and crop-0.5 maps,
   export the NIfTI, and stop for visual confirmation of full facial anatomy
   and absence of a detached background shell.
7. If that CSM gate fails, stop before the regularized sweep and resolve map
   support. If it passes, reuse the exact saved CSM CFL pair and hash for every
   wavelet and LLR run; do not rerun `ecalib` inside the sweep.

The recorded one-time `ecalib` cost is 92.54 s. The recorded λ=0 Wave command
took 767.65 s wall time (164.32 s internal reconstruction), followed by
117.98 s for NIfTI export. The first regularized run must be timed before
updating the remaining runtime estimate.

The accepted-5×5×5 rerun used hard `ecalib -m 1 -c 0.5` and saved both maps
and eigenvalues. It took 98.36 s. Its λ=0 Wave command took 747.71 s wall time
(161.74 s internal reconstruction; 747.09 s reported BART total), followed by
69.17 s for NIfTI export. The resulting λ=0 NIfTI is awaiting the required
visual anatomy-support confirmation before either positive-λ smoke test.

## 9.2 Publishable sweep driver

Run the sweep through a small, sensibly named driver rather than invoking every
reconstruction manually. The driver must:

- accept all paths, regularizer settings, and backend choices through its CLI;
- contain no dataset-specific absolute paths;
- call the pinned upstream Wave-MPRAGE BART wrapper directly where its behavior
  is used instead of copying that wrapper's implementation;
- accept an existing CSM basename and verify its recorded hash;
- use deterministic output names and never overwrite completed results by default;
- resume safely by validating each completed run's manifest and output;
- record the exact BART command, version, backend, λ, block size, iteration
  settings, maximum eigenvalue, internal runtime, wall time, and output hash;
- export magnitude/phase NIfTIs consistently for every result;
- have focused tests for command construction, parameter validation, naming,
  resume/skip behavior, and manifest generation;
- include concise function docstrings and comments where intent is not obvious.

Freeze the backend and optimizer settings after one smoke test; do not switch
CPU/GPU or stopping rules between parameter choices. Reuse the saved maximum
normal-operator eigenvalue (`-e 6.70e7`) so each run does not estimate it again.

## 9.3 Coarse wavelet pilot

Use FISTA and the requested three log-spaced weights:

```bash
bart wave \
    -w \
    -f \
    -i 100 \
    -t 1e-6 \
    -e 6.70e7 \
    -r <lambda> \
    coil_sens \
    wave_psf \
    wave_kspace \
    recon
```

- [x] λ = 0 rerun with the accepted 5×5×5 source and crop-0.5 CSM candidate
- [ ] λ = 1e-4
- [ ] λ = 1e-3
- [ ] λ = 1e-2

## 9.4 Coarse LLR pilot

The pinned Wave-MPRAGE wrapper demonstrates LLR with block size 8 and λ=0.002.
Keep block size fixed at 8 and bracket that reference value by a factor of ten:

```bash
bart wave \
    -l \
    -b 8 \
    -f \
    -i 100 \
    -t 1e-6 \
    -e 6.70e7 \
    -r <lambda> \
    coil_sens \
    wave_psf \
    wave_kspace \
    recon
```

- [ ] LLR block 8, λ = 2e-4
- [ ] LLR block 8, λ = 2e-3
- [ ] LLR block 8, λ = 2e-2

## 9.5 Regularization smoke-test gate

Do not launch all six positive-λ jobs immediately after the λ=0/CSM gate.
First run only these representative center values:

```text
wavelet: λ = 1e-3
LLR:     block 8, λ = 2e-3
```

For each smoke test, verify that BART actually selected the requested
regularizer/FISTA path, the output is finite and nonzero, the result differs
from λ=0, the command and convergence/runtime log are complete, and the NIfTI
geometry is unchanged. Export matched quicklooks and pause for explicit user
visual confirmation. This gate is intended to catch command, scaling,
convergence, excessive smoothing, and ineffective-regularization problems
before spending time on the rest of the coarse pilot.

Only after visual approval run the four remaining endpoints:

```text
wavelet: λ = 1e-4, 1e-2
LLR:     block 8, λ = 2e-4, 2e-2
```

- [ ] Wavelet λ=1e-3 smoke test exported and technically validated.
- [ ] LLR block-8 λ=2e-3 smoke test exported and technically validated.
- [ ] Explicit user visual approval received before remaining coarse jobs.

This gives seven images in the initial comparison: λ=0, three wavelet results,
and three LLR results. At the previously observed end-to-end rate, budget
roughly 1.5–2.5 hours sequentially, but replace this estimate after the first
timed FISTA run.

- [ ] Scripted sweep driver implemented.
- [ ] Per-run commands, status, runtime, CSM hash, and output paths recorded in machine-readable metadata.
- [ ] Coarse wavelet and LLR results compared visually and quantitatively.
- [ ] Fine λ refinement deferred; do not add intermediate λ values in this stage.
- [ ] LLR block-size refinement deferred; do not sweep block size in this stage.

**Current best method/λ:** `TBD`

---

# 10. Reference images and experiment interpretation

Primary practical reference:

```text
online R3×1 no-wave scanner DICOM
```

Also retain:

```text
offline GRAPPA-completed no-wave reconstruction
offline SENSE-completed no-wave reconstruction (deferred secondary diagnostic)
```

Important limitation:

$$
d_{\mathrm{NW,R3}}
\rightarrow
\hat k_{\mathrm{NW,full}}
\rightarrow
d_{\mathrm{Wave,R3}}^{\mathrm{synthetic}}.
$$

The no-wave completion method is part of the synthetic-data model.

Interpret this primarily as a **controlled regularization-selection experiment**, not a perfect physical simulation of an acquired Wave scan.

- [ ] State this limitation in analysis notes.
- [ ] Compare GRAPPA- and SENSE-derived results only in a future method-dependence study.

---

# 11. Image registration and intensity normalization

Before comparing online DICOM and offline reconstructions:

- [ ] Select only the 256 unfiltered product images from the 512-file DICOM
  folder; exclude the duplicated filtered distortion-correction series.
- [ ] Convert the unfiltered DICOM reference to floating point while recording
  its original unsigned-12-bit normalization.
- [ ] Correct the known rotation/flips and resolve the approximately half-voxel
  geometry convention before calculating residual metrics.
- [ ] Match voxel size, matrix, FOV, and cropping.
- [ ] Estimate one documented rigid/affine registration convention and apply
  it identically to λ=0, every wavelet result, and every LLR result. Do not
  independently optimize geometry for each reconstruction.
- [ ] Build and save one anatomy mask plus fixed background, homogeneous-WM,
  gray-matter/WM, aliasing, and edge ROIs in the common reference space.
- [ ] Select one robust intensity-matching rule and apply the same procedure to
  every reconstruction. Do not compare raw DICOM and BART voxel scales.

Possible normalization choices:

- WM/brain ROI scale,
- robust linear scale fit,
- percentile-based scaling.

Do not compare raw scanner and BART voxel values without scaling.

---

# 12. Evaluation metrics

Do not optimize λ using naïve SNR alone: stronger regularization can decrease apparent noise while increasing bias and smoothing.

## Quantitative

- [ ] SSIM within the fixed anatomy mask as the primary structural metric.
- [ ] PSNR after documented robust intensity matching and fixed registration.
- [ ] NRMSE and MAE within the same anatomy mask.
- [ ] Mean, SD, and coefficient of variation in homogeneous WM.
- [ ] Gray/white-matter CNR.
- [ ] Background noise level and spatial noise nonuniformity.
- [ ] Edge sharpness plus a gradient/detail-preservation metric.
- [ ] Residual R=3 aliasing/error ROI metric.
- [ ] Full-anatomy coverage and missed-anatomy fraction, including nose/lips.
- [ ] Record metrics per parameter in machine-readable CSV/JSON with the exact
  reference, mask, registration, and normalization provenance.

## Qualitative

Inspect:

- [ ] cortical sharpness,
- [ ] gray/white matter boundaries,
- [ ] small structures/vessels if relevant,
- [ ] residual R=3 aliasing,
- [ ] ringing,
- [ ] over-smoothing,
- [ ] noise texture,
- [ ] peripheral anatomy,
- [ ] jaw/neck/FOV-wrap regions if relevant.

Create side-by-side and difference images for the λ sweep.

Rank the coarse candidates using visual assessment together with SSIM, PSNR,
NRMSE, noise/CNR, sharpness, aliasing, and anatomy coverage. Do not select a
winner from PSNR or SSIM alone because the product DICOM includes scanner-side
processing and is a practical reference rather than exact synthetic truth.

---

# 13. Emergency Phase G — SENSE/ESPIRiT recovery branch (concluded)

The user explicitly activated this branch after the GRAPPA artifact persisted.
It established that SENSE removes the alias but none of the tested single-map
ESPIRiT support choices passed the full-anatomy/background visual gate. Stop
the threshold search and retain these outputs as a secondary diagnostic only.

```text
R3×1 no-wave raw
        ↓
same coil compression
        ↓
measured imaging and ACS in the identical 64→12 virtual-coil basis
        ↓
BART `ecalib -m 1` maps from compressed measured ACS only
        ↓
SigPy unregularized Cartesian SENSE with the exact acquisition mask
        ↓
complex common image x
        ↓
x × S_c
        ↓
coil-wise no-wave images
        ↓
FFT
        ↓
full coil-wise no-wave k-space
        ↓
same Wave synthesis
        ↓
same R3×1 mask
        ↓
same BART regularization sweep
```

For multi-map ESPIRiT:

$$
x_c = S_{c,1}x_1 + S_{c,2}x_2
$$

rather than forcing a single-map model.

- [x] Reuse the saved leading 12 columns of the validated nested coil basis.
- [x] Export measured image/refscan-union k-space with exact-zero missing samples and no GRAPPA values.
- [x] Verify imaging and ACS provenance use the identical basis file/hash and columns `[0,12)`.
- [x] Run BART `ecalib -m 1 -c 0.8` on the compressed measured no-wave ACS.
- [x] Run λ=0 SigPy SENSE with an explicit image/refscan union mask.
- [x] Record CG iterations, residual, wall time, and acquired-sample model residual.
- [x] Export magnitude/phase NIfTIs in corrected canonical RAS orientation.
- [ ] Inspect both axial index 75 and reverse-count index 180 before acceptance.
- [x] SENSE preprocessing kept unregularized to avoid pre-smoothing synthetic truth.
- [x] Coil-wise data regenerated as model-consistent `F(Sx)` k-space; do not feed it to Wave synthesis before visual approval.
- [ ] Synthetic Wave data generated.
- [ ] Same BART λ sweep repeated.
- [ ] Preferred λ compared against GRAPPA branch.

---

# 14. GRAPPA vs SENSE comparison (deferred secondary analysis)

Do not mistake a clean SENSE result for proof that GRAPPA cannot work. The two
methods have different calibration models and conditioning. A clean SENSE
result localizes the current failure to the GRAPPA branch; retain an independent
GRAPPA-reference audit for later.

Compare:

```text
A. GRAPPA-derived synthetic Wave
B. SENSE-derived synthetic Wave
```

Questions:

1. Are the full no-wave coil datasets similar?
2. Are the synthetic Wave k-spaces similar?
3. Are final Wave reconstructions similar?
4. Is the preferred regularizer unchanged?
5. Is the preferred λ in the same range?
6. Which no-wave completion better matches the online no-wave DICOM before Wave synthesis?

- [ ] Compare no-wave images.
- [ ] Compare synthetic Wave k-space.
- [ ] Compare NRMSE/SSIM.
- [ ] Compare visual ranking.
- [ ] Record best λ from each branch.

**GRAPPA best λ:** `TBD`

**SENSE best λ:** `TBD`

**Conclusion:** `TBD`

---

# 15. Suggested implementation structure

Adapt to the existing repository rather than duplicating existing utilities.

```text
recon/
├── grappa/
│   ├── coil_compression.py
│   ├── grappa_calibration.py
│   ├── grappa_apply.py
│   └── grappa_validation.py
│
├── synthetic_wave/
│   ├── wave_forward.py
│   ├── sampling_mask.py
│   └── bart_export.py
│
├── experiments/
│   ├── run_grappa_wave_synthesis.py
│   ├── run_sense_wave_synthesis.py        # deferred
│   ├── run_bart_regularization_sweep.py
│   └── compare_regularization.py
│
└── tests/
    ├── test_grappa_acquired_samples_unchanged.py
    ├── test_grappa_acs_holdout.py
    ├── test_wave_forward_no_wave_identity.py
    ├── test_sampling_mask.py
    └── test_bart_dimensions.py
```

---

# 16. Suggested CLI for GRAPPA synthesis

Example target interface:

```bash
python run_grappa_wave_synthesis.py \
    --twix scan.dat \
    --seq scan.seq \
    --out synthetic_wave \
    --ncc 12 \
    --grappa-kernel-ro 5 \
    --grappa-kernel-pe1 5 \
    --grappa-lambda 0.01 \
    --acs-pe1 24 \
    --acs-pe2 all
```

The loader should infer the ACS support from TWIX counters. These flags are validation/override controls, not the authoritative source of its location.

Useful optional flags:

```text
--reuse-coil-compression
--save-grappa-full-kspace
--save-grappa-coil-images
--validate-acs
--benchmark-grappa
--save-bart
```

Use the existing repository's naming conventions where possible.

---

# 17. Required diagnostics/logging

Record every run:

```text
input TWIX
input sequence
matrix size
sampling pattern
ACS size
physical coil count
compressed coil count
coil-compression energy retained
GRAPPA kernel size
GRAPPA regularization
GRAPPA calibration runtime
GRAPPA application runtime
Wave PSF source
BART version
ESPIRiT settings
BART Wave settings
output paths
```

- [ ] Save JSON/YAML run metadata.
- [ ] Print concise human-readable summary.
- [ ] Record random seeds where applicable.

---

# 18. Minimum tests before trusting the experiment

## GRAPPA

- [ ] Fully sampled synthetic test → remove R3 lines → GRAPPA → compare with truth.
- [x] Acquired lines remain unchanged.
- [x] ACS held-out error is measured and used for the coil-count decision.
- [x] No PE1/PE2 transpose in the audited `[256,256,256,24]` output.
- [x] Shared weights are reused across PE2.
- [x] 12-coil result compared against 16 and 24 coils.

## Wave synthesis

- [ ] With `PSF = 1`, Wave-forward path reproduces no-wave k-space.
- [ ] Full-sampling Wave synthesis is internally consistent.
- [ ] R3 mask is applied only after Wave encoding.
- [ ] BART zero mask matches intended acquired positions.

## BART

- [ ] Unregularized reconstruction runs.
- [ ] Regularized reconstruction runs.
- [ ] Output scaling is handled consistently across λ.
- [ ] Phase output is retained if needed.

---

# 19. Recommended execution order

## Phase A — data and mask verification

- [x] Load/index R3×1 no-wave TWIX and probe one image/refscan payload block.
- [x] Enumerate the product TWIX measurements and available `image`/`refscan` streams; do not assume the integrated Pulseq SET layout.
- [x] Locate the fully sampled ACS in the product TWIX and record its stream, dimensions, acquisition indices, and relevant MDH counters.
- [ ] Keep the product-TWIX loader minimal once the ACS location is confirmed.
- [x] Verify 256×256×256 matrix and 64 channels.
- [x] Verify product refscan ACS support: 256×24×256 after readout oversampling removal.
- [x] Verify exact R3×1 PE1 mask: image lines are `1 mod 3`; merge refscan PE1 lines 115...138 across all PE2 partitions.
- [x] Identify and preserve the 256-slice unfiltered `ND` online DICOM series as the reference.

Phase A implementation and reproducible output:

```text
scripts/inspect_product_dataset.py
phase_a_report_20260817_product.json   # machine-local, ignored by git
```

The TWIX has two measurements. Measurement 1 is selected because it contains the main 256³ MPRAGE image stream. It contains 21,760 unique image PE coordinates and 6,144 unique refscan coordinates; after overlap, the merged mask contains 25,856 unique PE1/PE2 coordinates, or 101 acquired PE1 lines per PE2 partition. There are no duplicate or out-of-range coordinates.

## Phase B — coil compression

- [x] Build one 64→24 maximum basis; Ncc=12 and 16 use nested leading columns.
- [x] Validate identical application and output shapes on image and refscan payload probes.
- [x] Measure covariance-energy retention for Ncc=12/16/24.
- [x] Validate 12 against 16/24 using held-out ACS GRAPPA NRMSE; retain the metric-preferred Ncc=24 comparison and use the user-selected Ncc=12 active baseline.

Phase B implementation and machine-local outputs:

```text
scripts/estimate_coil_compression.py
phase_b_coil_compression_20260817_product.json   # ignored by git
phase_b_coil_compression_20260817_product.npz    # ignored by git
```

The covariance pass used all 256 refscan PE2 partitions, PE2 chunks of 8, and a readout stride of 4 matching the reference utility. It accumulated 393,216 nonzero sample rows without retaining the design matrix. Runtime was 37.95 seconds. The saved `[64,24]` basis has Frobenius orthogonality error `6.61e-7`.

## Phase C — GRAPPA

- [x] Implement local 2D R3 calibration/application using pygrappa 0.26.3 as a test oracle.
- [x] Pool calibration examples across all 256 available refscan PE2 partitions using chunked normal-equation accumulation.
- [x] Train shared weights once.
- [x] Apply across all 256 PE2 planes in vectorized four-partition chunks.
- [x] Validate held-out ACS and completed no-wave k-space.
- [x] Save both the metric-preferred `[256,256,256,24]` comparison and user-selected active `[256,256,256,12]` complex64 k-space, each with a central RSS diagnostic.

### GRAPPA visual-acceptance troubleshooting

The central-slice diagnostic was insufficient. At stored axial index 180
(the user-reported slice 75 when counted from the opposite direction), the
same R3 ghost is present in the no-wave GRAPPA RSS image and the synthetic
Wave λ=0 image, but not in the scanner's unfiltered product DICOM. Therefore
the artifact originates in no-wave GRAPPA and propagates through the otherwise
consistent Wave forward/reconstruction path.

- [x] Confirm all coils are trained jointly; no coil is calibrated independently.
- [x] Confirm raw/local PE1 residue mapping, measured-sample preservation, FFT conventions, and `pygrappa` regular-grid source geometry.
- [x] Measure held-out ACS NRMSE per PE2 partition. The pooled Ncc=12 value
  `0.1724` hides a range of `0.1425...1.1320` and a median of `0.5122` because
  the energy-weighted aggregate is dominated by central kz partitions.
- [x] Test shared-kernel Tikhonov values. Reducing the parameter from `0.01`
  to `0.0001` changes aggregate NRMSE only from `0.1724` to `0.1690` and does
  not explain or fix the artifact.
- [x] Reconstruct a diagnostic Ncc=12 volume with one joint-multicoil 5×5
  kernel calibrated from each PE2 partition's own fully sampled refscan.
  This slightly reduces the ghost but does not remove it.
- [x] Compare the existing Ncc=12 and Ncc=24 RSS images at the problematic
  slice. Ncc=24 is less noisy but retains the same structured ghost.
- [x] Use `pygrappa` only as an ACS oracle for wider 2D kernels. A 7×9 kernel
  improves representative self-fit NRMSE only modestly and does not justify a
  second large 2D reconstruction.
- [x] Use `pygrappa.mdgrappa` only as a small ACS oracle for 3D support. A
  5×5×3 kernel improves representative NRMSE by about 9% in outer partitions
  and 34% centrally, but is substantially slower and remains unproven as a
  visual fix. A full 3D GRAPPA branch is not the default next step.
- [x] After explicit discussion, implement one final local 5×5×3 attempt as a
  resumable command-line job. `scripts/reconstruct_no_wave_grappa_3d.py` checkpoints the
  compressed ACS, pooled normal equations, and flushed reconstruction
  partitions; all 12 source coils jointly predict all 12 target coils.
- [x] Run the 5×5×3 job, export its RSS NIfTI, and confirm that visual aliasing persists.
- [x] Generalize the resumable implementation to 5×5×Kz with kernel-aware checkpoints.
- [x] Run joint-coil 5×5×5 at Ncc=12 and λ=0.01. The user visually confirmed
  that its reconstructed image no longer has the aliasing seen in shallower kernels.
- [ ] If a future run is interrupted, rerun the identical command with `--resume`; do not
  delete or mix individual checkpoint files before resuming.
- [x] Accept the 5×5×5 GRAPPA volume as an alias-free candidate for downstream Wave synthesis.
- [x] Retain SENSE as the second recommended approach, conditional on fixing
  the current ESPIRiT-support cutoff at the nose/mouth.

## Phase D — synthetic Wave

- [x] Center-embed the Ncc=12 coil images from 256 to 1024 readout voxels.
- [x] Generate the theoretical PSF from the supplied matching sequence.
- [x] Feed the extended no-wave coil images through `F_RO → PSF → F_PE1,PE2`.
- [x] Generate full `[1024,256,256,12]` Wave k-space.
- [x] Export direct-IFFT magnitude/phase diagnostics for the first few coils and pause for review.
- [x] Re-apply exact R3×1 sampling mask.
- [x] Save BART-formatted Wave k-space and the identical theoretical PSF.

## Phase E — BART sweep

- [x] Export measured no-wave ACS as BART `kspace_calib` after applying the active 64→12 compression basis.
- [x] Verify exact ACS coordinates/payload, zero exterior, BART dimensions, and finiteness.
- [x] Run the historical `bart ecalib -m 1 -c 0.8` and λ=0 reconstruction used for initial troubleshooting.
- [x] Trace the λ=0 R3 ghost to the GRAPPA-completed no-wave input.
- [x] Keep positive-λ runs paused until a visually acceptable no-wave completion is selected.
- [x] Select joint-coil 5×5×5 GRAPPA as the visually acceptable multi-coil no-wave completion.
- [x] Regenerate synthetic Wave and BART inputs in a new output tree using the accepted 5×5×5 source.
- [x] Generate hard `ecalib -m 1 -c 0.5` maps once and save eigenvalues/hash.
- [ ] Visually gate the crop-0.5 λ=0 result before positive regularization.
- [ ] Reuse the accepted maps and `-e 6.70e7` for every coarse run without rerunning `ecalib`.
- [ ] Run only wavelet `1e-3` and LLR block-8 `2e-3` first; validate outputs
  and stop for explicit user visual confirmation.
- [ ] After approval, run wavelet endpoints `1e-4` and `1e-2`.
- [ ] After approval, run LLR block-8 endpoints `2e-4` and `2e-2`.
- [ ] Record per-run commands, backend, iteration settings, runtimes, hashes, and NIfTI outputs.
- [ ] Defer fine λ and LLR block-size sweeps.

## Phase F — evaluation

- [ ] Select and convert the 256 unfiltered product DICOM images only.
- [ ] Resolve orientation, flips, half-voxel convention, and one shared registration.
- [ ] Apply one documented robust intensity-matching procedure to every output.
- [ ] Compute SSIM, PSNR, NRMSE, MAE, WM variation, CNR, background noise,
  sharpness/detail, aliasing, and anatomy-coverage metrics in fixed masks/ROIs.
- [ ] Save metric tables and provenance in CSV/JSON.
- [ ] Generate side-by-side and difference figures for λ=0, wavelet, LLR, and DICOM.
- [ ] Select the coarse best method/λ using metrics plus visual assessment.
- [ ] Keep fine sweep deferred.

## Emergency Phase G — SENSE recovery (concluded; GRAPPA selected)

- [x] Compress measured image/refscan data to the same 12 virtual coils.
- [x] Calibrate one-map BART ESPIRiT maps from compressed measured ACS only.
- [x] Run the no-wave λ=0 SigPy SENSE reconstruction and export diagnostics.
- [x] Obtain user visual approval that SENSE removes the GRAPPA aliasing.
- [x] Create SENSE-derived full no-wave coil data, but keep downstream Wave synthesis gated on visual approval.
- [x] Investigate the truncated nose/mouth. Strongly suppressed anatomy is
  colocated with the hard `ecalib -c 0.8` support boundary: 96.2% of those
  voxels have ESPIRiT map norm `<0.1`, whereas covered anatomy has map support.
- [ ] Recalibrate a map-support diagnostic while saving BART eigenvalue maps;
  test smooth maps (`ecalib -S`) and one or two less strict crop thresholds
  such as 0.5/0.7. Do not silently use `-c 0`, which may admit unstable maps.
- [x] Generate the first gated candidate with `ecalib -S -c 0.7` and save its
  eigenvalue map. Map-norm support above 0.1 increases from 41.0% to 62.3% of
  the volume and recovers 36.1% of the previous hard-crop background/support.
- [x] Obtain explicit user visual review of the 0.7 candidate. It restores the
  nose/lips but is rejected because it creates a detached bright background shell.
- [x] Diagnose the shell as weak-support amplification, not anatomy or an
  orientation/export error. In BART v1.0, Soft-SENSE at crop `c` remains
  nonzero down to eigenvalue `(2c-1)^2`; `-S -c 0.7` therefore admits values
  down to 0.16. The rejected shell has median eigenvalue 0.269, median map norm
  0.057, and 92.7% of its voxels have map norm `<0.25`. Unregularized SENSE
  amplifies noise/model mismatch in this ill-conditioned secondary support.
- [x] Run the user-selected isolation test with hard `ecalib -c 0.7` and no
  `-S`. Map support above 0.1 is 45.29%; 50 CG iterations converge to a
  final/initial residual ratio of `3.36e-9`. The acquired-data relative
  residual is 0.09907 and the detached Soft-SENSE shell is absent in the
  pipeline quicklook.
- [x] Obtain explicit user visual review of hard crop 0.7. The detached shell
  is removed and the lips are retained, but the nose remains cut off; reject
  this candidate as incomplete full-anatomy support.
- [x] Run the user-approved next candidate with hard `ecalib -c 0.6`, no `-S`.
  Map support above 0.1 is 49.19%; 50 CG iterations converge to a
  final/initial residual ratio of `1.13e-8`, with acquired-data relative
  residual 0.09840. The detached Soft-SENSE shell is absent in the quicklook.
- [x] Obtain explicit user visual review of hard crop 0.6. It removes the
  detached shell and retains more face than 0.7, but the nose remains partially
  cut off; reject it as the final full-anatomy baseline.
- [x] Stop the ESPIRiT threshold search and select joint-coil 5×5×5 GRAPPA for
  downstream synthetic-Wave generation.
- [x] Record the selection rationale: GRAPPA directly supplies completed
  multi-coil k-space, avoids the extra SENSE `F(Sx)` back-projection, preserves
  the full visual anatomy, and appears to have a better effective g-factor.
  The g-factor statement is qualitative pending an explicit measurement.
- [ ] Hold: compare accepted 5×5×5 GRAPPA and corrected-support SENSE image
  quality only after geometry and intensity normalization are finalized.
  Visual assessment remains primary. Report SSIM plus scale-matched NRMSE/PSNR
  against unfiltered DICOM, and separately report anatomy coverage, background
  noise, homogeneous-WM coefficient of variation, CNR, and edge sharpness.
  PSNR alone is insufficient because it is scale-sensitive and conflates
  scanner-processing differences with reconstruction quality.
- [ ] Repeat SENSE-derived Wave synthesis only as a future secondary analysis.
- [ ] Repeat the SENSE-derived BART sweep only if that branch is reactivated.
- [ ] Compare optimal parameters with SENSE only as a future method-dependence study.

This branch was explicitly authorized on 2026-08-19. Pause after the no-wave
NIfTIs and diagnostics for visual approval; authorization does not bypass that
acceptance gate or automatically launch the later positive-λ sweep.

## Phase H — future R3×2 and R3×3 synthetic masks (deferred)

The current experiment remains R3×1. Keep full Wave synthesis independent of the retrospective sampling mask so the validated `[1024,256,256,12]` full Wave k-space can support later acceleration studies without repeating GRAPPA completion or Wave encoding.

Candidate masks are:

```text
R3×2: PE1 acceleration 3, PE2 acceleration 2
R3×3: PE1 acceleration 3, PE2 acceleration 3
```

For each future mask, explicitly define and record both PE1 and PE2 residues/offsets, treatment of the fully sampled ACS region, acquired coordinate count, effective acceleration, and a unique output tag. Generate masks from a dedicated configurable mask builder and validate coordinates directly; do not generalize R3×1 with an unchecked `mask[::R1, ::R2]` expression.

The theoretical PSF and full Wave k-space must remain identical across acceleration comparisons. Only the retrospective mask and downstream BART reconstruction settings should change. Store each acceleration result separately so R3×1, R3×2, and R3×3 cannot overwrite or be confused with one another.

- [ ] Implement and validate an R3×2 synthetic sampling mask.
- [ ] Implement and validate an R3×3 synthetic sampling mask.
- [ ] Compare reconstruction stability and preferred regularization across acceleration factors.
- [ ] Do not begin Phase H unless explicitly requested.

## Phase I — final cleanup

Perform this only after the requested reconstruction and evaluation work is
complete, so cleanup does not disrupt reproducibility while the experiment is
still changing.

Repository organization was advanced early at the user's request on
2026-08-19. This pass retained every useful workflow, created the durable
filename map, renamed stage-labeled source files, audited function comments and
external-code boundaries, and validated all commands/tests. Dataset-output
inventory and deletion remain deferred until reconstruction and evaluation are
complete; no ignored historical artifact was removed in this pass.

- [x] Before moving or removing code, classify every script as canonical,
  reusable support, diagnostic/comparison, historical-but-useful, or truly
  obsolete. Retain scripts that may be useful for future datasets,
  troubleshooting, validation, alternative reconstruction methods, or
  reproducibility even when they are not used by the selected pipeline.
  Inactivity alone is not a reason to delete a script.
- [x] Create a checked-in script-name migration dictionary before renaming.
  Use a simple table such as `docs/script_name_map.md` with old path, new path,
  classification, current role, and rename/retention rationale. Keep it after
  cleanup as the durable lookup index for developers and update it whenever a
  mapped file moves again.
- [x] Inventory tracked files whose names contain pipeline stage labels such as
  `phase_a` or `phase_e`, rename them to descriptive task-based names, and
  update imports, tests, documentation, and command examples. Preserve names
  where “phase” is scientifically meaningful, such as magnitude/phase image
  components. Use `git mv` so individual-file history remains traceable in
  addition to the explicit migration dictionary.
- [x] Do not delete a script unless its behavior is fully redundant or invalid,
  it has no credible future diagnostic/reproducibility value, and its removal
  rationale is recorded in the cleanup inventory. Prefer moving inactive but
  useful workflows into a clearly labeled diagnostic or legacy-support area.
- [x] Add a concise leading docstring or maintenance comment to every function
  in experiment scripts and utilities. Explain purpose and non-obvious MRI,
  array-layout, or external-interface conventions without narrating obvious
  statements.
- [x] Audit code derived from the pinned external repositories. When an
  external repository already exposes the function needed, import and call it
  directly instead of maintaining a regenerated local copy; retain a local
  adapter only when layout, streaming, or interface conversion requires one,
  and document that boundary.
- [ ] Inventory temporary and intermediate outputs, verify which artifacts are
  required for provenance or downstream reconstruction, then remove unneeded
  files from the dataset output directory without deleting source data or the
  final calibrated/reconstructed results.
- [x] Run the complete test suite, verify documented commands and output paths,
  and finish with a clean Git worktree.

---

# 20. Open decisions

## GRAPPA kernel

Original baseline:

```text
5×5×1 over RO×PE1×PE2
```

Final selection: `5×5×5`, visually alias-free with full-anatomy support at
Ncc=12. Use its completed multi-coil k-space for the main synthetic-Wave path.
SENSE is retained only as a deferred secondary comparison.

## GRAPPA Tikhonov regularization

Initial:

```text
0.01
```

Final: `TBD`

## Coil compression

Baseline:

```text
64 → 12
```

Final: `TBD`

## ESPIRiT map count

Next acceptance candidate:

```text
1 map, hard crop 0.5, measured compressed no-wave ACS
```

Deferred alternative:

```text
2 maps / soft-SENSE
```

Final: `TBD after λ=0 Wave visual support gate`

## BART regularizer

Coarse methods:

```text
wavelet
LLR
```

Fine λ/block refinement: `deferred`

Final: `TBD after DICOM-referenced metrics and visual comparison`

---

# 21. Results log

## Dataset

```text
TWIX:
Sequence:
Online DICOM:
Scan date:
Matrix:
Acceleration:
ACS:
Physical coils:
```

## Coil compression

```text
Ncc tested: 12, 16, 24
Energy retained: 84.6410% / 90.1746% / 96.0338% for Ncc=12/16/24
ACS held-out NRMSE: 0.17240017 / 0.14574042 / 0.12384302
Selected Ncc: 12 active baseline by explicit user choice; 24 retained as the aggregate-NRMSE comparison
```

## GRAPPA

```text
Kernel: nominal 5×5×1; 10 acquired source locations per target type
Calibration lambda: 0.01 with pygrappa Frobenius scaling
Calibration runtime: approximately 44 s on the initial uncached pass
Application runtime: 904.54 s including final shared-filesystem writeback
Peak memory:
Validation NRMSE: 0.17240017 at active Ncc=12; 0.12384302 at comparison Ncc=24
Final diagnostic: joint-coil 5×5×5, 50 spatial source locations / 600 joint-coil features per target type
5×5×5 total wall time: 866.35 s; reconstruction portion: 794.45 s
5×5×5 output: [256,256,256,12] complex64, finite
Visual result: user-approved; aliasing seen with 5×5×1 and 5×5×3 is absent
Recommendation: use GRAPPA kernel 5×5×5 or SENSE with verified full-anatomy ESPIRiT support
```

## Synthetic Wave

```text
PSF: theoretical sequence-derived PSF; synthesis/BART logical SHA-256 e888fee32e89edf1d23c08ec0dd36c9f4777b56e2cde8bc1507a4288f58f10d4
Wave dimensions: full [1024,256,256,12]; BART [1024,256,256,12,1], complex64
Sampling-mask verification: TWIX image/refscan MDH union; 25,856 PE coordinates; 101 PE1 lines per partition; zero acquired mismatches; zero nonzero unacquired samples
```

## BART

```text
BART version: v1.0.00-dirty, fugue build
ESPIRiT maps: measured no-wave refscan ACS, active 64→12 basis, `ecalib -m 1 -c 0.8`; [256,256,256,12,1]; 92.54 s wall time
Lambda=0: unregularized CG, `wave -i 300 -t 0.001`; 164.32 s internal solve, 766.55 s BART total, 767.65 s external wall time
Lambda=0 NIfTI: [256,256,256], 1 mm isotropic, TWIX-derived IAL orientation; BART output already on the deoversampled logical readout grid
Wavelet lambda sweep:
LLR sweep:
Best setting:
```

## Emergency Phase G no-wave SENSE

```text
Measured input: [256,256,256,12], image/refscan PE1 union (101 lines), exact-zero missing samples
Coil basis: saved leading columns [0,12); identical basis SHA-256 for imaging and ACS
Acquired input verification: bitwise equal to the preserved acquired samples in the GRAPPA branch
ESPIRiT: BART `ecalib -m 1 -c 0.8`, [256,256,256,12,1], 88.32 s
SENSE: SigPy 0.1.27 CPU, lambda=0, explicit union mask, 50 CG iterations
CG time: 1227.55 s; final/initial normal-equation residual ratio 8.69e-10
Acquired model residual: ||MFSx-y||/||y|| = 0.100176
Model coil k-space: [256,256,256,12] complex64, retained for later Wave synthesis
NIfTI: canonical RAS, [256,256,256], 1 mm isotropic
DICOM orientation audit: whole-volume NCC 0.8985; axial 75/180 NCC 0.9150/0.8975
Remaining geometry difference: approximately -0.5 mm in A and S; resolve during comparison without silent resampling
Total reconstruction/export wall time: 1456.53 s
Map-support finding: `ecalib -c 0.8` produces near-binary map norm; 96.2% of
strongly SENSE-suppressed GRAPPA anatomy has map norm <0.1. The cutoff is
concentrated at the anterior nose/mouth boundary and is not a CG convergence issue.
```

## Quantitative comparison (planned; explicitly held)

```text
Reference: unfiltered product DICOM after resolving the remaining half-voxel geometry convention
Primary structural metric: SSIM within a documented anatomy mask
Scale-dependent metrics: robustly intensity-matched NRMSE and PSNR
Noise metrics: background noise and homogeneous-WM coefficient of variation
Other: CNR, edge sharpness, and full-anatomy coverage/missed-anatomy fraction
Status: do not calculate yet; first fix SENSE map support and freeze registration/normalization
```

## Final conclusion

```text
TBD
```

---

# 22. Completion criteria

The GRAPPA-based experiment is complete when:

- [x] R3×1 no-wave raw data load reproducibly.
- [x] Coil compression is validated.
- [x] Shared 2D GRAPPA weights are trained once from the 256×24×256 product refscan support.
- [x] Full coil-wise no-wave k-space is reconstructed.
- [x] Acquired samples remain unchanged.
- [x] Synthetic Wave encoding is applied to completed coil data.
- [x] Exact R3×1 mask is re-applied after Wave encoding.
- [x] BART reconstructs the synthetic Wave data successfully.
- [ ] A regularization sweep is completed.
- [ ] Metrics and visual comparisons are generated.
- [ ] A preferred regularization method/range is identified.
- [x] The SENSE-derived branch is explicitly activated as Emergency Phase G.

---

# 23. Immediate next action

The 5×5×5 GRAPPA image is visually approved and alias-free. SENSE also removes
the alias, but hard ESPIRiT crops 0.8, 0.7, and 0.6 progressively truncate
facial anatomy, while Soft-SENSE 0.7 restores the face at the cost of a detached
weak-support noise shell. **Stop the SENSE threshold search.** Continue the
main synthetic-Wave experiment from the accepted joint-coil 5×5×5 GRAPPA
multi-coil k-space. Keep GRAPPA-versus-SENSE metrics as a deferred secondary
analysis rather than a prerequisite for the regularization sweep.

Prerequisites now take priority: regenerate the synthetic Wave inputs from the
accepted 5×5×5 output in a new directory, calibrate one hard crop-0.5 CSM set,
and visually approve its λ=0 Wave reconstruction. Reuse that exact map set for
the requested coarse wavelet and LLR pilots. Afterward, perform the fixed-
geometry, fixed-normalization comparison against the unfiltered DICOM and defer
all fine parameter refinement.

```text
measured R3×1 no-wave k-space + measured ACS refscan
        ↓
same saved 64→12 coil compression for both streams
        ↓
joint-coil 5×5×5 GRAPPA trained from the compressed ACS
        ↓
accepted completed no-wave coil k-space [256,256,256,12]
        ↓
centered 3D IFFT to coil images
        ↓
center-embed RO 256 at indices 384:640 of ROext 1024
        ↓
generate theoretical [1024,256,256] PSF from the supplied sequence
        ↓
apply F_RO → theoretical PSF → F_PE1,PE2
        ↓
full Wave k-space [1024,256,256,12]
        ↓
direct-IFFT first few coils to magnitude/phase NIfTIs and pause for review
        ↓
visual approval
        ↓
re-apply exact R3×1 product sampling mask
        ↓
export BART Wave k-space, identical theoretical PSF, and provenance
        ↓
infer ESPIRiT maps from the compressed no-wave ACS
        ↓
hard `ecalib -m 1 -c 0.5`; save eigenvalues and CSM hash once
        ↓
run λ=0 BART Wave; export NIfTI and pause for CSM/full-anatomy review
        ↓
run wavelet 1e-3 and LLR block-8 2e-3 only
        ↓
export matched NIfTIs/quicklooks and pause for visual troubleshooting gate
        ↓
after approval, run wavelet {1e-4,1e-2} and LLR block-8 {2e-4,2e-2}
        ↓
align all outputs once to the unfiltered product DICOM reference
        ↓
apply one intensity-matching rule and fixed masks/ROIs
        ↓
report SSIM, PSNR, NRMSE, noise/CNR, sharpness, aliasing, and anatomy coverage
        ↓
select the coarse best method/λ; fine sweep remains deferred
```

**Phase C passed its held-out ACS, measured-sample preservation, finiteness, fill-count, and central-RSS checks.**
