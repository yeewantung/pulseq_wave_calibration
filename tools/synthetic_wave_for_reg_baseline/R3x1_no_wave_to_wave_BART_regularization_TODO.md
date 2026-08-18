# R3×1 No-Wave → Synthetic Wave → BART Regularization Tuning
## Implementation To-Do and Progress Tracker

**Purpose:** Build a reproducible offline experiment for tuning BART Wave reconstruction regularization using an acquired **R3×1 no-wave** dataset and its **online scanner DICOM** as the practical reference.

**Primary strategy:** Reconstruct the missing no-wave k-space with **2D GRAPPA**, preserving coil-wise k-space, then apply the existing Wave forward model and retrospectively re-apply the same R3×1 sampling mask.

**Deferred secondary strategy:** Repeat the experiment with **SENSE/ESPIRiT** to measure how much the synthetic Wave result depends on the no-wave completion method. Do not implement or run this branch unless explicitly requested.

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

## 1.2 GRAPPA first; SENSE second

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
- [ ] Implement SENSE/ESPIRiT branch second. (Deferred unless otherwise stated)
- [ ] Compare the two synthetic Wave datasets and their preferred BART regularization. (Deferred unless otherwise stated)

## 1.3 Use 2D GRAPPA on the 3D acquisition

Acquisition dimensions:

```text
RO  = kx
PE1 = ky   ← accelerated, R = 3
PE2 = kz   ← fully sampled, R = 1
coil
```

Baseline GRAPPA kernel:

```text
5 × 5 × 1
RO × PE1 × PE2
```

Use no PE2 neighbor samples in the baseline.

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
- [x] True 3D GRAPPA is reserved for a later optional comparison only.

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
sources/published_code/wave-mprage/recon
sources/published_code/wave-gre-flow-comp/recon
```

The current copies share the same `utils/twix_import.py` and `bart/run_wave_recon.sh`; several other utilities are identical or closely related. For this MPRAGE dataset, start from the Wave-MPRAGE path and consult the GRE path for newer/generalized behavior where useful.

- [x] Implement explicit local R=3 GRAPPA calibration and application routines.
- [x] Record the relevant `pygrappa` behavior or source version used as a reference.
- [ ] Reuse/adapt the existing BART CFL export, ESPIRiT calibration, Wave reconstruction wrapper, and NIfTI conversion instead of reimplementing them.
- [ ] Infer ESPIRiT maps from the fully sampled no-wave ACS after applying the same coil-compression matrix used for the synthetic Wave data.
- [ ] Decide before implementation whether these sibling repositories should be added as git submodules or whether the minimal stable utilities should be incorporated locally with provenance. Do not attempt a broader utility unification in this experiment.

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

## 4.1 Do not use generic whole-volume 3D mdgrappa as the baseline

Avoid starting with:

```python
mdgrappa(full_256x256x256_volume, ...)
```

Reasons:

- no PE2 acceleration,
- unnecessarily large temporary arrays,
- generic missing-point enumeration,
- poor scaling for 256³ data.

- [x] Baseline does not use a 5×5×5 kernel.
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

- [ ] ESPIRiT maps generated in the same virtual-coil basis.
- [ ] Reuse/adapt `recon/bart/bart_utils/bart_io.py` and `recon/bart/run_wave_recon.sh` from the existing Wave repositories.
- [ ] Build the BART ESPIRiT calibration CFL from the fully sampled no-wave ACS in that same virtual-coil basis.
- [ ] BART dimensions verified.
- [ ] Unregularized Wave reconstruction completed.
- [ ] Wavelet sweep completed.
- [ ] Optional LLR sweep completed.
- [ ] Optional multi-map ESPIRiT comparison completed.

---

# 9. Initial BART regularization sweep

Run the sweep through a small driver script rather than invoking every reconstruction manually. The driver should accept a configurable list of regularizers/weights, create deterministic output names, record the exact BART command and runtime, stop clearly on failures, and support resuming without overwriting completed outputs unless requested.

Wavelet baseline:

```bash
bart wave \
    -g \
    -w \
    -f \
    -i 50 \
    -r <lambda> \
    coil_sens \
    wave_psf \
    wave_kspace \
    recon
```

Initial sweep:

```text
0
1e-5
1e-4
5e-4
1e-3
5e-3
```

- [ ] λ = 0
- [ ] λ = 1e-5
- [ ] λ = 1e-4
- [ ] λ = 5e-4
- [ ] λ = 1e-3
- [ ] λ = 5e-3
- [ ] Scripted sweep driver implemented.
- [ ] Per-run commands, status, runtime, and output paths recorded in machine-readable metadata.
- [ ] Best coarse interval identified.
- [ ] Fine sweep completed.

**Current best λ:** `TBD`

---

# 10. Reference images and experiment interpretation

Primary practical reference:

```text
online R3×1 no-wave scanner DICOM
```

Also retain:

```text
offline GRAPPA-completed no-wave reconstruction
offline SENSE-completed no-wave reconstruction (later)
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
- [ ] Compare GRAPPA- and SENSE-derived results to estimate method dependence.

---

# 11. Image registration and intensity normalization

Before comparing online DICOM and offline reconstructions:

- [ ] Match orientation.
- [ ] Match voxel size/matrix/cropping.
- [ ] Register if needed.
- [ ] Select robust intensity normalization.
- [ ] Apply the same normalization strategy to every λ.

Possible normalization choices:

- WM/brain ROI scale,
- robust linear scale fit,
- percentile-based scaling.

Do not compare raw scanner and BART voxel values without scaling.

---

# 12. Evaluation metrics

Do not optimize λ using naïve SNR alone: stronger regularization can decrease apparent noise while increasing bias and smoothing.

## Quantitative

- [ ] NRMSE against aligned reference.
- [ ] SSIM.
- [ ] Mean/SD in homogeneous WM ROI.
- [ ] Edge sharpness.
- [ ] Residual aliasing/error ROI metric.
- [ ] Optional detail/gradient metric.

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

---

# 13. SENSE/ESPIRiT comparison branch (deferred unless otherwise stated)

This entire section is out of the current implementation scope. Do not implement or run it unless explicitly requested after the GRAPPA-derived experiment is complete.

After GRAPPA is validated:

```text
R3×1 no-wave raw
        ↓
same coil compression
        ↓
ESPIRiT maps
        ↓
SENSE / PICS / CG reconstruction
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

- [ ] SENSE reconstruction implemented.
- [ ] SENSE preprocessing kept minimally regularized to avoid pre-smoothing synthetic truth.
- [ ] Coil-wise data regenerated.
- [ ] Synthetic Wave data generated.
- [ ] Same BART λ sweep repeated.
- [ ] Preferred λ compared against GRAPPA branch.

---

# 14. GRAPPA vs SENSE comparison (deferred unless otherwise stated)

This comparison is also out of the current scope because it depends on the deferred SENSE-derived branch.

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
scripts/phase_a_inspect.py
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
scripts/phase_b_coil_compression.py
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

## Phase D — synthetic Wave

- [x] Center-embed the Ncc=12 coil images from 256 to 1024 readout voxels.
- [x] Generate the theoretical PSF from the supplied matching sequence.
- [x] Feed the extended no-wave coil images through `F_RO → PSF → F_PE1,PE2`.
- [x] Generate full `[1024,256,256,12]` Wave k-space.
- [x] Export direct-IFFT magnitude/phase diagnostics for the first few coils and pause for review.
- [x] Re-apply exact R3×1 sampling mask.
- [x] Save BART-formatted Wave k-space and the identical theoretical PSF.

## Phase E — BART sweep

- [ ] Generate/verify ESPIRiT maps.
- [ ] Run λ=0.
- [ ] Run coarse wavelet λ sweep.
- [ ] Identify useful interval.
- [ ] Run fine sweep.
- [ ] Optional LLR comparison.

## Phase F — evaluation

- [ ] Align online DICOM/offline outputs.
- [ ] Normalize intensities.
- [ ] Compute metrics.
- [ ] Generate side-by-side figures.
- [ ] Select preliminary best regularization.

## Phase G — SENSE comparison (deferred unless otherwise stated)

- [ ] Create SENSE-derived full no-wave coil data.
- [ ] Repeat Wave synthesis.
- [ ] Repeat BART sweep.
- [ ] Compare optimal parameters with GRAPPA branch.

Do not begin Phase G without an explicit request.

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

---

# 20. Open decisions

## GRAPPA kernel

Baseline:

```text
5×5 over RO×PE1
```

Final: `TBD`

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

Baseline:

```text
1 map
```

Optional:

```text
2 maps / soft-SENSE
```

Final: `TBD`

## BART regularizer

Primary:

```text
wavelet
```

Secondary:

```text
LLR
```

Final: `TBD`

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
```

## Synthetic Wave

```text
PSF: theoretical sequence-derived PSF; synthesis/BART logical SHA-256 e888fee32e89edf1d23c08ec0dd36c9f4777b56e2cde8bc1507a4288f58f10d4
Wave dimensions: full [1024,256,256,12]; BART [1024,256,256,12,1], complex64
Sampling-mask verification: TWIX image/refscan MDH union; 25,856 PE coordinates; 101 PE1 lines per partition; zero acquired mismatches; zero nonzero unacquired samples
```

## BART

```text
BART version:
ESPIRiT maps:
Wavelet lambda sweep:
LLR sweep:
Best setting:
```

## Quantitative comparison

```text
NRMSE:
SSIM:
WM noise:
Sharpness:
Other:
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
- [ ] BART reconstructs the synthetic Wave data successfully.
- [ ] A regularization sweep is completed.
- [ ] Metrics and visual comparisons are generated.
- [ ] A preferred regularization method/range is identified.
- [ ] The current run records the SENSE-derived branch as deferred unless it was explicitly requested; GRAPPA-based completion does not depend on Phase G.

---

# 23. Immediate next action

Synthetic Wave generation, visual review, exact R3×1 mask application, and BART Wave/PSF export are complete. **Begin Phase E next** by estimating ESPIRiT maps from the compressed no-wave ACS, then run the unregularized reconstruction and regularization sweep:

```text
GRAPPA-completed no-wave k-space [256,256,256,12]
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
run unregularized BART Wave reconstruction and regularization sweep
```

**Phase C passed its held-out ACS, measured-sample preservation, finiteness, fill-count, and central-RSS checks.**
