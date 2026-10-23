# Uncropped CSM cache and crop diagnostics

## Scope of this first phase

`scripts/diagnose_csm_crop.py` is an opt-in, **diagnostic-only** workflow for
prepared native MPRAGE standard-PCA, MPRAGE ROVir, or GRE standard-PCA ACS.
It calibrates one ESPIRiT map set once, saving both uncropped maps and
eigenvalues, then derives crop variants without repeating ecalib.

The cache diagnostic **does not change the normal/retro launchers**. Their
output layout and hard-0.6 default are unchanged. A separate generic LR
support-transfer correction is described below; it applies to newly prepared
LR maps, independently of this diagnostic. The diagnostic does not run Wave,
replace production CSMs, apply N4, alter
NIfTI collection, or select a winning crop. Routing alternate CSMs into
parallel reconstructions is a separate, not-yet-implemented phase.

This is independent of the MPRAGE-only metal-CSM extrapolation feature.
The cache supports ROVir without treating its basis as standard PCA; it
does not restore physical-coil directions already removed by that transform.

## Running the diagnostic

Initialize the normal Python environment and the compatible host BART build.
Use a dedicated, explicitly approved directory underneath `diagnostics/`:

```bash
python tools/wave_retro_lr_recon/scripts/diagnose_csm_crop.py \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --output "$CONFIRMED_DIAGNOSTICS_DIR" \
  --hard-crops 0.4 0.5 0.6 \
  --soft-crops 0.5 0.6 \
  --check-only
```

`--check-only` validates the prepared ACS/basis, reports the additional payload
storage estimate, and creates no files or directories. It reads and hashes the
ACS, so it can still take time on network storage. Remove `--check-only` to
perform calibration and generate the diagnostic variants. No GPU is used.

The recorded actual calibration command is:

```bash
bart ecalib -m 1 -c 0 INPUT_ACS CALIBRATION/coil_sens_uncropped CALIBRATION/eigenvalues
```

No fictional `ecalib -c 0.6` command is recorded for a postprocessed result.
All ecalib settings other than map count/crop remain BART defaults and the
executable SHA-256 is bound to the cache. Changing those calibration settings
requires a separate future interface and a new cache, not a crop-only reuse.

By default only hard 0.6 is requested. The candidates in the example are a
comparison set, not endorsed optimal values. Add `--slices RO LIN PAR` for
three zero-based **logical** slice indices; the default uses grid centers.
Select slices through the region of interest when reviewing a mouth defect.
These are not NIfTI coordinates or anatomically oriented views.

## ACS magnitude overlays (completed cache only)

To compare crop support against anatomy, add an overlay figure without running
ecalib or reconstruction again:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  python tools/wave_retro_lr_recon/scripts/plot_csm_acs_overlay.py \
  --diagnostics "$CONFIRMED_DIAGNOSTICS_DIR"
```

The background is the coil-wise centered 3D inverse FFT of the **same-basis
ACS**, combined by RSS without CSM masking. For ROVir this is projected ACS,
not the original physical-coil signal. Each view includes a plain ACS panel.
Hard crops have cyan support contours; soft crops have a translucent weight
layer and cyan contours at 0.1/0.5/0.9. Dashed yellow marks hard-0.6 support.
All panels share one whole-volume percentile window (default 99.5) and display
gamma (default 0.5); these do not change the underlying data. Logical axes
remain explicit: no RAS orientation or registration is inferred. Use
`--slices RO LIN PAR` for matching slices through the region of interest.

New files are `figures/acs_csm_overlay_<hash>.png` and the matching
`metadata/acs_csm_overlay_<hash>.json`. Existing calibration, weights, figures
and reconstructed images are preserved; no ACS image volume or full CSM copy
is saved. Provenance binds the source manifest, basis, ACS, calibration,
variant hashes, plotting implementation and display settings. Exact resumes
validate the PNG hash; changing display settings adds another figure. Source
or weight mismatches fail rather than silently plotting unrelated data.
The single-worker CPU FFT processes one coil at a time. An ACS overlay helps
locate support exclusions, but is not evidence of recovered tissue or SNR.

## Files and disk use

```text
CONFIRMED_DIAGNOSTICS_DIR/
  calibration/     coil_sens_uncropped.{hdr,cfl}, eigenvalues.{hdr,cfl}
  csm_variants/    hard_0p6/, soft_bart_1_0_0p6/, ...
  figures/        hash-named support comparison PNGs
  metadata/       calibration.json, hash-bound figure records
  logs/           ecalib.log
  scripts/        reserved for local execution notes/scripts
```

Each variant stores `weights.{hdr,cfl}` and `variant.json`. These scalar weights
plus the immutable uncropped CSM define the full candidate exactly:
`candidate[x,coil] = uncropped[x,coil] * weights[x]`.

Add `--materialize` to also write `coil_sens.{hdr,cfl}` and
`materialized.json` in **each requested variant**. It can be added on a later
invocation without recalibration. No full CSM copy is made by default. For a
256 × 256 × 192 grid with 24 virtual coils, one CSM is 2.25 GiB and each
complex64 scalar map is 0.09375 GiB. Five candidates use about 2.81 GiB total
with weights only, or 14.06 GiB with all five full CSMs materialized, excluding
small headers/figures/logs and BART scratch. The runner checks additional
payload capacity plus a 1-GiB reserve; this is not a filesystem reservation.

## Scientific semantics

- **Hard:** `abs(eigenvalue) >= crop`, inclusive, evaluated in float32.
- **Soft:** `soft_bart_1_0` implements the reviewed BART 1.0
  `src/calib/calib.c` `md_crop_weight_fun` / `md_scurve` literally:
  `t = (sqrt(abs(eigenvalue)) - crop) / (1-crop)`, then
  `w = (t >= 1) + (t >= -1) * (t/(1+t*t) + 0.5)`.
  This particular implementation has a discontinuity at `t=1`; weights can
  exceed one. We do not silently substitute a clipped smoothstep. Metadata
  records the maximum weight and number of voxels above one; plot intensities
  alone cannot reveal all such overshoot because the display range is [0,1].
  Revalidate the synthetic parity test when changing BART versions.
- Only **one** ESPIRiT map set is supported. BART `-S` single-map soft cropping
  is **not multi-map Soft-SENSE**. Both hard and soft preserve complex phase
  and relative coil scaling; neither runs coil-RSS renormalization.
- Figures show eigenvalue magnitude and weighted coil RSS, not reconstructed
  tissue intensity. A larger support does not prove recoverable signal, a
  higher SNR, or a better forward model. No winner is selected automatically.
- Legacy low-resolution processing Fourier-resampled CSMs and normalized
  every nonzero Fourier tail to unit coil RSS. That could expand support and
  remove soft weighting. New LR preparations use the independent native
  support/gain transfer below. The diagnostic still does not automatically
  substitute its candidate CSMs into a production run.
- GRE gets one shared coil basis and one shared map set for all echoes; this
  diagnostic does not process echo images or claim validation of phase,
  inter-echo scaling, or delta-B0 under a future changed reconstruction.

## Retrospective CSM support and gain transfer

New standard MPRAGE, standard GRE, canonical ROVir, and metal LR maps share
`same_fov_native_support_and_rss_v1`. This is a general CSM correction, not
metal detection and not a new CLI option. Native R3x1/R3x2/R3x3 maps continue
to be reused directly. Identical-grid calls to the resampler copy exactly.

For a reduced grid the implementation separates three quantities:

1. **Support:** nonzero native coil RSS is mapped by nearest native voxel
   center. There is no dilation, convex hull, threshold re-estimation or
   periodic wrapping. A Fourier tail outside this mapped support is set to zero.
2. **Coil direction:** native vectors are divided by their RSS, then Fourier
   resampled along LIN/PAR only. The resulting coil vector is normalized inside
   the mapped support. An almost-zero vector (RSS <= 1e-8) falls back to the
   nearest supported native unit vector; metadata records the fallback count.
3. **Scalar gain:** native RSS is bilinearly interpolated over supported PE
   neighbors, dividing by the interpolated binary support to avoid dilution by
   background zeros. It is applied **after** direction normalization. Hard unit
   support remains unit; soft weights and other native scalar gains are retained
   rather than forced to one. No clipping is imposed on gains above one.

The centered same-FOV mapping is
`native_index = (target_index - floor(target_N/2))*native_N/target_N + floor(native_N/2)`.
Nearest-neighbor half ties select the higher native index; edge interpolation
clamps rather than wraps. RO is unchanged. Odd/even dimensions and singleton
axes are covered by tests. Nearest-center support is a discrete representation
of the same native domain, **not** an exactly identical physical boundary at
every subvoxel scale or a volume-preserving mask average. A coarse voxel may
not resolve a thin hole. No rigid registration or anatomical rotation is added.

GRE continues to share one map across all echoes. This prevents echo-specific
map choices; it does not substitute for real-data phase/delta-B0 validation.
Existing lambda choices are not newly validated or automatically reselected
under this corrected LR operator. The synthetic GRE sweep also imports this
shared resampler for newly generated LR maps; its completed historical sweep
CSM cache remains a historical product, not proof of the new policy. No sweep
is launched or rewritten by this change.
The method preserves the support of the **selected normal CSM**: if a mouth
region is already zero in that normal CSM, it stays excluded. Restoring such
signal requires a separately reviewed change to the normal CSM (for example
crop diagnostics); LR resampling must not invent it through ringing.

Each generated LR CSM has `coil_sens_resampling.json`, binding the native CSM,
output CFL pair, target geometry, implementation and policy. Standard/ROVir
case manifests include this record; metal's existing map manifest already
binds the native candidate and the resampling implementation hash, with the
detailed transfer record beside its CFL pair. Exact resumes validate hashes
instead of accepting only a matching array shape. Source, basis, PSF, geometry
and reconstruction-state validation elsewhere are not relaxed.

**Existing LR maps are not overwritten or relabeled.** An unversioned old LR
map or a metal LR request bound to the old resampler is rejected on new
preparation, with existing artifacts left intact. Use a separately confirmed
fresh reconstruction root to evaluate the corrected path. Native images,
completed historical collections and existing NIfTIs are not edited by this
code change. Do not remove hashes/sidecars to force reuse. MPRAGE's whole-file
PSF implementation binding also covers its preparation adapter, so exact
normal-input reuse can report a code mismatch after this adapter change;
follow the existing documented hash-bound legacy-input reuse procedure rather
than weakening that check or relabeling old calibration as newly computed.

## Provenance, reuse and resource safety

The cache checks prepared native geometry/FOV, physical/virtual coil bounds,
ACS dimensions/byte count/SHA-256/finite values, basis file and logical PCA
hashes, and the exact source-manifest identity. ROVir projection basis and
the full recorded ROVir provenance are retained independently of standard PCA.
The scope is **prepared-ACS calibration**, not a new validation of raw TWIX,
PSF, trajectory, or overall production readiness. Existing validators for
those stages are not bypassed or changed.

Reuse is strict: changed source, basis, ACS, BART binary, implementation,
thread count, or corrupt output hashes are rejected. Completed calibration
and completed variants are reusable; additional crops are additive. Exact
completion records are written last. An interrupted calibration or partially
written variant is preserved and rejected, not automatically repaired or
overwritten. Review such a directory and explicitly approve a fresh location;
the runner does not delete partial files. One filesystem lock excludes
concurrent writers.

The CPU child defaults to one OpenMP thread, single-threaded BLAS, `nice 19`,
`SCHED_IDLE`, disabled CUDA visibility, and a 96-GiB address-space limit. It
requires 112 GiB available RAM to start and stops **only its own child** if
available RAM falls below 64 GiB or its RSS exceeds 64 GiB. These conservative
limits can refuse a busy host; they are not a guarantee against every OOM
condition. It does not wait indefinitely, kill another job, or launch GPU work.

## Validation

Pure tests cover inclusive thresholds, soft endpoints, phase preservation,
physical-coil bounds, basis/geometry/source mismatch, ROVir isolation,
shared-echo GRE provenance, additive variants, read-only checks, completion
reuse, partial-output rejection and concurrency. The optional tiny CPU test
compares derived hard/soft maps with separate direct BART calibrations on a
16-cubed synthetic phantom, never on a subject:

```bash
CSM_TEST_BART=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tools/wave_retro_lr_recon/tests -p test_csm_cache.py -v
```

Subject-specific launchers and real paths belong only in ignored `.local.sh`
or `.local.json` files, never in this public document or committed fixtures.
