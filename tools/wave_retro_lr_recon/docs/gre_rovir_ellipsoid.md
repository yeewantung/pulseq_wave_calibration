# Experimental GRE ROVir with sphere/ellipsoid null regions

## Scope

`scripts/gre_rovir_recon.py` is an experimental ROVir path for measured
Wave-GRE. It is separate from the MPRAGE ROVir workflow
([`mprage_rovir_reconstruction.md`](mprage_rovir_reconstruction.md)) and
does not change it. It was written to test a spherical null region over the
face in a pediatric GRE/SWI scan. A face box that spans all partitions also
covers the frontal lobe; a sphere centred anterior-inferior of the brain
does not.

The workflow binds to a completed standard-PCA GRE tree of the same scan
(`<recon_gre>/vccN`). From that tree it reuses:

- geometry;
- the calibrated PSF (copied, not recalibrated);
- the ESPIRiT crop;
- the retrospective sampling lattice;
- the Wavelet lambda (0.015).

It reads the physical coils from TWIX and checks that projecting them with
the standard PCA basis reproduces the standard ACS and Wave k-space.

## Null-region syntax

Indices are native BART array indices. RO is the readout. LIN 0 is
anterior. PAR 0 is inferior. Ellipsoid centres may lie outside the grid, and
radii are in millimetres, so a sphere stays round on 0.88 x 0.88 x 2.5 mm
voxels.

```text
--null-ellipsoid "ro=125,lin=15,par=-16,radius_mm=110"
--null-ellipsoid "ro=125,lin=15,par=-16,ro_radius_mm=100,lin_radius_mm=110,par_radius_mm=90"
--null-box "ro=0:249,lin=0:30,par=0:10"
```

The options are repeatable. The null region is the union of all given
shapes, and the ROVir signal region is its exact complement. The region ID
(`null_<sha12>`) is a hash of the binary union, computed the same way as the
MPRAGE candidate hash. A voxel centre belongs to an ellipsoid when
`sum(((i - centre) * voxel_mm / radius_mm)**2) <= 1`.

## Commands

The `vccN` in the output folder name below is the retained ROVir coil count.
Run the stages in this order:

1. `inspect` exports the physical-coil ACS and k-space. It also writes ACS
   RSS views with index ticks, for placing the sphere.
2. `roi` adds the masks and `bart rovir`. It writes overlays on the ACS and
   on the standard image, plus the energy curve (`roi/<id>/figures/`).
3. `run` adds `ccapply`, `ecalib`, and `bart wave` for the normal case and
   the retrospective cases (FISTA-r0 and Wavelet). It then converts to
   NIfTI and writes the standard-vs-ROVir QC (`rovir_<id>_vccN/qc/`).

```bash
# Activate the tool's Python environment and put a host-compatible BART
# build on PATH (see ../SETUP.md).
python scripts/gre_rovir_recon.py run TWIX.dat OUTPUT_ROOT SEQUENCE.seq \
    --standard-root /path/to/recon_gre/vcc24 \
    --null-ellipsoid "ro=125,lin=15,par=-16,radius_mm=110" \
    --virtual-coils 44 -g
```

Every stage is resumable, and completed stages are reused. Options:

- `--retro-case`: `native_r3x3` by default; `native_r3x2` is also
  supported.
- `--no-retro`: skip the retrospective cases.
- `--branch`: limit the reconstruction to `fista_r0` or `wavelet`.
- `--ecalib-crop`: override the crop inherited from the standard tree.

## Output layout

```text
OUTPUT_ROOT/
  physical_inputs/{manifest.json, bart/, figures/}
  roi/<region_id>/{manifest.json, masks/, inputs/, transform/, figures/}
  rovir_<region_id>_vcc<N>/
    manifest.json
    normal/{bart_inputs, bart_output, nifti}
    retro/<case>/{bart_inputs, bart_output, nifti}
    qc/{compare_*.png, comparison.json}
  logs/
```

## What to expect

ROVir maximizes the ratio of signal-region energy to null-region energy. The
face and the frontal brain are seen by the same anterior coils, so nulling
the face removes frontal-brain coil sensitivity too. The ESPIRiT-combined
image then has a smooth shading. In the QC figures, the "ROVir/standard
shading" row approximates the R = 1 SNR ratio. The retained coil count sets
the trade-off. Check `roi/<id>/figures/rovir_energy_curve.png` and the
shading row before you pick a count.

Measured on one pediatric scan (52 physical coils, sphere centre
`ro=125,lin=15,par=-16`, retrospective R3x3 FISTA-r0; R3x1 Wavelet was within
0.03). Values are relative to standard PCA with 24 virtual coils. The SNR
proxy is the median shading inside a brain mask, and the face value is the
face-to-brain intensity ratio relative to the standard image, inside the
110 mm sphere:

| Sphere radius | ROVir coils | Brain SNR proxy | Frontal SNR proxy | Face remaining |
|---|---|---|---|---|
| 110 mm | 24 | 0.52 | 0.33 | 0.02 |
| 110 mm | 44 | 0.77 | 0.66 | 0.11 |
| 110 mm | 48 | 0.87 | 0.77 | 0.26 |
| 55 mm | 24 | 0.72 | 0.60 | 0.50 |
| 55 mm | 48 | 0.91 | 0.90 | 0.75 |

The sphere kept a margin of at least 10 mm from the brain, so the SNR loss is
not caused by nulling brain voxels. Halving the radius does not halve the
suppression, because ROVir optimizes coil combinations. At matched brain SNR
the larger sphere still suppresses more face signal (110 mm with 44 coils
against 55 mm with 24 coils).
