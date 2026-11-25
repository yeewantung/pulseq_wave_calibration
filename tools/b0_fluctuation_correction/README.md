# B0-fluctuation correction for Wave-CAIPI GRE

This tool estimates and removes B0 fluctuations from measured Wave-CAIPI GRE
data. Respiration is the main source. The data themselves are the navigator:
no sequence change, no navigator echoes, no bellows.

It adapts the consistency navigator of Meineke and Nielsen to Wave encoding
(*Data consistency-driven determination of B0-fluctuations in gradient-echo
MRI*, Magn Reson Med 2019;81:3046-3055, doi:10.1002/mrm.27630).

The tool works on a standard GRE tree produced by `tools/wave_retro_lr_recon`,
for example `<recon_gre>/vcc24`. It writes a corrected tree in the same
layout. That tree is reconstructed with the existing BART `wave` commands and
NIfTI converter.

## Method

### Why the method transfers to Wave

A fluctuation that is constant during one readout multiplies the whole
acquired line by one phase. The time scale of breathing is much longer than
the 30 ms TR. This holds for a Wave line too: all 1000 oversampled samples of
one (LIN, PAR) line share the phase. Correcting the data is therefore a
per-line operation.

Only two steps of the method need the encoding model:

- reconstructing an image from the data;
- synthesizing data from that image.

### The Wave model

`wave_operator.py` implements BART's Wave model on the GPU:

    A = M · F_yz · PSF(kx, y, z) · F_x · Pad_x · maps

It checks out against measured data and BART:

- On a measured 3T Wave-GRE scan (R3x1, 24 virtual coils), fitting the
  BART-restored image through this operator gives a complex scale of
  1.006 + 0i.
- A conjugated or flipped PSF instead leaves about 100% residual.
- The GPU CG-SENSE (`recon.py`) reproduces BART's image with correlation
  0.9999, in about 4 s for 40 iterations.

### One navigator cycle

Each cycle has five steps:

1. Run CG-SENSE (warm start) on the currently corrected data.
2. Synthesize every acquired line, `A x`.
3. Compute per-line least-squares statistics over the whole Wave readout.
4. Smooth them with a Gaussian along the acquisition time. The time comes
   from the TWIX MDH timestamps; here LIN is the outer loop and PAR the inner
   loop, with TR 30 ms.
5. Update the per-line correction.

Updates are over-relaxed (default 2) because the least-squares image absorbs
part of the inconsistency in every cycle. Time-constant terms are removed from
each update, because they cannot be identified: a constant phase rotates the
image and a constant gradient adds a phase ramp.

### Methods (`--method`)

| method | per-line model | correction handed to BART |
|---|---|---|
| `linear_field` (default) | `f0(t) + g(t)·(r − r_ref)` inside the Wave forward model, linearized in `g`. Per-line Gauss-Newton over all coils and readout samples. | `exp(−iΦ0)·y − i·Σ B_p·A(r_p·x̂)` |
| `global` | one phase per line, shared by all coils | `exp(−iΦ0)·y` |
| `coil_affine` | per-coil phases constrained to be affine in each coil's signal centre of mass | per-coil line phase |
| `per_coil` | one free phase per coil and line, as in the paper (keep `--relaxation 1`) | per-coil line phase |

In `linear_field`, the gradient term is subtracted using the final image. With
the true parameters, this correction reaches the noise floor.

In `per_coil` with the 24 PCA virtual coils, the per-coil phases capture only
about a quarter of a real S-I gradient. Broad virtual coils are a poor proxy
for a spatial field. Modelling the field inside the encoding avoids that
limitation.

## Validation (`simulate`)

The ground truth is the CG image of a measured 3T Wave-GRE scan, using its
real PSF, its 24 ESPIRiT maps, the R3 sampling, and the TWIX timing. Onto it, the simulation
injects a breathing-like fluctuation:

- 1.5 Hz uniform plus 0.01 Hz/mm S-I gradient;
- a 0.35 Hz fundamental plus a second harmonic;
- noise at the measured residual level (an upper bound).

The gradient part is synthesized exactly.

| | NRMSE complex / magnitude | uniform trace corr / RMS error | S-I gradient corr / slope |
|---|---|---|---|
| noise only | 0.0175 / 0.0124 | | |
| uncorrected | 0.103 / 0.067 | | |
| `linear_field` (z) | 0.025 / 0.0145 | 0.9994 / 0.04 Hz | 0.977 / 0.99 |
| `global` | 0.029 / 0.018 | 0.997 / 0.13 Hz | – |
| `coil_affine` (z) | 0.029 / 0.018 | 0.997 / 0.13 Hz | 0.77 / 0.24 |

## Limitations

- **Fluctuations locked to the PAR sweep cannot be seen.** A fluctuation that
  repeats identically in every PAR sweep (72 × 30 ms = 2.16 s here) is a
  kz-only phase. Such data are consistent with a z-blurred image, so no
  consistency method can detect them. Breathing near 0.46 Hz is partly in
  this blind spot. Slow drifts are invisible as well; the paper reports the
  same high-pass behaviour.
- **Only uniform and linear fields are modelled.** Local, nonlinear
  fluctuations, such as near the sinuses or in the CSF, are only partly
  captured. The per-coil modes give some extra spatial freedom.
- **Bulk motion is not corrected.** Rigid motion is a different
  inconsistency. `b0_fluctuation/motion.py` is an experimental per-segment
  rigid alignment (Fourier-shear rotation, L-BFGS through the Wave
  operator). It recovers a known motion when the reference image is correct.
  With the sequential LIN order used here, though, the least-squares image
  absorbs a block of moved segments, and only the segments at the block
  edges stay inconsistent. On the real geometry it therefore did not recover
  a simulated 1.5 mm / 1° block. `segment_residual_profile` turns this into a
  detector: step-like motion shows as a per-segment residual 2–3× above its
  time neighbours at the block edges. Correcting motion needs a distributed
  phase-encoding order or navigators.
- **`linear_field` is linearized in the gradient.** Use it for gradient
  phases well below 1 rad across the FOV; 1 Hz across the head at 20 ms TE is
  0.13 rad.
- **Multi-echo data share one estimate.** The estimate comes from one echo
  (the last by default) and is scaled by TE to the other echoes.

## Usage

```bash
# Activate a Python environment with torch, numpy, matplotlib and pymapvbvd,
# and put a host-compatible BART build on PATH (see ../wave_retro_lr_recon/SETUP.md).
TOOL=tools/b0_fluctuation_correction/scripts/b0_fluctuation_correction.py

# 1) Estimate on the normal (R3x1) data. This writes the corrected tree
#    (normal + retro/native_r3x3), the estimation arrays, and figures.
python $TOOL estimate \
    --standard-root /path/to/recon_gre/vcc24 \
    --twix /path/to/meas.dat \
    --output-root /path/to/recon_gre_b0fluct \
    --method linear_field --axes z

# 2) Run BART wave (FISTA-r0 and Wavelet 0.015), export NIfTI, and write the
#    standard-vs-corrected QC.
python $TOOL reconstruct \
    --output-root /path/to/recon_gre_b0fluct \
    --twix /path/to/meas.dat --sequence /path/to/matching.seq -g

# Optional: injected-fluctuation validation on the same geometry.
python $TOOL simulate --standard-root /path/to/recon_gre/vcc24 \
    --twix /path/to/meas.dat --output-dir /path/to/validation --methods linear_field,global
```

### Options

- `--retro-strategy self` re-estimates on each retrospective case's own
  lines, which is a realistic test for a prospective R3x3 scan. The default
  `from-normal` masks the corrected R3x1 data.
- `--axes xyz` adds RO and LIN gradients. These are noisier; the S-I gradient
  dominates for breathing.
- `--fwhm-ms`, `--gradient-fwhm-ms`, `--iterations` and `--relaxation` tune
  the estimation.

## Output layout

```text
<output-root>/
  estimation/{manifest.json, <case>_b0_fluctuation_estimate.npz,
              <case>_cg_{uncorrected,corrected}.npy, figures/}
  normal/{bart_inputs, bart_output, nifti}
  retro/<case>/{bart_inputs, bart_output, nifti}
  qc/{compare_<case>_<branch>.png, manifest.json}
  logs/bart.log
```

Run the tests with:

```bash
python -m unittest discover tools/b0_fluctuation_correction/tests
```

The tests use a GPU when one is available.
