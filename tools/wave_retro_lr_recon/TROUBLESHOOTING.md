# MPRAGE reconstruction troubleshooting

## Inspect the native R3x1 PSF coefficients first

Native R3x1 preparation automatically creates:

```text
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png
```

This figure plots the processed `a`, `b`, and `c` phase-plane coefficients
actually used to construct the native and retrospective calibrated PSFs. Its
x-axis is the oversampled readout sample index, not a physical k-space unit.
The dotted vertical line marks the readout center. For sine-line processing,
the shaded region is the automatically selected or manually supplied half-open
fit interval `[min, max)`; the title identifies which selection mode was used.
The fitted model is evaluated across the complete readout.

If a reconstruction has unexpected ringing, displacement, structured ghosts,
or a marked failure relative to the FISTA/Wavelet comparison, inspect this
plot before changing reconstruction regularization. Look for:

- isolated spikes or abrupt discontinuities;
- non-finite or visibly unstable edge behavior;
- a sine-line curve dominated by implausible extrapolation outside its shaded
  fit interval; or
- a coefficient whose scale or trend is markedly inconsistent with the other
  scans acquired using the same sequence and calibration protocol.

The plot is a visual diagnostic, not an automatic pass/fail test. Smooth curves
do not prove that the PSF is correct, and unusual curves can reflect a genuine
calibration difference. Sampling validation, the refscan-derived sensitivity
maps, sequence/TWIX matching, and BART command records remain separate possible
causes of reconstruction failure.

## Compare coefficient-processing choices safely

The default remains nine-sample smoothing. If its coefficient PNG looks
unreliable, first try the optional automatic sine-plus-line model:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing sine-line
```

If that PNG is also unsatisfactory, provide both bounds as a manual override:

```bash
tools/wave_retro_lr_recon/scripts/sample_mprage_normal_recon.sh \
    /path/to/measured_wave_mprage.dat \
    /path/to/a_new_output_root \
    /path/to/matching_wave_mprage.seq \
    --psf-coefficient-processing sine-line \
    --psf-fit-kx-min START_INDEX \
    --psf-fit-kx-max END_INDEX
```

Use a new output root when changing processing mode or fit bounds. Existing
inputs are deliberately rejected when their manifest records different PSF
settings, preventing accidental reuse of a PSF from another comparison arm.
Automatic selection and validation fail explicitly; they never fall back to
smooth without the user requesting smooth in a separate run.

When reporting a problem, retain the coefficient PNG, the normal-input
`manifest.json`, and the BART command text files. Do not add TWIX paths,
subject identifiers, or other private machine data to tracked documentation.
