# MPRAGE reconstruction troubleshooting

## Inspect the native R3x1 PSF coefficients first

Native R3x1 preparation automatically creates:

```text
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png
```

This figure overlays the original `a`, `b`, and `c` coefficient samples as
scatter points with the processed curves actually used to construct the native
and retrospective calibrated PSFs. Its x-axis is the oversampled readout sample
index, not a physical k-space unit. The dotted vertical line marks the readout
center. For sine-line processing, the shaded region is the automatically
selected or manually supplied half-open fit interval `[min, max)`; the title
identifies which selection mode was used. The fitted model is evaluated across
the complete readout, while the scatter shows its agreement with the measured
coefficient observations.

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

Automatic sine-line processing keeps strict validation for `a` and `b`. When
only `c` fails, their fitted frequencies must agree within 2% and their mean
must agree with the sequence-trajectory frequency within 3%. The code then fits
`c` at that fixed common frequency with independent amplitude, phase, slope,
and intercept. Its relaxed safety gates require a finite well-conditioned fit,
raw RMSE no greater than 50% of the fitted data range, and endpoint-trim
full-readout relative L2 difference no greater than 2. If those gates fail,
the accepted result is the explicit hybrid `a/b=sine-line, c=9-point smooth`.
The manifest records the original rejection, common frequency, thresholds,
constrained candidate, outcome, and smooth window. For a smooth fallback, the
PNG draws the rejected constrained `c` as a dashed comparison when a finite
candidate is available and labels the accepted smooth curve; the fallback is
never silent.

If automatic range selection fails, either `a` or `b` fails, their frequencies
are inconsistent, or no finite `c` fallback can be produced, preparation stops
and retains two failure-specific files:

```text
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED.png
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED.json
```

The PNG overlays the raw samples and, when the upstream fit reached the
validation gates, the rejected candidate curve. It also shades the automatic
half-open fitting interval. Its title explicitly states that the candidate was
not used for reconstruction. The JSON preserves the numerical validation gates
and exact selected interval. These files live outside `normal/bart_inputs`, so
after choosing reviewed manual bounds the same output root can be rerun safely;
no ready-input manifest exists from the failed attempt.

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

Use a new output root when changing processing mode or fit bounds after BART
inputs were successfully prepared. Existing inputs are deliberately rejected
when their manifest records different PSF settings, preventing accidental reuse
of a PSF from another comparison arm. The exception is an automatic fit that
stopped before creating a ready-input manifest: its rejected-fit PNG/JSON do not
block a manual rerun in the same output root. Automatic selection and validation
fail explicitly except for the documented c-only recovery policy. That policy
never changes `a/b` to smooth and records any accepted smooth `c` fallback.

When reporting a problem, retain the coefficient PNG, the normal-input
`manifest.json`, and the BART command text files. Do not add TWIX paths,
subject identifiers, or other private machine data to tracked documentation.
