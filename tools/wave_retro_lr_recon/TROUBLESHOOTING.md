# Wave reconstruction troubleshooting

## Single- and multi-echo GRE checks

GRE support is code- and unit-tested but must not be described as real-data
validated until the user visually confirms the outputs. Start with
`normal/bart_inputs/manifest.json` and verify a positive ordered echo list,
`sequence_echo_count_match=true`, `sequence_echo_times_match=true`, identical
`shared_calibration_id` values, PSF shapes `1000 x 250 x 72`, and shared
selected lambda `0.015`. Echoes still require distinct PSF, measured-k-space,
BART-command, and NIfTI records. For the LIN low-resolution case, require
matrix `250 x 148 x 72`, crop `[51:199]`, and LR CSM shape
`250 x 148 x 72 x coil x 1`.

`normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png` must overlay raw `a/b/c`
scatter samples on the processed curves. The shared coefficient archive stores
processed keys `a/b/c` and raw keys `a_raw/b_raw/c_raw`. Reusing an older GRE
preparation upgrades these fields from the retained projection caches and
rewrites the plot.

For Pulseq 3D scans, `sKSpace.lPartitions` may remain `1` even when MDH PAR
counters cover the full volume. The manifest records this raw tag but resolves
the partition matrix from sequence `Nz` plus exact measured MDH PAR support.
Do not treat the raw tag alone as evidence of a single-partition acquisition.

Every BART output echo directory must contain `wave_command.txt`, while the
native map directory contains `ecalib_command.txt`. The conversion manifest
records both commands and the exact BART restoration inputs. If magnitude
scale or phase appears wrong, check the recorded formula:

```text
amplitude = kspace_norm * sqrt(extended_RO * LIN * PAR)
phase = 1j * (-1)**(LIN//2)
```

The scientific complex arrays are under each NIfTI branch's
`quantitative_complex/` directory. Magnitude NIfTIs are display-normalized
using the echo-1 99th percentile shared across all echoes; phase NIfTIs
are wrapped radians. NIfTIs must report stored axis codes RAS, with the
conversion manifest recording logical roles `(readout, phase, slice)`, array
flips `(false, true, false)`, and no interpolation. Do not troubleshoot
measured GRE by adding the synthetic BET/brain-mask evaluation workflow;
masking is a separate downstream presentation decision.

## MPRAGE checks

## `prepare_mprage_normal.py` is reported as `Killed`

A bare shell `Killed` message normally means that the operating system or a
memory cgroup sent `SIGKILL`; it is not a Python exception. MPRAGE preparation
uses host RAM. The sample script's `-g` option applies to the later BART Wave
commands and does not move TWIX preparation to the GPU.

Integrated refscan arrays can be deceptively large because mapVBVD represents
the sparse projection and ACS records on a dense five-set grid. For example,
complex64 shape `(1024, 72, 72, 5, 52)` occupies about 10.3 GiB. The current
preparation implementation consumes that reference before loading the image,
checks unmeasured image samples in bounded readout blocks, reuses full-grid
image storage for masking, and releases the 52-coil image immediately after
compression. Do not restore whole-volume boolean selections or retain image
and reference payloads together.

After a failure, record available host memory and check the kernel log when
permitted:

```bash
free -h
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
journalctl -k --since "-10 min" 2>/dev/null | grep -Ei 'oom|out of memory|killed process'
```

On the next user-initiated preparation run, `/usr/bin/time -v` can record
`Maximum resident set size`. A CUDA out-of-memory failure instead normally
appears as an explicit CUDA/PyTorch/BART error rather than a bare shell
`Killed` message. The pypulseq file-version warning is also independent of
host-memory termination.

## Inspect the native R3x1 PSF coefficients first

Native R3x1 preparation automatically creates:

```text
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_FULL_RANGE.png
OUTPUT_ROOT/normal/PSF_PLANE_COMPARISON.png
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

The first coefficient figure always uses `[-2*pi, 2*pi]`, making datasets easy
to compare without an outlier controlling the display. The full-range companion
autoscales a/b/c independently and must be inspected when the fixed plot clips
samples. The plane comparison has kx-y and kx-z rows with theoretical,
directly measured, fitted/calibrated, and wrapped-residual columns. White lines
mark the single spatial interval used across all kx samples.

Spatial-region selection runs for every dataset before coefficient processing.
If the full and central fits agree in complex phase, the original full-region
fit is returned unchanged. Only sustained disagreement triggers a search for
the widest stable center-containing inner interval. The y and z choices are
independent, use original full-FOV coordinates, and are recorded in
`processing_diagnostics.spatial_region_selection`. A constant `c` branch jump
may be removed only by recorded integer multiples of `2*pi`; `a` and `b` are
never gauge-shifted, and the pointwise complex PSF is unchanged by this step.

There is one controlled retry for cases where the full-y plane initially
passes those checks but automatic kx selection subsequently detects sustained
coefficient corruption. Preparation reuses the already evaluated central 50%
y fit (`[18, 54)` for a 72-sample calibration) and reruns all kx and a/b/c
validation. This updates both `a` and the sin-projection contribution to `c`;
it is not the c-only smoothing fallback. A retry is never automatically
accepted, never changes z, and never overrides manual y bounds. Its trigger,
initial selection, exact bounds, and outcome are recorded in
`processing_diagnostics.automatic_spatial_fallback`.

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
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED_FULL_RANGE.png
OUTPUT_ROOT/normal/PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED.json
```

The PNG overlays the raw samples and, when the upstream fit reached the
validation gates, the rejected candidate curve. It also shades the automatic
half-open fitting interval. Its title explicitly states that the candidate was
not used for reconstruction. The JSON preserves the numerical validation gates
and exact selected interval. These files live outside `normal/bart_inputs`, so
after choosing reviewed manual bounds the same output root can be rerun safely;
no ready-input manifest exists from the failed attempt.

If automatic spatial selection cannot find a reliable region, preparation
instead retains `PSF_PLANE_AUTOMATIC_REGION_REJECTED.png` and its JSON record.
Review the white interval boundaries and use paired `--psf-fit-y-min/max` or
`--psf-fit-z-min/max` arguments. These bounds index the calibration image plane,
not the oversampled kx readout, and therefore do not replace the existing kx
sine-line bounds.

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
