# MPRAGE metal-fringe intervention arms

These arms test reconstruction-side interventions for the coherent fringe
near dental metal. Each arm changes one variable relative to the accepted
PCA-12 one-map `-c 0` FISTA-r0 baseline. All arms are judged on the same
fixed-normalization review figures. The primary review is visual: fringe
reduction, preserved anatomy, and no worse peripheral Wave spreading.

The ROI-based `diagnose` stage of the set-4 consistency diagnostics is paused,
because the fringe is superimposed on anatomy. The Stage 3 outputs are kept
and read as inputs; no arm writes to the accepted root or the Stage 3 root.

| Arm | Changed variable | Status |
| --- | --- | --- |
| 1 | FISTA iteration cap 100 → 300, with per-iteration residual logging | approved (2026-10-01) |
| 2a | CSM: map 1 of the Stage 3 two-map calibration alone | planned |
| 2b | CSM: maps 1 and 2 of that calibration, without Soft-SENSE weighting | planned |
| 4 | Virtual coils 12 → 24 (existing PCA control, `--ecalib-crop 0`) | planned |
| 3 | Noise prewhitening before PCA-12 | planned |

Arms run one at a time in the order 1, 2, 4, 3. Each run needs its own
authorization and is followed by a visual review. Free disk space is checked
again before every run.

## Review figures (all arms)

`scripts/mprage_intervention_qc.py` (module `wave_retro_lr.intervention_qc`)
compares one arm with the accepted baseline:

- **Coordinates.** Boxes are inclusive voxel ranges of the RAS-stored NIfTI.
  Axis 0 is R-L, axis 1 is A-P, and axis 2 is S-I, with indices increasing
  toward R, A, and S. They are given explicitly on the command line.
- **Scale.** Each magnitude NIfTI is rescaled on export so that its 99th
  percentile equals 1. The review multiplies every magnitude by its sidecar
  `MagnitudeNormalization.InputPercentileValue` to restore the shared BART
  scale. Shape and affine must equal the baseline.
- **Normalization.** One constant C is the 99.5th percentile of the restored
  baseline in the central box. The image window is [0, C], the
  magnitude-difference window is [-0.2 C, 0.2 C], and the air window is
  [0, 0.1 C]. These windows are identical for every arm, and no per-arm gain
  is fitted.
- **Slices.** Five per orientation at an exactly uniform integer step, centred
  in each range, with the indices recorded.
- **Views.** For the central and metal-context boxes: axial, sagittal, and
  coronal montages, both full-field and cropped, with baseline, candidate, and
  difference rows. Also the air band, and the edge strips for residual Wave
  spreading, which appears on the middle sagittal slices at the anterior face
  edge and the posterior occiput/head edge. The strips are the A-P ranges
  outside the central box (for the pilot, anterior face A-P 152-255 and
  posterior occiput/head A-P 0-63), over the full S-I extent. They are shown on
  the five sagittal views that span the central R-L range (R-L 50, 80, 110,
  140, 170 for the pilot).
- **Statistics.** Descriptive only: medians, 99.5th percentiles, median
  ratios, relative RMS differences, and air mean and standard deviation.

The review writes `qc/figures/*.png` and `qc/qc_manifest.json` and refuses to
overwrite either.

## Arm 1: FISTA-r0 convergence control

`scripts/sample_mprage_fista_convergence.sh` holds the single BART command:

```bash
BART_DEBUG_LEVEL=5 bart wave [-g] -w -f -r 0 -i 300 -t 1e-6 \
  ACCEPTED/normal/bart_output/coil_sens ACCEPTED/normal/bart_inputs/psf \
  ACCEPTED/normal/bart_inputs/wave_kspace OUTPUT_ROOT/normal/bart_output/fista_r0_i300/image_wave
```

Python (`scripts/mprage_fista_convergence.py`, module
`wave_retro_lr.fista_convergence`) validates and records the run. It derives
the command from the accepted FISTA-r0 record, changing only `-i` and the
output image, and requires the shell's command to match it token by token,
including the device flag. Before running, it binds the accepted baseline by
file identity and hashes. It refuses an output root that equals, lies inside,
or contains the accepted root or a `--protected-root`.

**Reuse and failures.** A recorded control is reused only if it still meets
a fixed output contract. The contract comes from the implementation and the
accepted root, never from the manifest's own entries:

- the four required logs, plus both GPU logs exactly when `-g` is used;
- the environment log, the image, and the command record;
- the current implementation hashes and a fresh binding of the accepted
  baseline;
- a command argv and text equal to the command derived again from the
  accepted record;
- the convergence, timing, and GPU summaries, re-parsed from the verified logs;
- a closed set of files in every workflow directory.

So removing a record together with its file, or adding a file, stops reuse.
Without a manifest, any entry under the output root comes from an interrupted
run, even an empty directory or a single log, and the stage stops. The shell
writes with `noclobber`, so it never overwrites a fixed-name log. Before every
run it rechecks free disk space and, with `-g`, free GPU memory. If GPU memory
runs short, it stops instead of switching device.

**What is recorded.** Debug level 5 changes logging only. It prints one
`#It k: r` line per iteration, where r is the relative normal-equation
residual ‖Aᴴy − AᴴA x_k‖ / ‖Aᴴy‖ that BART tests against `-t`, with six
decimals. It also prints the final iteration count. A run to the cap prints
`#It 000` to `#It 299`; a tolerance stop at k ends with `#It k`. With λ = 0
and continuation 1, iterations 0 to 99 retrace the accepted run, up to GPU
floating-point variation.

`manifests/fista_convergence.json` records:

- the accepted bindings, the command, and the image hashes;
- the residual trace and the final count;
- BART's eigenvalue and timing lines;
- `/usr/bin/time -v` (wall clock and peak RSS);
- GPU samples taken once per second (device and BART process memory);
- the BART version and binary hash, and the environment log.

Stages:

- `print-command`
- `reconstruct`
- `convert` (the existing converter, suffix `BARTWaveMPRAGENormalFISTAR0I300`)
- `mprage_fista_convergence.py residual-figure`, which writes
  `qc/figures/residual_trace.png`

```text
OUTPUT_ROOT/
├── normal/bart_output/fista_r0_i300/   image_wave, wave_command.txt
├── normal/nifti/fista_r0_i300/         magnitude and phase NIfTIs
├── manifests/fista_convergence.json
├── logs/                               environment/, bart_version.txt, bart_binary.sha256,
│                                       bart_wave.debug5.log, bart_wave.time.txt, gpu_*.csv, convert log
└── qc/                                 figures/, qc_manifest.json
```

**Expected cost.**

- Runtime: about 2.5-3 minutes on an idle GPU, measured as 0.43 s per
  iteration plus about 26 s of setup. Budget 5-10 minutes on a shared GPU.
- Memory: an estimated 15-25 GiB host and 15-30 GiB GPU. Neither grows with
  the iteration count, and both are measured during the run.
- Disk: about 0.3 GB.
