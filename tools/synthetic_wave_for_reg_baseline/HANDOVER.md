# Synthetic-Wave regularization handover

Updated: 2026-08-20, America/New_York

Read `EXPERIMENT_PLAN.md` first. It is the active scientific and implementation
plan. The old R3x1 tracker is historical only.

## Immediate next action

Resume the new isolated Phase-A run in tmux with:

```bash
bash tools/synthetic_wave_for_reg_baseline/scripts/run_phase_a_remaining.sh \
  --stage all-before-review
```

This completes the focused GPU reconstructions, converts the normalized
unfiltered DICOM, and prepares the fixed BET-mask and L/R review figures. It
then stops for visual review. Only after inspecting both printed figure paths,
run the explicitly confirmed `all-after-review` command printed by the script.
All new results remain separate from the accepted historical sweep.

After the presentation stage, use the 2021-05-10 R1 data as the best available
baseline. Validate and freeze one partial-Fourier readout-completion method,
then perform the more rigorous R1 regularization refinement and pick a better,
more transferable parameter set. Finally apply that set back to R3 without
retuning as a cross-dataset transfer check.

## Environment on macha

```bash
source /path/to/user_workspace/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source /path/to/user_workspace/bart/bart_startup.sh
```

FSL BET:

```bash
export FSLDIR=/path/to/software/packages/fsl/6.0.6
. "${FSLDIR}/etc/fslconf/fsl.sh"
```

Always use GPU (`-g`) for every BART reconstruction. Do not silently fall back
to CPU; stop and document any command-specific incompatibility.

## Phase-A continuation state

The isolated output tree is:

```text
/path/to/data/20260817_product/
  synthetic_wave_grappa_5x5x5_ncc12_r3x2_phase_a
```

It references the accepted R3x2 BART inputs and lambda-zero reconstruction by
symlink; it does not overwrite the historical sweep. The corrected block-8 LLR
pilot at lambda `2e-4` is complete there and differs from accepted lambda zero
by relative L2 `0.0061681`, confirming a non-degenerate penalty.

The first `wave -v -g` attempt exposed a GPU-unsafe direct pointer access in
BART's complex-decomposition forward operator. BART commit `60dceb33` replaces
that loop with device-aware multidimensional primitives. The production
lambda-zero FISTA equivalence gate then passed: recombined split-complex versus
native-complex relative L2 was `1.36985e-6`, below the frozen `1e-5` limit.
The failed pre-fix CG attempt is retained under the Phase-A `diagnostics/`
directory.

`run_phase_a_remaining.sh` is resumable and always passes `-g`. It runs three
Wavelet values (`1e-6`, `1e-5`, `1e-4`) and the focused corrected block-8 LLR
range (`2e-5`, `5e-5`, `1e-4`, `2e-4`, `5e-4`). The accepted pilot is reused.
The QC and evaluation stages require explicit mask and L/R approval; there is
no automatic approval path.

## Repository state

Repository:

```text
/path/to/user_workspace/sources/published_code/pulseq_wave_calibration
```

The accepted R3x2 evaluation is commit:

```text
28f3f12 Add R3x2 regularization sweep evaluation
```

This cleanup/plan update may be uncommitted. Check `git status --short` before
working. The historical tracker was moved to `docs/archive/`, and the active
plan is `EXPERIMENT_PLAN.md`.

## Dataset roles and baseline status

Original R1 candidates:

```text
/path/to/data/2021_05_10_bay4_mprage_R1_subjects2and3
```

- `meas_MID00792_FID57743_tfl_mprage_R1_Qiyuan.dat` / DICOM `0013`.
- `meas_MID00809_FID57760_tfl_mprage_R1_Qiyuan.dat` / DICOM `0019`.
- Both are 32-coil R1 scans with a full 256 x 192 PE grid but partial readout.
- Both DICOM series contain 192 normalized unfiltered `ND,NORM` images at
  1 mm isotropic resolution.

Additional R1 candidates inspected on 2026-08-20:

```text
/path/to/data/2021_05_14_bay4_subject6/
  meas_MID00786_FID58692_tfl_mprage_R1_Qiyuan.dat

/path/to/data/2021_05_14_bay4_subject7/
  meas_MID00805_FID58714_tfl_mprage_R1_Qiyuan.dat
```

Both are also 32-coil R1 scans with complete 256 x 192 PE coverage, but both
have the identical 404-sample, center-148 partial readout. They do not solve
the baseline problem. Subject 6 contains prior `.nii`/`.mat` reconstructions;
no DICOM files were found in either added folder during this inspection.

No better R1 source is currently available, so the partial-readout limitation
is accepted for the later refinement stage. The 2021-05-10 scans are the
primary R1 dataset; freeze a completion method and designate independent
development and confirmation scans before the R1 sweep.

An additional candidate inspected on 2026-08-20 was:

```text
/path/to/data/20260601_cimax_mprage_invivo/
  meas_MID00077_FID25178_pulseq_mprage_nowave_full.dat
```

Despite the name and an R1x1 protocol-header value, its counters show R2x2 PE
sampling with a central 32 x 32 calibration region. Its nominal image matrix
is 256 x 256 x 192, but it is not an R1 baseline and was rejected for that
role.

R3 preliminary optimization and later parameter-transfer dataset:

```text
/path/to/data/20260817_product
```

Future DICOM reference:

```text
/path/to/data/20260817_product/mprage_product_unfiltered_normalize
```

Select the 256-image `ND,NORM` series UID
`1.3.12.2.1107.5.2.0.99923.3.2026082020033466358602277.0.0.0` and reject the
`DIS2D/DIS3D` series.

The R3 data may be tuned now for preliminary presentation results. Since it
will no longer be an untouched holdout, the later application of R1-selected
parameters must be called a cross-dataset transfer check rather than an
independent validation.

## R1 readout finding

For all four inspected R1 TWIX files:

```text
nominal oversampled Fourier columns: 512
stored MDH samples per line:          404
MDH center column:                    148
readout OS factor:                    2
nominal final columns:                256
```

Mapping input center 148 to nominal center 256 places the acquired samples at
nominal indices 108:512. This is one-sided/asymmetric readout support. The
existing inspector's `flagRemoveOS=True` result of 202 samples is not itself a
valid final 256-pixel reconstruction. Preserve the 404 samples and the center
metadata; test partial-Fourier completion and crop the oversampled image FOV.

## Frozen regularization decisions

Execution order is R3 presentation optimization first, followed by R1
partial-Fourier completion and final refinement. Parameters selected during
the first stage are provisional and R3-specific.

Wavelet coarse sweep on R1 development scan:

```text
lambda = 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
```

Correct LLR before rerunning it. Existing effective commands used `wave -l`
without `-v`. BART only creates the size-two real/imaginary ITER dimension
with `-v`; without it, the low-rank matrix has a singleton column dimension.
First validate `-l -v`, output recombination, and lambda-zero equivalence.

The custom `bart wave` internally divides input k-space by its global L2 norm
and does not expose PICS `-w`/`-S`. Record that norm and restore output scale in
the wrapper when needed. `-e` is maximum-eigenvalue/step-size information, not
intensity scaling.

## Evaluation decisions

- Convert normalized unfiltered DICOM to canonical RAS.
- Create one fixed FSL BET mask per reference; mask metrics, not displays.
- Visually approve mask and L/R orientation.
- Estimate one proper rigid transform from lambda zero and reuse it unchanged.
- Primary ranking uses brain-mask structural/error/detail metrics.
- Background noise and missed anatomy remain separately reported QC.
- The approved historical R3 orientation used permutation `[0,1,2]` and RAS
  flips `[true,false,true]`; revalidate on R1.

## Retrospective low-resolution integration

Source repository, not yet tested or merged:

```text
/path/to/user_workspace/sources/published_code/wave-retro-lr-recon
```

Merge it later, preserving history, under `tools/wave_retro_lr_recon/`. The
synthetic baseline should call its API/CLI through a manifest. Its current
phase-encoding-only behavior is the intended behavior. Never crop readout.
Use these requested physical XYZ resolutions:

```text
1.5 x 1.0 x 1.0 mm
1.0 x 1.5 x 1.0 mm
1.25 x 1.25 x 1.0 mm
```

For sagittal MPRAGE, physical X and Y are the two PE directions and physical Z
is readout. Keep physical Z/readout at 1 mm and its matrix unchanged. Crop the
no-wave PE support before Wave encoding and rebuild the target PE-grid PSF,
subject to a focused operator-ordering test.

## Accepted historical R3 outputs

Accepted R3x1 5x5x5-derived synthetic Wave tree:

```text
/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12
```

Accepted R3x2 sweep/evaluation:

```text
/path/to/data/20260817_product/synthetic_wave_grappa_5x5x5_ncc12_r3x2
```

The complete 25-case R3x2 evaluation is retained as a historical pilot. Its
LLR interpretation is limited by the missing `-v`, and its DICOM predates the
normalized reference. Do not overwrite it.

Canonical accepted GRAPPA 5x5x5 results after cleanup:

```text
/path/to/data/20260817_product/grappa_5x5x5_ncc12
```

The dataset-level canonical/historical inventory and deletion record is:

```text
/path/to/data/20260817_product/OUTPUTS_INDEX.md
```

The old manifest path below is retained as a compatibility symlink:

```text
.../synthetic_wave_theoretical_ncc12/grappa_diagnostics
```

Do not alter `mprage_bart` or `gre_bart`.

## Verification commands

```bash
python -m unittest discover \
  -s tools/synthetic_wave_for_reg_baseline/tests \
  -p 'test_*.py'

git status --short
```

The last complete run before this documentation/cleanup pass had 68 passing
tests.
