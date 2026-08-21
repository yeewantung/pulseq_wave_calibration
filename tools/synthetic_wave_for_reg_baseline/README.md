# Synthetic Wave regularization baseline

This directory contains the R3 presentation-optimization, R1 parameter-
refinement, and cross-dataset transfer workflow for synthetic Wave-MPRAGE
reconstruction.

Start with:

- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for the active scientific plan;
- [`HANDOVER.md`](HANDOVER.md) for exact cross-session state; and
- [`docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md`](docs/archive/R3x1_no_wave_to_wave_BART_regularization_HISTORICAL.md)
  only when historical R3x1 development detail is needed.

## Layout

```text
scripts/       Reconstruction programs and reusable algorithm/I/O modules
requirements/  Incremental Python dependency sets grouped by task
docs/          Maintenance indexes, including the old-to-new filename map
tests/         Unit and reference-oracle tests
```

Machine-local dataset notes and generated reconstruction artifacts are ignored
by git. Large artifacts should live in dataset output trees rather than at this
directory's root; paths and scan filenames remain configuration/CLI inputs.

## Current experiment

First optimize provisionally on the 2026 R3x1 product and synthetic R3x2 Wave
data and generate presentation results. Then use the 2021-05-10 R1 scans as
the best available baseline after validating and freezing their partial-
Fourier readout completion. The R1 Wavelet coarse sweep will use lambda values
`1e-6`, `1e-5`, `1e-4`, `1e-3`, and `1e-2`, plus lambda zero. LLR must use the
verified BART real/imaginary split (`-v`) and output recombination.

After R1 refinement, apply the selected parameter back to R3 without retuning
as a cross-dataset transfer check. Because R3 is used for the preliminary
optimization, it is not an untouched independent validation dataset.

## Which programs matter

The current production path is:

1. `inspect_product_dataset.py` audits a new product TWIX/DICOM dataset.
2. `estimate_coil_compression.py` estimates the Ncc=12 compression basis.
3. `reconstruct_no_wave_grappa_3d.py` completes the no-wave k-space with the
   accepted joint-coil 5×5×5 GRAPPA kernel.
4. `synthesize_wave_kspace.py` applies the theoretical sequence PSF.
5. `export_bart_wave_inputs.py` masks and exports the synthetic Wave data.
6. `export_bart_calibration_acs.py` exports measured no-wave ACS for one
   reusable BART ESPIRiT calibration.
7. `run_bart_wave_lambda0.py` runs the unregularized acceptance reconstruction.
8. `run_bart_regularization.py` calls the pinned upstream wrapper for one
   hashed, resumable wavelet or LLR case.
9. `prepare_regularization_evaluation.py` consolidates complete NIfTI pairs,
   selects one exact DICOM series by UID and unfiltered `ND` metadata, and
   records hashes and conversion provenance.
10. `review_regularization_orientation.py` canonicalizes both inputs to RAS,
    audits all signed axis mappings, and produces explicitly labeled L/R QC
    figures without accepting a correction or running registration.
11. `evaluate_regularization_volume.py` requires that recorded approval,
    estimates one proper rigid transform from lambda zero, applies it unchanged
    to every magnitude volume, and writes whole-volume metrics and plots.

For the current R3 presentation optimization, `run_r3_presentation_optimization.sh` provides a
resumable tmux-friendly orchestration entry point. Its default
`all-before-review` stage runs the focused GPU reconstructions and prepares the
normalized-DICOM, BET-mask, and orientation QC package. It stops before the
visual gate. After reviewing the printed BET and L/R figures, the separate
`all-after-review --confirm-reviewed-mask-and-lr` stage records that explicit
decision, evaluates the fixed-mask sweep, and creates the compact comparison
package. `validate_bart_split_complex.py` records the required lambda-zero
equivalence of native and recombined `wave -l -v` representations.

For the full-head normalized DICOM reference, the runner invokes
`prepare_reference_brain_mask.py` with BET robust center estimation and a fixed
fractional threshold of `0.55`. The manifest records both settings.

`run_ecalib_intensity_pilot.sh` is the isolated tmux entry point for testing
BART `ecalib -I`. It runs one GPU Wave lambda-zero reconstruction and creates a
brain-median-scaled intensity-profile comparison. It does not run or select
regularization, and it treats no-wave GRAPPA/SENSE only as comparison
references.

The measured ACS calibration is shared from the parent full-Wave BART inputs;
the R3x2 directory supplies the retrospectively masked k-space and linked PSF.

For acceleration comparisons,
`export_bart_wave_inputs_retrospective.py` reuses the validated full synthetic
Wave volume and theoretical PSF, builds an explicit Cartesian PE1×PE2 lattice
plus fully sampled ACS, and exports a separate BART input tree. It does not
repeat GRAPPA completion or Wave encoding.

`reconstruct_no_wave_grappa_2d.py`, `prepare_no_wave_sense.py`, and
`run_no_wave_sense.py` are retained diagnostic alternatives, not the selected
production path. `export_multicoil_nifti.py` and `export_grappa_rss.py` are
visualization helpers. The old-to-new filename dictionary and the reason each
program was retained are in [`docs/script_name_map.md`](docs/script_name_map.md).

## Setup and tests

From the repository root:

```bash
git submodule update --init external/wave-mprage
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/bart_reconstruction.txt
python -m unittest discover \
    -s tools/synthetic_wave_for_reg_baseline/tests \
    -p 'test_*.py'
```

The DICOM-referenced evaluation tools additionally use:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/evaluation.txt
```

The optional SENSE diagnostic additionally uses:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/sense_diagnostics.txt
```

Wave-MPRAGE is pinned as a submodule because the current scripts import its
BART CFL and TWIX-to-NIfTI utilities at runtime. Wave-GRE remains an optional
reference and is not a submodule because no current code depends on it.

The NIfTI orientation/export helpers and the in-memory BART CFL writer are
called directly from that submodule. Coil-compression covariance accumulation
remains a local streaming adapter because the product refscan is read in PE2
chunks rather than materialized as the four-dimensional array expected by the
reference utility. The theoretical-trajectory adapter likewise adds strict
integrated-tail/sequence-definition checks and lightweight provenance without
importing the upstream all-in-one reconstruction program and its unrelated
GPU/SENSE dependencies. Regularized production runs should call the pinned
`external/wave-mprage/recon/bart/run_wave_recon.sh` wrapper directly; the local
lambda-zero runner is retained for its timing, map montage, and acceptance
manifest checks. `run_bart_regularization.py` provides the publishable CLI,
provenance, validation, and safe completed-run reuse around that direct call.

Evaluation intentionally has a hard orientation gate. Run the preparation and
orientation-review programs, inspect `orientation_signed_axis_choice.png`, and
record the user's L/R decision before registering volumes or calculating
metrics. The signed-axis search is diagnostic only; its top-scoring mapping is
not silently applied.

Run a script with `--help` for its dataset-independent CLI, for example:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/reconstruct_no_wave_grappa_3d.py --help
```
