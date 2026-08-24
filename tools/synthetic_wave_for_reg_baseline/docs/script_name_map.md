# Script and dependency name migration map

This index preserves lookup continuity after stage-labeled files were renamed
by function. Inactive workflows are retained when they remain useful for
diagnostics, comparison, validation, reproducibility, or future datasets.

## Scripts

| Previous path | Current path | Classification | Current role and retention rationale |
|---|---|---|---|
| `scripts/phase_a_inspect.py` | `scripts/inspect_product_dataset.py` | Reusable support | Inspects TWIX/DICOM metadata, acquisition counters, and product sampling geometry. |
| `scripts/phase_b_coil_compression.py` | `scripts/estimate_coil_compression.py` | Canonical support | Estimates the nested physical-to-virtual coil basis used by all current paths. |
| `scripts/phase_c_grappa.py` | `scripts/reconstruct_no_wave_grappa_2d.py` | Retained diagnostic | Preserves the earlier 2D/partitionwise GRAPPA implementation for comparison and regression diagnosis. |
| `scripts/run_grappa_3d.py` | `scripts/reconstruct_no_wave_grappa_3d.py` | Canonical workflow | Runs the accepted resumable joint-coil 5×5×Kz GRAPPA completion; Kz=5 is selected. |
| `scripts/phase_c_export_coil_nifti.py` | `scripts/export_multicoil_nifti.py` | Diagnostic/export | Exports per-coil magnitude/phase volumes for visual inspection of completed k-space. |
| `scripts/phase_d_synthesize_wave.py` | `scripts/synthesize_wave_kspace.py` | Canonical workflow | Generates theoretical-PSF synthetic Wave k-space from completed multi-coil no-wave k-space. |
| `scripts/phase_d_finalize_bart_inputs.py` | `scripts/export_bart_wave_inputs.py` | Canonical workflow | Applies the measured mask and exports validated BART Wave inputs. |
| `scripts/phase_e_prepare_bart_acs.py` | `scripts/export_bart_calibration_acs.py` | Canonical support | Exports measured compressed ACS for BART ESPIRiT calibration. |
| `scripts/phase_e_run_lambda0.py` | `scripts/run_bart_wave_lambda0.py` | Validation workflow | Runs the timed unregularized Wave reconstruction and exports review NIfTIs; retained as a focused acceptance tool. |
| `scripts/phase_e_utils.py` | `scripts/bart_cfl.py` | Reusable support | Provides bounded-memory BART CFL I/O, hashing, and validation helpers. |
| `scripts/prepare_no_wave_sense.py` | unchanged | Retained diagnostic | Prepares exact measured no-wave inputs for the alternative SENSE investigation. |
| `scripts/run_no_wave_sense.py` | unchanged | Retained diagnostic | Reproduces the SENSE/ESPIRiT support investigation and its visual findings. |

Unchanged files were also reviewed rather than silently retained:

| Current path | Classification | Retention rationale |
|---|---|---|
| `scripts/grappa_3d_r3.py` | Canonical algorithm | Implements the accepted joint-coil 3D R=3 kernel geometry. |
| `scripts/grappa_r3.py` | Retained diagnostic algorithm | Supports the useful 2D GRAPPA regression comparison and oracle tests. |
| `scripts/wave_synthesis.py` | Canonical algorithm | Contains bounded-memory extended-readout and theoretical-PSF adapters. |
| `scripts/validate_full_sampling_wave_operator.py` | Evaluation gate | Runs real-data all-coil `PSF=1` no-Wave identity and full-sampling Wave inverse checks without BART or presentation processing. |
| `scripts/sampling_mask.py` | Canonical support | Reconstructs and validates the authoritative product mask from inspection metadata. |
| `scripts/export_grappa_rss.py` | Diagnostic/export | Produces compact RSS NIfTIs from measured or GRAPPA-completed multicoil k-space. |
| `scripts/prepare_no_wave_sense.py` | Retained diagnostic | Builds a measured-only alternative input for future GRAPPA/SENSE investigation. |
| `scripts/run_no_wave_sense.py` | Retained diagnostic | Reproduces the SENSE/ESPIRiT anatomy-support comparison. |
| `scripts/run_no_wave_r3x1_pics_sweep.py` | Presentation comparison | Builds a retrospective R3x1 no-Wave input, runs a GPU PICS CG-SENSE control and compact Wavelet/FISTA sweep, exports magnitude NIfTIs, and saves exact-grid direct-FFT metrics. |
| `scripts/run_previous_non_bart_wave_cg_sense.py` | Historical-algorithm presentation comparison | Adapts accepted BART-formatted inputs to the existing Torch Wave PCG-SENSE implementation and saves exact-grid direct-FFT metrics without defining a new reconstruction algorithm. |
| `scripts/presentation_metrics.py` | Shared presentation evaluation | Restores NIfTI export normalization and computes the standard fixed-mask metric set against the hash-bound approved direct-FFT R1 reference without registration or interpolation. |
| `scripts/build_presentation_nifti_collection.py` | Presentation packaging | Copies accepted magnitude NIfTIs byte-for-byte into a manifested collection and records pending reconstructions as JSON placeholders. |
| `scripts/run_bart_regularization.py` | Canonical workflow | Calls the pinned Wave wrapper for one validated, hashed wavelet or LLR case. |
| `scripts/run_wavelet_sweep.sh` | Canonical workflow | Runs/resumes the compact R1 Wavelet sweep and its reference-neutral review. |
| `scripts/run_llr_sweep.sh` | Canonical workflow | Gates split-complex lambda zero, then runs/resumes the compact block-8 LLR sweep and review. |
| `scripts/run_regularization_refinement.sh` | Canonical workflow | Runs/resumes the missing higher-lambda Wavelet and multi-block LLR cases selected from direct-FFT coarse metrics. |
| `scripts/review_regularization_sweep.py` | Canonical workflow | Builds a shared-window Wavelet or LLR review without DICOM or BET ranking. |
| `scripts/validate_llr_lambda_zero.py` | Canonical workflow | Validates recombined split-complex LLR lambda zero against native-complex FISTA lambda zero. |
| `scripts/prepare_r1_reference_comparison.py` | Qualitative reference audit | Audits and converts a matched SOS unfiltered DICOM pair, exports direct no-Wave FFT RSS, and creates a no-ranking comparison. |
| `scripts/validate_metrics_geometry.py` | Evaluation gate | Hash-validates the approved direct-FFT reference, metrics-only mask, and required reconstruction cases on one exact RAS grid without registration or interpolation. |
| `scripts/evaluate_direct_fft_regularization.py` | Evaluation workflow | Computes fixed-mask coarse metrics and shared-window reviews against the approved fully sampled direct-FFT RSS reference without registration or DICOM ranking. |
| `scripts/export_bart_wave_inputs_retrospective.py` | Acceleration-comparison workflow | Builds a validated PE1×PE2 retrospective mask from reusable full Wave data and exports a separate BART input tree. |
| `scripts/prepare_regularization_evaluation.py` | Evaluation workflow | Consolidates complete reconstruction NIfTIs and converts one strictly selected unfiltered DICOM reference with hashes and provenance. |
| `scripts/review_regularization_orientation.py` | Evaluation gate | Produces RAS-canonical, L/R-labeled no-registration comparisons and ranks signed-axis hypotheses without applying one. |
| `scripts/evaluate_regularization_volume.py` | Evaluation workflow | Applies one approved orientation correction and one shared lambda-zero rigid transform, then writes registered volumes, fixed-mask 3D metrics, provenance, and sweep plots. |

## Tests

| Previous path | Current path |
|---|---|
| `tests/test_phase_a_inspect.py` | `tests/test_inspect_product_dataset.py` |
| `tests/test_phase_b_coil_compression.py` | `tests/test_estimate_coil_compression.py` |
| `tests/test_phase_e_utils.py` | `tests/test_bart_cfl.py` |

## Requirement sets

| Previous path | Current path | Scope |
|---|---|---|
| `requirements/phase-a.txt` | `requirements/inspection.txt` | Dataset/TWIX inspection. |
| `requirements/phase-b.txt` | `requirements/coil_compression.txt` | Inspection plus coil compression. |
| `requirements/phase-c.txt` | `requirements/grappa.txt` | GRAPPA reconstruction and image export. |
| `requirements/phase-d.txt` | `requirements/wave_synthesis.txt` | GRAPPA plus theoretical Wave synthesis. |
| `requirements/phase-e.txt` | `requirements/bart_reconstruction.txt` | Wave synthesis plus external BART workflows. |
| `requirements/sense.txt` | `requirements/sense_diagnostics.txt` | BART requirements plus the retained SigPy diagnostic. |
| none | `requirements/evaluation.txt` | BART requirements plus DICOM selection and plotting dependencies for whole-volume evaluation. |

## Historical generated artifacts

Ignored machine-local reports use purpose-based names such as
`product_dataset_inspection_*.json` and `coil_compression_*.npz`. Their internal
provenance remains intact after the one-time rename. New commands should also
use descriptive output names or explicit user-supplied paths.
