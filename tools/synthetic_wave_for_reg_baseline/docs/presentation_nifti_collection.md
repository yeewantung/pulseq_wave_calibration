# Presentation magnitude NIfTI collection

The active private collection contains the requested 20 presentation slots
plus three supplemental retrospective FISTA-lambda-zero controls. All 23 are
finite canonical-RAS magnitude NIfTIs. Files are not resampled or
cross-normalized while being collected; the six retrospective-resolution
images retain their native matrices and voxel sizes.

Canonical reconstruction output trees retain both magnitude and phase NIfTIs.
This presentation collection intentionally copies only magnitude; phase stays
beside the reconstruction and is recorded in its case manifest.

The collection manifest records the requested display order, source and copied
file hashes, source manifests, shapes, voxel sizes, and orientation. The
tracked builder is:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/build_presentation_nifti_collection.py \
    --config /path/to/presentation_nifti_collection.local.json \
    --refresh
```

Use an ignored local configuration copied from
`requirements/presentation_nifti_collection.example.json`. `--refresh` may
replace an owned placeholder with a newly available NIfTI, but it refuses to
overwrite a changed collected NIfTI or replace an available NIfTI with a
placeholder.

## Completed GRAPPA entry

- Synthetic no-Wave R3x1 GRAPPA is available. It uses the accepted joint-coil
  5x5x5/Ncc=12 implementation on the same retrospective R3x1-plus-ACS sampling
  contract used for the no-Wave PICS comparisons. Its reconstruction manifest
  records finite output, bitwise preservation of acquired samples, magnitude
  and phase NIfTIs, and complete exact-grid direct-FFT metrics.

The no-Wave sweep includes `1e-4`, `1e-3`, `1e-2`, `1.5e-2`, `2e-2`, and
`5e-2`. It uses a separate no-Wave PICS scaling contract (`-S`) and therefore
does not claim that the custom-Wave `lambda=1.5e-2` transfers numerically.
The originally requested `1.5e-2` result remains retained in the canonical
sweep, but the presentation collection now uses the explicit `1e-3` choice.

All reconstruction launchers evaluate every completed NIfTI against the
approved direct-FFT R1 RSS reference and approved BET mask on their exact
shared grid. Export normalization is restored first, one unconstrained
least-squares intensity scale is fitted inside the fixed brain mask, and no
registration or interpolation is allowed. The complete metric dictionary is
stored at `direct_fft_metrics.metrics` in each case `manifest.json`; useful
presentation fields include `nrmse_brain`, `ssim_3d_brain_bbox`,
`ssim_axial_brain_mean`, and `gradient_ncc_brain_edge`. These are reference
similarity/QC measures, not true SNR.

`presentation_metrics.csv` contains one row per requested display slot in the
same order as the collection manifest. Its companion
`presentation_metrics_manifest.json` hash-binds the source tables. DICOM rows
are explicitly qualitative-only, the GRAPPA row is complete, standard
reconstructions use exact-grid direct-FFT metrics, and retrospective-resolution
rows retain their documented matched-grid fidelity and native descriptive
measures.

Rebuild the table with path-local inputs:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/build_presentation_metrics_csv.py \
    --collection-manifest /path/to/collection_manifest.json \
    --regularization-metrics /path/to/regularization_metrics.csv \
    --regularization-metrics /path/to/additional_regularization_metrics.csv \
    --retrospective-matched-metrics /path/to/matched_fidelity_metrics.csv \
    --retrospective-native-metrics /path/to/native_resolution_metrics.csv \
    --retrospective-supplement /path/to/additional_retrospective_metrics.local.json \
    --output /path/to/presentation_metrics.csv \
    --refresh
```

The regularization option is repeatable. Rows are joined to collection entries
by source-NIfTI SHA-256, and conflicting records for one hash are rejected.

Additional retrospective solver batches use an ignored local copy of
`requirements/presentation_retrospective_metrics_supplement.example.json` to
map collection keys to their case names without colliding with the original
retrospective batch.

The ignored `export_presentation_orientation_tiffs.local.sh` launcher exports
sagittal, coronal, and axial slices into `orientation_slices/`. Standard
256-cubed cases use index 128. Each retrospective-resolution case uses the
center of each orientation axis: balanced `(102,102,128)`, lower-X
`(86,128,128)`, and lower-Y `(128,86,128)` for sagittal/coronal/axial. These
indices apply to both Wavelet and FISTA-lambda-zero retrospective cases. TIFFs
are 16-bit grayscale with per-volume positive-voxel p99.5 display scaling; the
slice manifest records each actual index, scaling, orientation convention,
source hashes, and output hashes. The presentation collection itself remains
magnitude-NIfTI-only.

The supplemental FISTA-lambda-zero analysis uses the full-resolution
FISTA-lambda-zero reconstruction for native sharpness ratios and direct FFT
RSS for matched-grid fidelity. It records that no separate visual-approval
artifact was created; the descriptive analysis was requested directly after
the native/matched geometry review. No automatic resolution selection is made.

The no-Wave R3x1 sweep evaluation is stored beside the collection under
`no_wave_r3x1_sweep_evaluation/`. Its CSV and PNG/PDF curve plot are hash-bound
to all six Wavelet cases, the CG-SENSE control, GRAPPA, and the approved
direct-FFT metrics reference. Among tested values, `1e-3` leads brain NRMSE,
3D SSIM, and edge-ratio closeness; `1e-4` leads edge-gradient NCC. The plot
highlights the explicitly selected presentation value `1e-3`; the former
`1.5e-2` entry is worse on all four plotted metrics but remains available in
the canonical sweep.

Refresh the TIFFs using the ignored local launcher:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/export_presentation_orientation_tiffs.local.sh
```

## Tmux launch pattern

First validate an ignored launcher without creating outputs:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_no_wave_r3x1_pics_sweep.local.sh \
    --validate-only

tools/synthetic_wave_for_reg_baseline/scripts/run_no_wave_r3x1_grappa.local.sh \
    --validate-only

tools/synthetic_wave_for_reg_baseline/scripts/run_previous_non_bart_wave_cg_sense.local.sh \
    --validate-only
```

Then start a tmux session, run one launcher inside it, detach with `Ctrl-b d`,
and later reattach with `tmux attach -t SESSION_NAME`. Do not launch the two
GPU jobs on the same GPU concurrently. The PICS no-Wave path requires BART GPU
`-g`, while the legacy Torch path stops unless CUDA is visible. The GRAPPA path
uses the accepted NumPy/SciPy implementation and does not invoke BART or use a
GPU. All launchers require the evaluation dependencies because they calculate
the saved direct-FFT metrics after NIfTI export.

For cases reconstructed before phase export was enabled, rerun the same local
launcher with `--resume` (already present in the supplied local launchers).
The workflow reads the saved complex image, exports magnitude and phase in a
temporary directory, verifies regenerated magnitude against the accepted one
voxel-for-voxel, and installs only the missing phase files. Neither solver is
rerun and the accepted magnitude is not replaced.

For a future reconstruction-backed placeholder, change only its corresponding
ignored local collection entry to `available`, point it at the accepted
magnitude NIfTI and case manifest, and rerun the collection builder with
`--refresh`.

## Cleanup boundary

Do not merge or remove sweep directories until the presentation collection is
complete and frozen. The later cleanup starts with a hash/manifests audit,
chooses canonical runs, merges dataset indexes and organization, archives
superseded sweep trees, and deletes nothing until every downstream reference
has been checked.
