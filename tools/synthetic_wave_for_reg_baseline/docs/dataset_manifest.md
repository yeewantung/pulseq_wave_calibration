# Dataset manifest contract

The dataset manifest is the authoritative boundary between acquisition-specific
state and reusable reconstruction code. Start a new dataset by copying
`configs/incoming_r1_dataset.example.json`, replacing every example path and
measured expectation, and validating it before any large reconstruction.

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/validate_dataset_manifest.py \
    /path/to/dataset.json --check-inputs
```

All paths are absolute or relative to the manifest file, except
`outputs.inspection_report`, which must be a contained relative path and is
resolved below `outputs.root`. Generated
reports store the manifest path, SHA-256, and a fully resolved snapshot so a
later run does not depend on the current working directory.

The other contained output prefixes keep calibration and reconstruction files
discoverable:

- `outputs.coil_compression_prefix` writes the basis `.npz` and report `.json`;
- `outputs.source_reconstruction_prefix` is shared by the compatible no-Wave
  source-reconstruction entry point.

## Scientific fields

- `geometry` uses logical `(readout, phase_encode_1, phase_encode_2)` order and
  records matrix sizes and FOV in millimetres.
- `sampling.source_acceleration_pe1_pe2` describes the acquired no-Wave data.
  `sampling.synthetic_wave_acceleration_pe1_pe2` describes the mask applied
  only after synthetic Wave encoding. They are intentionally separate.
- `sampling.require_complete_source_grid` states whether the source image
  stream itself must cover every PE coordinate. It should be `true` for the
  incoming fully sampled R1 development scan.
- `reconstruction` holds coil counts, GRAPPA settings, and BART settings.
  `bart.use_gpu` must remain `true`; a null maximum eigenvalue means it will be
  measured for the dataset rather than copied from the R3 experiment.
- `evaluation.ranking_reference` can be `grappa`, `nifti`, or `dicom`. The
  separate DICOM-ranking flag must agree with that choice. Keep it disabled and
  use GRAPPA during current development; change the manifest after the incoming
  dataset and its DICOM processing have passed qualification.
- `evaluation.brain_mask.usage` is fixed to `metrics_only`. A mask is never an
  input to reconstruction or ordinary image display.

## Inspection entry point

The inspector remains compatible with its original explicit `--twix`,
`--dicom-dir`, and `--output` arguments. For a new dataset, prefer the manifest
form so paths and expectations cannot drift between commands:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/inspect_product_dataset.py \
    --dataset-manifest /path/to/dataset.json --probe-samples
```

The report compares measured matrix, acceleration, coil count, readout
oversampling, source-grid completeness, optional ACS support, and DICOM image
type against the contract. It is written even when a check fails, then the
command exits nonzero so a tmux workflow cannot continue silently.

Downstream manifest-backed commands require this report to pass and require its
stored manifest SHA-256 to equal the current manifest. If an expectation or
path changes, rerun inspection rather than reusing stale approval.

After inspection passes, coil compression can use only the manifest plus
runtime chunking options:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/estimate_coil_compression.py \
    --dataset-manifest /path/to/dataset.json
```

The existing joint-coil GRAPPA entry point accepts the same contract and
derives its matrix, Ncc, kernel, regularization, input basis, and output prefix:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/reconstruct_no_wave_grappa_3d.py \
    --dataset-manifest /path/to/dataset.json --resume
```

That program remains intentionally restricted to measured PE1 stride/residue
`3/[1]`, source acceleration `3x1`, and full-PE2 refscan coverage. It exits
before reconstruction for an R1 manifest. A direct fully sampled source path is
the next required consumer for the incoming R1 acquisition; R1 data must not be
silently routed through the R3 interpolation operator.

Coil compression refuses an existing basis/report prefix. GRAPPA likewise
refuses existing checkpoints or results unless `--resume` is explicit, so a
new dataset contract cannot overwrite an accepted run accidentally.
