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
