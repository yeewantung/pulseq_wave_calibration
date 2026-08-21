# Dataset manifest contract

The dataset manifest is the authoritative boundary between acquisition-specific
state and reusable reconstruction code. Start a new dataset by copying
`configs/incoming_r1_dataset.example.json`, replacing every example path and
measured expectation, and validating it before any large reconstruction.

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/validate_dataset_manifest.py \
    /path/to/dataset.json --check-inputs
```

Input paths are absolute or relative to the manifest file. Every named output
path or prefix must be relative, cannot contain `..`, and is resolved below
`outputs.root`. Generated reports store the manifest path and SHA-256 so a
later run cannot silently consume a different contract revision.

The other contained output prefixes keep calibration and reconstruction files
discoverable:

- `outputs.coil_compression_prefix` writes the basis `.npz` and report `.json`;
- `outputs.source_reconstruction_prefix` is shared by the compatible no-Wave
  source-reconstruction entry point.
- `outputs.wave_synthesis_dir` contains the reusable, unmasked full-Wave data;
- `outputs.bart_export_dir` contains the separately masked target BART inputs.
- `outputs.lambda0_reconstruction_dir` contains the unregularized BART Wave
  acceptance reconstruction and its NIfTI review files.

## Scientific fields

- `geometry` uses logical `(readout, phase_encode_1, phase_encode_2)` order and
  records matrix sizes and FOV in millimetres.
- `sampling.source_acceleration_pe1_pe2` describes the acquired no-Wave data.
  `sampling.synthetic_wave_acceleration_pe1_pe2` describes the mask applied
  only after synthetic Wave encoding. They are intentionally separate.
- The target-mask residue and half-open PE1 ACS bounds are explicit. The
  current mask kind is a Cartesian image lattice union a PE1 ACS band that is
  fully sampled across PE2.
- `sampling.require_complete_source_grid` states whether the source image
  stream itself must cover every PE coordinate. It should be `true` for the
  incoming fully sampled R1 development scan.
- `reconstruction` holds coil counts, GRAPPA settings, and BART settings.
  `coil_compression_source` explicitly selects `image` or `refscan`; use the
  complete image stream for an R1 acquisition that has no PAT refscan.
  `bart.calibration_source` separately selects measured `image` or `refscan`
  data for ESPIRiT calibration; there is no automatic fallback. `bart.use_gpu`
  must remain `true`; a null maximum eigenvalue means it will be measured for
  the dataset rather than copied from the R3 experiment. The ESPIRiT crop,
  optional `-I` setting, and lambda-zero iteration/tolerance are explicit.
- `wave_synthesis` records the extended readout allocation, trajectory
  calibration dimensions, orientation/sign convention, and diagnostic coils.
- `inputs.dicom.enabled: false` makes the DICOM directory null, disables its
  metadata gate, and requires empty image-type token lists.
- `evaluation.ranking_reference` can be `none`, `grappa`, `nifti`, or `dicom`.
  The separate DICOM-ranking flag must agree with that choice. The current R1
  contract uses `none` because Prescan Normalize contaminated its DICOM
  intensity profile.
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

There is no implicit calibration-stream fallback. The incoming R1 example uses
`image`, while a current R3 contract should use `refscan`. The selected stream
and its dimensions are recorded in the coil-compression report.

For a fully sampled R1 source, validate the runtime TWIX layout without reading
its payload:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/assemble_fully_sampled_no_wave_kspace.py \
    --dataset-manifest /path/to/dataset.json --validate-only
```

After that passes, run the chunked, resumable direct assembly. It applies the
validated coil basis but performs no GRAPPA or other sample interpolation:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/assemble_fully_sampled_no_wave_kspace.py \
    --dataset-manifest /path/to/dataset.json --resume
```

The command requires a duplicate-free complete PE grid, a centered complete
readout, zero-origin compact TWIX support, and exact runtime
`[RO, coil, PE1, PE2]` geometry. Its output is
`[RO, PE1, PE2, virtual coil]` complex64 k-space under the configured source
prefix. Resume state is bound to the manifest, inspection report, TWIX file
identity, coil-basis hash, matrix, and coil counts.

For the existing R3 source, the joint-coil GRAPPA entry point accepts the same
contract and derives its matrix, Ncc, kernel, regularization, input basis, and
output prefix:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/reconstruct_no_wave_grappa_3d.py \
    --dataset-manifest /path/to/dataset.json --resume
```

The GRAPPA program remains intentionally restricted to measured PE1 stride/residue
`3/[1]`, source acceleration `3x1`, and full-PE2 refscan coverage. It exits
before reconstruction for an R1 manifest. Conversely, direct assembly requires
R1 and cannot accept accelerated source data. The two paths intentionally write
the same downstream k-space layout.

Coil compression, GRAPPA, and direct R1 assembly refuse unrelated existing
results. Their manifest routes support conservative `--resume` reuse bound to
the current contract and persisted provenance.

## Wave synthesis and target-mask export

The next two consumers use the same passed contract and the validated source
report. Run full Wave encoding first:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/synthesize_wave_kspace.py \
    --dataset-manifest /path/to/dataset.json --resume
```

This command derives subject, matrix/FOV, virtual-coil count, sequence, TWIX,
extended readout, trajectory settings, diagnostics, and output paths from the
manifest. It accepts an existing output only when `--resume` finds a complete
matching synthesis with intact source-report and PSF provenance.

Inspect the magnitude and phase montages under the configured
`outputs.wave_synthesis_dir`. Only after approving them, export the
manifest-defined synthetic target sampling:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/export_bart_wave_inputs.py \
    --dataset-manifest /path/to/dataset.json \
    --visual-review-approved --resume
```

The exporter requires unmasked full-Wave k-space, applies the target lattice
and ACS band after Wave encoding, verifies all acquired samples bitwise and all
omitted samples as exact zero, and links the validated PSF into a separate
`outputs.bart_export_dir/bart_inputs` tree. It never edits the accepted
synthesis manifest. A resumed run must match the dataset SHA-256, synthesis
manifest SHA-256, mask configuration, and output hashes.

## Measured ACS export

After the target BART tree is complete, export the calibration data with:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/export_bart_calibration_acs.py \
    --dataset-manifest /path/to/dataset.json --resume
```

For `bart.calibration_source: image`, the command requires a fully sampled R1
source prepared by the direct, interpolation-free path. It copies the declared
PE1 ACS band across every PE2 partition from that already compressed source;
it neither applies GRAPPA nor compresses the data a second time. For
`calibration_source: refscan`, it reads the measured rectangular TWIX refscan
and applies the accepted coil basis, preserving the compatible product route.

Both branches write `kspace_calib.hdr/.cfl` into the configured target
`bart_inputs` directory, verify the measured ACS payload, require exact zeros
outside its support, and attach calibration provenance to the existing BART
input manifest. Completed reuse is bound to the dataset, target-export
manifest, source identity, mask support, shape, and complete CFL hash.

## Tmux-friendly R1 execution

The small wrapper keeps long preparation separate from the visual-review gate.
On Macha, run:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_synthetic_wave_dataset.sh prepare
```

After reviewing the full-Wave diagnostics, continue in tmux with:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_synthetic_wave_dataset.sh \
    reconstruct \
    /path/to/data/20260821_product_synthetic_wave_r1_ncc12_r3x2/dataset_manifest.json \
    --confirm-full-wave-reviewed
```

The reconstruction branch sources the host BART setup and calls the
manifest-aware lambda-zero runner. Every `bart wave` reconstruction includes
`-g`; `ecalib -I` remains disabled in the current contract.
