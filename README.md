# Wave-MPRAGE retrospective LR + undersampling

Advanced batch patch for an existing [`HarmonizedMRI/wave-mprage`](https://github.com/HarmonizedMRI/wave-mprage) reconstruction output.

The program reuses the already generated coil-compressed k-space, low-resolution ESPIRiT maps, and FLASH PSF coefficient fits. It does **not** repeat TWIX-to-k-space conversion, coil compression, ESPIRiT calibration, or projection PSF fitting for each requested case.

## What it does

For every requested case, the script:

1. loads `kspace_cc` once and infers its sampling mask in memory;
2. infers the source `Ry` and `Rz` from the k-space mask and verifies them against the matching `.seq` definitions;
3. converts desired physical resolution `[x, y, z]` in millimetres to an integer target matrix at unchanged FOV;
4. center-crops the two phase-encoding dimensions while preserving the Python center indices;
5. adds retrospective undersampling only on axes that were originally fully sampled;
6. interpolates `csm_acs` directly to the target matrix and renormalizes it;
7. rebuilds the calibrated PSF on the target `y_norm`/`z_norm` grid from reusable `a(kx)`, `b(kx)`, and `c(kx)` coefficients;
8. runs the upstream preconditioned Wave CG-SENSE implementation;
9. writes case outputs under `retro-LR-us/` without moving or changing the original reconstruction files.

Sampling masks are not saved. They are inexpensive to infer/recreate and can be storage-heavy in a large batch.

## Repository setup

```bash
git clone <this-repository-url>
cd wave-mprage-retro-lr-us
./scripts/setup_upstream.sh
uv sync
```

When this bundle is committed as a new Git repository, the setup script adds `HarmonizedMRI/wave-mprage` as `external/wave-mprage` submodule. Outside a Git worktree it performs a normal recursive clone instead.

The external submodule is expected at `external/wave-mprage`. A different checkout may be supplied with `--wave-mprage-repo`.

## Source output folder

The source folder remains unchanged. Typical reusable files are:

```text
out/
├── a_fit_all_projy_72kyline_.npy
├── b_fit_all_projz_72kzline_.npy
├── c_fit_all_projy_72kyline_.npy
├── c_fit_all_projz_72kzline_.npy
├── csm_acs_.npy
├── csm_full_.npy
├── kspace_cc.npy                     # or the standard tagged kspace_*_cc_*.npy name
├── wave_mprage_manifest.json         # recommended
└── nifti/
```

The source manifest should record the matching TWIX and Pulseq paths. Accepted top-level manifest names are:

```text
wave_mprage_manifest.json
reconstruction_manifest.json
recon_manifest.json
manifest.json
```

If no manifest is present, provide the two paths directly with `--seq` and `--twix`. The input interface remains centered on one understandable folder: `--wave-mprage-out-folder`.

## Case file

Resolution is specified in **physical XYZ order**, matching sequence FOV/matrix definitions. For the current sagittal acquisition:

```text
physical x -> logical PAR / Rz dimension
physical y -> logical LIN / Ry dimension
physical z -> logical readout dimension
```

Only physical x and y are retrospectively cropped. Physical z/readout resolution must remain unchanged.

```json
{
  "cases": [
    {"resolution_mm": [1.0, 1.0, 1.0], "acceleration": [3, 1]},
    {"resolution_mm": [1.5, 1.0, 1.0], "acceleration": [3, 1]},
    {"resolution_mm": [1.5, 1.0, 1.0], "acceleration": [3, 2]}
  ]
}
```

Acceleration is `[Ry, Rz]`. An already accelerated axis must stay unchanged. Examples:

```text
source R3x1 -> target R3x2   allowed
source R1x1 -> target R2x2   allowed
source R3x1 -> target R6x1   rejected
```

## Run

When the source manifest records the `.dat` and `.seq` paths:

```bash
uv run python recon/recon_wave_mprage_retro_lr_us_batch.py \
  --wave-mprage-out-folder /path/to/out \
  --cases configs/cases.example.json \
  --save-intermediate standard \
  --save-nifti-phase
```

Without a manifest:

```bash
uv run python recon/recon_wave_mprage_retro_lr_us_batch.py \
  --wave-mprage-out-folder /path/to/out \
  --seq /path/to/matching.seq \
  --twix /path/to/source.dat \
  --cases configs/cases.example.json
```

Validate discovery, geometry, acceleration, and cases without reconstruction:

```bash
uv run python recon/recon_wave_mprage_retro_lr_us_batch.py \
  --wave-mprage-out-folder /path/to/out \
  --cases configs/cases.example.json \
  --validate-only
```

## Output layout

The script adds only `retro-LR-us/` to the original output folder:

```text
out/
├── ... original wave-mprage files remain here ...
├── nifti/
└── retro-LR-us/
    ├── batch_info.json
    ├── nifti/
    │   ├── res1x1x1mm_R3x1/
    │   │   ├── sub-retro_part-mag_MPRAGE.nii.gz
    │   │   ├── sub-retro_part-mag_MPRAGE.json
    │   │   ├── sub-retro_part-phase_MPRAGE.nii.gz
    │   │   └── sub-retro_part-phase_MPRAGE.json
    │   └── res1.5x1x1mm_R3x2/
    ├── res1x1x1mm_R3x1/
    │   ├── image.npy
    │   └── case_info.json
    └── res1.5x1x1mm_R3x2/
        ├── image.npy
        └── case_info.json
```

Folder names use the **actual achieved physical XYZ resolution**, rounded to at most two decimals. Full-precision requested and achieved resolutions, integer matrices, crop bounds, and acceleration are retained in `case_info.json`.

## Intermediate saving

```text
--save-intermediate none
    case_info.json and requested NIfTI outputs only

--save-intermediate standard     default
    image.npy + case_info.json + requested NIfTI outputs

--save-intermediate all
    standard outputs plus:
    kspace_lr_undersampled.npy
    csm_target.npy               normalized, unoversampled readout CSM
    psf_target.npy
```

Neither source nor case sampling masks are saved at any level.

## PSF coefficient reuse

Preferred processed coefficient files are:

```text
psf_coefficients_processed*.npz    keys: a, b, c
psf_coefficients_processed*.npy    shape: (3, Nx_os) or (Nx_os, 3)
psf_integrated_calib_fit*.npy      same numeric shape
```

If none exists, the script finds the standard raw projection fits:

```text
a_fit_all_projy_*.npy
b_fit_all_projz_*.npy
c_fit_all_projy_*.npy
c_fit_all_projz_*.npy
```

It reconstructs the upstream combination:

```text
a = a_projy
b = b_projz
c = c_projy + c_projz
```

and applies the same upstream `smooth` or `sine-line` coefficient processing. For a source reconstructed with sine-line processing, record its fit range in the manifest or pass:

```text
--psf-coefficient-processing sine-line
--psf-fit-kx-min INTEGER
--psf-fit-kx-max INTEGER
```

## Tests

```bash
uv run --group dev pytest
```

The included tests cover center preservation, acceleration inference, undersampling restrictions, physical-XYZ resolution conversion, achieved-resolution folder naming, and the no-saved-mask policy.
