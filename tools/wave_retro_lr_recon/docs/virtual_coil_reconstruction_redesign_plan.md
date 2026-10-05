# Virtual-coil and reconstruction-profile redesign plan

Updated: 2026-10-05, America/New_York

## Status and purpose

This is an approved design direction, not an implemented user contract. The
next implementation session must begin read-only, verify the current Git state
and all affected consumers, and present any material deviation from this plan
before editing. Do not launch scientific reconstructions or create production
output directories during implementation.

The redesign covers the standard, non-ROVir MPRAGE and multi-echo GRE normal
and retrospective launchers. It will make the standard PCA virtual-coil count
configurable, reduce the default reconstruction workload, isolate every new
coil-count product from historical outputs, and extend the single existing
NIfTI collection instead of creating a second collection.

## Evidence and default virtual-coil decision

The current standard preparation paths hard-code 12 virtual coils. A read-only
audit of 31 existing standard normal-input manifests found the following
retained-energy distribution:

| Modality | Manifests | Ncc=12 retained energy |
| --- | ---: | ---: |
| MPRAGE | 22 | 0.748-0.911; mean 0.866 |
| GRE | 9 | 0.847-0.922; mean 0.884 |
| Combined | 31 | 0.748-0.922; mean 0.872 |

This confirms that Ncc=12 is frequently below 90% retained ACS energy and can
be an aggressive fixed truncation. Retained energy alone does not prove image
quality, and retaining more coils also retains more noise modes and increases
memory and runtime. Nevertheless, a fixed Ncc=24 is the approved new default
because it is a conservative, reproducible choice for the existing 40-, 52-,
and 64-channel acquisitions and gives downstream networks a consistent channel
contract.

Every standard launcher must accept an explicit `--virtual-coils N` override.
The requested value must be positive and no larger than the physical receive
coil count. Preparation must record the requested count, physical count,
compression-basis identity, leading singular values, and retained energy. A
different count is a different immutable product, never an in-place upgrade.
If a later image review is unsatisfactory, troubleshooting may explicitly
rerun another count; that run follows the same `vccN` source and collection
layout described below.

The implementation should report cumulative retained energy at useful fixed
counts, including 12, 20, and 24 when available. These values are diagnostics,
not an automatic subject-specific Ncc selector. Do not silently vary Ncc by
subject.

ROVir keeps its existing, separately selected virtual-coil contract. The
standard-PCA `--virtual-coils` option must not change, reinterpret, or silently
override a canonical ROVir transform. A launcher that supports `--rovir` must
reject an incompatible simultaneous standard-PCA coil-count request with a
clear message.

## Reconstruction-profile CLI

The four user-facing standard launchers are:

- `scripts/sample_mprage_normal_recon.sh`;
- `scripts/sample_mprage_retro_lr_recon.sh`;
- `scripts/sample_gre_normal_recon.sh`; and
- `scripts/sample_gre_retro_lr_recon.sh`.

They must expose three mutually exclusive reconstruction-profile flags:

- `--reg-full`: run both FISTA lambda zero and the reviewed Wavelet branch;
- `--wavelet-only`: run only the reviewed Wavelet branch; and
- `--fista-only`: run only FISTA lambda zero.

Normal reconstruction defaults to `--wavelet-only`. Retrospective
reconstruction defaults to `--fista-only`. Supplying more than one profile is
an error. The selected profile must be recorded in a resumable run manifest,
and an existing branch may be reused only after its command, inputs, output,
geometry, finite-value, coil-processing, and regularization records pass.

MPRAGE R1 has no approved positive Wavelet parameter. A request that resolves
to Wavelet-only for R1 must fail explicitly rather than silently substitute
FISTA. ROVir behavior remains governed by its canonical manifests and must not
be broadened incidentally by this standard-PCA redesign.

## Default retrospective cases

The new standard MPRAGE retrospective default contains only:

1. `native_r3x2`, approximately 1-mm isotropic;
2. `lr_y_1p5mm_r3x2`, approximately 1 x 1.5 x 1 mm; and
3. `native_r3x3`, approximately 1-mm isotropic.

The existing LR-X and LR-XY implementations may remain available for backward
compatibility, but they are not part of the new default batch and must not be
prepared merely as an unused side effect.

The existing GRE/SWI retrospective set already matches the approved default:

1. `native_r3x2`, approximately 0.9 x 0.9 x 2.5 mm;
2. `lin_low_resolution_r3x2`, approximately 0.9 x 1.5 x 2.5 mm; and
3. `native_r3x3` on the native spatial grid.

## Reconstruction and collection layout

The user supplies one reconstruction root. New standard-PCA runs always write
below a count-specific directory, including when the explicit count is 12:

```text
RECONSTRUCTION_ROOT/
|-- normal/                         # historical standard VCC=12, untouched
|-- retro/                          # historical standard VCC=12, untouched
|-- vcc24/
|   |-- normal/
|   |   |-- bart_inputs/
|   |   |-- bart_output/
|   |   `-- nifti/
|   `-- retro/
|       |-- native_r3x2/
|       |-- modality-specific LIN-low-resolution R3x2/
|       `-- native_r3x3/
|-- vcc20/                          # example troubleshooting rerun
|   |-- normal/
|   `-- retro/
`-- nifti_collection/               # the only collection root
    |-- vcc24/
    |   |-- original_nifti/
    |   |-- head_masked_nifti/      # MPRAGE presentation derivatives only
    |   `-- manifest.json
    |-- vcc20/
    |   `-- ...
    |-- rovir/
    |   `-- ...
    `-- manifest.json               # top-level discovery and hash index
```

There must not be a second `nifti_collection` under `vcc24`. The existing
collection launcher remains the single user entry point. It discovers complete
`vccN/normal` and `vccN/retro` outputs and appends or safely synchronizes them
under `nifti_collection/vccN`. If canonical ROVir outputs exist, it collects
them under `nifti_collection/rovir`. Standard variants and ROVir are additive;
one never replaces another.

Within each `vccN` collection subtree, normal and retrospective cases remain
distinguishable by their relative case path and manifest records. The builder
must allow the approved asymmetric defaults: a normal product may contain only
Wavelet while retrospective products contain only FISTA. It must not require
the same reconstruction branches to exist for normal and retro.

Existing root-level standard VCC=12 reconstructions and their established
collection layout are legacy-compatible inputs. Do not move, rename, rewrite,
or relabel them. New explicit `--virtual-coils 12` work goes below `vcc12/`, so
it also cannot overwrite a historical root-level product.

Every collection entry must validate and record the source virtual-coil count,
compression-basis identity, source manifest identity, reconstruction profile,
case, geometry, acceleration, method, lambda, and copied-file hashes. A
`vccN` directory containing a manifest with a different Ncc is a hard error.
Interrupted variants must be resumable without weakening provenance checks.

## Regularization values and required Ncc=24 reassessment

The reviewed MPRAGE native-R3x1 Wavelet value `3.5e-2` and GRE/SWI shared-echo
value `1.5e-2` were selected with Ncc=12 inputs. Increasing Ncc changes the
retained data subspace, noise contribution, ecalib maps, and the balance
between the data-fidelity and regularization terms. The old values may remain
near-optimal, but they must not be relabeled as Ncc=24 optima without a new
assessment. There is no valid fixed rescaling by channel-count or retained-
energy ratio.

After implementation and source review, run compact Ncc=24 native-R3x1 sweeps
for MPRAGE and GRE separately. Include FISTA lambda zero as a control. A
reasonable initial grid is:

- MPRAGE: `0.025, 0.030, 0.035, 0.040, 0.045, 0.050`;
- GRE/SWI: `0.010, 0.0125, 0.015, 0.0175, 0.020`.

Expand or refine only if the reviewed optimum lies on a boundary. MPRAGE must
use the same pure-image-lattice, resolution-matched direct-FFT, approved BET
brain-mask, fixed-window, and manual-selection contract as the accepted
synthetic workflow. GRE must keep one candidate lambda shared across all
echoes within a case and jointly review magnitude, phase, relative inter-echo
scaling, delta-B0 consistency, finite values, and convergence. Do not select a
winner automatically.

Post-implementation status: the MPRAGE Ncc=24 native-R3x1 sweep and manual
review selected `lambda=3e-2`. The measured default is isolated under
`wavelet_selected_vcc24`; no retrospective case inherited that value. The GRE
Ncc=24 native-R3x1 shared-echo sweep and manual review independently retained
`lambda=1.5e-2`. Its measured normal default is also isolated under
`wavelet_selected_vcc24`; no GRE retrospective case inherited the reviewed
label.

The default retrospective workflow is FISTA-only, so this reassessment does
not require case-specific retrospective Wavelet sweeps before historical
FISTA reprocessing. If retrospective Wavelet branches will be used through
`--reg-full`, their Ncc=12 selections must be described as transferred values
until separately validated at Ncc=24.

## Implementation order and validation

1. Begin read-only: inventory the four launchers, preparation APIs, R3x3
   launchers, converters, manifests, collection builders, tests, documentation,
   ignored local launchers, and downstream consumers.
2. Parameterize MPRAGE and GRE preparation with `virtual_coils=24`; remove
   scientific hard-coding while keeping strict physical-coil validation.
3. Add immutable `vccN` source layout, reuse gates, provenance, and clear
   legacy-root compatibility. Recompute the PCA basis and matched CSM for each
   Ncc. Reuse a source-compatible accepted PSF/trajectory when the workflow
   already provides a strict hash-bound reuse path; do not weaken PSF checks.
4. Add the three reconstruction profiles and the reduced default MPRAGE case
   set. Propagate the profile and Ncc through focused R3x3 launchers.
5. Extend the existing MPRAGE and GRE collection builders to discover `vccN`
   and ROVir variants under the one collection root, without requiring branch
   symmetry between normal and retro.
6. Add focused unit and command-contract tests for defaults, explicit
   overrides, mutually exclusive flags, mismatched reuse, legacy discovery,
   multiple `vccN` variants, ROVir isolation, asymmetric branches, resumability,
   privacy, and exact collection provenance.
7. Run both tool test suites, upstream Wave-MPRAGE and Wave-GRE tests where
   applicable, privacy checks, shell syntax checks, and `git diff --check`.
8. Present the code and exact proposed production commands for review. Do not
   create a production root or launch BART jobs until the user confirms the
   exact output directory and name.

Preserve frozen historical output trees. Keep real paths only in ignored
`.local.sh` or `.local.json` files. Do not modify external submodules, reset or
discard unrelated changes, commit, push, or launch scientific jobs without
explicit user authorization.
