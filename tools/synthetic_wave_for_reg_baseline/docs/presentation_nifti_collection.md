# Presentation magnitude NIfTI collection

The active private collection contains the requested 20 presentation slots.
It currently has 16 finite canonical-RAS magnitude NIfTIs and four explicit
JSON placeholders. Files are not resampled or cross-normalized while being
collected; the three retrospective-resolution images retain their native
matrices and voxel sizes.

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

## Current pending entries

- Synthetic no-Wave R3x1 GRAPPA remains an explicit placeholder, as requested.
- Synthetic no-Wave R3x1 CG-SENSE and Wavelet `lambda=1.5e-2` are produced by
  the no-Wave PICS sweep after its tmux launcher is run.
- Previous non-BART synthetic-Wave R3x2 CG-SENSE is produced by the legacy
  Torch PCG-SENSE adapter after its tmux launcher is run.

The no-Wave sweep includes `1e-4`, `1e-3`, `1e-2`, `1.5e-2`, `2e-2`, and
`5e-2`. It uses a separate no-Wave PICS scaling contract (`-S`) and therefore
does not claim that the custom-Wave `lambda=1.5e-2` transfers numerically.
The requested `1.5e-2` result is retained as one presentation case within that
compact no-Wave-specific sweep.

## Tmux launch pattern

First validate either ignored launcher without creating outputs:

```bash
tools/synthetic_wave_for_reg_baseline/scripts/run_no_wave_r3x1_pics_sweep.local.sh \
    --validate-only

tools/synthetic_wave_for_reg_baseline/scripts/run_previous_non_bart_wave_cg_sense.local.sh \
    --validate-only
```

Then start a tmux session, run one launcher inside it, detach with `Ctrl-b d`,
and later reattach with `tmux attach -t SESSION_NAME`. Do not launch both jobs
on the same GPU concurrently. The no-Wave path requires BART GPU `-g`; the
legacy Torch path stops unless CUDA is visible.

After either job completes, change only the corresponding ignored local
collection entries from `placeholder` to `available`, point them at the new
magnitude NIfTI and case manifest, and rerun the collection builder with
`--refresh`.

## Cleanup boundary

Do not merge or remove sweep directories until the presentation collection is
complete and frozen. The later cleanup starts with a hash/manifests audit,
chooses canonical runs, merges dataset indexes and organization, archives
superseded sweep trees, and deletes nothing until every downstream reference
has been checked.
