# niftis_to_brain_masks_batch

Batch HD-BET brain masks for measured Wave reconstructions. Each mask is cut
from one subject/contrast's fully sampled R3x1 baseline and stored **beside the
data it describes**, so any later analysis reuses it instead of recomputing it.

> Every `path/to/...` below is a placeholder. **Replace it with the real path on
> your machine** — for example
> `~/Library/CloudStorage/Dropbox-PartnersHealthCare/<user>/SubtleMRI_DataShare/LowDose_R01_DataShare/waveCAIPI_scans`.
> Quote any path containing spaces.

## Requirements

HD-BET must be installed in the active environment; this tool shells out to it.

```bash
pip install HD-BET
hd-bet --help          # confirm it is on PATH
```

It pulls in nnU-Net and torch. When adding it to an environment that already has
torch, constrain the version so it is not upgraded underneath other work:

```bash
printf 'torch==2.10.0\n' > /tmp/constraints.txt
pip install -c /tmp/constraints.txt HD-BET
```

## Generate

```bash
python scripts/generate_brain_masks.py "path/to/waveCAIPI_scans"
```

| option | meaning |
| --- | --- |
| `--subjects FMP_199 …` | subjects to mask; default every `FMP_*` found |
| `--contrasts MPRAGE_preGad …` | contrasts to mask; default every contrast holding a baseline |
| `--output-root PATH` | mirror the `<subject>/<contrast>/masks` tree elsewhere instead of writing beside the data |
| `--force` | regenerate existing masks, discarding their approval |
| `--verify-baseline` | also re-hash baselines when deciding to skip |
| `--device {auto,mps,cpu,cuda}` | prediction device; default `auto` |
| `--enable-tta` | eight mirrored passes: better masks, ~8× the compute |
| `--hd-bet NAME` | HD-BET executable name, if not `hd-bet` |
| `--verbose` | ask HD-BET for verbose progress |

## Where masks go

Next to the head mask already shipped with the share:

```
waveCAIPI_scans/FMP_199/MPRAGE_preGad/masks/
    head_mask_from_normal.nii.gz     shipped with the share
    brain_mask_hdbet.nii.gz          this tool
    brain_mask_hdbet.json            provenance and approval
    brain_mask_hdbet_qc.png          boundary figure for visual review
```

That means **this writes into the reconstruction share**, which for the
PartnersHealthCare folder syncs to everyone who has it. `--output-root` mirrors
the same `<subject>/<contrast>/masks` layout somewhere else if you want to stage
a run first; consumers take a matching flag to read from the mirror.

Note this differs from `niftis_to_slides`, whose `--output-root` is the parent of
a single created `slides_output/` folder. Here the root *is* the top of the
mirrored tree, with no wrapper folder, so it lines up with the share it mirrors:
`path/to/output/FMP_199/MPRAGE_preGad/masks/brain_mask_hdbet.nii.gz`.

The sidecar records the mask's SHA-256, its voxel count and volume, the baseline
it was cut from with that file's SHA-256, and a `generator` block naming the
HD-BET version, the device and whether test-time augmentation was on.

## Reuse, and when a mask is stale

A subject is skipped when its mask, sidecar and QC figure are all present and
the sidecar still matches the mask on disk. `--force` regenerates anyway,
discarding approval.

Because each sidecar records the baseline's digest, a mask left behind by a
superseded reconstruction can be detected. That check is off by default, since
hashing a baseline forces a download of an online-only Dropbox file;
`--verify-baseline` turns it on.

## Nothing is approved automatically

Masks land as `visual_review_required`. Look at every `brain_mask_hdbet_qc.png`
— three orientations, three slices each, boundary in green, every panel labelled
L/R/A/P/S/I so a left-right flip cannot slip through — then:

```bash
python scripts/approve_brain_masks.py "path/to/waveCAIPI_scans" --list
python scripts/approve_brain_masks.py "path/to/waveCAIPI_scans" \
    --approved-by "Your Name" \
    --note "what you actually checked"
```

`--list` reports every mask, its volume and its status without writing anything.
`--reapprove` re-stamps masks that are already approved. `--subjects`,
`--contrasts` and `--output-root` narrow the set exactly as they do above.

Approval is enforced by consumers, not merely advertised: `niftis_to_slides`
raises rather than scoring against an unapproved mask, and re-checks the mask's
digest, so a mask edited after approval is refused.

## Scope and validation

Contrasts and branches are discovered, not hard-coded. `MPRAGE_*` resolves via
`optimal_wavelet`, the GRE/SWI trees via `selected_wavelet`, and the baseline is
matched on `part-mag` so the differing filename stems (`sub-normal_…` versus
`sub-native_r3x1_…`) both work. On the current share that discovers 47
baselines: 30 MPRAGE and 17 SWI. Narrow a run with `--subjects` / `--contrasts`.

Every mask is checked before it is installed. It must land on exactly the
baseline's shape and affine — HD-BET writes on its input grid, but that is
asserted rather than assumed — and must cover between 200k voxels and 80% of the
volume. A subject that fails is reported and the batch continues, so one bad
scan does not cost the whole run; the exit status is non-zero if any failed.

## Device and runtime

`--device auto` is the default: it takes the best device this machine has, in
the order cuda, mps, cpu. An explicit `--device mps` on a machine without Metal
falls back to cpu with a warning rather than failing, so the same command works
on an Apple silicon laptop, a CUDA box and a plain CPU server.

On Apple silicon that resolves to Metal, which is markedly faster than CPU.
nnU-Net logs `perform_everything_on_device=True is only supported for cuda
devices! Setting this to False` and does resampling and export on CPU while the
network runs on Metal; that is expected, not a fault, and explains the worker
processes it spawns.

The **first** run downloads roughly 110 MB of weights before predicting
anything, which on a slow link can take longer than the masking itself and
consumes no CPU while it happens. HD-BET's output is streamed rather than
captured so that download stays visible instead of looking like a hang.

Test-time augmentation is off by default; `--enable-tta` turns on the eight
mirrored passes for slightly better masks at roughly eight times the compute.

## What the masks look like

On the 30 MPRAGE baselines of the current share, masks run 1373–1763 mL
(mean ≈ 1570). That is generous for brain-only tissue, which is HD-BET's normal
behaviour: it keeps CSF and some dura.

Post-contrast masks come out larger than their pre-contrast pair in **all 14
subjects** that have both, by ≈ 60 mL. Gadolinium brightens vessels and dura at
exactly the boundary the network is deciding. It does not invalidate anything,
but brain-mask metrics for postGad include marginally more non-brain tissue than
preGad, so cross-contrast comparisons carry that small bias; within-contrast
comparisons do not.

## Tests

```bash
python -m pytest tests/ -q
```

Discovery, reuse detection, staleness, the approval gate and device fallback are
covered against temporary trees; no test touches real data or runs HD-BET.
