# niftis_to_slides

Turn measured Wave-MPRAGE reconstruction NIfTIs into a comparison slide deck:
three-orientation center-slice strips, optional masked NRMSE / SSIM / PSNR, and
a PowerPoint deck built to the reviewed slide pattern.

> Every `path/to/...` below is a placeholder. **Replace it with the real path on
> your machine** — for example
> `~/Library/CloudStorage/Dropbox-PartnersHealthCare/<user>/SubtleMRI_DataShare/LowDose_R01_DataShare/waveCAIPI_scans`.
> Quote any path containing spaces.

## Retrospective comparisons without R3x1 data

For a collection containing only retrospective reconstructions, use the separate
visual-comparison entry point (supports MPRAGE and SWI magnitude images):

```bash
python scripts/build_retrospective_slides.py "path/to/collection" \
    --output "path/to/collection/nifti_comparison.pptx"
```

It discovers `<subject>/<contrast>/<container>/<branch>/retro/<variant>/`
automatically, keeping echoes and reconstruction suffixes in separate comparison
groups. All magnitude inputs in this layout are included; phase images are
excluded. Each group prefers `native_r3x2` as its **display reference** (override
with `--reference-variant`), falling back to its first naturally sorted variant.
There is no fully sampled reference and no accuracy scoring.

Each group shares a canonical RAS grid, a display window (foreground p99.5),
and least-squares intensity matching. Anisotropic voxel dimensions are respected
in the sagittal/coronal/axial strips. `--scale none` disables gain matching;
`--display-percentile` adjusts the shared window. The template and +20%
brightness curve are the same as for the reference-based pipeline.

The PPTX is written directly at `--output` (default:
`<input>/nifti_comparison.pptx`). A sibling `<deck-stem>_assets/` directory holds
16-bit TIFF strips and `manifest.csv`, recording source paths relative to the
input, the display reference, window, gain and any geometry resampling.

To insert R3x1 wavelet baselines into an already reviewed retrospective deck:

```bash
python scripts/insert_baseline_slides.py "path/to/collection" \
    --reference-root "path/to/reference-share" \
    --deck "path/to/collection/nifti_comparison.pptx" \
    --output "path/to/updated/nifti_comparison.pptx"
```

The original deck and its sibling `<deck-stem>_assets/manifest.csv` are used to
match each comparison group to a unique wavelet `normal/` reconstruction in the
same subject, contrast and container (e.g. `vcc24_original`), with the same echo
and acquisition tags. Missing or ambiguous matches stop the run. One baseline
page is inserted before each group's retrospective pages, which retain their
original slide XML and embedded images. Baselines are mapped to each group's
existing display grid, window and gain-matching mode, then rendered with the
same physical aspect ratio, 13-inch centered picture and +20% brightness curve.
No accuracy metrics are computed. New TIFFs and `baseline_manifest.csv` go into
the output deck's sibling assets directory; original comparison assets are not
copied when output is written elsewhere.

To add the original no-regularization FISTA reconstructions after each uncond
pair in a deck that already includes baselines:

```bash
python scripts/insert_fista_slides.py "path/to/collection" \
    --reference-root "path/to/reference-share" \
    --deck "path/to/collection/nifti_comparison.pptx" \
    --output "path/to/updated/nifti_comparison.pptx"
```

This reads the original retrospective `manifest.csv` beside the input deck
(override with `--manifest` if needed). Raw `FISTAR0` magnitude files must match
the same subject, contrast, container, variant, echo and acquisition; model
outputs are excluded. All matches and insertion positions are checked before
rendering. Each group's order becomes R3x1 wavelet, R3x2 uncond, R3x3 uncond,
R3x2 FISTA, R3x3 FISTA. The new pages retain the group's existing display grid,
window, intensity matching, physical aspect ratio, +20% brightness and layout.
Existing slide XML and embedded images are preserved. New TIFFs are written
under `original_fista/` in the output assets directory, with a separate
`fista_manifest.csv` recording their provenance and display settings.

## All-subject R3x1 wavelet deck

```bash
python scripts/build_wavelet_slides.py "path/to/reference-share" \
    --output "path/to/reference-share/all_subjects_r3x1_wavelet.pptx"
```

This creates one image page per subject/contrast/echo/acquisition, with its
identifiers in the title and no divider or comparison pages. Duplicate versions
are resolved in this order: `original_nifti/rovir_optimal_wavelet`,
`vcc24_original/wavelet_selected_vcc24`, `original_nifti/optimal_wavelet`, then
`original_nifti/selected_wavelet`. ROVir versions are labelled in the page
title. Masked copies, phase images, model outputs and retrospective cases are
excluded. Ambiguous matches or subjects lacking any supported baseline stop
the run, rather than silently dropping a subject.

Each volume defines its own canonical RAS display grid and foreground p99.5
window (`--display-percentile` to change it); its gain is 1. The renderer retains
physical voxel aspect ratios and uses the same +20% brightness and centered
13-inch picture layout. The sibling assets directory contains 16-bit TIFFs,
`manifest.csv` with provenance/display settings, and `selection.csv` showing all
supported unmasked candidates and which one was selected for each scan.

## Two shares are required

The super-resolution share holds SR outputs only. It contains **no** `normal/`
baselines and **no** masks, so it cannot supply the R3x1 wavelet baseline that
defines the strip dimensions, the display window and the metric reference. Both
roots are therefore mandatory:

```bash
python scripts/build_slides.py \
    "path/to/waveCAIPI_scans_subtle" \
    --reference-root "path/to/waveCAIPI_scans" \
    --output-root "path/to/output"
```

The positional input share decides **which** subjects and contrasts appear. The
reference share supplies the baseline, the Pre-SR reconstructions and the masks.

`--output-root` is the **parent** of the `slides_output/` folder the tool
creates; it does not need to exist beforehand. It defaults to the input share,
which for the PartnersHealthCare folder means writing several hundred megabytes
into a shared clinical-data directory. Point it somewhere else unless you mean
to publish the outputs.

## Options

| option | meaning |
| --- | --- |
| `--reference-root PATH` | **required**; share holding baselines, Pre-SR volumes and masks |
| `--output-root PATH` | parent of the created `slides_output/`; default the input share |
| `--subjects FMP_199 …` | subjects to include; default every `FMP_*` in the input share |
| `--contrasts MPRAGE_preGad …` | contrasts to include |
| `--metric-masks head brain …` | foregrounds to score over; default `head` |
| `--display-mask {head,intensity,brain}` | foreground setting the window and display scaling; default `head` |
| `--brain-mask-root PATH` | mirror root for brain masks; default reads beside the data |
| `--no-metrics` | skip scoring, captions and the CSV |
| `--no-summary` | skip the cross-dataset charts and the evaluation section |
| `--mask-fraction F` | intensity-mask threshold as a fraction of the baseline p99; default 0.05 |
| `--scale {none,lsq,median,percentile}` | intensity match before scoring and slicing; default `lsq` |
| `--interpolation-order {0,1,3}` | spline order onto the baseline grid; default 3 |
| `--display-percentile P` | baseline percentile used as the window top; default 99.5 |
| `--deck-name NAME.pptx` | filename of the deck inside `slides_output/` |
| `--template PATH` | styling template; default `assets/slide_template.pptx` |

## Outputs

Everything lands in a `slides_output/` folder created under `--output-root`,
so pointing two runs at different roots keeps their results side by side:

```
<output-root>/slides_output/
    pptx_img/               16-bit LZW TIFF strips, one per indexed volume
    summary/                cross-dataset charts, one per metric and mask
    metrics.csv             one row per scored volume per mask
    wave_sr_comparison.pptx the deck, opening with an evaluation section
```

The brightened 8-bit rasters PowerPoint embeds are built in memory, so no
duplicate image files are written.

Image filenames carry the full provenance, for example
`FMP199_preGad_fista_r0_lr_x_1p5mm_r3x2_Post-SR_shd.tiff`: subject, contrast,
regularization branch, retrospective case (which itself encodes native versus
low resolution and the acceleration), SR stage and SR model.

`metrics.csv` is long format — one row per volume **per mask**, labelled by
`mask_mode` and `mask_source` — so several masks never collide in one row.

## The slide pattern

| element | value |
| --- | --- |
| evaluation | the deck opens with one section of cross-dataset charts, three per scored mask |
| section | one `Section Header` per subject/contrast, `FMP 199 MPRAGE preGad` |
| first slide | the fully sampled baseline, `Wave R3x1 – wavelet`, uncaptioned |
| comparison | `Wave Fake R3x2 – LR x 1.5mm, no reg, SHD` |
| order | retro case > regularization > SR stage (no SR, SHD, retrained U-Net) |
| picture | width 13 in, exactly centered, sent to back |
| brightness | +20%, baked into the raster and recorded for the pane |
| axis group | copied onto low-resolution slides only |
| caption | one line per scored mask, below the picture |

A caption with both masks reads:

```
 head mask     NRMSE 0.0743     SSIM 0.8720     PSNR 31.50 dB
brain mask     NRMSE 0.0563     SSIM 0.9317     PSNR 29.38 dB
```

(taken verbatim from `FMP 199 MPRAGE preGad`, `1mm iso, no reg, no SR`)

Every title word is derived, not tabled: the native case is named from its
measured voxel size and the acceleration comes from the case folder name, so a
new `lr_z_2mm_r4x1` case labels itself without a code change.

## Why brightness is baked in

PowerPoint renders the pixels it stores and treats `brightnessContrast` as pane
state. A picture whose raster is uncorrected therefore renders uncorrected no
matter what the Format Picture pane reads — it only becomes correct once the
pane value is touched, which rewrites the raster.

So the correction is applied to the embedded PNG, using PowerPoint's own +20%
curve rather than a flat gain. That curve (`POWERPOINT_BRIGHT20_LUT`) was
measured by pairing the rasters of a hand-corrected deck against the strips they
were made from; it sits near a 1.25x gain but rolls off in shadows and
highlights. Regenerating hand-corrected images reproduces them to within 2/255,
with 97% of pixels within 1/255.

The LUT is calibrated for **+20% only**. A different percentage would need its
own measurement.

Because these pictures carry no `.wdp` backup layer, changing the brightness
value on a generated slide applies the new correction on top of the baked one.
Rebuild the deck instead of adjusting brightness in PowerPoint.

## Metrics

Metrics are never taken over the whole field of view: air outnumbers anatomy and
would inflate PSNR and SSIM. Each metric runs inside a foreground mask, and 3D
SSIM is averaged over mask voxels rather than over the bounding box, so
background contributes nothing.

| mask | meaning |
| --- | --- |
| `head` | the whole-head mask shipped with the reference share; falls back to `intensity` when absent |
| `intensity` | baseline samples above `--mask-fraction` of their own p99 |
| `brain` | the approved mask written beside the reconstruction by `niftis_to_brain_masks_batch` |

`--metric-masks` takes several: each scored mask gets its own labelled CSV rows,
its own caption line on the slide and its own summary charts. `--display-mask`
is separate and defaults to `head`, so changing what is scored never re-windows
the pictures.

Brain masks are read from `<subject>/<contrast>/masks/brain_mask_hdbet.nii.gz`
inside the reference share, or from a mirror given by `--brain-mask-root`. This
tool never creates one, and refuses to score against a mask that has not been
visually approved or that changed after approval.

`--no-metrics` skips scoring, captions and the CSV. It does **not** change the
images: the display mask and the least-squares intensity match still set the
window and the rendering, so strips look the same either way.

Volumes are resampled onto the baseline grid (cubic by default) and intensity
matched inside the mask before scoring, so a global gain difference between
reconstructions is not charged as error.

### Head and brain masks answer different questions

The head mask includes scalp and skull, which are bright in T1 and blur badly
under low-resolution acquisition. Restricting to brain roughly halves the
apparent resolution penalty — on the current data the LR-to-native NRMSE ratio
falls from 2.71× (head) to 1.79× (brain). Scoring both and comparing is usually
more informative than picking one.

PSNR is the one metric whose head/brain ordering is not stable: the brain mask
lowers the p99 peak (no bright scalp fat), which pushes PSNR down, while also
removing edge-concentrated error, which pushes it up. For LR cases the second
effect dominates, for native cases the first does. NRMSE and SSIM carry no such
ambiguity.

## End-to-end with brain masks

```bash
# 1. once per share: generate and approve brain masks
python ../niftis_to_brain_masks_batch/scripts/generate_brain_masks.py \
    "path/to/waveCAIPI_scans" --contrasts MPRAGE_preGad MPRAGE_postGad
python ../niftis_to_brain_masks_batch/scripts/approve_brain_masks.py \
    "path/to/waveCAIPI_scans" --approved-by "Your Name"

# 2. any number of times: build decks scoring both masks
python scripts/build_slides.py \
    "path/to/waveCAIPI_scans_subtle" \
    --reference-root "path/to/waveCAIPI_scans" \
    --output-root "path/to/output" \
    --metric-masks head brain
```

## Styling template

`assets/slide_template.pptx` carries the theme, the slide layouts and the x/y
axis indicator group, on one `TEMPLATE ASSETS` slide that the builder drops
after copying the group. Nothing about the styling is synthesised in code.

To restyle, edit that template in PowerPoint, keeping the `Title and Content`
and `Section Header` layouts and a group named `Group 8`.

## Scope

The reference-based pipeline is MPRAGE only. SWI is deliberately excluded there:
it uses a `selected_wavelet` branch,
different retrospective case names, and a baseline file named `sub-native_r3x1_…`
rather than `sub-normal_…`, so it needs its own naming map. (The masking tool
does handle SWI, since it only needs the baseline.)

Strip width follows the baseline grid, so subjects whose baseline matrix differs
produce differently sized pictures at the fixed 13 in width — a 220-row baseline
renders 4.78 in tall against 5.20 in for a 192-row one.

## Tests

```bash
python -m pytest tests/ -q
```

The suite builds miniature shares and synthetic volumes on a temporary path; it
never touches the real data.
