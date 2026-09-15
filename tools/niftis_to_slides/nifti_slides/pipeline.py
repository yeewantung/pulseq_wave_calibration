"""Drive NIfTI discovery, strip rendering, metrics and deck assembly.

Outputs land under ``<output_root>/slides_output``::

    pptx_img/    16-bit center-slice strips, one per indexed volume
    summary/     cross-dataset grouped bar charts, one per metric and mask
    metrics.csv  one row per scored volume and mask, written when metrics run
    <name>.pptx  the assembled deck

The brightened 8-bit rasters PowerPoint embeds are built in memory and handed
straight to the deck, so no duplicate image files are left on disk.

One subject/contrast is processed at a time, so only its baseline, its masks and
one candidate volume are held in memory at once.

Masks have two separate jobs. The *display* mask sets the grayscale window and
the intensity match applied before slicing, and is built whether or not metrics
run, so ``--no-metrics`` changes only the numbers and never the pictures. The
*scoring* masks are independent and may be several: each is scored, each gets
its own labelled CSV rows, its own caption line and its own summary charts.
Adding or switching a scoring mask therefore never re-windows the images.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import nibabel as nib

from . import brain_masks as brain_masks_module
from . import deck as deck_module
from . import images as images_module
from . import metrics as metrics_module
from . import summary as summary_module
from .discovery import (
    CONTRAST_ORDER,
    Case,
    group_by_section,
    index_cases,
    subject_contrast_dir,
)


SLIDES_DIRNAME = "slides_output"
IMAGE_DIRNAME = "pptx_img"
SUMMARY_DIRNAME = "summary"
METRICS_FILENAME = "metrics.csv"
EVALUATION_SECTION = "Evaluation across all datasets"

METRIC_FIELDS = (
    "subject",
    "subject_number",
    "contrast",
    "branch",
    "variant",
    "stage",
    "sr_model",
    "image_file",
    "mask_mode",
    "mask_source",
    "mask_voxels",
    "nrmse",
    "ssim_3d_mask",
    "ssim_3d_bbox",
    "psnr_p99_db",
    "intensity_scale",
    "scale_mode",
    "display_mask_mode",
    "baseline_path",
    "resampled_to_baseline",
    "interpolation_order",
    "shape_native",
    "voxel_size_native_mm",
    "source_path",
)


@dataclass(frozen=True)
class RunOptions:
    """Everything the pipeline needs beyond the two share roots.

    Attributes:
        output_root: Folder that receives ``slides_output``.
        template_path: Styling template deck.
        deck_name: Filename of the assembled deck.
        subjects: Subjects to process, or ``None`` for every subject found.
        contrasts: Contrasts to process.
        with_metrics: Whether to score volumes, caption slides and write the CSV.
        metric_masks: Masks to score over; every one gets its own CSV rows,
            caption line and charts.
        display_mask: Mask that sets the grayscale window and the display
            intensity match. Held apart from ``metric_masks`` so that changing
            what is scored never changes what the pictures look like.
        brain_mask_root: Mirror root holding approved brain masks in the same
            ``<subject>/<contrast>/masks`` layout. Leave unset to read the masks
            stored beside the reconstructions, which is where the masking tool
            writes them.
        mask_fraction: Intensity-mask threshold as a fraction of the p99.
        scale_mode: Intensity match applied before scoring and before slicing.
        interpolation_order: Spline order used to reach the baseline grid.
        display_percentile: Baseline percentile used as the display window top.
        with_summary: Whether to write cross-dataset charts and lead the deck
            with an evaluation section.
    """

    output_root: Path
    template_path: Path
    deck_name: str = "wave_sr_comparison.pptx"
    subjects: Sequence[str] | None = None
    contrasts: Sequence[str] = field(default=CONTRAST_ORDER)
    with_metrics: bool = True
    metric_masks: Sequence[str] = field(default=("head",))
    display_mask: str = "head"
    brain_mask_root: Path | None = None
    mask_fraction: float = 0.05
    scale_mode: str = "lsq"
    interpolation_order: int = 3
    display_percentile: float = 99.5
    with_summary: bool = True


@dataclass(frozen=True)
class RunResult:
    """What one run produced.

    Attributes:
        deck_path: The assembled deck.
        image_dir: Folder holding the 16-bit strips.
        metrics_path: The metrics CSV, or ``None`` when metrics were skipped.
        summary_paths: Cross-dataset charts written, in deck order.
        section_count: Subject/contrast sections written.
        slide_count: Picture slides written.
        metric_rows: Rows scored, one per volume per mask.
    """

    deck_path: Path
    image_dir: Path
    metrics_path: Path | None
    summary_paths: list[Path]
    section_count: int
    slide_count: int
    metric_rows: int


def run(
    input_root: Path,
    reference_root: Path,
    options: RunOptions,
    *,
    log: Callable[[str], None] = print,
) -> RunResult:
    """Render every indexed volume and assemble the comparison deck.

    Args:
        input_root: Share whose subjects drive the run, typically the SR share.
        reference_root: Share holding the fully sampled baselines and the
            unmodified retrospective reconstructions.
        options: Run configuration.
        log: Progress sink.

    Returns:
        A summary of what was written.

    Raises:
        ValueError: If no volumes were indexed.
    """
    slides_dir = options.output_root / SLIDES_DIRNAME
    image_dir = slides_dir / IMAGE_DIRNAME
    image_dir.mkdir(parents=True, exist_ok=True)

    cases = index_cases(
        input_root,
        reference_root,
        subjects=options.subjects,
        contrasts=options.contrasts,
    )
    if not cases:
        raise ValueError(
            f"No volumes indexed under {input_root} with baselines in {reference_root}."
        )

    sections = group_by_section(cases)
    log(f"Indexed {len(cases)} volumes across {len(sections)} sections.")

    builder = deck_module.DeckBuilder(options.template_path)
    metric_rows: list[dict[str, object]] = []
    slide_count = 0

    for (subject, contrast), section_cases in sections.items():
        title = deck_module.section_title(subject, contrast)
        builder.add_section(title)
        log(f"\n=== {title} ===")
        slide_count += _build_section(
            builder=builder,
            section_cases=section_cases,
            image_dir=image_dir,
            reference_root=reference_root,
            brain_mask_root=options.brain_mask_root,
            options=options,
            metric_rows=metric_rows,
            log=log,
        )

    metrics_path = None
    if options.with_metrics and metric_rows:
        metrics_path = slides_dir / METRICS_FILENAME
        _write_metrics(metrics_path, metric_rows)

    summary_paths = _add_evaluation_section(
        builder, metric_rows, slides_dir, options, log
    )

    deck_path = slides_dir / options.deck_name
    builder.save(deck_path)

    log(f"\nDeck:   {deck_path}")
    log(f"Strips: {image_dir}")
    if metrics_path is not None:
        log(f"Metrics: {metrics_path}")

    return RunResult(
        deck_path=deck_path,
        image_dir=image_dir,
        metrics_path=metrics_path,
        summary_paths=summary_paths,
        section_count=len(sections),
        slide_count=slide_count,
        metric_rows=len(metric_rows),
    )


def _brain_mask_path(
    reference_root: Path, brain_mask_root: Path | None, case: Case
) -> Path | None:
    """Locate the approved brain mask stored beside one reconstruction.

    Masks normally live in the reconstruction share itself, written there by the
    ``niftis_to_brain_masks_batch`` tool. ``brain_mask_root`` overrides that with
    a mirror of the same ``<subject>/<contrast>/masks`` layout.

    Args:
        reference_root: Share holding the reconstructions.
        brain_mask_root: Mirror root, or ``None`` to look beside the data.
        case: Baseline case of the section.

    Returns:
        The approved mask, or ``None`` when none has been generated.

    Raises:
        PermissionError: If a mask exists but is not approved.
        ValueError: If a mask's sidecar is missing, malformed or stale.
    """
    if brain_mask_root is not None:
        contrast_dir = brain_mask_root / case.subject / case.contrast
    else:
        contrast_dir = subject_contrast_dir(
            reference_root, case.subject, case.contrast
        )
    if contrast_dir is None or not contrast_dir.is_dir():
        return None
    return brain_masks_module.find_mask(contrast_dir)


def _build_section(
    *,
    builder: deck_module.DeckBuilder,
    section_cases: Sequence[Case],
    image_dir: Path,
    reference_root: Path,
    brain_mask_root: Path | None,
    options: RunOptions,
    metric_rows: list[dict[str, object]],
    log: Callable[[str], None],
) -> int:
    """Render and add every slide of one subject/contrast.

    Args:
        builder: Deck under construction.
        section_cases: Cases of this section, baseline first.
        image_dir: Folder receiving 16-bit strips.
        reference_root: Share holding the shipped head masks.
        brain_mask_root: Mirror root for brain masks, or None to look beside
            the data.
        options: Run configuration.
        metric_rows: Accumulating metric rows, appended to in place.
        log: Progress sink.

    Returns:
        The number of picture slides added.

    Raises:
        ValueError: If the section does not start with its baseline.
    """
    baseline_case = section_cases[0]
    if not baseline_case.is_baseline:
        raise ValueError(
            f"Section {baseline_case.subject}/{baseline_case.contrast} has no baseline."
        )

    baseline_image, baseline_data = images_module.load_magnitude(baseline_case.path)
    head_mask_path = _head_mask_path(reference_root, baseline_case)
    brain_mask_path = _brain_mask_path(reference_root, brain_mask_root, baseline_case)

    wanted_masks = [options.display_mask]
    if options.with_metrics:
        wanted_masks += [mode for mode in options.metric_masks]

    foregrounds: dict[str, metrics_module.Foreground] = {}
    for mode in wanted_masks:
        if mode in foregrounds:
            continue
        foregrounds[mode] = metrics_module.build_foreground(
            baseline_data,
            baseline_image,
            mode=mode,
            head_mask_path=head_mask_path,
            brain_mask_path=brain_mask_path,
            intensity_fraction=options.mask_fraction,
        )
        log(
            f"    mask {mode:9s} {foregrounds[mode].source} "
            f"({foregrounds[mode].voxel_count} voxels)"
        )

    display = foregrounds[options.display_mask]
    grid = images_module.build_baseline_grid(
        baseline_case.path,
        display.mask,
        display_percentile=options.display_percentile,
    )
    log(f"    baseline {grid.data.shape}, display window {grid.display_max:.4g}")

    added = 0
    for case in section_cases:
        image = nib.load(str(case.path))
        volume, resampled = images_module.resample_to_baseline(
            image, grid.image, interpolation_order=options.interpolation_order
        )

        captions: list[str] = []
        if case.is_baseline:
            shown = volume
        else:
            shown, _ = metrics_module.scale_to_baseline(
                grid.data, volume, display.mask, options.scale_mode
            )
            if options.with_metrics:
                for mode in options.metric_masks:
                    foreground = foregrounds[mode]
                    scored = metrics_module.compute_metrics(
                        grid.data,
                        volume,
                        foreground.mask,
                        scale_mode=options.scale_mode,
                    )
                    captions.append(
                        deck_module.format_caption(
                            mode, scored.nrmse, scored.ssim_mask, scored.psnr_db
                        )
                    )
                    metric_rows.append(
                        _metric_row(
                            case=case,
                            scored=scored,
                            mask_mode=mode,
                            foreground=foreground,
                            resampled=resampled,
                            options=options,
                            baseline_path=baseline_case.path,
                        )
                    )

        _, strip = images_module.orientation_strip(shown)
        tiff_path = image_dir / f"{case.image_stem}.tiff"
        width_px, height_px = images_module.write_strip_tiff(
            strip, grid.display_max, tiff_path
        )

        raster, raster_width, raster_height = deck_module.brightened_png_bytes(tiff_path)
        builder.add_picture_slide(
            deck_module.SlideSpec(
                title=_title_for(case),
                image=raster,
                width_px=raster_width,
                height_px=raster_height,
                with_axes=_wants_axes(case),
                captions=tuple(captions),
            )
        )
        added += 1
        log(f"    {case.image_stem}" + (f"  [{len(captions)} mask]" if captions else ""))

    return added


def _add_evaluation_section(
    builder: deck_module.DeckBuilder,
    metric_rows: Sequence[dict[str, object]],
    slides_dir: Path,
    options: RunOptions,
    log: Callable[[str], None],
) -> list[Path]:
    """Chart every scored mask and lead the deck with those charts.

    Args:
        builder: Deck under construction, with its sections already added.
        metric_rows: Every metric row of the run.
        slides_dir: Output folder receiving ``summary/``.
        options: Run configuration.
        log: Progress sink.

    Returns:
        The chart paths written, in deck order.
    """
    if not options.with_summary or not metric_rows:
        return []

    summary_dir = slides_dir / SUMMARY_DIRNAME
    charts: list[tuple[str, Path]] = []
    for mode in options.metric_masks:
        charts.extend(
            summary_module.write_summary_charts(
                metric_rows, summary_dir, mask_mode=mode
            )
        )
    if not charts:
        return []

    builder.add_section(EVALUATION_SECTION)
    for title, path in charts:
        builder.add_figure_slide(title, path)
    builder.move_last_slides_to_front(len(charts) + 1)

    log(f"\nSummary charts: {summary_dir} ({len(charts)})")
    return [path for _, path in charts]


def _title_for(case: Case) -> str:
    """Slide title for one case.

    Args:
        case: Indexed volume.

    Returns:
        The baseline title, or the comparison title built from the case.
    """
    if case.is_baseline:
        return deck_module.BASELINE_TITLE
    return deck_module.comparison_title(
        case.variant, case.branch, case.stage, case.sr_model, _voxel_size(case)
    )


def _wants_axes(case: Case) -> bool:
    """Whether the x/y axis indicator belongs on this slide.

    The indicator marks which in-plane direction was downsampled, so it belongs
    on the low-resolution cases only.

    Args:
        case: Indexed volume.

    Returns:
        True for retrospective low-resolution cases.
    """
    return not case.is_baseline and not case.variant.startswith("native")


def _voxel_size(case: Case) -> tuple[float, float, float]:
    """Voxel size of a reconstruction, in millimetres.

    Args:
        case: Indexed volume.

    Returns:
        The three voxel dimensions.
    """
    zooms = nib.load(str(case.path)).header.get_zooms()[:3]
    return tuple(float(value) for value in zooms)


def _head_mask_path(reference_root: Path, case: Case) -> Path | None:
    """Locate the whole-head mask shipped beside a reconstruction.

    Args:
        reference_root: Share holding the masks.
        case: Baseline case of the section.

    Returns:
        The mask path, or ``None`` when the share carries none.
    """
    directory = subject_contrast_dir(reference_root, case.subject, case.contrast)
    if directory is None:
        return None
    candidate = directory / metrics_module.HEAD_MASK_RELATIVE
    return candidate if candidate.is_file() else None


def _metric_row(
    *,
    case: Case,
    scored: metrics_module.VolumeMetrics,
    mask_mode: str,
    foreground: metrics_module.Foreground,
    resampled: bool,
    options: RunOptions,
    baseline_path: Path,
) -> dict[str, object]:
    """Assemble one CSV row, labelled by the mask it was scored over.

    Args:
        case: Scored volume.
        scored: Its metrics.
        mask_mode: Mask the metrics were taken over.
        foreground: That mask.
        resampled: Whether the volume needed resampling onto the baseline grid.
        options: Run configuration.
        baseline_path: The baseline it was scored against.

    Returns:
        One row keyed by :data:`METRIC_FIELDS`.
    """
    header = nib.load(str(case.path))
    return {
        "subject": case.subject,
        "subject_number": case.subject_number,
        "contrast": case.contrast.replace("MPRAGE_", ""),
        "branch": case.branch,
        "variant": case.variant,
        "stage": case.stage,
        "sr_model": case.sr_model,
        "image_file": f"{case.image_stem}.tiff",
        "mask_mode": mask_mode,
        "mask_source": foreground.source,
        "mask_voxels": foreground.voxel_count,
        "nrmse": scored.nrmse,
        "ssim_3d_mask": scored.ssim_mask,
        "ssim_3d_bbox": scored.ssim_bbox,
        "psnr_p99_db": scored.psnr_db,
        "intensity_scale": scored.intensity_scale,
        "scale_mode": options.scale_mode,
        "display_mask_mode": options.display_mask,
        "baseline_path": str(baseline_path),
        "resampled_to_baseline": resampled,
        "interpolation_order": options.interpolation_order,
        "shape_native": str(tuple(int(value) for value in header.shape)),
        "voxel_size_native_mm": str(
            tuple(round(float(value), 4) for value in header.header.get_zooms()[:3])
        ),
        "source_path": str(case.path),
    }


def _write_metrics(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write the metric table in deck order.

    Args:
        path: Destination CSV.
        rows: Metric rows, one per volume per scored mask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
