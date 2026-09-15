#!/usr/bin/env python3
"""Render Wave-MPRAGE reconstruction NIfTIs into a comparison slide deck."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from nifti_slides.discovery import CONTRAST_ORDER  # noqa: E402
from nifti_slides.metrics import MASK_MODES, SCALE_MODES  # noqa: E402
from nifti_slides.pipeline import RunOptions, run  # noqa: E402


DEFAULT_TEMPLATE = TOOL_ROOT / "assets" / "slide_template.pptx"


def main(argv: Sequence[str] | None = None) -> int:
    """Index both shares, render center-slice strips and assemble the deck.

    Args:
        argv: Optional command-line argument vector; ``None`` reads process
            arguments.

    Returns:
        Zero once the deck, the strips and any metric table are written.
    """
    args = _parser().parse_args(argv)

    options = RunOptions(
        output_root=args.output_root or args.input_root,
        template_path=args.template,
        deck_name=args.deck_name,
        subjects=args.subjects,
        contrasts=args.contrasts,
        with_metrics=not args.no_metrics,
        metric_masks=args.metric_masks,
        display_mask=args.display_mask,
        brain_mask_root=args.brain_mask_root,
        mask_fraction=args.mask_fraction,
        scale_mode=args.scale,
        interpolation_order=args.interpolation_order,
        display_percentile=args.display_percentile,
        with_summary=not args.no_summary,
    )
    result = run(args.input_root, args.reference_root, options)

    print(
        f"\n{result.section_count} sections, {result.slide_count} picture slides, "
        f"{result.metric_rows} metric rows, {len(result.summary_paths)} charts."
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "Share whose FMP_* subjects drive the run, typically the "
            "super-resolution share."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help=(
            "Share holding the fully sampled R3x1 baselines, the unmodified "
            "retrospective reconstructions and the whole-head masks. The SR "
            "share carries none of these."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Folder receiving slides_output/; defaults to the input share, "
            "which for a shared clinical folder means writing into it."
        ),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Styling template carrying the theme, layouts and axis indicator.",
    )
    parser.add_argument(
        "--deck-name",
        default="wave_sr_comparison.pptx",
        help="Filename of the assembled deck inside slides_output/.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject folders to process; defaults to every FMP_* in the input share.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=list(CONTRAST_ORDER),
        help="Contrast folders to process.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip scoring, slide captions and the metric CSV.",
    )
    parser.add_argument(
        "--metric-masks",
        nargs="+",
        choices=list(MASK_MODES),
        default=["head"],
        help=(
            "Foregrounds to score over. Each one gets its own labelled CSV "
            "rows, its own caption line and its own summary charts."
        ),
    )
    parser.add_argument(
        "--display-mask",
        choices=list(MASK_MODES),
        default="head",
        help=(
            "Foreground setting the grayscale window and the display intensity "
            "match. Held apart from --metric-masks so changing what is scored "
            "never changes what the pictures look like."
        ),
    )
    parser.add_argument(
        "--brain-mask-root",
        type=Path,
        default=None,
        help=(
            "Approved output of prepare_brain_masks.py; required whenever "
            "'brain' is scored or displayed."
        ),
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the cross-dataset charts and the evaluation section.",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.05,
        help="Intensity-mask threshold as a fraction of the baseline p99.",
    )
    parser.add_argument(
        "--scale",
        choices=list(SCALE_MODES),
        default="lsq",
        help="Intensity match applied inside the mask before scoring and slicing.",
    )
    parser.add_argument(
        "--interpolation-order",
        type=int,
        choices=[0, 1, 3],
        default=3,
        help="Spline order used to resample onto the baseline grid.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.5,
        help="Baseline percentile used as the shared grayscale window top.",
    )
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotImplementedError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
