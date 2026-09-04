#!/usr/bin/env python3
"""Generate GRE whole-head-mask candidates for explicit visual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre_head_mask import (  # noqa: E402
    DEFAULT_GRE_CORE_RELATIVE_THRESHOLDS,
    DEFAULT_GRE_MAXIMUM_GROWTH_DISTANCES_MM,
    DEFAULT_GRE_RELATIVE_THRESHOLDS,
    derive_gre_head_mask_candidates,
)


def _parser() -> argparse.ArgumentParser:
    """Build the local GRE head-mask derivation command parser.

    Returns:
        Parser requiring an exact source NIfTI and new output directory.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("magnitude_nifti", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--relative-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_GRE_RELATIVE_THRESHOLDS,
    )
    parser.add_argument(
        "--core-relative-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_GRE_CORE_RELATIVE_THRESHOLDS,
    )
    parser.add_argument(
        "--maximum-growth-distances-mm",
        nargs="+",
        type=float,
        default=DEFAULT_GRE_MAXIMUM_GROWTH_DISTANCES_MM,
    )
    parser.add_argument("--background-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--smoothing-mm", type=float, default=1.0)
    parser.add_argument("--boundary-width-mm", type=float, default=5.0)
    parser.add_argument("--opening-radius-mm", type=float, default=0.0)
    parser.add_argument("--closing-radius-mm", type=float, default=1.5)
    parser.add_argument("--dilation-radius-mm", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the requested non-ranking GRE head-mask sweep.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the complete review package is atomically installed.
    """

    args = _parser().parse_args(argv)
    manifest = derive_gre_head_mask_candidates(
        args.magnitude_nifti,
        args.output_directory,
        relative_thresholds=args.relative_thresholds,
        core_relative_thresholds=args.core_relative_thresholds,
        maximum_growth_distances_mm=args.maximum_growth_distances_mm,
        background_mad_multiplier=args.background_mad_multiplier,
        smoothing_mm=args.smoothing_mm,
        boundary_width_mm=args.boundary_width_mm,
        opening_radius_mm=args.opening_radius_mm,
        closing_radius_mm=args.closing_radius_mm,
        dilation_radius_mm=args.dilation_radius_mm,
    )
    print(
        f"GRE head-mask candidates: {args.output_directory.expanduser().resolve()} "
        f"({manifest['ready_for_visual_review_count']}/{manifest['candidate_count']} ready)"
    )
    print("No candidate was ranked or selected; review overlays/ in all three planes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
