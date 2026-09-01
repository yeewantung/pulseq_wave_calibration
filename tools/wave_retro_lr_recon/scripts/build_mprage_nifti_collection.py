#!/usr/bin/env python3
"""Build canonical-copy and whole-head-masked MPRAGE NIfTI collections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.nifti_collection import (  # noqa: E402
    HeadMaskParameters,
    build_mprage_nifti_collection,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse collection options and build the requested MPRAGE derivatives.

    Args:
        argv: Optional command-line argument vector; ``None`` reads process
            arguments.

    Returns:
        Zero after the collection and its manifest are installed successfully.
    """
    args = _parser().parse_args(argv)
    parameters = HeadMaskParameters(
        relative_threshold=args.relative_threshold,
        core_relative_threshold=args.core_relative_threshold,
        maximum_growth_distance_mm=args.maximum_growth_distance_mm,
        background_mad_multiplier=args.background_mad_multiplier,
        smoothing_mm=args.smoothing_mm,
        boundary_width_mm=args.boundary_width_mm,
        opening_radius_mm=args.opening_radius_mm,
        closing_radius_mm=args.closing_radius_mm,
        dilation_radius_mm=args.dilation_radius_mm,
    )
    manifest = build_mprage_nifti_collection(
        args.output_root,
        require_retro=args.require_retro,
        parameters=parameters,
    )
    print(
        "MPRAGE NIfTI collection complete: "
        f"{args.output_root.expanduser().resolve() / 'nifti_collection'} "
        f"({len(manifest['cases'])} case groups)"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the MPRAGE NIfTI collection command-line parser.

    Returns:
        Parser exposing the output root, required-case mode, and conservative
        whole-head mask parameters.
    """
    defaults = HeadMaskParameters()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help=(
            "Reconstruction root containing branch-specific normal/nifti "
            "and optional retro cases."
        ),
    )
    parser.add_argument(
        "--require-retro",
        action="store_true",
        help="Require all four retrospective case NIfTI directories.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=defaults.relative_threshold,
        help="Low foreground threshold relative to the smoothed positive p99.",
    )
    parser.add_argument(
        "--core-relative-threshold",
        type=float,
        default=defaults.core_relative_threshold,
        help="Trusted head-core threshold relative to the smoothed positive p99.",
    )
    parser.add_argument(
        "--maximum-growth-distance-mm",
        type=float,
        default=defaults.maximum_growth_distance_mm,
        help="Maximum low-threshold foreground growth beyond the trusted core.",
    )
    parser.add_argument(
        "--background-mad-multiplier",
        type=float,
        default=defaults.background_mad_multiplier,
        help="Robust boundary-noise multiplier used by the low threshold.",
    )
    parser.add_argument(
        "--smoothing-mm",
        type=float,
        default=defaults.smoothing_mm,
        help="Gaussian smoothing standard deviation in millimetres.",
    )
    parser.add_argument(
        "--boundary-width-mm",
        type=float,
        default=defaults.boundary_width_mm,
        help="Boundary-shell width for robust noise estimation.",
    )
    parser.add_argument(
        "--opening-radius-mm",
        type=float,
        default=defaults.opening_radius_mm,
        help="Physical binary-opening radius; zero disables opening.",
    )
    parser.add_argument(
        "--closing-radius-mm",
        type=float,
        default=defaults.closing_radius_mm,
        help="Physical binary-closing radius.",
    )
    parser.add_argument(
        "--dilation-radius-mm",
        type=float,
        default=defaults.dilation_radius_mm,
        help="Final physical dilation radius; zero disables dilation.",
    )
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
