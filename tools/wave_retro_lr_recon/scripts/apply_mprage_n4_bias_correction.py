#!/usr/bin/env python3
"""Create an optional ANTs N4 derivative from canonical MPRAGE NIfTIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.n4_bias_correction import (  # noqa: E402
    N4Parameters,
    apply_mprage_n4_bias_correction,
    apply_mprage_shared_n4_bias_field,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and create or validate one N4 derivative.

    Args:
        argv: Optional command-line arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful creation or exact reuse.
    """

    args = _parser().parse_args(argv)
    if args.reference_derivative is not None:
        manifest = apply_mprage_shared_n4_bias_field(
            args.source_nifti_directory,
            args.output_directory,
            args.reference_derivative,
        )
    else:
        manifest = apply_mprage_n4_bias_correction(
            args.source_nifti_directory,
            args.output_directory,
            executable=args.executable,
            parameters=N4Parameters(
                shrink_factor=args.shrink_factor,
                bspline_spacing_mm=args.bspline_spacing_mm,
                convergence_iterations=tuple(args.convergence_iterations),
                convergence_threshold=args.convergence_threshold,
            ),
        )
    print(
        "MPRAGE N4 derivative ready: "
        f"{args.output_directory.expanduser().resolve()} "
        f"({manifest['status']})"
    )
    return 0


def _iterations(value: str) -> list[int]:
    """Parse an ``x``-separated positive iteration schedule.

    Args:
        value: Text such as ``50x50x50x50``.

    Returns:
        Positive iteration counts.

    Raises:
        argparse.ArgumentTypeError: If parsing or validation fails.
    """

    try:
        result = [int(token) for token in value.split("x")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("iterations must be x-separated integers") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("iterations must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the opt-in N4 command-line parser.

    Returns:
        Parser with source, destination, executable, and fitting parameters.
    """

    defaults = N4Parameters()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_nifti_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--reference-derivative",
        type=Path,
        help=(
            "reuse a validated normal-reconstruction N4 field in physical space "
            "instead of fitting a new field"
        ),
    )
    parser.add_argument("--executable", default="N4BiasFieldCorrection")
    parser.add_argument("--shrink-factor", type=int, default=defaults.shrink_factor)
    parser.add_argument(
        "--bspline-spacing-mm", type=float, default=defaults.bspline_spacing_mm
    )
    parser.add_argument(
        "--convergence-iterations",
        type=_iterations,
        default=list(defaults.convergence_iterations),
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=defaults.convergence_threshold,
    )
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
