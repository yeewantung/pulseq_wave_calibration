#!/usr/bin/env python3
"""Prepare only native-grid R3x3 retrospective Wave-MPRAGE BART inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.mprage import prepare_retro_mprage_r3x3  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the native R3x3 preparation command-line interface.

    Returns:
        Parser for R3x1 TWIX, shared output root, and matching sequence.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "twix", type=Path, help="Native R1 or residue-1 R3x1 Wave-MPRAGE TWIX file."
    )
    parser.add_argument("output", type=Path, help="User-selected dataset output root.")
    parser.add_argument("seq", type=Path, help="Matching integrated Wave-MPRAGE sequence.")
    parser.add_argument(
        "--psf-coefficient-processing",
        choices=("smooth", "sine-line"),
        default="sine-line",
        help=(
            "PSF coefficient processing. Sine-line selects its range automatically "
            "unless both manual kx bounds are supplied."
        ),
    )
    parser.add_argument("--psf-fit-kx-min", type=int)
    parser.add_argument("--psf-fit-kx-max", type=int)
    parser.add_argument("--psf-fit-y-min", type=int)
    parser.add_argument("--psf-fit-y-max", type=int)
    parser.add_argument("--psf-fit-z-min", type=int)
    parser.add_argument("--psf-fit-z-max", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and prepare one native R3x3 case.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after compatible BART inputs are ready.
    """
    args = _parser().parse_args(argv)
    prepare_retro_mprage_r3x3(
        args.twix,
        args.output,
        args.seq,
        psf_coefficient_processing=args.psf_coefficient_processing,
        psf_fit_kx_min=args.psf_fit_kx_min,
        psf_fit_kx_max=args.psf_fit_kx_max,
        psf_fit_y_min=args.psf_fit_y_min,
        psf_fit_y_max=args.psf_fit_y_max,
        psf_fit_z_min=args.psf_fit_z_min,
        psf_fit_z_max=args.psf_fit_z_max,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
