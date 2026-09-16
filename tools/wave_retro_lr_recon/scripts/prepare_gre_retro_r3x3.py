#!/usr/bin/env python3
"""Prepare only native-grid R3x3 retrospective multi-echo Wave-GRE inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre import prepare_retro_gre_r3x3  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the native-R3x3 GRE preparation CLI.

    Returns:
        Parser for measured GRE TWIX, shared output root, and matching sequence.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "twix", type=Path, help="Measured regular-R3x1 multi-echo Wave-GRE TWIX file."
    )
    parser.add_argument("output", type=Path, help="User-selected reconstruction root.")
    parser.add_argument("seq", type=Path, help="Matching integrated Wave-GRE sequence.")
    parser.add_argument(
        "--psf-coefficient-processing",
        choices=("smooth", "sine-line"),
        default="sine-line",
        help=(
            "Existing normal PSF processing contract; sine-line uses automatic "
            "bounds when neither manual kx bound is supplied."
        ),
    )
    parser.add_argument("--psf-fit-kx-min", type=int)
    parser.add_argument("--psf-fit-kx-max", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and prepare native-R3x3 GRE inputs.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after compatible multi-echo inputs are ready.
    """

    args = _parser().parse_args(argv)
    prepare_retro_gre_r3x3(
        args.twix,
        args.output,
        args.seq,
        psf_coefficient_processing=args.psf_coefficient_processing,
        psf_fit_kx_min=args.psf_fit_kx_min,
        psf_fit_kx_max=args.psf_fit_kx_max,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
