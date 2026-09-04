#!/usr/bin/env python3
"""Prepare native measured single- or multi-echo Wave-GRE BART inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre import prepare_normal_gre  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the normal GRE preparation CLI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("twix", type=Path, help="Measured Wave-GRE TWIX file.")
    parser.add_argument("output", type=Path, help="Exact user-selected output root.")
    parser.add_argument("seq", type=Path, help="Matching integrated Wave-GRE sequence.")
    parser.add_argument(
        "--psf-coefficient-processing", choices=("smooth", "sine-line"), default="smooth"
    )
    parser.add_argument("--psf-fit-kx-min", type=int)
    parser.add_argument("--psf-fit-kx-max", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare normal GRE inputs from parsed command-line paths.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero after successful preparation.
    """

    args = _parser().parse_args(argv)
    prepare_normal_gre(
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
