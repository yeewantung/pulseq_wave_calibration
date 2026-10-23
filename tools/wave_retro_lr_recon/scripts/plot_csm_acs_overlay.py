#!/usr/bin/env python3
"""Overlay completed CSM crop variants on same-basis ACS magnitude; never run BART."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wave_retro_lr.csm_overlay import plot_acs_overlay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse plotting options, create an additive PNG/JSON pair and return a shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True, help="Existing user-confirmed CSM cache directory")
    parser.add_argument("--slices", nargs=3, type=int, metavar=("RO", "LIN", "PAR"))
    parser.add_argument("--percentile", type=float, default=99.5, help="Shared whole-volume ACS upper-window percentile")
    parser.add_argument("--gamma", type=float, default=.5, help="Common display exponent, never modifies ACS data")
    parser.add_argument("--alpha", type=float, default=.3, help="Soft-weight overlay opacity")
    args = parser.parse_args(argv)
    try:
        result = plot_acs_overlay(args.diagnostics, slices=args.slices, percentile=args.percentile,
                                  gamma=args.gamma, alpha=args.alpha)
        print(f"ACS/CSM overlay: {result['figure']['path']}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
