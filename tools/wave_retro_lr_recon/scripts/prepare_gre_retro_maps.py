#!/usr/bin/env python3
"""Fourier-resample one native GRE CSM set to the exact LIN-low-resolution grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre import prepare_retro_gre_sensitivity_maps  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the low-resolution CSM preparation CLI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Prepared GRE output root.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare low-resolution maps below the selected output root.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero after successful resampling.
    """

    args = _parser().parse_args(argv)
    prepare_retro_gre_sensitivity_maps(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
