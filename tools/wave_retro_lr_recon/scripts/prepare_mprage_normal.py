#!/usr/bin/env python3
"""Prepare native measured Wave-MPRAGE BART inputs from TWIX data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.mprage import prepare_normal_mprage  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI paths and prepare native measured-Wave BART inputs.

    Args:
        argv: Optional argument vector; ``None`` reads the process arguments.

    Returns:
        Zero after preparation completes successfully.
    """
    args = _parser().parse_args(argv)
    prepare_normal_mprage(args.twix, args.output, args.seq, reuse=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the TWIX, output-root, and sequence command-line interface.

    Returns:
        Configured argument parser for normal MPRAGE input preparation.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("twix", type=Path, help="Wave-encoded Siemens TWIX file.")
    parser.add_argument("output", type=Path, help="User-selected dataset output root.")
    parser.add_argument("seq", type=Path, help="Matching integrated Wave-MPRAGE sequence.")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
