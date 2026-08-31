#!/usr/bin/env python3
"""Prepare native and direct-crop LR R3x2 Wave-MPRAGE BART inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.mprage import prepare_retro_mprage  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI paths and prepare native plus retrospective MPRAGE inputs.

    Args:
        argv: Optional argument vector; ``None`` reads the process arguments.

    Returns:
        Zero after all BART input sets are ready.
    """
    args = _parser().parse_args(argv)
    prepare_retro_mprage(args.twix, args.output, args.seq)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the retrospective MPRAGE preparation command-line interface.

    Returns:
        Configured parser for TWIX, shared output root, and sequence paths.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("twix", type=Path, help="Wave-encoded Siemens TWIX file.")
    parser.add_argument(
        "output", type=Path, help="Same dataset output root used by the normal script."
    )
    parser.add_argument("seq", type=Path, help="Matching integrated Wave-MPRAGE sequence.")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
