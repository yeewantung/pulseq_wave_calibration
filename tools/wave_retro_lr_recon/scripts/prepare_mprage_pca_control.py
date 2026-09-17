#!/usr/bin/env python3
"""Prepare a higher-channel standard-PCA Wave-MPRAGE control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.pca_control import prepare_mprage_pca_control  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the PCA-control preparation command interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("twix", type=Path)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("accepted_normal_root", type=Path)
    parser.add_argument("feasibility_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--virtual-coils", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one standard-PCA control from the supplied paths.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful preparation.
    """
    args = _parser().parse_args(argv)
    manifest = prepare_mprage_pca_control(
        args.twix,
        args.sequence,
        args.accepted_normal_root,
        args.feasibility_root,
        args.output_root,
        virtual_coils=args.virtual_coils,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
