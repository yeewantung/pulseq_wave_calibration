#!/usr/bin/env python3
"""Build a validated, unmasked GRE magnitude/phase NIfTI collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre_nifti_collection import build_gre_nifti_collection  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the GRE collection command-line parser.

    Returns:
        Parser requiring one GRE reconstruction output root.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        help=(
            "GRE reconstruction root containing normal and optional retro outputs; "
            "the collection is written below nifti_collection."
        ),
    )
    parser.add_argument(
        "--require-retro",
        action="store_true",
        help="Require native-R3x2 and LIN-low-resolution-R3x2 outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the collection requested by command-line arguments.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Zero after successful validation and collection creation.
    """

    args = _parser().parse_args(argv)
    manifest = build_gre_nifti_collection(
        args.output_root,
        require_retro=args.require_retro,
    )
    print(
        f"GRE NIfTI collection complete: "
        f"{args.output_root.expanduser().resolve() / 'nifti_collection'} "
        f"({manifest['nifti_count']} NIfTIs; no masking)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
