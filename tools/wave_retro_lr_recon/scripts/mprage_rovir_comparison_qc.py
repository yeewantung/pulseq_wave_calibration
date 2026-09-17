#!/usr/bin/env python3
"""Create fixed-window QC for ROVir-24 and ROVir-48 Wave-MPRAGE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_control import write_mprage_rovir_comparison_qc  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the ROVir comparison-QC command interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rovir24_nifti", type=Path)
    parser.add_argument("rovir48_nifti", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write fixed-window QC and print its manifest.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful QC export.
    """
    args = _parser().parse_args(argv)
    manifest = write_mprage_rovir_comparison_qc(
        args.rovir24_nifti, args.rovir48_nifti, args.output_directory
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
