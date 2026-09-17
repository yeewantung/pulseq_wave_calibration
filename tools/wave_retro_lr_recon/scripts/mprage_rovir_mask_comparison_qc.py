#!/usr/bin/env python3
"""Create scale-restored QC for two ROVir-24 negative-ROI choices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_control import write_mprage_rovir_mask_comparison_qc  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the negative-ROI comparison interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ro000_030_nifti", type=Path)
    parser.add_argument("ro000_020_nifti", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write negative-ROI comparison QC.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful QC export.
    """
    args = _parser().parse_args(argv)
    manifest = write_mprage_rovir_mask_comparison_qc(
        args.ro000_030_nifti, args.ro000_020_nifti, args.output_directory
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
