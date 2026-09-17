#!/usr/bin/env python3
"""Create fixed-window center-slice QC for an MPRAGE PCA control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.pca_control import write_mprage_pca_control_qc  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the Ncc comparison-QC command interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("baseline_nifti", type=Path)
    parser.add_argument("control_nifti", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the fixed-window Ncc=12 versus control figure.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful QC export.
    """
    args = _parser().parse_args(argv)
    manifest = write_mprage_pca_control_qc(
        args.baseline_nifti, args.control_nifti, args.output_directory
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
