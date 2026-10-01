#!/usr/bin/env python3
"""Write fixed-normalization review figures comparing one arm with the baseline.

Observation boxes are inclusive voxel ranges of the RAS-stored NIfTI, given
explicitly on the command line. Implementation: ``wave_retro_lr.intervention_qc``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import intervention_qc  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser for one review.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path, help="Accepted magnitude NIfTI or its directory.")
    parser.add_argument("--candidate", required=True, type=Path, help="Arm magnitude NIfTI or its directory.")
    parser.add_argument("--candidate-label", required=True, help="Short arm label for figure rows.")
    parser.add_argument("--output-dir", required=True, type=Path, help="QC directory of the arm.")
    parser.add_argument("--central", required=True, help="Central box r0:r1,a0:a1,s0:s1 (inclusive).")
    parser.add_argument("--metal", required=True, help="Metal-context box r0:r1,a0:a1,s0:s1 (inclusive).")
    parser.add_argument("--air-si-min", required=True, type=int, help="Lowest S-I index of the air band.")
    parser.add_argument("--slices", type=int, default=5, help="Slices per montage (default 5).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one review and print its normalization and statistics.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the review is written.
    """
    args = _parser().parse_args(argv)
    manifest = intervention_qc.write_review(
        args.baseline,
        args.candidate,
        args.candidate_label,
        args.output_dir,
        central=intervention_qc.parse_box(args.central),
        metal=intervention_qc.parse_box(args.metal),
        air_si_min=args.air_si_min,
        count=args.slices,
    )
    summary = {
        "status": manifest["status"],
        "normalization": manifest["normalization"],
        "slices": manifest["slices"],
        "statistics": manifest["statistics"],
        "figures": sorted(manifest["figures"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
