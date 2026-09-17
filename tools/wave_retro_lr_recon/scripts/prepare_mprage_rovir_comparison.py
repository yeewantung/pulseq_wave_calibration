#!/usr/bin/env python3
"""Prepare matched ROVir-24 and ROVir-48 Wave-MPRAGE inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_control import prepare_mprage_rovir_comparison  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the ROVir comparison preparation interface.

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
    parser.add_argument("--channel-counts", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--partition-progress-interval", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare comparison inputs and print their shared manifest.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful preparation.
    """
    args = _parser().parse_args(argv)
    manifest = prepare_mprage_rovir_comparison(
        args.twix,
        args.sequence,
        args.accepted_normal_root,
        args.feasibility_root,
        args.output_root,
        channel_counts=args.channel_counts,
        partition_progress_interval=args.partition_progress_interval,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
