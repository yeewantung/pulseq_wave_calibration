#!/usr/bin/env python3
"""Migrate one MPRAGE NIfTI collection to original-only storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.collection_archive import prune_mprage_head_masks  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Parse migration options and prune one collection's mask products.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Zero after printing the migration summary.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--validate-hashes",
        action="store_true",
        help="Recompute every existing collection hash before migration.",
    )
    args = parser.parse_args(argv)
    result = prune_mprage_head_masks(
        args.output_root, validate_hashes=args.validate_hashes
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
