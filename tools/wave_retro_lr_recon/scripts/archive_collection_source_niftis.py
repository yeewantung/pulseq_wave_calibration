#!/usr/bin/env python3
"""Archive source NIfTI pairs after verifying byte-identical collection copies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.collection_archive import (  # noqa: E402
    archive_collection_source_niftis,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and optionally archive one output root's source NIfTIs.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Zero after printing the JSON-native archival summary.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report exact source files without deleting them.",
    )
    parser.add_argument(
        "--trust-manifest-hashes",
        action="store_true",
        help=(
            "Require matching recorded hashes and copy presence without rereading "
            "all payloads; use only after representative full SHA-256 checks."
        ),
    )
    args = parser.parse_args(argv)
    result = archive_collection_source_niftis(
        args.output_root,
        dry_run=args.dry_run,
        verify_hashes=not args.trust_manifest_hashes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
