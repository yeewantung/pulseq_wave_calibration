#!/usr/bin/env python3
"""Record one explicit, hash-bound MPRAGE ROVir reconstruction choice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_selection import record_mprage_rovir_selection  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """Build the explicit ROVir selection-recording interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feasibility_root", type=Path)
    parser.add_argument("reconstruction_root", type=Path)
    parser.add_argument("comparison_qc_manifest", type=Path)
    parser.add_argument("--selection-scope", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--virtual-coils", required=True, type=int)
    parser.add_argument("--ecalib-crop", required=True, type=float)
    parser.add_argument("--reviewer-note", required=True)
    parser.add_argument("--confirm-user-selection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and record one user-approved ROVir choice.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the immutable selection record is written.
    """
    args = _parser().parse_args(argv)
    manifest = record_mprage_rovir_selection(
        args.feasibility_root,
        args.reconstruction_root,
        args.comparison_qc_manifest,
        selection_scope=args.selection_scope,
        candidate_id=args.candidate_id,
        virtual_coils=args.virtual_coils,
        ecalib_crop=args.ecalib_crop,
        reviewer_note=args.reviewer_note,
        confirm_user_selection=args.confirm_user_selection,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
