#!/usr/bin/env python3
"""Record visual approval of generated brain masks.

Run this only after looking at each ``brain_mask_hdbet_qc.png``. Approval is
what lets downstream metrics score against a mask; without it they refuse. Each
mask's digest is re-checked here, so a mask replaced or edited after generation
cannot be approved by accident.

With ``--list`` nothing is written: it just reports what exists and its status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from brain_mask_batch.layout import (  # noqa: E402
    STATUS_APPROVED,
    approve,
    discover_targets,
    is_complete,
    read_sidecar,
)


def main(argv: Sequence[str] | None = None) -> int:
    """List or approve the masks discovered under a share.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Zero once the requested masks are listed or approved.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.list and not args.approved_by:
        parser.error("--approved-by is required unless --list is given")

    targets = discover_targets(
        args.reconstruction_root,
        subjects=args.subjects,
        contrasts=args.contrasts,
        output_root=args.output_root,
    )
    present = [target for target in targets if is_complete(target)]
    if not present:
        raise SystemExit(
            f"No complete brain masks found under {args.reconstruction_root}. "
            "Run generate_brain_masks.py first."
        )

    if args.list:
        _report(present)
        return 0

    approved = 0
    for target in present:
        payload = read_sidecar(target.sidecar_path)
        if payload.get("approved") and not args.reapprove:
            print(f"  {target.subject:8s} {target.contrast:16s} already approved")
            continue
        approve(target.sidecar_path, approved_by=args.approved_by, note=args.note)
        approved += 1
        print(f"  {target.subject:8s} {target.contrast:16s} approved")

    print(f"\nApproved {approved} of {len(present)} masks as {args.approved_by!r}.")
    return 0


def _report(targets) -> None:
    """Print each mask's size and review status."""
    print(f"{'subject':10s} {'contrast':18s} {'voxels':>10s} {'mL':>8s}  status")
    for target in targets:
        payload = read_sidecar(target.sidecar_path)
        status = payload.get("status", "?")
        marker = "" if status == STATUS_APPROVED else "   <- review"
        print(
            f"{target.subject:10s} {target.contrast:18s} "
            f"{int(payload.get('voxel_count', 0)):>10d} "
            f"{float(payload.get('volume_ml', 0.0)):>8.1f}  {status}{marker}"
        )
        print(f"           QC: {target.qc_path}")


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reconstruction_root",
        type=Path,
        help="Share the masks were generated against.",
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--contrasts", nargs="+", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Mirror root used at generation time, if masks are not beside the data.",
    )
    parser.add_argument(
        "--approved-by",
        default="",
        help="Who reviewed the QC figures; recorded in every sidecar.",
    )
    parser.add_argument("--note", default="", help="Optional reviewer comment.")
    parser.add_argument(
        "--reapprove",
        action="store_true",
        help="Re-stamp masks that are already approved.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Report masks and their status without approving anything.",
    )
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
