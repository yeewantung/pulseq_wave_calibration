#!/usr/bin/env python3
"""Record explicit visual approval of the R3 BET mask and L/R orientation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-mask-manifest", required=True, type=Path)
    parser.add_argument("--orientation-report", required=True, type=Path)
    parser.add_argument(
        "--confirm-reviewed-mask-and-lr",
        action="store_true",
        help="Confirm that the BET boundary and labeled L/R figures were visually reviewed.",
    )
    parser.add_argument("--notes", default="Explicit approval recorded by the R3 presentation runner.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_reviewed_mask_and_lr:
        raise ValueError("Explicit --confirm-reviewed-mask-and-lr is required")
    mask_manifest_path = args.brain_mask_manifest.expanduser().resolve()
    orientation_report_path = args.orientation_report.expanduser().resolve()
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    orientation_report = json.loads(orientation_report_path.read_text(encoding="utf-8"))
    if orientation_report.get("status") != "orientation_approved" or not orientation_report.get(
        "decision_fields", {}
    ).get("user_approved_best_signed_axis_mapping"):
        raise ValueError("Orientation report must already contain explicit signed-axis approval")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    mask_manifest["status"] = "approved"
    mask_manifest["approval"] = {
        "mask_boundary_visually_approved": True,
        "left_right_orientation_visually_approved": True,
        "orientation_report": str(orientation_report_path),
        "approved_mapping": orientation_report["decision_fields"]["approved_mapping"],
        "reviewed_at_utc": reviewed_at,
        "notes": args.notes,
    }
    _write_json(mask_manifest_path, mask_manifest)
    print(f"Recorded R3 presentation visual QC approval: {mask_manifest_path}")
    return mask_manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
