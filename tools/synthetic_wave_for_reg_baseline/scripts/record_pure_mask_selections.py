#!/usr/bin/env python3
"""Record five explicit user-reviewed pure-mask selections without ranking."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pure_mask_rerun import CASE_IDS, load_json, sha256_file, validate_config, write_json_atomic


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and record explicit manual selections.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after a complete hash-bound decision record is written.
    """
    args = _parser().parse_args(argv)
    record(
        args.config,
        confirmed_output_root=args.confirm_output_root,
        confirm_manual_visual_review=args.confirm_manual_visual_review,
        reviewer_note=args.reviewer_note,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the explicit selection-recording interface.

    Returns:
        Parser requiring output-root and visual-review acknowledgements.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--confirm-output-root", required=True, type=Path)
    parser.add_argument("--confirm-manual-visual-review", action="store_true")
    parser.add_argument("--reviewer-note", required=True)
    return parser


def _utc_now() -> str:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        ISO-8601 UTC string.
    """
    return datetime.now(timezone.utc).isoformat()


def _normalized_setting(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one explicit method/block/lambda selection.

    Args:
        item: Local configuration selection object.

    Returns:
        JSON-native setting matching sweep manifests.
    """
    return {
        "method": str(item["method"]),
        "block_size": None if item.get("block_size") is None else int(item["block_size"]),
        "lambda": float(item["lambda"]),
    }


def record(
    config_path: str | Path,
    *,
    confirmed_output_root: Path,
    confirm_manual_visual_review: bool,
    reviewer_note: str,
) -> dict[str, Any]:
    """Hash-bind five manual choices to reviewed shortlist candidates.

    Args:
        config_path: Ignored local rerun configuration containing decisions.
        confirmed_output_root: Exact previously approved workflow root.
        confirm_manual_visual_review: Explicit acknowledgement of visual review.
        reviewer_note: Nonempty human review note stored with the decisions.

    Returns:
        Completed selection manifest.
    """
    if not confirm_manual_visual_review:
        raise ValueError("--confirm-manual-visual-review is required.")
    if not reviewer_note.strip():
        raise ValueError("--reviewer-note must be nonempty.")
    validated = validate_config(config_path)
    root = Path(validated["layout"]["root"])
    if confirmed_output_root.expanduser().resolve() != root:
        raise ValueError("Confirmed output root differs from the local rerun configuration.")
    shortlist_path = root / "evaluation" / "review" / "shortlist_manifest.json"
    shortlist = load_json(shortlist_path, "manual shortlist manifest")
    if shortlist.get("status") != "complete" or shortlist.get(
        "automatic_shortlist_selection_performed"
    ) is not False:
        raise ValueError("A complete explicitly manual shortlist is required.")
    raw = validated["config"]["snapshot"].get("evaluation", {}).get(
        "manual_final_selections"
    )
    if not isinstance(raw, Mapping) or set(raw) != set(CASE_IDS):
        raise ValueError("evaluation.manual_final_selections must contain all five cases.")
    shortlist_settings = shortlist["inputs"]["manual_shortlist"]
    sweep_binding = shortlist["inputs"]["sweep_manifest"]
    evaluation_binding = shortlist["inputs"]["evaluation_manifest"]
    sweep_path = Path(sweep_binding["path"])
    evaluation_path = Path(evaluation_binding["path"])
    if (
        sha256_file(sweep_path) != sweep_binding["sha256"]
        or sha256_file(evaluation_path) != evaluation_binding["sha256"]
    ):
        raise ValueError("Reviewed shortlist inputs changed after rendering.")
    sweep = load_json(sweep_path, "shortlist sweep")
    selections = {}
    for case_id in CASE_IDS:
        setting = _normalized_setting(raw[case_id])
        if setting not in shortlist_settings[case_id]:
            raise ValueError(f"{case_id} final selection is absent from its reviewed shortlist.")
        record_item = next(
            item
            for item in sweep["candidate_manifests"]
            if item["case_id"] == case_id and item["setting"] == setting
        )
        candidate_path = Path(record_item["manifest"])
        if sha256_file(candidate_path) != record_item["manifest_sha256"]:
            raise ValueError(f"{case_id} selected candidate manifest changed.")
        candidate = load_json(candidate_path, "selected candidate manifest")
        for part in ("magnitude", "phase"):
            output = candidate["outputs"][part]
            if sha256_file(output["path"]) != output["sha256"]:
                raise ValueError(f"{case_id} selected candidate {part} changed.")
        selections[case_id] = {
            "setting": setting,
            "candidate_manifest": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "magnitude": candidate["outputs"]["magnitude"],
            "phase": candidate["outputs"]["phase"],
        }
    output_path = root / "evaluation" / "review" / "selection_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"Selection record already exists: {output_path}")
    completed = {
        "format_version": 1,
        "status": "complete",
        "shortlist_manifest": {
            "path": str(shortlist_path),
            "sha256": sha256_file(shortlist_path),
        },
        "selection_method": "explicit manual visual and metric tradeoff review",
        "reviewer_note": reviewer_note.strip(),
        "automatic_selection_performed": False,
        "composite_score_used": False,
        "selections": selections,
        "recorded_at_utc": _utc_now(),
    }
    write_json_atomic(output_path, completed)
    print(f"Pure-mask manual selection manifest: {output_path}")
    return completed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
