#!/usr/bin/env python3
"""Record explicit approval of a manifested native/matched visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument(
        "--approval-statement",
        required=True,
        help="Exact or concise user statement conveying approval.",
    )
    parser.add_argument(
        "--approval-source",
        default="explicit user response",
        help="Non-identifying description of where approval was received.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    review_path = args.review_manifest.expanduser().resolve()
    review = _load_json(review_path)
    if review.get("status") != "complete":
        raise ValueError("Review manifest is not complete.")
    output_records = {
        Path(record["path"]).name: record for record in review.get("outputs", [])
    }
    required = {"native_grid_comparison.png", "matched_1mm_grid_comparison.png"}
    if not required.issubset(output_records):
        raise ValueError("Review manifest lacks the native or matched figure.")
    figure_records = []
    for name in sorted(required):
        record = output_records[name]
        path = Path(record["path"]).expanduser().resolve()
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"Review figure hash changed: {path}")
        figure_records.append({"path": str(path), "sha256": digest})
    output_path = (
        review_path.parent / "visual_approval.json"
        if args.output is None
        else args.output.expanduser().resolve()
    )
    if output_path.exists():
        raise FileExistsError(f"Approval record already exists: {output_path}")
    payload = {
        "format_version": 1,
        "status": "approved",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_source": str(args.approval_source),
        "approval_statement": str(args.approval_statement),
        "review_manifest": {
            "path": str(review_path),
            "sha256": sha256_file(review_path),
        },
        "approved_outputs": [
            "native_grid_comparison",
            "matched_grid_comparison",
        ],
        "figures": figure_records,
        "authorization": (
            "Visual-review gate only; permits descriptive resolution-tradeoff "
            "metrics under the frozen scientific constraints."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"Visual approval record: {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
