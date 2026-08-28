#!/usr/bin/env python3
"""Record an explicit, hash-bound choice from a regularization metric table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic


REPORTED_METRICS = (
    "nrmse_brain",
    "ssim_3d_brain_bbox",
    "ncc_brain",
    "gradient_ncc_brain_edge",
    "edge_preservation_ratio",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _magnitude_record(manifest: dict[str, Any]) -> dict[str, Any]:
    magnitude = next(
        item for item in manifest.get("nifti_outputs", []) if item.get("part") == "mag"
    )
    path = Path(magnitude["nifti"]).expanduser().resolve()
    if sha256_file(path) != magnitude["nifti_sha256"]:
        raise ValueError(f"Magnitude NIfTI is missing or changed: {path}")
    return _record(path)


def _metric_row(path: Path, case_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one metric row for {case_id}")
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the chosen run and write the user decision without retuning."""
    if not args.confirm_user_selection:
        raise ValueError("Explicit --confirm-user-selection is required")
    provenance_path = args.metrics_provenance.expanduser().resolve()
    selected_manifest_path = args.selected_manifest.expanduser().resolve()
    baseline_manifest_path = args.baseline_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    provenance = _read_json(provenance_path)
    selected_manifest = _read_json(selected_manifest_path)
    baseline_manifest = _read_json(baseline_manifest_path)
    if provenance.get("status") != "complete":
        raise ValueError("Metric evaluation is not complete")
    if selected_manifest.get("status") != "complete" or baseline_manifest.get(
        "status"
    ) != "complete":
        raise ValueError("Selected and baseline reconstructions must be complete")

    metrics_path = Path(provenance["metrics_csv"]["path"]).resolve()
    if sha256_file(metrics_path) != provenance["metrics_csv"]["sha256"]:
        raise ValueError("Metric CSV is missing or changed")
    selected_metrics = _metric_row(metrics_path, args.selected_case_id)
    baseline_metrics = _metric_row(metrics_path, args.baseline_case_id)
    selected_nifti = _magnitude_record(selected_manifest)
    baseline_nifti = _magnitude_record(baseline_manifest)
    if selected_metrics["source_nifti_sha256"] != selected_nifti["sha256"]:
        raise ValueError("Selected metric row does not match the selected NIfTI")
    if baseline_metrics["source_nifti_sha256"] != baseline_nifti["sha256"]:
        raise ValueError("Baseline metric row does not match the baseline NIfTI")

    config = selected_manifest["config"]
    payload = {
        "format_version": 1,
        "status": "user_selected",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "case_id": args.selected_case_id,
            "regularizer": config["regularizer"],
            "lambda": config["lambda"],
            "lambda_label": config["lambda_label"],
            "optimizer": config["optimizer"],
            "backend": config["backend"],
        },
        "selected_reconstruction": {
            "manifest": _record(selected_manifest_path),
            "magnitude_nifti": selected_nifti,
        },
        "baseline": {
            "case_id": args.baseline_case_id,
            "manifest": _record(baseline_manifest_path),
            "magnitude_nifti": baseline_nifti,
            "metrics": {
                name: float(baseline_metrics[name]) for name in REPORTED_METRICS
            },
        },
        "evidence": {
            "metrics_provenance": _record(provenance_path),
            "metrics_csv": _record(metrics_path),
            "selected_metrics": {
                name: float(selected_metrics[name]) for name in REPORTED_METRICS
            },
        },
        "decision": {
            "approval_source": "Explicit user decision after quantitative review",
            "rationale": args.rationale,
            "automatic_selection_performed": False,
        },
    }
    if output_path.exists() and not args.refresh:
        raise FileExistsError(f"Selection record exists; use --refresh: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    print(f"Recorded regularization selection: {output_path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-provenance", required=True, type=Path)
    parser.add_argument("--selected-case-id", required=True)
    parser.add_argument("--selected-manifest", required=True, type=Path)
    parser.add_argument("--baseline-case-id", required=True)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--confirm-user-selection", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, StopIteration, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
