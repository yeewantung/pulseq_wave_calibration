#!/usr/bin/env python3
"""Record explicit visual approval of a frozen R1-to-R3 transfer review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument(
        "--confirm-qualitative-transfer-reviewed",
        action="store_true",
        help="Confirm visual review without calculating metrics or retuning lambda.",
    )
    parser.add_argument(
        "--notes",
        default="The regularized reconstruction is visibly smoother and acceptable.",
    )
    return parser.parse_args(argv)


def _validate_recorded_file(path_value: str, expected_sha256: str) -> None:
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Recorded review artifact is missing or changed: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen transfer package and append the user's decision."""
    if not args.confirm_qualitative_transfer_reviewed:
        raise ValueError("Explicit --confirm-qualitative-transfer-reviewed is required")
    manifest_path = args.review_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "awaiting_qualitative_transfer_assessment":
        raise ValueError("Review manifest is not awaiting qualitative assessment")
    if manifest.get("qualitative_transfer_only") is not True:
        raise ValueError("Review manifest is not marked qualitative-transfer-only")
    if manifest.get("regularizer") != "wavelet":
        raise ValueError("R3 transfer approval requires the frozen Wavelet review")

    cases = manifest.get("cases", [])
    if [case.get("lambda_label") for case in cases] != ["0", "1.5e-2"]:
        raise ValueError("Review must compare FISTA lambda zero with Wavelet 1.5e-2")
    if float(cases[1].get("lambda", -1)) != 0.015:
        raise ValueError("Frozen Wavelet lambda is not 1.5e-2")

    source = manifest["lambda_zero_manifest"]
    _validate_recorded_file(source["path"], source["sha256"])
    for case in cases:
        _validate_recorded_file(case["run_manifest"], case["run_manifest_sha256"])
        nifti = case["nifti"]
        _validate_recorded_file(nifti["nifti"], nifti["nifti_sha256"])
        _validate_recorded_file(nifti["sidecar"], nifti["sidecar_sha256"])
    outputs = manifest["outputs"]
    _validate_recorded_file(outputs["comparison"], outputs["comparison_sha256"])
    _validate_recorded_file(
        outputs["relative_l2_plot"], outputs["relative_l2_plot_sha256"]
    )

    manifest["status"] = "qualitative_transfer_approved"
    manifest["approval"] = {
        "decision": "accept_frozen_r1_wavelet_transfer_on_r3",
        "visually_reviewed": True,
        "regularized_result_visibly_smoother": True,
        "metric_ranking_performed": False,
        "lambda_retuned_on_r3": False,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Recorded R3 qualitative-transfer approval: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc
