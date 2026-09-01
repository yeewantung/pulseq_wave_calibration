#!/usr/bin/env python3
"""Render an explicitly declared cross-family pure-mask review shortlist."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_pure_mask_sweeps import (
    _load_sweep,
    _plot_family,
    logical_to_physical_xyz,
    scale_candidate_for_display,
)
from pure_mask_rerun import CASE_IDS, load_json, sha256_file, validate_config, write_json_atomic


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and validate or render a manual shortlist.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after validation or figure generation.
    """
    args = _parser().parse_args(argv)
    result = render(
        args.config,
        stage=args.stage,
        validate_only=args.validate_only,
        confirmed_output_root=args.confirm_output_root,
        resume=args.resume,
    )
    if args.validate_only:
        print(
            f"Validated {result['shortlist_candidate_count']} explicitly listed "
            "shortlist candidates; no output was written."
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the cross-family shortlist command interface.

    Returns:
        Parser requiring one completed evaluated sweep.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", default="fine", choices=("coarse", "fine"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _utc_now() -> str:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        ISO-8601 UTC string.
    """
    return datetime.now(timezone.utc).isoformat()


def _shortlist(
    config: Mapping[str, Any], sweep: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Validate a manual shortlist against manifest-listed candidates.

    Args:
        config: Complete ignored local rerun configuration.
        sweep: Completed manifest-backed sweep.

    Returns:
        Ordered per-case shortlist settings.
    """
    evaluation = config.get("evaluation")
    raw = evaluation.get("manual_shortlist") if isinstance(evaluation, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != set(CASE_IDS):
        raise ValueError("evaluation.manual_shortlist must explicitly contain all five cases.")
    available = {
        (
            record["case_id"],
            record["setting"]["method"],
            record["setting"]["block_size"],
            float(record["setting"]["lambda"]),
        )
        for record in sweep["candidate_manifests"]
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for case_id in CASE_IDS:
        settings = raw[case_id]
        if not isinstance(settings, list) or not 2 <= len(settings) <= 6:
            raise ValueError(f"{case_id} manual shortlist must contain two to six candidates.")
        normalized = []
        keys = set()
        for item in settings:
            if not isinstance(item, Mapping):
                raise ValueError(f"{case_id} shortlist entries must be objects.")
            setting = {
                "method": str(item["method"]),
                "block_size": None if item.get("block_size") is None else int(item["block_size"]),
                "lambda": float(item["lambda"]),
            }
            key = (case_id, setting["method"], setting["block_size"], setting["lambda"])
            if key not in available:
                raise ValueError(f"{case_id} shortlist setting is absent from the sweep: {setting}")
            if key in keys:
                raise ValueError(f"{case_id} repeats shortlist setting: {setting}")
            keys.add(key)
            normalized.append(setting)
        result[case_id] = normalized
    return result


def render(
    config_path: str | Path,
    *,
    stage: str,
    validate_only: bool,
    confirmed_output_root: Path | None,
    resume: bool,
) -> dict[str, Any]:
    """Validate and optionally render one manual cross-family shortlist.

    Args:
        config_path: Ignored local rerun configuration.
        stage: Evaluated sweep stage supplying candidates.
        validate_only: Perform no writes when true.
        confirmed_output_root: Exact user-approved root required for writes.
        resume: Reuse a complete identical shortlist package.

    Returns:
        Validation summary or completed shortlist manifest.
    """
    validated = validate_config(config_path)
    sweep_path, sweep = _load_sweep(validated, stage)
    evaluation_path = Path(validated["layout"]["root"]) / "evaluation" / stage / "evaluation_manifest.json"
    evaluation = load_json(evaluation_path, f"pure-mask {stage} evaluation")
    if (
        evaluation.get("status") != "complete"
        or evaluation.get("stage") != stage
        or evaluation.get("scientific_scope", {}).get(
            "automatic_composite_selection_performed"
        )
        is not False
    ):
        raise ValueError("Manual shortlist requires completed non-composite evaluation.")
    evaluation_sweep = evaluation.get("sweep_manifest", {})
    if (
        Path(evaluation_sweep.get("path", "")).resolve() != sweep_path
        or evaluation_sweep.get("sha256") != sha256_file(sweep_path)
    ):
        raise ValueError("Manual shortlist evaluation is not bound to its sweep manifest.")
    shortlist = _shortlist(validated["config"]["snapshot"], sweep)
    count = sum(len(values) for values in shortlist.values())
    if validate_only:
        if confirmed_output_root is not None:
            raise ValueError("--confirm-output-root is not used with --validate-only.")
        return {"status": "validated", "shortlist_candidate_count": count}
    root = Path(validated["layout"]["root"])
    if confirmed_output_root is None or confirmed_output_root.expanduser().resolve() != root:
        raise ValueError("Shortlist rendering requires the exact user-confirmed output root.")
    review_root = root / "evaluation" / "review"
    manifest_path = review_root / "shortlist_manifest.json"
    expected_inputs = {
        "sweep_manifest": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
        "evaluation_manifest": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
        },
        "manual_shortlist": shortlist,
    }
    if manifest_path.is_file() and resume:
        prior = load_json(manifest_path, "shortlist manifest")
        if prior.get("status") == "complete" and prior.get("inputs") == expected_inputs:
            return prior
    if review_root.exists() and any(review_root.iterdir()):
        raise FileExistsError(f"Review output is not safely reusable: {review_root}")
    review_root.mkdir(parents=True, exist_ok=True)
    config = validated["config"]["snapshot"]
    axis_order = config["evaluation"]["logical_to_canonical_axis_order"]
    axis_flips = config["evaluation"]["logical_to_canonical_axis_flips"]
    preparation = load_json(root / "preparation_manifest.json", "preparation manifest")
    figures = []
    for case_id in CASE_IDS:
        case = load_json(preparation["cases"][case_id]["case_manifest"], "prepared case")
        reference = logical_to_physical_xyz(
            np.load(case["direct_fft_reference"]["path"], allow_pickle=False),
            axis_order=axis_order,
            axis_flips=axis_flips,
        )
        selected = []
        for setting in shortlist[case_id]:
            record = next(
                record
                for record in sweep["candidate_manifests"]
                if record["case_id"] == case_id and record["setting"] == setting
            )
            candidate = load_json(record["manifest"], "shortlist candidate")
            magnitude = logical_to_physical_xyz(
                np.load(candidate["outputs"]["magnitude"]["path"], allow_pickle=False),
                axis_order=axis_order,
                axis_flips=axis_flips,
            )
            metric_row = next(
                row
                for row in evaluation["rows"]
                if row["case_id"] == case_id
                and row["method"] == setting["method"]
                and row["block_size"] == setting["block_size"]
                and float(row["lambda"]) == setting["lambda"]
            )
            scale = float(metric_row["intensity_scale_lsq"])
            label = (
                f"{setting['method']} b={setting['block_size']} "
                f"λ={setting['lambda']:g} (LSQ×{scale:.3g})"
            )
            selected.append((label, scale_candidate_for_display(magnitude, scale)))
        figure_path = review_root / f"{case_id}_cross_family_shortlist.png"
        _plot_family(
            figure_path,
            reference,
            selected,
            title=f"{case_id}: manually declared cross-family shortlist",
        )
        figures.append({"path": str(figure_path), "sha256": sha256_file(figure_path)})
    completed = {
        "format_version": 1,
        "status": "complete",
        "inputs": expected_inputs,
        "figures": figures,
        "automatic_shortlist_selection_performed": False,
        "automatic_winner_selection_performed": False,
        "final_selection_requires_explicit_user_visual_review": True,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, completed)
    print(f"Pure-mask manual shortlist manifest: {manifest_path}")
    return completed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
