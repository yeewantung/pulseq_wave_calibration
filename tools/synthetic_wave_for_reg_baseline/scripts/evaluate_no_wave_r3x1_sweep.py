#!/usr/bin/env python3
"""Plot no-Wave R3x1 Wavelet metrics against the direct-FFT reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bart_cfl import sha256_file
from presentation_metrics import validate_metrics_reference_manifest


METRIC_FIELDS = (
    "intensity_scale_lsq",
    "ncc_brain",
    "ncc_full_fov",
    "nrmse_brain",
    "mae_brain",
    "nmae_brain",
    "psnr_p99_db",
    "ssim_3d_brain_bbox",
    "ssim_axial_brain_mean",
    "ssim_axial_brain_slice_count",
    "gradient_ncc_brain_edge",
    "edge_preservation_ratio",
    "background_std_normalized_p99",
    "background_mean_abs_normalized_p99",
    "background_p95_abs_normalized_p99",
    "anatomy_missed_brain_fraction",
)

PLOT_METRICS = (
    ("nrmse_brain", "Brain NRMSE ↓", "min"),
    ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑", "max"),
    ("gradient_ncc_brain_edge", "Edge gradient NCC ↑", "max"),
    ("edge_preservation_ratio", "Edge magnitude ratio → 1", "one"),
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validated_record(manifest_path: Path, reference_hash: str) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Reconstruction manifest is not complete: {manifest_path}")
    direct = manifest.get("direct_fft_metrics", {})
    if direct.get("status") != "complete":
        raise ValueError(f"Direct-FFT metrics are incomplete: {manifest_path}")
    if direct.get("metrics_reference_manifest", {}).get("sha256") != reference_hash:
        raise ValueError(f"Metrics-reference hash differs: {manifest_path}")
    nifti = manifest.get("nifti", {})
    magnitude = Path(nifti.get("magnitude_nifti", ""))
    if not magnitude.is_file():
        raise FileNotFoundError(magnitude)
    magnitude_hash = sha256_file(magnitude)
    if magnitude_hash != nifti.get("magnitude_nifti_sha256"):
        raise ValueError(f"Magnitude NIfTI hash differs: {magnitude}")
    if direct.get("candidate", {}).get("magnitude_nifti_sha256") != magnitude_hash:
        raise ValueError(f"Metrics are not bound to the magnitude NIfTI: {manifest_path}")
    metrics = direct.get("metrics", {})
    if any(
        field not in metrics or not math.isfinite(float(metrics[field]))
        for field in METRIC_FIELDS
    ):
        raise ValueError(f"Metric dictionary is incomplete or non-finite: {manifest_path}")

    config = manifest.get("config", {})
    regularizer = str(config.get("regularizer", ""))
    if regularizer == "cg_sense":
        method = "cg_sense"
        lambda_value = None
    elif regularizer == "wavelet":
        method = "wavelet"
        lambda_value = float(config["lambda"])
        if not math.isfinite(lambda_value) or lambda_value <= 0:
            raise ValueError(f"Invalid Wavelet lambda: {manifest_path}")
    elif manifest.get("purpose") == "no-Wave R3x1 presentation GRAPPA reconstruction":
        method = "grappa"
        lambda_value = None
    else:
        raise ValueError(f"Unrecognized no-Wave reconstruction method: {manifest_path}")
    return {
        "method": method,
        "lambda": lambda_value,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "magnitude_nifti": str(magnitude),
        "magnitude_nifti_sha256": magnitude_hash,
        **{field: float(metrics[field]) for field in METRIC_FIELDS},
    }


def metric_leaders(wavelet_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return descriptive per-metric leaders without selecting a lambda."""
    if not wavelet_records:
        raise ValueError("At least one Wavelet record is required.")
    leaders: dict[str, Any] = {}
    for field, _label, objective in PLOT_METRICS:
        if objective == "min":
            leader = min(wavelet_records, key=lambda row: row[field])
        elif objective == "max":
            leader = max(wavelet_records, key=lambda row: row[field])
        else:
            leader = min(wavelet_records, key=lambda row: abs(row[field] - 1.0))
        leaders[field] = {
            "lambda": leader["lambda"],
            "value": leader[field],
            "objective": objective,
        }
    return leaders


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "method",
        "lambda",
        "manifest",
        "manifest_sha256",
        "magnitude_nifti",
        "magnitude_nifti_sha256",
        *METRIC_FIELDS,
    )
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _plot(
    path_png: Path,
    path_pdf: Path,
    wavelet: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    leaders: dict[str, Any],
    highlight_lambda: float,
    presentation_selection: bool,
) -> None:
    colors = {"cg_sense": "#444444", "grappa": "#7a5195"}
    labels = {"cg_sense": "CG-SENSE control", "grappa": "GRAPPA control"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    lambdas = [row["lambda"] for row in wavelet]
    for axis, (field, label, _objective) in zip(axes.ravel(), PLOT_METRICS):
        values = [row[field] for row in wavelet]
        axis.semilogx(
            lambdas,
            values,
            color="#006ba4",
            marker="o",
            linewidth=1.8,
            label="Wavelet/FISTA",
        )
        leader = leaders[field]
        axis.scatter(
            [leader["lambda"]],
            [leader["value"]],
            marker="*",
            s=120,
            color="#008000",
            zorder=5,
            label="descriptive metric leader",
        )
        highlighted = next(
            (row for row in wavelet if math.isclose(row["lambda"], highlight_lambda)),
            None,
        )
        if highlighted is not None:
            axis.scatter(
                [highlighted["lambda"]],
                [highlighted[field]],
                marker="s",
                s=65,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.5,
                zorder=6,
                label=(
                    f"selected presentation λ={highlight_lambda:g}"
                    if presentation_selection
                    else f"highlighted λ={highlight_lambda:g}"
                ),
            )
        for control in controls:
            axis.axhline(
                control[field],
                color=colors[control["method"]],
                linestyle="--" if control["method"] == "cg_sense" else ":",
                linewidth=1.1,
                label=labels[control["method"]],
            )
        axis.set_xlabel("Wavelet λ")
        axis.set_ylabel(label)
        axis.grid(alpha=0.28, which="both")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels_found = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels_found, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="outside lower center",
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(
        "No-Wave R3×1 Wavelet sweep vs fully sampled direct-FFT RSS\n"
        "Stars mark single-metric leaders; no lambda is automatically selected"
    )
    figure.savefig(path_png, dpi=200)
    figure.savefig(path_pdf)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sweep_path = args.sweep_manifest.expanduser().resolve()
    grappa_path = args.grappa_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (sweep_path, grappa_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sweep = _load_json(sweep_path)
    if sweep.get("status") != "complete":
        raise ValueError("No-Wave sweep manifest is not complete")
    reference_record = sweep.get("metrics_reference_manifest", {})
    reference_path = Path(reference_record.get("path", "")).resolve()
    context = validate_metrics_reference_manifest(reference_path)
    if context["manifest_sha256"] != reference_record.get("sha256"):
        raise ValueError("Sweep metrics-reference hash changed")

    manifest_path = output_dir / "no_wave_r3x1_sweep_evaluation_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"Nonempty output is not an owned evaluation: {output_dir}")
        if not args.refresh:
            raise FileExistsError("No-Wave sweep evaluation exists; use --refresh.")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_manifests = sorted(sweep_path.parent.glob("*/manifest.json"))
    records = [
        _validated_record(path, context["manifest_sha256"])
        for path in case_manifests
    ]
    records.append(_validated_record(grappa_path, context["manifest_sha256"]))
    wavelet = sorted(
        (row for row in records if row["method"] == "wavelet"),
        key=lambda row: row["lambda"],
    )
    if [row["lambda"] for row in wavelet] != [
        float(value) for value in sweep["wavelet_lambdas"]
    ]:
        raise ValueError("Wavelet case manifests do not match the sweep lambda list")
    controls = [row for row in records if row["method"] in {"cg_sense", "grappa"}]
    if sorted(row["method"] for row in controls) != ["cg_sense", "grappa"]:
        raise ValueError("Expected exactly one CG-SENSE and one GRAPPA control")
    leaders = metric_leaders(wavelet)

    csv_path = output_dir / "no_wave_r3x1_sweep_metrics.csv"
    png_path = output_dir / "no_wave_r3x1_sweep_metrics_vs_lambda.png"
    pdf_path = output_dir / "no_wave_r3x1_sweep_metrics_vs_lambda.pdf"
    ordered = sorted(
        records,
        key=lambda row: (
            {"cg_sense": 0, "grappa": 1, "wavelet": 2}[row["method"]],
            -1.0 if row["lambda"] is None else row["lambda"],
        ),
    )
    _write_csv(csv_path, ordered)
    _plot(
        png_path,
        pdf_path,
        wavelet,
        controls,
        leaders,
        args.highlight_lambda,
        args.presentation_selection,
    )
    interpretation = (
        "Among tested values, lambda 1e-3 leads brain NRMSE, 3D SSIM, and "
        "edge-ratio closeness; lambda 1e-4 leads edge-gradient NCC. Lambda "
        "1.5e-2 is worse on all four plotted reference-similarity metrics. "
    )
    if args.presentation_selection:
        interpretation += (
            f"Lambda {args.highlight_lambda:g} is the explicit user-selected "
            "no-Wave presentation value; the evaluation made no automatic selection."
        )
    else:
        interpretation += "No lambda is selected without visual or user review."
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "no-Wave R3x1 Wavelet sweep metric curves",
        "scientific_scope": {
            "reference": "approved fully sampled direct-FFT RSS",
            "comparison_grid": "exact native 1 mm grid",
            "registration_performed": False,
            "interpolation_performed": False,
            "automatic_lambda_selection_performed": False,
            "metric_leaders_are_descriptive_only": True,
        },
        "inputs": {
            "sweep_manifest": {
                "path": str(sweep_path),
                "sha256": sha256_file(sweep_path),
            },
            "grappa_manifest": {
                "path": str(grappa_path),
                "sha256": sha256_file(grappa_path),
            },
            "metrics_reference_manifest": {
                "path": str(reference_path),
                "sha256": context["manifest_sha256"],
            },
        },
        "wavelet_lambdas": [row["lambda"] for row in wavelet],
        "highlighted_current_presentation_lambda": args.highlight_lambda,
        "presentation_selection": {
            "selected": bool(args.presentation_selection),
            "lambda": args.highlight_lambda if args.presentation_selection else None,
            "selection_mode": (
                "explicit_user_choice"
                if args.presentation_selection
                else "not_selected_by_evaluation"
            ),
        },
        "descriptive_metric_leaders": leaders,
        "interpretation": interpretation,
        "outputs": {
            "metrics_csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
            "plot_png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "plot_pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    print(f"No-Wave sweep evaluation: {manifest_path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-manifest", required=True, type=Path)
    parser.add_argument("--grappa-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--highlight-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--presentation-selection",
        action="store_true",
        help="Record the highlighted lambda as an explicit user presentation choice.",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
