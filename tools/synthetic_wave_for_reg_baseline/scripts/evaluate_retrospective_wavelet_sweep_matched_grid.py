#!/usr/bin/env python3
"""Evaluate a completed retro-LR Wavelet sweep on the original matched grid.

This script performs evaluation only. It reproduces the established
retrospective fidelity calculation: linear resampling to the 1 mm RAS grid,
the original fixed BET brain/edge masks, and one mask-restricted LSQ scale.
It never calls a reconstruction runner or creates native-grid references.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from nibabel.processing import resample_from_to

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = TOOL_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "wave_retro_lr_recon"))

from analyze_retrospective_low_resolution import (
    _canonical_image,
    matched_fidelity_metrics,
)
from wave_retro_lr.bart_io import sha256_file
from wave_retro_lr.pipeline import _load_json, _resolve_path, _write_json


PLOT_METRICS = (
    ("nrmse_brain", "Brain NRMSE ↓", "min"),
    ("ssim_axial_brain_bbox_mean", "Axial SSIM ↑", "max"),
    ("gradient_ncc_fixed_edge", "Edge-gradient NCC ↑", "max"),
    ("edge_gradient_preservation_ratio", "Edge ratio → 1", "one"),
)


def _record_path(record: Mapping[str, Any], parent: Path) -> Path:
    path = Path(str(record["path"])).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _verified_record_path(
    record: Mapping[str, Any], parent: Path, label: str
) -> Path:
    path = _record_path(record, parent)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} hash changed: {path}")
    return path


def _original_context(analysis_path: Path) -> dict[str, Any]:
    """Load the exact references and masks used by the accepted LR analysis."""
    analysis = _load_json(analysis_path)
    if analysis.get("status") != "complete":
        raise ValueError("Original retrospective analysis is not complete.")
    review_record = analysis["inputs"]["review_manifest"]
    review_path = _verified_record_path(
        review_record, analysis_path.parent, "original review manifest"
    )
    review = _load_json(review_path)
    by_key = {str(item["key"]): item for item in review["inputs"]}
    references: dict[str, dict[str, Any]] = {}
    for key in ("full_resolution_fista_lambda0", "direct_fft_rss"):
        record = by_key[key]
        path = _verified_record_path(record, review_path.parent, key)
        image, data = _canonical_image(path)
        references[key] = {"path": path, "record": record, "image": image, "data": data}

    full_image = references["full_resolution_fista_lambda0"]["image"]
    direct_image = references["direct_fft_rss"]["image"]
    if full_image.shape != direct_image.shape or not np.allclose(
        full_image.affine, direct_image.affine, atol=1e-5
    ):
        raise ValueError("Original full-resolution references no longer share one grid.")

    mask_records = {
        Path(str(item["path"])).name: item for item in analysis["fixed_masks"]["outputs"]
    }
    brain_path = _verified_record_path(
        mask_records["approved_bet_on_reconstruction_grid.nii.gz"],
        analysis_path.parent,
        "original matched-grid brain mask",
    )
    edge_path = _verified_record_path(
        mask_records["fixed_reference_edge_mask.nii.gz"],
        analysis_path.parent,
        "original matched-grid edge mask",
    )
    brain_image = nib.as_closest_canonical(nib.load(str(brain_path)))
    edge_image = nib.as_closest_canonical(nib.load(str(edge_path)))
    for label, image in (("brain", brain_image), ("edge", edge_image)):
        if image.shape != full_image.shape or not np.allclose(
            image.affine, full_image.affine, atol=1e-5
        ):
            raise ValueError(f"Original {label} mask no longer matches the 1 mm grid.")
    brain = np.asarray(brain_image.dataobj) > 0
    edge = np.asarray(edge_image.dataobj) > 0
    if int(brain.sum()) != int(analysis["fixed_masks"]["brain_voxel_count"]):
        raise ValueError("Original brain-mask voxel count changed.")
    if int(edge.sum()) != int(analysis["fixed_masks"]["edge_voxel_count"]):
        raise ValueError("Original edge-mask voxel count changed.")
    return {
        "analysis": analysis,
        "analysis_sha256": sha256_file(analysis_path),
        "review_path": review_path,
        "references": references,
        "brain_path": brain_path,
        "edge_path": edge_path,
        "brain": brain,
        "edge": edge,
        "full_image": full_image,
    }


def _candidates(sweep_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep = _load_json(sweep_path)
    if sweep.get("status") != "complete":
        raise ValueError("Retrospective Wavelet sweep is not complete.")
    case_by_name = {str(case["case_name"]): case for case in sweep["cases"]}
    records = []
    for run in sweep["lambda_runs"]:
        lambda_value = float(run["lambda"])
        for item in run["cases"]:
            case_name = str(item["case_name"])
            case_manifest = Path(item["case_manifest"]).expanduser().resolve()
            magnitude = Path(item["magnitude_nifti"]).expanduser().resolve()
            phase = Path(item["phase_nifti"]).expanduser().resolve()
            for path in (case_manifest, magnitude, phase):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if sha256_file(case_manifest) != item["case_manifest_sha256"]:
                raise ValueError(f"Case manifest changed after the sweep: {case_manifest}")
            records.append(
                {
                    "lambda": lambda_value,
                    "case_name": case_name,
                    "case": case_by_name[case_name],
                    "case_manifest": str(case_manifest),
                    "case_manifest_sha256": item["case_manifest_sha256"],
                    "magnitude_nifti": str(magnitude),
                    "magnitude_nifti_sha256": sha256_file(magnitude),
                    "phase_nifti": str(phase),
                    "phase_nifti_sha256": sha256_file(phase),
                }
            )
    expected = len(sweep["wavelet_lambdas"]) * len(sweep["cases"])
    if len(records) != expected:
        raise ValueError(f"Found {len(records)} candidates; expected {expected}.")
    records.sort(key=lambda item: (item["case_name"], item["lambda"]))
    return sweep, records


def metric_leaders(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return direct-FFT per-case metric leaders without selecting a lambda."""
    direct_rows = [row for row in rows if row["reference"] == "direct_fft_rss"]
    leaders: dict[str, Any] = {}
    for case_name in sorted({str(row["case_name"]) for row in direct_rows}):
        case_rows = [row for row in direct_rows if row["case_name"] == case_name]
        leaders[case_name] = {}
        for field, _label, objective in PLOT_METRICS:
            if objective == "min":
                leader = min(case_rows, key=lambda row: float(row[field]))
            elif objective == "max":
                leader = max(case_rows, key=lambda row: float(row[field]))
            else:
                leader = min(case_rows, key=lambda row: abs(float(row[field]) - 1.0))
            leaders[case_name][field] = {
                "lambda": float(leader["lambda"]),
                "value": float(leader[field]),
                "objective": objective,
            }
    return leaders


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot(
    png_path: Path,
    pdf_path: Path,
    rows: Sequence[Mapping[str, Any]],
    leaders: Mapping[str, Any],
) -> None:
    direct_rows = [row for row in rows if row["reference"] == "direct_fft_rss"]
    case_names = sorted({str(row["case_name"]) for row in direct_rows})
    colors = ("#006ba4", "#ff800e", "#3e9651")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (field, label, _objective) in zip(axes.ravel(), PLOT_METRICS):
        for index, case_name in enumerate(case_names):
            case_rows = [row for row in direct_rows if row["case_name"] == case_name]
            color = colors[index % len(colors)]
            axis.plot(
                [float(row["lambda"]) for row in case_rows],
                [float(row[field]) for row in case_rows],
                marker="o",
                linewidth=1.6,
                color=color,
                label=str(case_rows[0]["case_label"] or case_name),
            )
            leader = leaders[case_name][field]
            axis.scatter(
                [leader["lambda"]], [leader["value"]], marker="*", s=110, color=color, zorder=5
            )
        axis.set_xscale("symlog", linthresh=1e-5, linscale=0.8)
        axis.set_xlabel("Wavelet λ (0 is FISTA control)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle(
        "Retrospective low-resolution Wavelet sweep\n"
        "Original matched 1 mm direct-FFT metrics; stars are per-metric leaders"
    )
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)


def run(config_path: Path, *, validate_only: bool, resume: bool) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("Matched-grid evaluation config format_version must be 1.")
    sweep_path = _resolve_path(
        config.get("sweep_manifest"), config_path.parent, "sweep_manifest"
    )
    analysis_path = _resolve_path(
        config.get("original_analysis_manifest"),
        config_path.parent,
        "original_analysis_manifest",
    )
    output_dir = _resolve_path(
        config.get("output_dir"), config_path.parent, "output_dir"
    )
    context = _original_context(analysis_path)
    sweep, candidates = _candidates(sweep_path)
    if validate_only:
        print(f"Matched-grid evaluation inputs: {len(candidates)} candidates")
        print("Reference grid: original 1 mm RAS retrospective-analysis grid")
        print("Candidate interpolation: linear; reconstruction calls: none")
        return {"status": "validated", "candidate_count": len(candidates)}

    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.is_file() and resume:
        prior = _load_json(manifest_path)
        if (
            prior.get("status") == "complete"
            and prior.get("candidate_inputs") == candidates
            and prior.get("original_analysis_manifest", {}).get("sha256")
            == context["analysis_sha256"]
        ):
            print(f"Reusing matched-grid evaluation: {manifest_path}")
            return prior
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Matched-grid output is not empty; use --resume: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    running = {
        "format_version": 1,
        "status": "running",
        "purpose": "original matched-grid retrospective Wavelet sweep evaluation",
        "candidate_inputs": candidates,
        "original_analysis_manifest": {
            "path": str(analysis_path),
            "sha256": context["analysis_sha256"],
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, running)

    full_image = context["full_image"]
    zooms = full_image.header.get_zooms()[:3]
    rows = []
    for candidate_record in candidates:
        candidate_path = Path(candidate_record["magnitude_nifti"])
        candidate_image, _candidate_native = _canonical_image(candidate_path)
        matched = np.abs(
            np.asarray(resample_from_to(candidate_image, full_image, order=1).dataobj)
        ).astype(np.float32)
        for reference_key in ("full_resolution_fista_lambda0", "direct_fft_rss"):
            metrics = matched_fidelity_metrics(
                context["references"][reference_key]["data"],
                matched,
                context["brain"],
                context["edge"],
                zooms,
            )
            case = candidate_record["case"]
            rows.append(
                {
                    "reference": reference_key,
                    "case_name": candidate_record["case_name"],
                    "case_label": case.get("label"),
                    "lambda": candidate_record["lambda"],
                    "candidate_voxel_volume_mm3": float(
                        np.prod(candidate_image.header.get_zooms()[:3])
                    ),
                    "candidate_nifti": str(candidate_path),
                    "candidate_nifti_sha256": candidate_record["magnitude_nifti_sha256"],
                    **metrics,
                }
            )
    rows.sort(key=lambda row: (row["reference"], row["case_name"], row["lambda"]))
    leaders = metric_leaders(rows)
    csv_path = output_dir / "matched_fidelity_metrics.csv"
    png_path = output_dir / "direct_fft_metrics_vs_lambda.png"
    pdf_path = output_dir / "direct_fft_metrics_vs_lambda.pdf"
    _write_csv(csv_path, rows)
    _plot(png_path, pdf_path, rows, leaders)

    manifest = {
        **running,
        "status": "complete",
        "sweep_manifest": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
        "scientific_scope": {
            "comparison_grid": "original full-resolution 1 mm canonical RAS grid",
            "candidate_resampling": "linear interpolation",
            "candidate_registration_performed": False,
            "brain_mask": "original fixed approved BET mask",
            "edge_mask": "original full-resolution FISTA-lambda-zero-derived fixed edge mask",
            "references": ["full_resolution_fista_lambda0", "direct_fft_rss"],
            "fidelity_scaling": "one least-squares scalar inside the fixed BET mask",
            "native_grid_reference_created": False,
            "reconstruction_called": False,
            "automatic_lambda_selection_performed": False,
        },
        "fixed_inputs": {
            "review_manifest": str(context["review_path"]),
            "brain_mask": {"path": str(context["brain_path"]), "voxel_count": int(context["brain"].sum())},
            "edge_mask": {"path": str(context["edge_path"]), "voxel_count": int(context["edge"].sum())},
        },
        "direct_fft_metric_leaders": leaders,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (csv_path, png_path, pdf_path)
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    print(f"Matched-grid sweep evaluation: {manifest_path}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args.config, validate_only=args.validate_only, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
