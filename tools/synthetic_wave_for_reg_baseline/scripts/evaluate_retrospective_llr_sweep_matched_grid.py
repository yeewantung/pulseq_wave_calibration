#!/usr/bin/env python3
"""Evaluate a completed retro-LR LLR sweep on the original matched grid.

This script performs evaluation only. It linearly resamples candidates to the
accepted 1 mm RAS grid and reuses the original fixed references, BET brain
mask, edge mask, LSQ scaling, and retrospective fidelity metric function.
"""

from __future__ import annotations

import argparse
import csv
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

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.pipeline import _load_json, _resolve_path, _write_json  # noqa: E402

from analyze_retrospective_low_resolution import (  # noqa: E402
    _canonical_image,
    matched_fidelity_metrics,
)
from evaluate_retrospective_wavelet_sweep_matched_grid import (  # noqa: E402
    PLOT_METRICS,
    _original_context,
)


def _case_record(
    item: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    method: str,
    block_size: int | None,
    lambda_value: float,
) -> dict[str, Any]:
    case_manifest = Path(item["case_manifest"]).expanduser().resolve()
    magnitude = Path(item["magnitude_nifti"]).expanduser().resolve()
    phase = Path(item["phase_nifti"]).expanduser().resolve()
    for path in (case_manifest, magnitude, phase):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(case_manifest) != item["case_manifest_sha256"]:
        raise ValueError(f"Case manifest changed after the sweep: {case_manifest}")
    return {
        "method": method,
        "block_size": block_size,
        "lambda": lambda_value,
        "case_name": case["case_name"],
        "case": dict(case),
        "case_manifest": str(case_manifest),
        "case_manifest_sha256": item["case_manifest_sha256"],
        "magnitude_nifti": str(magnitude),
        "magnitude_nifti_sha256": sha256_file(magnitude),
        "phase_nifti": str(phase),
        "phase_nifti_sha256": sha256_file(phase),
    }


def _candidates(sweep_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep = _load_json(sweep_path)
    if sweep.get("status") != "complete":
        raise ValueError("Retrospective LLR sweep is not complete.")
    case_by_name = {str(case["case_name"]): case for case in sweep["cases"]}
    records = []
    for run in sweep["llr_runs"]:
        for item in run["cases"]:
            case = case_by_name[str(item["case_name"])]
            records.append(
                _case_record(
                    item,
                    case=case,
                    method="llr",
                    block_size=int(run["block_size"]),
                    lambda_value=float(run["lambda"]),
                )
            )
    control = sweep["fista_lambda0_control"]
    for item in control["cases"]:
        case = case_by_name[str(item["case_name"])]
        records.append(
            _case_record(
                item,
                case=case,
                method="fista_lambda0_control",
                block_size=None,
                lambda_value=0.0,
            )
        )
    expected_llr = len(sweep["llr_settings"]) * len(sweep["cases"])
    if len([record for record in records if record["method"] == "llr"]) != expected_llr:
        raise ValueError("LLR sweep candidate count differs from its declared settings.")
    records.sort(
        key=lambda item: (
            item["method"],
            -1 if item["block_size"] is None else item["block_size"],
            item["case_name"],
            item["lambda"],
        )
    )
    return sweep, records


def metric_leaders(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return direct-FFT leaders for every block/case/metric."""
    llr_rows = [
        row
        for row in rows
        if row["reference"] == "direct_fft_rss" and row["method"] == "llr"
    ]
    leaders: dict[str, Any] = {}
    for block_size in sorted({int(row["block_size"]) for row in llr_rows}):
        block_rows = [row for row in llr_rows if int(row["block_size"]) == block_size]
        block_leaders = {}
        for case_name in sorted({str(row["case_name"]) for row in block_rows}):
            case_rows = [row for row in block_rows if row["case_name"] == case_name]
            case_leaders = {}
            for field, _label, objective in PLOT_METRICS:
                if objective == "min":
                    leader = min(case_rows, key=lambda row: float(row[field]))
                elif objective == "max":
                    leader = max(case_rows, key=lambda row: float(row[field]))
                else:
                    leader = min(case_rows, key=lambda row: abs(float(row[field]) - 1.0))
                case_leaders[field] = {
                    "lambda": float(leader["lambda"]),
                    "value": float(leader[field]),
                    "objective": objective,
                }
            block_leaders[case_name] = case_leaders
        leaders[str(block_size)] = block_leaders
    return leaders


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_block(
    png_path: Path,
    pdf_path: Path,
    rows: Sequence[Mapping[str, Any]],
    leaders: Mapping[str, Any],
    block_size: int,
) -> None:
    direct = [row for row in rows if row["reference"] == "direct_fft_rss"]
    llr = [
        row
        for row in direct
        if row["method"] == "llr" and int(row["block_size"]) == block_size
    ]
    controls = {
        str(row["case_name"]): row
        for row in direct
        if row["method"] == "fista_lambda0_control"
    }
    case_names = sorted({str(row["case_name"]) for row in llr})
    colors = ("#006ba4", "#ff800e", "#3e9651")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (field, label, _objective) in zip(axes.ravel(), PLOT_METRICS):
        for index, case_name in enumerate(case_names):
            case_rows = [row for row in llr if row["case_name"] == case_name]
            color = colors[index % len(colors)]
            title = str(case_rows[0]["case_label"] or case_name)
            axis.semilogx(
                [float(row["lambda"]) for row in case_rows],
                [float(row[field]) for row in case_rows],
                marker="o",
                linewidth=1.6,
                color=color,
                label=title,
            )
            axis.axhline(
                float(controls[case_name][field]),
                color=color,
                linestyle=":",
                linewidth=1.0,
            )
            leader = leaders[str(block_size)][case_name][field]
            axis.scatter(
                [leader["lambda"]], [leader["value"]], marker="*", s=110, color=color, zorder=5
            )
        axis.set_xlabel("LLR λ")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle(
        f"Retrospective low-resolution corrected LLR, block {block_size}\n"
        "Original matched 1 mm direct-FFT metrics; dotted lines are FISTA λ=0 controls"
    )
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)


def run(config_path: Path, *, validate_only: bool, resume: bool) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("LLR evaluation config format_version must be 1.")
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
        llr_count = len([record for record in candidates if record["method"] == "llr"])
        print(f"Matched-grid LLR evaluation inputs: {llr_count} LLR candidates")
        print("Controls: 3 completed retrospective FISTA lambda-zero cases")
        print("Candidate interpolation: linear; reconstruction calls: none")
        return {"status": "validated", "llr_candidate_count": llr_count}

    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.is_file() and resume:
        prior = _load_json(manifest_path)
        if (
            prior.get("status") == "complete"
            and prior.get("candidate_inputs") == candidates
            and prior.get("original_analysis_manifest", {}).get("sha256")
            == context["analysis_sha256"]
        ):
            print(f"Reusing matched-grid LLR evaluation: {manifest_path}")
            return prior
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"LLR evaluation output is not empty; use --resume: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned LLR evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    running = {
        "format_version": 1,
        "status": "running",
        "purpose": "original matched-grid retrospective corrected-LLR sweep evaluation",
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
        candidate_image, _native = _canonical_image(candidate_path)
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
                    "method": candidate_record["method"],
                    "block_size": candidate_record["block_size"],
                    "lambda": candidate_record["lambda"],
                    "case_name": candidate_record["case_name"],
                    "case_label": case.get("label"),
                    "candidate_voxel_volume_mm3": float(
                        np.prod(candidate_image.header.get_zooms()[:3])
                    ),
                    "candidate_nifti": str(candidate_path),
                    "candidate_nifti_sha256": candidate_record["magnitude_nifti_sha256"],
                    **metrics,
                }
            )
    rows.sort(
        key=lambda row: (
            row["reference"],
            row["method"],
            -1 if row["block_size"] is None else row["block_size"],
            row["case_name"],
            row["lambda"],
        )
    )
    leaders = metric_leaders(rows)
    csv_path = output_dir / "matched_fidelity_metrics.csv"
    _write_csv(csv_path, rows)
    plot_paths = []
    for block_size in sorted({int(item["block_size"]) for item in sweep["llr_settings"]}):
        png_path = output_dir / f"direct_fft_metrics_vs_lambda_block-{block_size}.png"
        pdf_path = output_dir / f"direct_fft_metrics_vs_lambda_block-{block_size}.pdf"
        _plot_block(png_path, pdf_path, rows, leaders, block_size)
        plot_paths.extend((png_path, pdf_path))

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
            "fista_lambda0_role": "evaluation control only",
            "reconstruction_called": False,
            "automatic_llr_selection_performed": False,
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
            for path in (csv_path, *plot_paths)
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    print(f"Matched-grid LLR evaluation: {manifest_path}")
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
