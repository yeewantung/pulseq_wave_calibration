#!/usr/bin/env python3
"""Evaluate exact-grid Wave candidates against an approved direct-FFT RSS reference."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from skimage import __version__ as skimage_version

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic
from evaluate_regularization_volume import (
    _gradient_magnitude,
    build_fixed_masks,
    compute_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-reference-manifest", required=True, type=Path)
    parser.add_argument("--geometry-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}: {path}")


def _finite_magnitude(path: Path, label: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or not np.any(data > 0):
        raise ValueError(f"{label} is non-finite or has no positive voxels: {path}")
    if float(np.min(data)) < -1e-6:
        raise ValueError(f"{label} contains negative magnitude values: {path}")
    return image, np.abs(data)


def _load_candidate(case: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    case_id = case["case_id"]
    nifti_path = Path(case["magnitude_nifti"]).resolve()
    sidecar_path = Path(case["magnitude_sidecar"]).resolve()
    _verify_hash(nifti_path, case["magnitude_nifti_sha256"], f"{case_id} NIfTI")
    _verify_hash(sidecar_path, case["magnitude_sidecar_sha256"], f"{case_id} sidecar")
    _, normalized = _finite_magnitude(nifti_path, case_id)
    sidecar = _load_json(sidecar_path, f"{case_id} sidecar")
    normalization = sidecar.get("MagnitudeNormalization", {})
    if normalization.get("Method") != "positive-finite-percentile":
        raise ValueError(f"Unsupported NIfTI normalization for {case_id}")
    input_percentile = float(normalization.get("InputPercentileValue", 0.0))
    output_percentile = float(normalization.get("OutputPercentileValue", 0.0))
    if input_percentile <= 0 or output_percentile <= 0:
        raise ValueError(f"Invalid NIfTI normalization values for {case_id}")
    restored = normalized * np.float32(input_percentile / output_percentile)
    if not np.isfinite(restored).all() or not np.any(restored > 0):
        raise ValueError(f"Restored magnitude is invalid for {case_id}")
    return restored, {
        "method": normalization["Method"],
        "percentile": float(normalization["Percentile"]),
        "input_percentile_value": input_percentile,
        "output_percentile_value": output_percentile,
        "restoration_multiplier": input_percentile / output_percentile,
    }


def _save_mask(
    mask: np.ndarray,
    path: Path,
    reference_image: nib.spatialimages.SpatialImage,
) -> dict[str, Any]:
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference_image.affine, header),
        str(path),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "voxel_count": int(mask.sum()),
    }


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    columns = list(records[0])
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _plane(data: np.ndarray, plane: str, center: Sequence[int]) -> np.ndarray:
    x, y, z = center
    if plane == "coronal":
        return data[:, y, :].T
    if plane == "axial":
        return data[:, :, z].T
    raise ValueError(plane)


def _directions(axis: plt.Axes, plane: str) -> None:
    labels = (
        ((0.02, 0.5, "L"), (0.98, 0.5, "R"), (0.5, 0.98, "S"), (0.5, 0.02, "I"))
        if plane == "coronal"
        else ((0.02, 0.5, "L"), (0.98, 0.5, "R"), (0.5, 0.98, "A"), (0.5, 0.02, "P"))
    )
    for x, y, text in labels:
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            color="white",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
        )


def _plot_masks(
    reference: np.ndarray,
    masks: dict[str, np.ndarray],
    center: Sequence[int],
    vmax: float,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
    for axis, plane in zip(axes, ("coronal", "axial")):
        axis.imshow(
            _plane(reference, plane, center),
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=vmax,
        )
        axis.contour(
            _plane(masks["brain"], plane, center),
            levels=[0.5],
            colors="#00ff66",
            linewidths=0.8,
        )
        axis.contour(
            _plane(masks["edge"], plane, center),
            levels=[0.5],
            colors="#ffd000",
            linewidths=0.45,
        )
        axis.set_title(f"{plane}: metric mask=green, edge support=yellow")
        axis.set_axis_off()
        _directions(axis, plane)
    figure.suptitle("Approved direct-FFT metric support")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _method_records(
    records: list[dict[str, Any]],
    regularizer: str,
    block_size: int | None = None,
) -> list[dict[str, Any]]:
    return sorted(
        (
            record
            for record in records
            if record["regularizer"] == regularizer
            and (block_size is None or record["block_size"] == block_size)
        ),
        key=lambda record: record["lambda"],
    )


def _plot_metrics(records: list[dict[str, Any]], regularizer: str, path: Path) -> None:
    rows = _method_records(records, regularizer)
    positive = [row for row in rows if row["lambda"] > 0]
    zero = next(row for row in rows if row["lambda"] == 0)
    title = "LLR" if regularizer == "llr" else "Wavelet"
    metrics = (
        ("nrmse_brain", "Brain NRMSE ↓"),
        ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑"),
        ("gradient_ncc_brain_edge", "Edge gradient NCC ↑"),
        ("edge_preservation_ratio", "Edge magnitude ratio → 1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.ravel(), metrics):
        axis.semilogx(
            [row["lambda"] for row in positive],
            [row[key] for row in positive],
            marker="o",
            label="positive λ",
        )
        axis.axhline(
            zero[key],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="matched λ=0",
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.set_xlabel(f"{title} λ")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{title} metrics against direct FFT RSS")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_llr_metrics(records: list[dict[str, Any]], path: Path) -> None:
    rows = _method_records(records, "llr")
    blocks = sorted({int(row["block_size"]) for row in rows if row["lambda"] > 0})
    zero = next(row for row in rows if row["lambda"] == 0)
    metrics = (
        ("nrmse_brain", "Brain NRMSE ↓"),
        ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑"),
        ("gradient_ncc_brain_edge", "Edge gradient NCC ↑"),
        ("edge_preservation_ratio", "Edge magnitude ratio → 1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.ravel(), metrics):
        for block in blocks:
            block_rows = [
                row
                for row in rows
                if row["block_size"] == block and row["lambda"] > 0
            ]
            axis.semilogx(
                [row["lambda"] for row in block_rows],
                [row[key] for row in block_rows],
                marker="o",
                label=f"block {block}",
            )
        axis.axhline(
            zero[key],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="matched λ=0",
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.set_xlabel("LLR λ")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("LLR block-size and lambda metrics against direct FFT RSS")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_llr_heatmaps(records: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in records
        if row["regularizer"] == "llr" and row["lambda"] > 0
    ]
    blocks = sorted({int(row["block_size"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    lookup = {(int(row["block_size"]), float(row["lambda"])): row for row in rows}
    metrics = (
        ("nrmse_brain", "Brain NRMSE ↓", "viridis_r"),
        ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑", "viridis"),
        ("gradient_ncc_brain_edge", "Edge gradient NCC ↑", "viridis"),
        ("edge_preservation_ratio", "Edge ratio → 1", "coolwarm"),
    )
    figure_height = max(9.0, 4.0 + 0.55 * len(lambdas))
    figure, axes = plt.subplots(
        2, 2, figsize=(10, figure_height), constrained_layout=True
    )
    for axis, (key, title, cmap) in zip(axes.ravel(), metrics):
        matrix = np.asarray(
            [
                [
                    lookup[(block, value)][key]
                    if (block, value) in lookup
                    else np.nan
                    for block in blocks
                ]
                for value in lambdas
            ]
        )
        # A ragged grid is scientifically useful here: block 4 is sampled near
        # its local optimum while blocks 8 and 16 extend the upper boundary.
        # Mark unrun combinations explicitly rather than implying measurements.
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#d9d9d9")
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=color_map, aspect="auto")
        axis.set_xticks(range(len(blocks)), labels=blocks)
        axis.set_yticks(range(len(lambdas)), labels=[f"{value:g}" for value in lambdas])
        axis.set_xlabel("LLR block size")
        axis.set_ylabel("λ")
        axis.set_title(title)
        for row_index in range(len(lambdas)):
            for column_index in range(len(blocks)):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        "not run",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=5,
                    )
                    continue
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.5g}",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.55 else "white",
                    fontsize=6,
                )
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("LLR block-size × lambda metrics against direct FFT RSS")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_common_window(
    reference: np.ndarray,
    records: list[dict[str, Any]],
    scaled_volumes: dict[str, np.ndarray],
    regularizer: str,
    center: Sequence[int],
    vmax: float,
    path: Path,
    block_size: int | None = None,
) -> None:
    rows = _method_records(records, regularizer, block_size)
    method_title = "LLR" if regularizer == "llr" else "Wavelet"
    if regularizer == "llr" and block_size is not None:
        matched_zero = next(
            row
            for row in records
            if row["regularizer"] == "llr" and row["lambda"] == 0
        )
        rows = [matched_zero] + [row for row in rows if row["lambda"] > 0]
        method_title = f"LLR block {block_size}"
    volumes = [("Direct FFT RSS", reference)] + [
        (f"λ={row['lambda_label']}", scaled_volumes[row["case_id"]]) for row in rows
    ]
    figure, axes = plt.subplots(
        2,
        len(volumes),
        figsize=(2.6 * len(volumes), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (volume_title, volume) in enumerate(volumes):
        for row_index, plane in enumerate(("coronal", "axial")):
            axis = axes[row_index, column]
            axis.imshow(
                _plane(volume, plane, center),
                cmap="gray",
                origin="lower",
                vmin=0,
                vmax=vmax,
            )
            axis.set_title(f"{volume_title}\n{plane}", fontsize=9)
            axis.set_axis_off()
            _directions(axis, plane)
    figure.suptitle(
        f"{method_title} sweep — shared direct-FFT intensity window"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _metric_leaders(records: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "lowest_nrmse_brain": ("nrmse_brain", "min"),
        "highest_ssim_3d_brain_bbox": ("ssim_3d_brain_bbox", "max"),
        "highest_ncc_brain": ("ncc_brain", "max"),
        "highest_gradient_ncc_brain_edge": ("gradient_ncc_brain_edge", "max"),
        "edge_preservation_ratio_closest_to_one": ("edge_preservation_ratio", "closest_one"),
    }
    leaders: dict[str, Any] = {}
    for regularizer in ("wavelet", "llr"):
        positive = [
            row
            for row in records
            if row["regularizer"] == regularizer and row["lambda"] > 0
        ]
        leaders[regularizer] = {}
        for label, (key, objective) in definitions.items():
            if objective == "min":
                selected = min(positive, key=lambda row: row[key])
            elif objective == "max":
                selected = max(positive, key=lambda row: row[key])
            else:
                selected = min(positive, key=lambda row: abs(row[key] - 1.0))
            leaders[regularizer][label] = {
                "case_id": selected["case_id"],
                "lambda": selected["lambda"],
                "value": selected[key],
            }
    leaders["llr_by_block"] = {}
    blocks = sorted(
        {
            int(row["block_size"])
            for row in records
            if row["regularizer"] == "llr" and row["lambda"] > 0
        }
    )
    for block in blocks:
        block_rows = [
            row
            for row in records
            if row["regularizer"] == "llr"
            and row["block_size"] == block
            and row["lambda"] > 0
        ]
        leaders["llr_by_block"][str(block)] = {}
        for label, (key, objective) in definitions.items():
            if objective == "min":
                selected = min(block_rows, key=lambda row: row[key])
            elif objective == "max":
                selected = max(block_rows, key=lambda row: row[key])
            else:
                selected = min(block_rows, key=lambda row: abs(row[key] - 1.0))
            leaders["llr_by_block"][str(block)][label] = {
                "case_id": selected["case_id"],
                "lambda": selected["lambda"],
                "value": selected[key],
            }
    return leaders


def run(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = args.metrics_reference_manifest.expanduser().resolve()
    geometry_path = args.geometry_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    masks_dir = output_dir / "derived_masks"
    plots_dir.mkdir()
    masks_dir.mkdir()

    metrics_reference = _load_json(metrics_path, "metrics-reference manifest")
    geometry = _load_json(geometry_path, "geometry report")
    if metrics_reference.get("status") != "approved_for_metrics":
        raise ValueError("Metrics-reference manifest is not approved")
    if geometry.get("status") != "passed":
        raise ValueError("Geometry/provenance gate has not passed")
    if geometry["metrics_reference_manifest"]["sha256"] != sha256_file(metrics_path):
        raise ValueError("Geometry report references a different metrics manifest")
    policy = geometry.get("geometry_policy", {})
    if policy.get("registration_performed") or policy.get("interpolation_performed"):
        raise ValueError("Geometry report contains registration or interpolation")

    reference_record = metrics_reference["ranking_reference"]
    reference_path = Path(reference_record["path"]).resolve()
    _verify_hash(reference_path, reference_record["sha256"], "direct FFT RSS reference")
    reference_image, reference = _finite_magnitude(reference_path, "direct FFT RSS reference")
    mask_record = metrics_reference["brain_mask"]
    mask_path = Path(mask_record["path"]).resolve()
    _verify_hash(mask_path, mask_record["sha256"], "approved brain mask")
    mask_image = nib.load(str(mask_path))
    brain = np.asarray(mask_image.dataobj) > 0
    if brain.shape != reference.shape or not np.array_equal(mask_image.affine, reference_image.affine):
        raise ValueError("Approved brain mask is not on the exact reference grid")

    masks, mask_metadata = build_fixed_masks(reference, brain)
    mask_metadata["reference_kind"] = "direct_fft_rss"
    mask_metadata["approved_brain_mask"] = {
        "path": str(mask_path),
        "sha256": mask_record["sha256"],
    }
    derived_masks = {
        "edge": _save_mask(masks["edge"], masks_dir / "edge_support.nii.gz", reference_image),
        "background": _save_mask(
            masks["background"], masks_dir / "background_support.nii.gz", reference_image
        ),
    }
    center = [int(round(value)) for value in np.argwhere(brain).mean(axis=0)]
    reference_positive = reference[brain & (reference > 0)]
    display_vmax = float(np.percentile(reference_positive, 99.5))
    _plot_masks(
        reference,
        masks,
        center,
        display_vmax,
        plots_dir / "fixed_metric_support.png",
    )

    records: list[dict[str, Any]] = []
    scaled_volumes: dict[str, np.ndarray] = {}
    reference_gradient = _gradient_magnitude(reference)
    for index, case in enumerate(geometry["cases"], start=1):
        case_id = case["case_id"]
        print(f"[{index:02d}/{geometry['case_count']:02d}] {case_id}", flush=True)
        candidate, normalization = _load_candidate(case)
        metric_values, scaled = compute_metrics(
            reference,
            candidate,
            masks,
            mask_metadata,
            reference_gradient=reference_gradient,
        )
        records.append(
            {
                "case_id": case_id,
                "regularizer": case["regularizer"],
                "lambda": case["lambda"],
                "lambda_label": case["lambda_label"],
                "block_size": "" if case["block_size"] is None else case["block_size"],
                "role": "solver_control" if case["lambda"] == 0 else "candidate",
                "source_nifti": case["magnitude_nifti"],
                "source_nifti_sha256": case["magnitude_nifti_sha256"],
                "export_normalization_restoration_multiplier": normalization[
                    "restoration_multiplier"
                ],
                **metric_values,
            }
        )
        scaled_volumes[case_id] = scaled
        del candidate

    records.sort(
        key=lambda row: (
            row["regularizer"],
            int(row["block_size"]) if row["block_size"] != "" else 0,
            row["lambda"],
        )
    )
    metrics_csv = output_dir / "regularization_metrics.csv"
    _write_csv(records, metrics_csv)
    _plot_metrics(records, "wavelet", plots_dir / "wavelet_metrics.png")
    _plot_llr_metrics(records, plots_dir / "llr_block_size_lambda_metrics.png")
    _plot_llr_heatmaps(records, plots_dir / "llr_block_size_lambda_heatmaps.png")
    _plot_common_window(
        reference,
        records,
        scaled_volumes,
        "wavelet",
        center,
        display_vmax,
        plots_dir / "wavelet_common_reference_window.png",
    )
    for block_size in sorted(
        {
            int(row["block_size"])
            for row in records
            if row["regularizer"] == "llr" and row["lambda"] > 0
        }
    ):
        _plot_common_window(
            reference,
            records,
            scaled_volumes,
            "llr",
            center,
            display_vmax,
            plots_dir / f"llr_block-{block_size}_common_reference_window.png",
            block_size=block_size,
        )

    provenance = {
        "format_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "R1 regularization metrics against fully sampled direct FFT RSS",
        "selection_status": "no_parameter_selected",
        "metrics_reference_manifest": {
            "path": str(metrics_path),
            "sha256": sha256_file(metrics_path),
        },
        "geometry_report": {
            "path": str(geometry_path),
            "sha256": sha256_file(geometry_path),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": reference_record["sha256"],
            "display_window": [0.0, display_vmax],
            "display_window_rule": "0 to p99.5 of positive reference voxels inside approved mask",
        },
        "fixed_masks": {"metadata": mask_metadata, "derived": derived_masks},
        "processing": {
            "registration_performed": False,
            "interpolation_performed": False,
            "bias_correction_performed": False,
            "histogram_matching_performed": False,
            "candidate_export_normalization": "undone from each NIfTI sidecar before metrics",
            "intensity_matching": "one unconstrained LSQ scalar per candidate inside the fixed approved mask",
            "display_scaling": "same direct-FFT-derived window for reference and all LSQ-scaled candidates",
        },
        "metric_definitions": {
            "nrmse_brain": "brain-mask RMSE divided by brain-mask RMS(reference)",
            "nmae_brain": "brain-mask MAE divided by brain-mask mean absolute reference",
            "psnr_p99_db": "20*log10(reference brain-positive p99 / brain-mask RMSE)",
            "ssim_3d_brain_bbox": "3D SSIM in fixed-mask bounding box after shared reference p1-p99 clipping",
            "ncc_brain": "normalized cross-correlation inside the fixed approved mask",
            "gradient_ncc_brain_edge": "NCC of sigma-0.7 gradient magnitudes in the fixed reference-derived edge support",
            "edge_preservation_ratio": "mean candidate/reference gradient magnitude in fixed edge support; target 1",
            "background_metrics": "scaled candidate magnitude in fixed background support, normalized by reference p99",
        },
        "metric_leaders": _metric_leaders(records),
        "metrics_csv": {
            "path": str(metrics_csv),
            "sha256": sha256_file(metrics_csv),
            "row_count": len(records),
        },
        "plots": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(plots_dir.glob("*.png"))
        ],
        "case_records": records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "scikit_image": skimage_version,
            "matplotlib": matplotlib.__version__,
        },
        "notes": [
            "Lambda-zero cases are solver controls, not the quantitative reference.",
            "Metric leaders are reported per metric only; no composite rank or parameter selection is performed.",
            "Outside-brain support includes skull and neck in this full-head acquisition; its background fields are diagnostic and are not noise or SNR estimates.",
            "DICOM intensities are not loaded or used.",
        ],
    }
    provenance_path = output_dir / "metrics_provenance.json"
    write_json_atomic(provenance_path, provenance)
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Provenance: {provenance_path}")
    return provenance


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
