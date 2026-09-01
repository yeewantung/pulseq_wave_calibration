#!/usr/bin/env python3
"""Evaluate pure-mask sweeps from explicit manifests without automatic selection."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage.metrics import structural_similarity

from pure_mask_rerun import (
    CASE_IDS,
    load_json,
    logical_array_sha256,
    sha256_file,
    validate_config,
    write_json_atomic,
)


METRIC_OBJECTIVES = {
    "nrmse_brain": "min",
    "rmse_brain": "min",
    "mae_brain": "min",
    "ncc_brain": "max",
    "ssim_3d_brain_bbox": "max",
    "gradient_ncc_fixed_edge": "max",
    "edge_gradient_preservation_ratio": "one",
}

EVALUATION_DERIVATION_VERSION = 2

CURVE_METRICS = (
    ("nrmse_brain", "NRMSE (brain)", "min"),
    ("ssim_3d_brain_bbox", "3D SSIM (brain bbox)", "max"),
    ("ncc_brain", "NCC (brain)", "max"),
    ("gradient_ncc_fixed_edge", "Gradient NCC (fixed edge)", "max"),
    ("edge_gradient_preservation_ratio", "Edge-gradient preservation ratio", "one"),
    (
        "background_std_normalized_p99_qc",
        "Background SD / reference p99 (QC)",
        "min",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and validate or evaluate one completed sweep.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after validation or completed evaluation.
    """
    args = _parser().parse_args(argv)
    result = evaluate(
        args.config,
        stage=args.stage,
        validate_only=args.validate_only,
        confirmed_output_root=args.confirm_output_root,
        resume=args.resume,
        refresh_derived_outputs=args.refresh_derived_outputs,
    )
    if args.validate_only:
        print(
            f"Validated {result['candidate_count']} manifest-backed candidates; "
            "no evaluation output was written."
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the manifest-backed evaluation command interface.

    Returns:
        Parser for coarse or fine evaluation.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("coarse", "fine"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-derived-outputs",
        action="store_true",
        help=(
            "Replace only manifest-proven evaluator-owned metrics and figures; "
            "never reconstruct candidates"
        ),
    )
    return parser


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp.

    Returns:
        ISO-8601 UTC string.
    """
    return datetime.now(timezone.utc).isoformat()


def _canonical_mask(
    path: Path,
    *,
    expected_shape_xyz: tuple[int, int, int],
    expected_fov_mm_xyz: tuple[float, float, float],
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load the approved BET mask in canonical RAS orientation.

    Args:
        path: Hash-validated approved BET NIfTI.
        expected_shape_xyz: Required native physical XYZ matrix.
        expected_fov_mm_xyz: Required physical XYZ field of view in millimeters.

    Returns:
        Canonical image and nonempty boolean mask.
    """
    image = nib.as_closest_canonical(nib.load(str(path)))
    values = np.asarray(image.dataobj)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    actual_fov = zooms * np.asarray(image.shape, dtype=np.float64)
    if (
        values.shape != expected_shape_xyz
        or not np.isfinite(values).all()
        or not np.isfinite(image.affine).all()
        or not np.isfinite(zooms).all()
        or np.any(zooms <= 0)
        or not np.allclose(actual_fov, expected_fov_mm_xyz, rtol=0.0, atol=1e-3)
    ):
        raise ValueError("Approved BET mask fails native dimension, FOV, or finite-value gates.")
    mask = values > 0
    if not np.any(mask):
        raise ValueError("Approved BET mask must be nonempty.")
    return image, mask


def logical_to_physical_xyz(
    array: np.ndarray, *, axis_order: Sequence[int], axis_flips: Sequence[bool]
) -> np.ndarray:
    """Map logical RO/LIN/PAR arrays into the fixed canonical physical grid.

    Args:
        array: Logical three-dimensional reconstruction or reference.
        axis_order: Accepted permutation from logical axes to physical XYZ.
        axis_flips: Accepted post-permutation axis flips.

    Returns:
        Physical XYZ float32 array.
    """
    if tuple(int(value) for value in axis_order) != (2, 1, 0):
        raise ValueError("Sagittal MPRAGE logical-to-physical order must be PAR/LIN/RO.")
    if len(axis_flips) != 3 or any(not isinstance(value, bool) for value in axis_flips):
        raise ValueError("logical_to_canonical_axis_flips must contain three booleans.")
    result = np.transpose(np.asarray(array), axes=tuple(axis_order))
    for axis, flip in enumerate(axis_flips):
        if flip:
            result = np.flip(result, axis=axis)
    return np.asarray(result, dtype=np.float32)


def _target_affine(
    approved_image: nib.Nifti1Image,
    target_shape: tuple[int, int, int],
    target_zooms: tuple[float, float, float],
) -> np.ndarray:
    """Build a same-FOV target affine preserving the approved physical center.

    Args:
        approved_image: Canonical approved BET grid.
        target_shape: Target physical XYZ matrix.
        target_zooms: Target physical XYZ voxel sizes in millimeters.

    Returns:
        Center-preserving target-grid affine.
    """
    source_affine = np.asarray(approved_image.affine, dtype=np.float64)
    source_zooms = np.asarray(approved_image.header.get_zooms()[:3], dtype=np.float64)
    directions = source_affine[:3, :3] / source_zooms[None, :]
    target_affine = source_affine.copy()
    target_affine[:3, :3] = directions * np.asarray(target_zooms)[None, :]
    source_center = (np.asarray(approved_image.shape, dtype=np.float64) - 1.0) / 2.0
    target_center = (np.asarray(target_shape, dtype=np.float64) - 1.0) / 2.0
    center_world = source_affine[:3, :3] @ source_center + source_affine[:3, 3]
    target_affine[:3, 3] = center_world - target_affine[:3, :3] @ target_center
    return target_affine


def _gradient_magnitude(data: np.ndarray, zooms: Sequence[float]) -> np.ndarray:
    """Calculate a sigma-0.7-mm physical gradient magnitude.

    Args:
        data: Physical XYZ magnitude volume.
        zooms: Physical voxel sizes in millimeters.

    Returns:
        Float32 gradient magnitude per millimeter.
    """
    spacing = np.asarray(zooms, dtype=float)
    smoothed = gaussian_filter(
        np.asarray(data, dtype=np.float32), sigma=tuple(0.7 / spacing)
    )
    components = np.gradient(smoothed, *spacing, edge_order=1)
    return np.sqrt(sum(component * component for component in components)).astype(np.float32)


def _ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    """Calculate masked zero-mean normalized cross-correlation.

    Args:
        first: Reference array.
        second: Candidate array.
        mask: Fixed boolean comparison support.

    Returns:
        Finite correlation coefficient when the denominator is nonzero.
    """
    first_values = first[mask].astype(np.float64)
    second_values = second[mask].astype(np.float64)
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator <= 0:
        raise ValueError("NCC denominator is zero inside the fixed mask.")
    return float(np.dot(first_values, second_values) / denominator)


def _brain_bbox(mask: np.ndarray) -> tuple[slice, slice, slice]:
    """Return the minimal three-dimensional bounding box of a mask.

    Args:
        mask: Nonempty boolean mask.

    Returns:
        Three half-open slices.
    """
    coordinates = np.argwhere(mask)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0) + 1
    return tuple(slice(int(start), int(stop)) for start, stop in zip(low, high, strict=True))


def evaluation_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    brain: np.ndarray,
    edge: np.ndarray,
    background: np.ndarray,
    zooms: Sequence[float],
) -> dict[str, float]:
    """Calculate the approved separate fidelity, detail, and QC metrics.

    Args:
        reference: Resolution-matched direct-FFT RSS magnitude.
        candidate: BART Wave candidate magnitude on the same grid.
        brain: Fixed approved or nearest-neighbor-derived BET support.
        edge: Fixed reference-derived edge support.
        background: Fixed QC-only outside-brain support.
        zooms: Physical XYZ voxel sizes in millimeters.

    Returns:
        LSQ scale, brain fidelity, SSIM, gradient, and background-QC values.
    """
    if any(array.shape != reference.shape for array in (candidate, brain, edge, background)):
        raise ValueError("Evaluation arrays must share one exact grid.")
    candidate_values = candidate[brain].astype(np.float64)
    reference_values = reference[brain].astype(np.float64)
    denominator = float(np.dot(candidate_values, candidate_values))
    if denominator <= 0:
        raise ValueError("Candidate is zero inside the approved BET mask.")
    scale = float(np.dot(reference_values, candidate_values) / denominator)
    scaled = scale_candidate_for_display(candidate, scale)
    residual = scaled[brain].astype(np.float64) - reference_values
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bbox = _brain_bbox(brain)
    low, high = np.percentile(reference_values, [1.0, 99.0])
    data_range = float(high - low)
    if data_range <= 0:
        raise ValueError("Direct-FFT reference has no SSIM intensity range.")
    reference_crop = np.clip(reference[bbox], low, high)
    candidate_crop = np.clip(scaled[bbox], low, high)
    ssim = float(structural_similarity(reference_crop, candidate_crop, data_range=data_range))
    reference_gradient = _gradient_magnitude(reference, zooms)
    candidate_gradient = _gradient_magnitude(scaled, zooms)
    reference_p99 = float(np.percentile(reference_values[reference_values > 0], 99.0))
    missed_threshold = 0.05 * reference_p99
    return {
        "intensity_scale_lsq": scale,
        "nrmse_brain": rmse
        / max(float(np.sqrt(np.mean(reference_values**2))), 1e-12),
        "rmse_brain": rmse,
        "mae_brain": mae,
        "ncc_brain": _ncc(reference, scaled, brain),
        "ssim_3d_brain_bbox": ssim,
        "gradient_ncc_fixed_edge": _ncc(reference_gradient, candidate_gradient, edge),
        "edge_gradient_preservation_ratio": float(np.mean(candidate_gradient[edge]))
        / max(float(np.mean(reference_gradient[edge])), 1e-12),
        "background_mean_abs_normalized_p99_qc": float(np.mean(np.abs(scaled[background])))
        / max(reference_p99, 1e-12),
        "background_std_normalized_p99_qc": float(np.std(scaled[background]))
        / max(reference_p99, 1e-12),
        "missed_anatomy_brain_fraction_qc": float(np.mean(scaled[brain] < missed_threshold)),
    }


def scale_candidate_for_display(
    candidate: np.ndarray, intensity_scale_lsq: float
) -> np.ndarray:
    """Map one magnitude candidate into its direct-FFT reference intensity units.

    Args:
        candidate: Candidate magnitude array on the reference grid.
        intensity_scale_lsq: Positive BET-restricted least-squares scale.

    Returns:
        Float32 candidate multiplied by the validated scale.

    Raises:
        ValueError: If the scale is nonfinite/nonpositive or the result is nonfinite.
    """
    scale = float(intensity_scale_lsq)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Candidate LSQ intensity scale must be finite and positive.")
    scaled = np.asarray(candidate, dtype=np.float32) * np.float32(scale)
    if not np.isfinite(scaled).all():
        raise ValueError("LSQ-scaled candidate contains nonfinite values.")
    return scaled


def metric_leaders(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report independent metric leaders by case and regularizer family.

    Args:
        rows: Candidate metric records.

    Returns:
        Nested leaders without any composite score or automatic winner.
    """
    result: dict[str, Any] = {}
    for case_id in CASE_IDS:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        result[case_id] = {}
        families = sorted(
            {
                "wavelet" if row["method"] in {"wavelet", "fista_lambda0"} else f"llr_block-{row['block_size']}"
                for row in case_rows
            }
        )
        for family in families:
            if family == "wavelet":
                family_rows = [
                    row for row in case_rows if row["method"] in {"wavelet", "fista_lambda0"}
                ]
            else:
                block = int(family.split("-")[-1])
                family_rows = [
                    row
                    for row in case_rows
                    if row["method"] == "llr" and int(row["block_size"]) == block
                ]
                family_rows += [row for row in case_rows if row["method"] == "fista_lambda0"]
            leaders = {}
            for metric, objective in METRIC_OBJECTIVES.items():
                if objective == "min":
                    leader = min(family_rows, key=lambda row: float(row[metric]))
                elif objective == "max":
                    leader = max(family_rows, key=lambda row: float(row[metric]))
                else:
                    leader = min(family_rows, key=lambda row: abs(float(row[metric]) - 1.0))
                leaders[metric] = {
                    "method": leader["method"],
                    "block_size": leader["block_size"],
                    "lambda": leader["lambda"],
                    "value": leader[metric],
                    "objective": objective,
                }
            result[case_id][family] = leaders
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a nonempty metric table atomically.

    Args:
        path: Destination CSV.
        rows: Homogeneous row mappings.

    Returns:
        None.
    """
    if not rows:
        raise ValueError("Cannot write an empty pure-mask metric table.")
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _orthogonal_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract central sagittal, coronal, and axial slices.

    Args:
        volume: Physical XYZ array.

    Returns:
        Central slices in three physical orientations.
    """
    centers = tuple(size // 2 for size in volume.shape)
    return (
        volume[centers[0], :, :].T,
        volume[:, centers[1], :].T,
        volume[:, :, centers[2]].T,
    )


def _plot_family(
    path: Path,
    reference: np.ndarray,
    candidates: Sequence[tuple[str, np.ndarray]],
    *,
    title: str,
) -> None:
    """Save a shared-window orthogonal reference/candidate review figure.

    Args:
        path: Destination PNG.
        reference: Physical XYZ direct-FFT reference.
        candidates: Candidate labels and LSQ-scaled physical XYZ magnitudes in
            direct-FFT reference intensity units.
        title: Figure title.

    Returns:
        None.
    """
    positive = reference[reference > 0]
    upper = float(np.percentile(positive, 99.5))
    volumes = [("direct FFT", reference), *candidates]
    figure, axes = plt.subplots(
        len(volumes), 3, figsize=(9, max(2.2 * len(volumes), 4)), squeeze=False
    )
    for row, (label, volume) in enumerate(volumes):
        for column, image in enumerate(_orthogonal_slices(volume)):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=upper, origin="lower")
            axes[row, column].axis("off")
        axes[row, 0].set_title(label, loc="left", fontsize=8)
    for column, label in enumerate(("sagittal", "coronal", "axial")):
        axes[0, column].set_xlabel(label)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _family_rows(
    rows: Sequence[Mapping[str, Any]], family: str
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Separate one FISTA control and the regularized rows for a plot family.

    Args:
        rows: Metric records from one case.
        family: ``wavelet`` or an ``llr_block-N`` family identifier.

    Returns:
        The unique FISTA lambda-zero row and lambda-sorted regularized rows.

    Raises:
        ValueError: If the family is invalid or its required rows are incomplete.
    """
    controls = [row for row in rows if row["method"] == "fista_lambda0"]
    if len(controls) != 1:
        raise ValueError("Metric curves require exactly one FISTA lambda-zero control.")
    if family == "wavelet":
        regularized = [row for row in rows if row["method"] == "wavelet"]
    elif family.startswith("llr_block-"):
        try:
            block_size = int(family.removeprefix("llr_block-"))
        except ValueError as exc:
            raise ValueError(f"Invalid LLR family identifier: {family}") from exc
        regularized = [
            row
            for row in rows
            if row["method"] == "llr" and int(row["block_size"]) == block_size
        ]
    else:
        raise ValueError(f"Unknown metric-curve family: {family}")
    regularized.sort(key=lambda row: float(row["lambda"]))
    if not regularized or any(float(row["lambda"]) <= 0 for row in regularized):
        raise ValueError(f"{family} metric curves require positive regularized lambdas.")
    return controls[0], regularized


def _plot_metric_curves(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    include_matched_1mm: bool,
    title: str,
) -> None:
    """Save separate native and optional matched-grid metric-versus-lambda curves.

    Args:
        path: Destination PNG.
        rows: Metric records from one case.
        family: ``wavelet`` or an ``llr_block-N`` family identifier.
        include_matched_1mm: Plot matched-1-mm metrics for an LR case when true.
        title: Figure title.

    Returns:
        None. The figure is written to ``path``.
    """
    control, regularized = _family_rows(rows, family)
    lambdas = np.asarray([float(row["lambda"]) for row in regularized])
    grids = [("Native grid", "", "#0072B2", "o", "-")]
    if include_matched_1mm:
        grids.append(("Matched 1 mm", "matched_1mm_", "#D55E00", "s", "--"))

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for axis, (metric, label, objective) in zip(axes.flat, CURVE_METRICS, strict=True):
        for grid_label, prefix, color, marker, linestyle in grids:
            values = [float(row[f"{prefix}{metric}"]) for row in regularized]
            control_value = float(control[f"{prefix}{metric}"])
            axis.plot(
                lambdas,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=f"{grid_label}: regularized",
            )
            axis.axhline(
                control_value,
                color=color,
                linestyle=":",
                linewidth=1.2,
                label=f"{grid_label}: FISTA λ=0",
            )
        if objective == "one":
            axis.axhline(
                1.0,
                color="black",
                linestyle="-.",
                linewidth=1.0,
                label="Ideal = 1",
            )
        direction = {"min": "↓", "max": "↑", "one": "→ 1"}[objective]
        axis.set_title(f"{label} ({direction})", fontsize=9)
        axis.set_xscale("log")
        axis.set_xlabel("Regularization λ (log scale)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=6)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _validate_refresh_tree(
    output_dir: Path, manifest_path: Path, prior: Mapping[str, Any]
) -> None:
    """Prove that an existing evaluation tree contains only manifest-owned files.

    Args:
        output_dir: Existing stage-specific evaluation directory.
        manifest_path: Existing evaluation manifest inside ``output_dir``.
        prior: Parsed existing evaluation manifest.

    Returns:
        None after ownership and recorded-hash validation.

    Raises:
        ValueError: If a path escapes the tree, a file is unrecorded, or a
            recorded evaluator output has changed.
    """
    resolved_root = output_dir.resolve()
    allowed: dict[Path, str | None] = {manifest_path.resolve(): None}
    records = list(prior.get("outputs", []))
    records.extend(prior.get("derived_native_bet_masks", {}).values())
    for record in records:
        if not isinstance(record, Mapping) or "path" not in record:
            raise ValueError("Existing evaluation manifest has an invalid owned-file record.")
        path = Path(str(record["path"])).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"Existing evaluator-owned path escapes its output tree: {path}")
        allowed[path] = str(record.get("sha256")) if record.get("sha256") else None
    actual: set[Path] = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Evaluation refresh refuses symbolic links: {path}")
        if path.is_file():
            actual.add(path.resolve())
    unknown = sorted(actual - set(allowed))
    missing = sorted(set(allowed) - actual)
    if unknown:
        raise ValueError(f"Evaluation refresh found unowned files: {unknown}")
    if missing:
        raise ValueError(f"Evaluation refresh found missing manifest-owned files: {missing}")
    for path, expected_hash in allowed.items():
        if expected_hash is not None and sha256_file(path) != expected_hash:
            raise ValueError(f"Manifest-owned evaluation file changed: {path}")


def _load_sweep(validated: dict[str, Any], stage: str) -> tuple[Path, dict[str, Any]]:
    """Load one completed owned sweep and validate every listed candidate.

    Args:
        validated: Fresh local configuration validation.
        stage: ``coarse`` or ``fine``.

    Returns:
        Sweep manifest path and parsed payload.
    """
    root = Path(validated["layout"]["root"])
    path = root / "sweeps" / stage / "sweep_manifest.json"
    sweep = load_json(path, f"pure-mask {stage} sweep manifest")
    if sweep.get("status") != "complete" or sweep.get("stage") != stage:
        raise ValueError(f"Pure-mask {stage} sweep is not complete.")
    preparation_path = root / "preparation_manifest.json"
    preparation = load_json(preparation_path, "pure-mask preparation manifest")
    preparation_binding = sweep.get("preparation_manifest", {})
    if (
        preparation.get("status") != "complete"
        or Path(preparation_binding.get("path", "")).resolve() != preparation_path
        or preparation_binding.get("sha256") != sha256_file(preparation_path)
    ):
        raise ValueError("Sweep is not hash-bound to the completed preparation manifest.")
    records = sweep.get("candidate_manifests")
    if not isinstance(records, list) or len(records) != int(sweep.get("candidate_count", -1)):
        raise ValueError("Sweep candidate manifest list is incomplete.")
    allowed_roots = [root / "sweeps" / stage]
    if stage == "fine":
        allowed_roots.append(root / "sweeps" / "coarse")
    seen: set[tuple[Any, ...]] = set()
    control_count = {case_id: 0 for case_id in CASE_IDS}
    for record in records:
        case_id = record.get("case_id")
        if case_id not in CASE_IDS:
            raise ValueError(f"Sweep contains an unknown case identifier: {case_id!r}.")
        candidate_path = Path(record["manifest"]).resolve()
        if not any(candidate_path.is_relative_to(candidate_root) for candidate_root in allowed_roots):
            raise ValueError(f"Candidate manifest is outside the owned sweep trees: {candidate_path}")
        if sha256_file(candidate_path) != record["manifest_sha256"]:
            raise ValueError(f"Candidate manifest changed: {candidate_path}")
        candidate = load_json(candidate_path, "candidate manifest")
        if candidate.get("status") != "complete" or candidate.get("phase_available") is not True:
            raise ValueError(f"Candidate is incomplete or lacks phase: {candidate_path}")
        if candidate.get("case_id") != case_id or candidate.get("setting") != record.get("setting"):
            raise ValueError(f"Candidate identity differs from its sweep record: {candidate_path}")
        setting = candidate["setting"]
        key = (case_id, setting["method"], setting["block_size"], float(setting["lambda"]))
        if key in seen:
            raise ValueError(f"Sweep repeats candidate {key}.")
        seen.add(key)
        if setting["method"] == "fista_lambda0":
            control_count[case_id] += 1
        prepared_case_path = Path(preparation["cases"][case_id]["case_manifest"])
        prepared_binding = candidate.get("prepared_case_manifest", {})
        if (
            Path(prepared_binding.get("path", "")).resolve() != prepared_case_path
            or prepared_binding.get("sha256") != sha256_file(prepared_case_path)
        ):
            raise ValueError(f"Candidate is not bound to the prepared {case_id} inputs.")
        prepared_case = load_json(prepared_case_path, f"{case_id} prepared case")
        expected_shape = tuple(
            int(value) for value in prepared_case["case"]["target_logical_matrix_ro_lin_par"]
        )
        for part in ("magnitude", "phase"):
            output = candidate["outputs"][part]
            if record.get(part) != output:
                raise ValueError(f"Candidate {part} record differs from its manifest.")
            if sha256_file(output["path"]) != output["sha256"]:
                raise ValueError(f"Candidate {part} changed: {output['path']}")
            values = np.load(output["path"], mmap_mode="r", allow_pickle=False)
            if (
                values.shape != expected_shape
                or values.dtype != np.float32
                or output.get("shape") != list(expected_shape)
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Candidate {part} shape/type/finite gate failed: {output['path']}")
    if any(count != 1 for count in control_count.values()):
        raise ValueError("Every evaluated case must contain exactly one FISTA lambda-zero control.")
    if stage == "coarse":
        for case_id in CASE_IDS:
            if sum(record["case_id"] == case_id for record in records) != 23:
                raise ValueError(f"Coarse sweep must contain 23 candidates for {case_id}.")
    return path, sweep


def evaluate(
    config_path: str | Path,
    *,
    stage: str,
    validate_only: bool,
    confirmed_output_root: Path | None,
    resume: bool,
    refresh_derived_outputs: bool = False,
) -> dict[str, Any]:
    """Validate and optionally evaluate one manifest-backed sweep stage.

    Args:
        config_path: Ignored local rerun configuration.
        stage: ``coarse`` or ``fine``.
        validate_only: Perform no writes when true.
        confirmed_output_root: Exact user-approved output root required for writes.
        resume: Reuse a complete matching evaluation.
        refresh_derived_outputs: Replace only a proven evaluator-owned output tree.

    Returns:
        Validation summary or completed evaluation manifest.
    """
    validated = validate_config(config_path)
    sweep_path, sweep = _load_sweep(validated, stage)
    candidate_count = len(sweep["candidate_manifests"])
    if validate_only:
        if refresh_derived_outputs:
            raise ValueError("--refresh-derived-outputs is not used with --validate-only.")
        if confirmed_output_root is not None:
            raise ValueError("--confirm-output-root is not used with --validate-only.")
        return {"status": "validated", "candidate_count": candidate_count}
    configured_root = Path(validated["layout"]["root"])
    if confirmed_output_root is None or confirmed_output_root.expanduser().resolve() != configured_root:
        raise ValueError("Evaluation requires the exact user-confirmed output root.")
    output_dir = configured_root / "evaluation" / stage
    manifest_path = output_dir / "evaluation_manifest.json"
    current_sweep_hash = sha256_file(sweep_path)
    if manifest_path.is_file():
        prior = load_json(manifest_path, "evaluation manifest")
        prior_matches = (
            prior.get("status") == "complete"
            and prior.get("sweep_manifest", {}).get("sha256") == current_sweep_hash
        )
        if (
            resume
            and prior_matches
            and prior.get("derivation_version") == EVALUATION_DERIVATION_VERSION
            and not refresh_derived_outputs
        ):
            return prior
        if refresh_derived_outputs:
            if not prior_matches:
                raise ValueError(
                    "Evaluation refresh requires the existing manifest to match the current sweep."
                )
            _validate_refresh_tree(output_dir, manifest_path, prior)
            print("Existing evaluator-owned output tree passed the refresh gate.", flush=True)
        elif output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                "Evaluation outputs use an older derivation. Run the explicit "
                "refresh evaluation action after review."
            )
    elif refresh_derived_outputs:
        raise FileNotFoundError(
            "Evaluation refresh requires an existing ownership manifest."
        )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = validated["config"]["snapshot"]
    evaluation_config = config.get("evaluation")
    if not isinstance(evaluation_config, Mapping):
        raise ValueError("evaluation must define the accepted logical/canonical transform.")
    axis_order = evaluation_config.get("logical_to_canonical_axis_order")
    axis_flips = evaluation_config.get("logical_to_canonical_axis_flips")
    bet_path = Path(validated["source"]["approved_bet_mask"]["path"])
    geometry = validated["geometry"]
    native_ro, native_lin, native_par = (
        int(value) for value in geometry["logical_matrix_ro_lin_par"]
    )
    approved_image, approved_mask = _canonical_mask(
        bet_path,
        expected_shape_xyz=(native_par, native_lin, native_ro),
        expected_fov_mm_xyz=tuple(
            float(value) for value in geometry["physical_fov_mm_xyz"]
        ),
    )
    preparation = load_json(
        configured_root / "preparation_manifest.json", "pure-mask preparation manifest"
    )
    records_by_case: dict[str, list[Mapping[str, Any]]] = {
        case_id: [
            record for record in sweep["candidate_manifests"] if record["case_id"] == case_id
        ]
        for case_id in CASE_IDS
    }
    rows: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    derived_mask_records: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        print(
            f"Evaluating {case_id}: {len(records_by_case[case_id])} candidates.",
            flush=True,
        )
        case_path = Path(preparation["cases"][case_id]["case_manifest"])
        case = load_json(case_path, f"{case_id} prepared case")
        geometry = case["case"]
        target_shape = tuple(int(value) for value in geometry["target_physical_matrix_xyz"])
        zooms = tuple(float(value) for value in geometry["achieved_resolution_mm_xyz"])
        affine = _target_affine(approved_image, target_shape, zooms)
        reference_record = case["direct_fft_reference"]
        if sha256_file(reference_record["path"]) != reference_record["sha256"]:
            raise ValueError(f"{case_id} direct-FFT reference file changed.")
        reference_logical = np.load(reference_record["path"], allow_pickle=False)
        if (
            reference_logical.shape
            != tuple(int(value) for value in geometry["target_logical_matrix_ro_lin_par"])
            or reference_logical.dtype != np.float32
            or logical_array_sha256(reference_logical) != reference_record["logical_sha256"]
            or not np.isfinite(reference_logical).all()
        ):
            raise ValueError(f"{case_id} direct-FFT reference validation failed.")
        reference = logical_to_physical_xyz(
            reference_logical, axis_order=axis_order, axis_flips=axis_flips
        )
        if reference.shape != target_shape:
            raise ValueError(f"{case_id} physical reference shape differs from geometry.")
        reference_image = nib.Nifti1Image(reference, affine)
        native_mask_image = resample_from_to(approved_image, reference_image, order=0)
        native_brain = np.asarray(native_mask_image.dataobj) > 0
        if not np.any(native_brain):
            raise ValueError(f"{case_id} derived approved BET mask is empty.")
        interior = binary_erosion(native_brain, iterations=2)
        reference_gradient = _gradient_magnitude(reference, zooms)
        edge_threshold = float(np.percentile(reference_gradient[interior], 80.0))
        edge = interior & (reference_gradient >= edge_threshold)
        background = ~binary_dilation(native_brain, iterations=8)
        if not np.any(edge) or not np.any(background):
            raise ValueError(f"{case_id} fixed edge/background support is empty.")
        case_output = output_dir / case_id
        case_output.mkdir(exist_ok=refresh_derived_outputs)
        native_mask_path = case_output / "approved_bet_mask_native.npy"
        np.save(native_mask_path, native_brain)
        derived_mask_records[case_id] = {
            "path": str(native_mask_path),
            "sha256": sha256_file(native_mask_path),
            "logical_sha256": logical_array_sha256(native_brain),
            "shape": list(native_brain.shape),
            "derivation": "nearest-neighbor mapping of the one approved BET mask",
        }
        candidates_for_plots: dict[str, list[tuple[str, np.ndarray]]] = {
            "wavelet": [],
            **{f"llr_block-{block}": [] for block in (4, 8, 16)},
        }
        for record in records_by_case[case_id]:
            candidate_manifest = load_json(record["manifest"], "candidate manifest")
            setting = candidate_manifest["setting"]
            magnitude_logical = np.load(
                candidate_manifest["outputs"]["magnitude"]["path"], allow_pickle=False
            )
            phase = np.load(candidate_manifest["outputs"]["phase"]["path"], allow_pickle=False)
            candidate = logical_to_physical_xyz(
                magnitude_logical, axis_order=axis_order, axis_flips=axis_flips
            )
            native_metrics = evaluation_metrics(
                reference, candidate, native_brain, edge, background, zooms
            )
            candidate_image = nib.Nifti1Image(candidate, affine)
            matched_reference = np.asarray(
                resample_from_to(reference_image, approved_image, order=1).dataobj
            ).astype(np.float32)
            matched_candidate = np.asarray(
                resample_from_to(candidate_image, approved_image, order=1).dataobj
            ).astype(np.float32)
            matched_gradient = _gradient_magnitude(
                matched_reference, approved_image.header.get_zooms()[:3]
            )
            matched_interior = binary_erosion(approved_mask, iterations=2)
            matched_edge = matched_interior & (
                matched_gradient >= np.percentile(matched_gradient[matched_interior], 80.0)
            )
            matched_background = ~binary_dilation(approved_mask, iterations=8)
            matched_metrics = evaluation_metrics(
                matched_reference,
                matched_candidate,
                approved_mask,
                matched_edge,
                matched_background,
                approved_image.header.get_zooms()[:3],
            )
            row = {
                "case_id": case_id,
                "method": setting["method"],
                "block_size": setting["block_size"],
                "lambda": setting["lambda"],
                "phase_available": True,
                "phase_finite": bool(np.isfinite(phase).all()),
                **native_metrics,
                **{f"matched_1mm_{key}": value for key, value in matched_metrics.items()},
            }
            rows.append(row)
            label = (
                "FISTA λ=0"
                if setting["method"] == "fista_lambda0"
                else f"{setting['method']} λ={float(setting['lambda']):g}"
            )
            display_candidate = scale_candidate_for_display(
                candidate, native_metrics["intensity_scale_lsq"]
            )
            display_label = (
                f"{label} (LSQ×{native_metrics['intensity_scale_lsq']:.3g})"
            )
            if setting["method"] in {"fista_lambda0", "wavelet"}:
                candidates_for_plots["wavelet"].append((display_label, display_candidate))
            if setting["method"] == "fista_lambda0":
                for block in (4, 8, 16):
                    candidates_for_plots[f"llr_block-{block}"].append(
                        (display_label, display_candidate)
                    )
            elif setting["method"] == "llr":
                candidates_for_plots[f"llr_block-{setting['block_size']}"].append(
                    (display_label, display_candidate)
                )
        for family, candidates in candidates_for_plots.items():
            if not candidates:
                continue
            figure_path = case_output / f"{family}_shared_window_orthogonal.png"
            _plot_family(
                figure_path,
                reference,
                candidates,
                title=f"{case_id}: FISTA control versus {family}",
            )
            figure_paths.append(figure_path)
            curve_path = case_output / f"{family}_metric_curves.png"
            _plot_metric_curves(
                curve_path,
                rows=[row for row in rows if row["case_id"] == case_id],
                family=family,
                include_matched_1mm=case_id.startswith("lr_"),
                title=f"{case_id}: {family} metrics versus regularization λ",
            )
            figure_paths.append(curve_path)
            print(
                f"Rendered {case_id} {family} montage and metric curves.",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            row["case_id"],
            row["method"],
            -1 if row["block_size"] is None else row["block_size"],
            row["lambda"],
        )
    )
    csv_path = output_dir / "metrics.csv"
    _write_csv(csv_path, rows)
    leaders = metric_leaders(rows)
    completed = {
        "format_version": 1,
        "derivation_version": EVALUATION_DERIVATION_VERSION,
        "status": "complete",
        "stage": stage,
        "sweep_manifest": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
        "approved_bet_mask": validated["source"]["approved_bet_mask"],
        "scientific_scope": {
            "reference": "resolution-matched direct FFT RSS from the fully sampled no-Wave source",
            "native_brain_mask": "nearest-neighbor mapping of the one approved BET mask",
            "matched_1mm_brain_mask": "exact original approved BET mask",
            "matched_1mm_resampling": "same fixed linear rule for reference and candidate",
            "candidate_specific_registration_performed": False,
            "dicom_intensity_ranking_performed": False,
            "background_values_are_qc_only": True,
            "automatic_composite_selection_performed": False,
            "automatic_winner_selection_performed": False,
            "final_selection_requires_manual_visual_user_review": True,
        },
        "visualization": {
            "candidate_intensity_scaling": (
                "per-candidate BET-restricted least-squares scalar to the "
                "resolution-matched direct-FFT reference"
            ),
            "orthogonal_figure_window": "shared reference positive-value p99.5",
            "metric_curves": (
                "native grid for every case; matched 1 mm additionally for LR cases"
            ),
            "fista_lambda_zero_curve_role": "horizontal control line",
            "automatic_composite_selection_performed": False,
        },
        "metric_leaders_by_case_and_family": leaders,
        "derived_native_bet_masks": derived_mask_records,
        "rows": rows,
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (csv_path, *figure_paths)
        ],
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, completed)
    print(f"Pure-mask {stage} evaluation manifest: {manifest_path}")
    return completed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
