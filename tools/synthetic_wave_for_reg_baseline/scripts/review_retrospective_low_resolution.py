#!/usr/bin/env python3
"""Create configured physical-coordinate retrospective-resolution visual QC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


PLANES = ("sagittal", "coronal", "axial")
PLANE_AXES = {
    "sagittal": (0, 1, 2),
    "coronal": (1, 0, 2),
    "axial": (2, 0, 1),
}
DIRECTION_LABELS = {
    "sagittal": ("P", "A", "I", "S"),
    "coronal": ("L", "R", "I", "S"),
    "axial": ("L", "R", "P", "A"),
}


@dataclass
class ReviewVolume:
    key: str
    title: str
    path: Path
    image: nib.spatialimages.SpatialImage
    data: np.ndarray
    display_scale: float
    case_manifest: Path | None = None
    export_sidecar: Path | None = None


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_magnitude(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI does not exist: {path}")
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3 or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError(f"Expected one canonical RAS 3D image: {path}")
    linear = np.asarray(image.affine[:3, :3], dtype=float)
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-5) or np.any(
        np.diag(linear) <= 0
    ):
        raise ValueError(f"QC currently requires an axis-aligned canonical RAS grid: {path}")
    data = np.abs(np.asarray(image.dataobj, dtype=np.float32))
    if not np.isfinite(data).all() or float(data.max()) <= 0:
        raise ValueError(f"Magnitude data must be finite and nonzero: {path}")
    return image, data


def _validated_canonical_export_sidecar(path: Path) -> Path:
    """Require provenance from the corrected canonical-RAS Wave exporter."""
    sidecar = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Wave NIfTI export sidecar does not exist: {sidecar}")
    payload = _load_json(sidecar)
    if payload.get("NIfTICanonicalRAS") is not True:
        raise ValueError(
            f"Wave NIfTI predates the canonical-RAS exporter contract: {path}"
        )
    if payload.get("NIfTIAffineAxisFlips") != [True, False, True]:
        raise ValueError(f"Unexpected Wave affine-axis convention in {sidecar}")
    return sidecar


def _positive_percentile(data: np.ndarray, percentile: float) -> float:
    positive = data[data > 0]
    if not positive.size:
        raise ValueError("Display volume has no positive voxels.")
    value = float(np.percentile(positive, percentile))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Display percentile is not positive and finite.")
    return value


def geometry_record(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    shape = np.asarray(image.shape, dtype=int)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    center = nib.affines.apply_affine(image.affine, (shape - 1.0) / 2.0)
    return {
        "shape_xyz": shape.tolist(),
        "voxel_size_mm_xyz": zooms.tolist(),
        "fov_mm_xyz": (shape * zooms).tolist(),
        "center_mm_ras": center.tolist(),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
    }


def validate_shared_physical_geometry(
    volumes: Sequence[ReviewVolume], *, center_tolerance_mm: float = 0.01
) -> None:
    if not volumes:
        raise ValueError("No review volumes were supplied.")
    reference_geometry = geometry_record(volumes[0].image)
    reference_center = np.asarray(reference_geometry["center_mm_ras"])
    reference_fov = np.asarray(reference_geometry["fov_mm_xyz"])
    for volume in volumes[1:]:
        current = geometry_record(volume.image)
        center_error = float(
            np.max(np.abs(np.asarray(current["center_mm_ras"]) - reference_center))
        )
        if center_error > center_tolerance_mm:
            raise ValueError(
                f"{volume.key} grid center differs by {center_error:g} mm; "
                "physical-coordinate slice matching is unsafe."
            )
        if not np.allclose(current["fov_mm_xyz"], reference_fov, atol=0.01, rtol=0.0):
            raise ValueError(
                f"{volume.key} FOV {current['fov_mm_xyz']} differs from "
                f"{reference_geometry['fov_mm_xyz']}."
            )


def world_slice_index(
    image: nib.spatialimages.SpatialImage,
    plane: str,
    world_point_mm_ras: Sequence[float],
) -> tuple[int, float]:
    """Return the nearest native slice and its actual RAS coordinate."""
    if plane not in PLANE_AXES:
        raise ValueError(f"Unsupported plane: {plane}")
    slice_axis = PLANE_AXES[plane][0]
    continuous = nib.affines.apply_affine(
        np.linalg.inv(image.affine), np.asarray(world_point_mm_ras, dtype=float)
    )
    index = int(np.rint(continuous[slice_axis]))
    if not 0 <= index < image.shape[slice_axis]:
        raise ValueError(
            f"World location {world_point_mm_ras} is outside {plane} axis of the image."
        )
    voxel = continuous.copy()
    voxel[slice_axis] = index
    actual_world = nib.affines.apply_affine(image.affine, voxel)
    return index, float(actual_world[slice_axis])


def _axis_edges_mm(image: nib.spatialimages.SpatialImage, axis: int) -> tuple[float, float]:
    scale = float(image.affine[axis, axis])
    origin = float(image.affine[axis, 3])
    first = origin - 0.5 * scale
    last = origin + (image.shape[axis] - 0.5) * scale
    return min(first, last), max(first, last)


def native_plane(
    volume: ReviewVolume,
    plane: str,
    world_point_mm_ras: Sequence[float],
) -> tuple[np.ndarray, tuple[float, float, float, float], int, float]:
    slice_axis, horizontal_axis, vertical_axis = PLANE_AXES[plane]
    index, actual_world = world_slice_index(volume.image, plane, world_point_mm_ras)
    if slice_axis == 0:
        pixels = volume.data[index, :, :].T
    elif slice_axis == 1:
        pixels = volume.data[:, index, :].T
    else:
        pixels = volume.data[:, :, index].T
    horizontal = _axis_edges_mm(volume.image, horizontal_axis)
    vertical = _axis_edges_mm(volume.image, vertical_axis)
    return pixels, (*horizontal, *vertical), index, actual_world


def _directions(axis: plt.Axes, plane: str) -> None:
    left, right, bottom, top = DIRECTION_LABELS[plane]
    style = dict(
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1),
    )
    axis.text(0.02, 0.5, left, transform=axis.transAxes, va="center", **style)
    axis.text(0.98, 0.5, right, transform=axis.transAxes, ha="right", va="center", **style)
    axis.text(0.5, 0.02, bottom, transform=axis.transAxes, ha="center", **style)
    axis.text(0.5, 0.98, top, transform=axis.transAxes, ha="center", va="top", **style)


def _render_native_comparison(
    volumes: Sequence[ReviewVolume],
    world_point: Sequence[float],
    output_path: Path,
    figure_title: str,
) -> dict[str, dict[str, Any]]:
    figure, axes = plt.subplots(
        len(PLANES), len(volumes), figsize=(3.2 * len(volumes), 9.5), constrained_layout=True
    )
    slice_records: dict[str, dict[str, Any]] = {volume.key: {} for volume in volumes}
    for row, plane in enumerate(PLANES):
        for column, volume in enumerate(volumes):
            pixels, extent, index, actual_world = native_plane(volume, plane, world_point)
            axes[row, column].imshow(
                pixels / volume.display_scale,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_axis_off()
            axes[row, column].set_title(
                f"{volume.title}\n{plane}: native index {index}, {actual_world:.2f} mm",
                fontsize=8,
            )
            _directions(axes[row, column], plane)
            slice_records[volume.key][plane] = {
                "native_index": index,
                "actual_world_coordinate_mm": actual_world,
            }
    figure.suptitle(figure_title, fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return slice_records


def _render_matched_grid_comparison(
    volumes: Sequence[ReviewVolume],
    reference_image: nib.spatialimages.SpatialImage,
    world_point: Sequence[float],
    output_path: Path,
    figure_title: str,
) -> None:
    matched: list[ReviewVolume] = []
    for volume in volumes:
        if volume.image.shape == reference_image.shape and np.allclose(
            volume.image.affine, reference_image.affine, atol=1e-5
        ):
            data = volume.data
        else:
            data = np.abs(
                np.asarray(resample_from_to(volume.image, reference_image, order=1).dataobj)
            ).astype(np.float32, copy=False)
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite values after matching {volume.key} to the reference grid.")
        matched.append(
            ReviewVolume(
                key=volume.key,
                title=volume.title,
                path=volume.path,
                image=reference_image,
                data=data,
                display_scale=volume.display_scale,
                case_manifest=volume.case_manifest,
            )
        )

    figure, axes = plt.subplots(
        len(PLANES), len(matched), figsize=(3.2 * len(matched), 9.5), constrained_layout=True
    )
    for row, plane in enumerate(PLANES):
        for column, volume in enumerate(matched):
            pixels, extent, index, actual_world = native_plane(volume, plane, world_point)
            axes[row, column].imshow(
                pixels / volume.display_scale,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_axis_off()
            axes[row, column].set_title(
                f"{volume.title}\n{plane}: matched index {index}, {actual_world:.2f} mm",
                fontsize=8,
            )
            _directions(axes[row, column], plane)
    figure.suptitle(figure_title, fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _resolution_text(case: dict[str, Any]) -> str:
    achieved = case["achieved_resolution_mm_xyz"]
    return " x ".join(f"{float(value):.3g}" for value in achieved)


def _load_volume(
    key: str,
    title: str,
    path: Path,
    percentile: float,
    *,
    case_manifest: Path | None = None,
    require_canonical_export: bool = False,
) -> ReviewVolume:
    path = path.expanduser().resolve()
    export_sidecar = (
        _validated_canonical_export_sidecar(path) if require_canonical_export else None
    )
    image, data = _canonical_magnitude(path)
    return ReviewVolume(
        key=key,
        title=title,
        path=path,
        image=image,
        data=data,
        display_scale=_positive_percentile(data, percentile),
        case_manifest=case_manifest,
        export_sidecar=export_sidecar,
    )


def _load_retro_volumes(
    batch_manifest: Path,
    percentile: float,
    *,
    title_template: str = "Retro LR corrected LLR\\n{resolution_mm} mm",
    require_canonical_export: bool = True,
) -> list[ReviewVolume]:
    batch = _load_json(batch_manifest)
    if batch.get("status") != "complete":
        raise ValueError(f"Retrospective batch is not complete: {batch_manifest}")
    volumes = []
    for case_manifest_value in batch.get("case_manifests", []):
        case_manifest = Path(case_manifest_value).expanduser().resolve()
        payload = _load_json(case_manifest)
        if payload.get("status") != "complete":
            raise ValueError(f"Retrospective case is not complete: {case_manifest}")
        outputs = [Path(value).expanduser().resolve() for value in payload["reconstruction"]["nifti_outputs"]]
        magnitude = [path for path in outputs if "_part-mag_" in path.name]
        if len(magnitude) != 1:
            raise ValueError(f"Expected one magnitude NIfTI in {case_manifest}; found {magnitude}")
        case = payload["case"]
        title = title_template.format(
            case_name=str(case["case_name"]),
            case_label=str(case.get("label", case["case_name"])),
            resolution_mm=_resolution_text(case),
        ).replace("\\n", "\n")
        volumes.append(
            _load_volume(
                str(case["case_name"]),
                title,
                magnitude[0],
                percentile,
                case_manifest=case_manifest,
                require_canonical_export=require_canonical_export,
            )
        )
    if len(volumes) != 3:
        raise ValueError(f"Expected three retrospective cases; found {len(volumes)}")
    return volumes


def _write_geometry_table(path: Path, volumes: Sequence[ReviewVolume]) -> None:
    fieldnames = [
        "key",
        "title",
        "path",
        "shape_xyz",
        "voxel_size_mm_xyz",
        "fov_mm_xyz",
        "center_mm_ras",
        "display_scale",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for volume in volumes:
            geometry = geometry_record(volume.image)
            writer.writerow(
                {
                    "key": volume.key,
                    "title": volume.title.replace("\n", " "),
                    "path": str(volume.path),
                    "shape_xyz": " x ".join(map(str, geometry["shape_xyz"])),
                    "voxel_size_mm_xyz": " x ".join(
                        f"{value:.9g}" for value in geometry["voxel_size_mm_xyz"]
                    ),
                    "fov_mm_xyz": " x ".join(
                        f"{value:.9g}" for value in geometry["fov_mm_xyz"]
                    ),
                    "center_mm_ras": ", ".join(
                        f"{value:.9g}" for value in geometry["center_mm_ras"]
                    ),
                    "display_scale": f"{volume.display_scale:.12g}",
                }
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path-agnostic JSON review configuration. Cannot be mixed with legacy inputs.",
    )
    parser.add_argument("--batch-manifest", type=Path, help="Legacy product-review input.")
    parser.add_argument("--grappa-nifti", type=Path, help="Legacy product-review input.")
    parser.add_argument("--full-resolution-nifti", type=Path, help="Legacy product-review input.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--slice-world-mm",
        type=float,
        nargs=3,
        metavar=("R", "A", "S"),
        help="RAS world point defining the three orthogonal review slices; default is FOV center.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.5,
        help="Per-volume positive-voxel percentile mapped to white for display only.",
    )
    args = parser.parse_args(argv)
    legacy = (args.batch_manifest, args.grappa_nifti, args.full_resolution_nifti)
    if args.config is not None:
        if any(value is not None for value in legacy) or args.output_dir is not None:
            parser.error("--config cannot be mixed with legacy input/output arguments")
    elif any(value is None for value in (*legacy, args.output_dir)):
        parser.error(
            "either --config or all of --batch-manifest, --grappa-nifti, "
            "--full-resolution-nifti, and --output-dir is required"
        )
    return args


def _legacy_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_manifest": args.batch_manifest.expanduser().resolve(),
        "output_dir": args.output_dir.expanduser().resolve(),
        "volumes": [
            {
                "key": "grappa",
                "title": "No-Wave GRAPPA\n1 x 1 x 1 mm",
                "path": args.grappa_nifti,
                "require_canonical_export": False,
            },
            {
                "key": "full_resolution_llr",
                "title": "Wave R3x2 corrected LLR\n1 x 1 x 1 mm",
                "path": args.full_resolution_nifti,
                "require_canonical_export": True,
            },
        ],
        "matched_grid_reference_key": "full_resolution_llr",
        "retrospective_title_template": "Retro LR corrected LLR\\n{resolution_mm} mm",
        "native_figure_title": (
            "Retrospective low-resolution native-grid review\n"
            "Same RAS locations and physical FOV; no spatial resampling"
        ),
        "matched_figure_title": (
            "Retrospective low-resolution matched-grid review\n"
            "LR magnitudes linearly resampled to the full-resolution 1 mm RAS grid"
        ),
        "scientific_scope": {
            "regularization": "corrected LLR block 8, lambda 2e-5",
            "grappa_role": "temporary qualitative anatomical comparison",
            "full_resolution_llr_role": "same-regularizer resolution reference",
            "dicom_used": False,
            "bet_mask_used": False,
            "quantitative_ranking_performed": False,
        },
        "provenance_manifests": [],
        "config_path": None,
    }


def _configured_settings(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("Review config format_version must be 1.")
    configured_volumes = config.get("reference_volumes")
    if not isinstance(configured_volumes, list) or not configured_volumes:
        raise ValueError("Review config reference_volumes must be a non-empty list.")
    volumes = []
    keys: set[str] = set()
    for record in configured_volumes:
        if not isinstance(record, dict):
            raise ValueError("Each reference_volumes entry must be an object.")
        key = str(record["key"])
        if key in keys:
            raise ValueError(f"Duplicate reference volume key: {key}")
        keys.add(key)
        volumes.append(
            {
                "key": key,
                "title": str(record["title"]).replace("\\n", "\n"),
                "path": _resolve_config_path(config_path, str(record["path"])),
                "require_canonical_export": bool(
                    record.get("require_canonical_export", False)
                ),
            }
        )
    matched_key = str(config["matched_grid_reference_key"])
    if matched_key not in keys:
        raise ValueError("matched_grid_reference_key is not a reference volume key.")
    display = config.get("display", {})
    if not isinstance(display, dict):
        raise ValueError("Review config display must be an object.")
    scope = config.get("scientific_scope", {})
    if not isinstance(scope, dict):
        raise ValueError("Review config scientific_scope must be an object.")
    provenance_values = config.get("provenance_manifests", [])
    if not isinstance(provenance_values, list):
        raise ValueError("Review config provenance_manifests must be a list.")
    return {
        "batch_manifest": _resolve_config_path(config_path, str(config["batch_manifest"])),
        "output_dir": _resolve_config_path(config_path, str(config["output_dir"])),
        "volumes": volumes,
        "matched_grid_reference_key": matched_key,
        "retrospective_title_template": str(
            config.get(
                "retrospective_title_template",
                "Retrospective low resolution\\n{case_label}\\n{resolution_mm} mm",
            )
        ),
        "native_figure_title": str(
            display.get(
                "native_figure_title",
                "Retrospective low-resolution native-grid review\\n"
                "Same RAS locations and physical FOV; no spatial resampling",
            )
        ).replace("\\n", "\n"),
        "matched_figure_title": str(
            display.get(
                "matched_figure_title",
                "Retrospective low-resolution matched-grid review\\n"
                "Low-resolution magnitudes linearly resampled to the configured reference grid",
            )
        ).replace("\\n", "\n"),
        "scientific_scope": scope,
        "provenance_manifests": [
            _resolve_config_path(config_path, str(value)) for value in provenance_values
        ],
        "config_path": config_path,
        "display_percentile": display.get("percentile"),
        "slice_world_mm": display.get("slice_world_mm_ras"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = (
        _legacy_settings(args)
        if getattr(args, "config", None) is None
        else _configured_settings(args.config)
    )
    configured_percentile = settings.get("display_percentile")
    percentile = float(
        args.display_percentile if configured_percentile is None else configured_percentile
    )
    if not 90.0 <= percentile <= 100.0:
        raise ValueError("Display percentile must be between 90 and 100.")
    batch_manifest = settings["batch_manifest"]
    output_dir = settings["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Review output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    volumes = [
        _load_volume(
            record["key"],
            record["title"],
            record["path"],
            percentile,
            require_canonical_export=record["require_canonical_export"],
        )
        for record in settings["volumes"]
    ]
    volumes.extend(
        _load_retro_volumes(
            batch_manifest,
            percentile,
            title_template=settings["retrospective_title_template"],
        )
    )
    validate_shared_physical_geometry(volumes)
    reference_volume = next(
        volume for volume in volumes if volume.key == settings["matched_grid_reference_key"]
    )
    reference_image = reference_volume.image
    reference_center = np.asarray(geometry_record(reference_image)["center_mm_ras"])
    world_point = (
        reference_center
        if settings.get("slice_world_mm") is None and args.slice_world_mm is None
        else np.asarray(
            settings.get("slice_world_mm")
            if settings.get("slice_world_mm") is not None
            else args.slice_world_mm,
            dtype=float,
        )
    )

    native_path = output_dir / "native_grid_comparison.png"
    matched_path = output_dir / "matched_1mm_grid_comparison.png"
    geometry_path = output_dir / "input_geometry.csv"
    slice_records = _render_native_comparison(
        volumes, world_point, native_path, settings["native_figure_title"]
    )
    _render_matched_grid_comparison(
        volumes,
        reference_image,
        world_point,
        matched_path,
        settings["matched_figure_title"],
    )
    _write_geometry_table(geometry_path, volumes)

    manifest = {
        "format_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "physical-coordinate visual review of retrospective low-resolution reconstructions",
        "scientific_scope": settings["scientific_scope"],
        "review_config": (
            None
            if settings["config_path"] is None
            else {
                "path": str(settings["config_path"]),
                "sha256": sha256_file(settings["config_path"]),
                "snapshot": _load_json(settings["config_path"]),
            }
        ),
        "batch_manifest": {
            "path": str(batch_manifest),
            "sha256": sha256_file(batch_manifest),
        },
        "display": {
            "slice_world_point_mm_ras": world_point.tolist(),
            "planes": list(PLANES),
            "native_grid": "nearest native slice; physical-mm extent; no spatial resampling",
            "matched_grid": "linear interpolation to the configured reference RAS grid",
            "matched_grid_reference_key": reference_volume.key,
            "intensity": (
                f"each volume divided by its own positive-voxel p{percentile:g}; "
                "display-only scaling, not a signal or SNR comparison"
            ),
            "window_after_scaling": [0.0, 1.0],
            "slice_records": slice_records,
        },
        "inputs": [
            {
                "key": volume.key,
                "title": volume.title,
                "path": str(volume.path),
                "sha256": sha256_file(volume.path),
                "case_manifest": (
                    None
                    if volume.case_manifest is None
                    else {
                        "path": str(volume.case_manifest),
                        "sha256": sha256_file(volume.case_manifest),
                    }
                ),
                "export_sidecar": (
                    None
                    if volume.export_sidecar is None
                    else {
                        "path": str(volume.export_sidecar),
                        "sha256": sha256_file(volume.export_sidecar),
                    }
                ),
                "geometry": geometry_record(volume.image),
                "display_scale": volume.display_scale,
            }
            for volume in volumes
        ],
        "provenance_manifests": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in settings["provenance_manifests"]
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (native_path, matched_path, geometry_path)
        ],
    }
    manifest_path = output_dir / "review_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Retrospective low-resolution review manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
