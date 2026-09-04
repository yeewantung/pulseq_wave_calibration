"""Generate reviewable GRE whole-head-mask parameter candidates."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from .bart_io import sha256_file
from .gre import GRE_BART_ARRAY_AXIS_FLIPS, GRE_SHARED_WAVELET_LAMBDA
from .nifti_collection import HeadMaskParameters, create_whole_head_mask


DEFAULT_GRE_RELATIVE_THRESHOLDS = (0.01, 0.015, 0.02, 0.03)
DEFAULT_GRE_CORE_RELATIVE_THRESHOLDS = (0.03, 0.05, 0.08)
DEFAULT_GRE_MAXIMUM_GROWTH_DISTANCES_MM = (8.0, 12.0, 16.0)


def derive_gre_head_mask_candidates(
    magnitude_nifti: str | Path,
    output_directory: str | Path,
    *,
    relative_thresholds: Sequence[float] = DEFAULT_GRE_RELATIVE_THRESHOLDS,
    core_relative_thresholds: Sequence[float] = DEFAULT_GRE_CORE_RELATIVE_THRESHOLDS,
    maximum_growth_distances_mm: Sequence[
        float
    ] = DEFAULT_GRE_MAXIMUM_GROWTH_DISTANCES_MM,
    background_mad_multiplier: float = 6.0,
    smoothing_mm: float = 1.0,
    boundary_width_mm: float = 5.0,
    opening_radius_mm: float = 0.0,
    closing_radius_mm: float = 1.5,
    dilation_radius_mm: float = 0.0,
) -> dict[str, Any]:
    """Build a non-ranking GRE head-mask sweep for visual review.

    Args:
        magnitude_nifti: Canonical normal native-R3x1 selected-Wavelet echo-1
            GRE magnitude NIfTI.
        output_directory: New exact directory selected by the user.
        relative_thresholds: Low foreground thresholds relative to positive p99.
        core_relative_thresholds: Trusted-core thresholds relative to positive p99.
        maximum_growth_distances_mm: Maximum physical growth distances from the core.
        background_mad_multiplier: Boundary-noise multiplier shared by candidates.
        smoothing_mm: Gaussian smoothing standard deviation.
        boundary_width_mm: Boundary shell width for noise estimation.
        opening_radius_mm: Physical opening radius.
        closing_radius_mm: Physical closing radius.
        dilation_radius_mm: Final conservative dilation radius.

    Returns:
        JSON-native sweep manifest. No candidate is ranked or selected.

    Raises:
        FileExistsError: If the requested output directory already exists.
        FileNotFoundError: If the NIfTI or matching JSON sidecar is absent.
        ValueError: If the source is not the required corrected GRE magnitude or
            the candidate grid is invalid.

    Side Effects:
        Atomically installs candidate masks, overlay PNGs, review instructions,
        and a manifest below ``output_directory``. Reconstruction outputs are
        never modified.
    """

    source = Path(magnitude_nifti).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new GRE mask-sweep directory: {destination}")
    image, data, sidecar, metadata = _load_gre_mask_source(source)
    parameter_sets = _candidate_parameters(
        relative_thresholds=relative_thresholds,
        core_relative_thresholds=core_relative_thresholds,
        maximum_growth_distances_mm=maximum_growth_distances_mm,
        background_mad_multiplier=background_mad_multiplier,
        smoothing_mm=smoothing_mm,
        boundary_width_mm=boundary_width_mm,
        opening_radius_mm=opening_radius_mm,
        closing_radius_mm=closing_radius_mm,
        dilation_radius_mm=dilation_radius_mm,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        masks = staging / "masks"
        overlays = staging / "overlays"
        masks.mkdir()
        overlays.mkdir()
        records: list[dict[str, Any]] = []
        for parameters in parameter_sets:
            candidate_id = _candidate_id(parameters)
            try:
                mask_image, details = create_whole_head_mask(image, parameters)
            except ValueError as exc:
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "rejected_by_sanity_check",
                        "parameters": asdict(parameters),
                        "error": str(exc),
                    }
                )
                continue
            mask_path = masks / f"{candidate_id}.nii.gz"
            overlay_path = overlays / f"{candidate_id}.png"
            nib.save(mask_image, str(mask_path))
            _write_mask_overlay(
                data,
                np.asarray(mask_image.dataobj, dtype=bool),
                overlay_path,
                candidate_id=candidate_id,
                parameters=parameters,
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "status": "ready_for_visual_review",
                    "parameters": asdict(parameters),
                    "details": details,
                    "mask": _relative_file_record(mask_path, staging),
                    "overlay": _relative_file_record(overlay_path, staging),
                }
            )

        manifest = {
            "format_version": 1,
            "status": "gre_head_mask_parameter_candidates_ready",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "magnitude_nifti": str(source),
                "magnitude_nifti_sha256": sha256_file(source),
                "json_sidecar": str(sidecar),
                "json_sidecar_sha256": sha256_file(sidecar),
                "shape": list(image.shape),
                "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
                "axis_codes": list(nib.aff2axcodes(image.affine)),
                "metadata": metadata,
            },
            "candidate_grid": {
                "relative_thresholds": [float(value) for value in relative_thresholds],
                "core_relative_thresholds": [
                    float(value) for value in core_relative_thresholds
                ],
                "maximum_growth_distances_mm": [
                    float(value) for value in maximum_growth_distances_mm
                ],
                "fixed_parameters": {
                    "background_mad_multiplier": float(background_mad_multiplier),
                    "smoothing_mm": float(smoothing_mm),
                    "boundary_width_mm": float(boundary_width_mm),
                    "opening_radius_mm": float(opening_radius_mm),
                    "closing_radius_mm": float(closing_radius_mm),
                    "dilation_radius_mm": float(dilation_radius_mm),
                },
            },
            "candidate_count": len(records),
            "ready_for_visual_review_count": sum(
                record["status"] == "ready_for_visual_review" for record in records
            ),
            "selection": {
                "status": "not_selected",
                "automatic_ranking": False,
                "automatic_selection": False,
                "required_review": (
                    "Inspect all three anatomical planes and select one candidate "
                    "that retains the complete outer head while excluding disconnected "
                    "background and table signal."
                ),
            },
            "candidates": records,
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "REVIEW_INSTRUCTIONS.txt").write_text(
            "Review every PNG in overlays/. Choose no candidate from coverage alone.\n"
            "The accepted mask must retain the complete scalp/head in sagittal, coronal, "
            "and axial views while excluding disconnected background and table signal.\n"
            "Record the chosen candidate_id only after visual review.\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _load_gre_mask_source(
    path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray, Path, dict[str, Any]]:
    """Load and validate the required GRE mask-derivation source.

    Args:
        path: Candidate magnitude NIfTI path.

    Returns:
        Loaded image, float32 magnitude, sidecar path, and sidecar metadata.

    Raises:
        FileNotFoundError: If the NIfTI or sidecar is absent.
        ValueError: If geometry, image role, echo, case, branch, or orientation
            provenance differs from the reviewed GRE contract.
    """

    if not path.is_file():
        raise FileNotFoundError(f"GRE mask-source NIfTI does not exist: {path}")
    sidecar = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"GRE mask-source JSON does not exist: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"GRE mask-source sidecar is not a JSON object: {sidecar}")
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if len(image.shape) != 3 or not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("GRE mask source must be one finite nonnegative 3D magnitude.")
    if tuple(nib.aff2axcodes(image.affine)) != ("R", "A", "S"):
        raise ValueError("GRE mask source must be stored in canonical RAS orientation.")
    if "normal" not in path.parts or "selected_wavelet" not in path.parts:
        raise ValueError("Use the normal selected-Wavelet GRE magnitude as mask source.")
    if metadata.get("ImagePart") != "mag":
        raise ValueError("GRE mask source sidecar must identify ImagePart='mag'.")
    if int(metadata.get("EchoNumber", 0)) != 1:
        raise ValueError("Derive the shared GRE head mask from echo 1.")
    if metadata.get("CaseID") != "native_r3x1":
        raise ValueError("Derive the GRE head mask from the native_r3x1 normal grid.")
    orientation = metadata.get("OrientationPolicy", {})
    if orientation.get("array_axis_flips") != list(GRE_BART_ARRAY_AXIS_FLIPS):
        raise ValueError("GRE mask source predates the reviewed orientation correction.")
    selected_lambda = float(metadata.get("GRESelectedWaveletLambda", np.nan))
    if not np.isclose(selected_lambda, GRE_SHARED_WAVELET_LAMBDA, atol=0, rtol=0):
        raise ValueError("GRE mask source does not use the reviewed shared Wavelet lambda.")
    return image, data, sidecar, metadata


def _candidate_parameters(
    *,
    relative_thresholds: Sequence[float],
    core_relative_thresholds: Sequence[float],
    maximum_growth_distances_mm: Sequence[float],
    background_mad_multiplier: float,
    smoothing_mm: float,
    boundary_width_mm: float,
    opening_radius_mm: float,
    closing_radius_mm: float,
    dilation_radius_mm: float,
) -> tuple[HeadMaskParameters, ...]:
    """Expand and validate the GRE threshold/core/growth candidate grid.

    Args:
        relative_thresholds: Candidate low thresholds.
        core_relative_thresholds: Candidate core thresholds.
        maximum_growth_distances_mm: Candidate growth distances.
        background_mad_multiplier: Fixed boundary-noise multiplier.
        smoothing_mm: Fixed Gaussian smoothing scale.
        boundary_width_mm: Fixed boundary-shell width.
        opening_radius_mm: Fixed opening radius.
        closing_radius_mm: Fixed closing radius.
        dilation_radius_mm: Fixed dilation radius.

    Returns:
        Deterministically ordered unique mask parameter sets.

    Raises:
        ValueError: If a swept sequence is empty, duplicated, non-finite, or
            contains values incompatible with the mask algorithm.
    """

    grids = []
    for name, values, positive in (
        ("relative_thresholds", relative_thresholds, True),
        ("core_relative_thresholds", core_relative_thresholds, True),
        ("maximum_growth_distances_mm", maximum_growth_distances_mm, False),
    ):
        grid = tuple(float(value) for value in values)
        if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all():
            raise ValueError(f"{name} must contain unique finite values.")
        if any(value <= 0 if positive else value < 0 for value in grid):
            raise ValueError(f"{name} contains an invalid value.")
        grids.append(grid)
    candidates = []
    for relative, core, growth in product(*grids):
        if core < relative:
            raise ValueError("Every GRE core threshold must be at least its low threshold.")
        candidates.append(
            HeadMaskParameters(
                relative_threshold=relative,
                core_relative_threshold=core,
                maximum_growth_distance_mm=growth,
                background_mad_multiplier=float(background_mad_multiplier),
                smoothing_mm=float(smoothing_mm),
                boundary_width_mm=float(boundary_width_mm),
                opening_radius_mm=float(opening_radius_mm),
                closing_radius_mm=float(closing_radius_mm),
                dilation_radius_mm=float(dilation_radius_mm),
            )
        )
    return tuple(candidates)


def _candidate_id(parameters: HeadMaskParameters) -> str:
    """Return a filesystem-safe identity for one parameter set.

    Args:
        parameters: Candidate head-mask parameters.

    Returns:
        Stable identifier encoding the swept values.
    """

    def token(value: float) -> str:
        """Format one nonnegative decimal value for a filename token."""

        return f"{value:g}".replace(".", "p")

    return (
        f"rel-{token(parameters.relative_threshold)}_"
        f"core-{token(parameters.core_relative_threshold)}_"
        f"grow-{token(parameters.maximum_growth_distance_mm)}mm"
    )


def _write_mask_overlay(
    magnitude: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    *,
    candidate_id: str,
    parameters: HeadMaskParameters,
) -> None:
    """Write orthogonal GRE magnitude views with the mask boundary overlaid.

    Args:
        magnitude: Canonical-RAS source magnitude.
        mask: Candidate binary mask on the same grid.
        output_path: Destination PNG path.
        candidate_id: Stable candidate label.
        parameters: Candidate parameters displayed in the title.

    Returns:
        None.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    positive = magnitude[magnitude > 0]
    upper = float(np.percentile(positive, 99.5))
    figure = Figure(figsize=(12, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)
    planes = ((0, "sagittal"), (1, "coronal"), (2, "axial"))
    fractions = (0.25, 0.5, 0.75)
    for row, (dimension, label_name) in enumerate(planes):
        for column, fraction in enumerate(fractions):
            axis = axes[row, column]
            index = min(
                magnitude.shape[dimension] - 1,
                int(round((magnitude.shape[dimension] - 1) * fraction)),
            )
            image_plane = np.rot90(np.take(magnitude, index, axis=dimension))
            mask_plane = np.rot90(np.take(mask, index, axis=dimension))
            axis.imshow(image_plane, cmap="gray", vmin=0, vmax=upper, origin="lower")
            if np.any(mask_plane) and np.any(~mask_plane):
                axis.contour(
                    mask_plane.astype(float),
                    levels=(0.5,),
                    colors=("red",),
                    linewidths=0.8,
                )
            axis.set_title(f"{label_name} {fraction:.0%} (index {index})")
            axis.axis("off")
    figure.suptitle(
        f"{candidate_id}\n"
        f"relative={parameters.relative_threshold:g}, "
        f"core={parameters.core_relative_threshold:g}, "
        f"growth={parameters.maximum_growth_distance_mm:g} mm"
    )
    figure.savefig(output_path, dpi=150, format="png")


def _relative_file_record(path: Path, root: Path) -> dict[str, Any]:
    """Describe one generated file relative to its sweep root.

    Args:
        path: Existing generated file.
        root: Staging root used for relative paths.

    Returns:
        Relative path and SHA-256 identity.
    """

    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON object.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping.

    Returns:
        None.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
