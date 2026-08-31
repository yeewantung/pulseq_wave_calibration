"""Build canonical and whole-head-masked MPRAGE NIfTI collections."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from .bart_io import sha256_file


COLLECTION_BUILDER = "wave_retro_lr.nifti_collection"
RETRO_CASES = (
    "native_r3x2",
    "lr_x_1p5mm_r3x2",
    "lr_y_1p5mm_r3x2",
    "lr_xy_1p25mm_r3x2",
)


@dataclass(frozen=True)
class HeadMaskParameters:
    """Configure conservative whole-head foreground extraction.

    Attributes:
        relative_threshold: Minimum foreground threshold as a fraction of the
            smoothed positive-voxel 99th percentile.
        core_relative_threshold: Higher relative threshold used to identify a
            trusted central head component for optional distance-limited growth.
        maximum_growth_distance_mm: Maximum physical distance that low-threshold
            foreground may extend beyond the trusted core; zero disables this
            constraint.
        background_mad_multiplier: Robust boundary-noise multiplier added to
            the boundary median.
        smoothing_mm: Gaussian smoothing standard deviation in millimetres.
        boundary_width_mm: Physical width of the boundary shell used to
            estimate background noise.
        opening_radius_mm: Physical radius used to remove narrow foreground
            bridges and protrusions before connected-component selection.
        closing_radius_mm: Physical radius used to connect nearby head tissue.
        dilation_radius_mm: Physical radius added after hole filling so the
            mask conservatively retains the outer head surface.
    """

    relative_threshold: float = 0.02
    core_relative_threshold: float = 0.05
    maximum_growth_distance_mm: float = 12.0
    background_mad_multiplier: float = 6.0
    smoothing_mm: float = 1.0
    boundary_width_mm: float = 5.0
    opening_radius_mm: float = 0.0
    closing_radius_mm: float = 1.5
    dilation_radius_mm: float = 0.0


def build_mprage_nifti_collection(
    output_root: str | Path,
    *,
    require_retro: bool = False,
    parameters: HeadMaskParameters | None = None,
) -> dict[str, Any]:
    """Create canonical-copy and whole-head-masked MPRAGE NIfTI trees.

    Args:
        output_root: Reconstruction root containing ``normal/nifti`` and,
            when available, the four ``retro/<case>/nifti`` directories.
        require_retro: Whether all four retrospective case directories must
            contain complete magnitude/phase NIfTI and JSON pairs.
        parameters: Optional whole-head mask extraction parameters.

    Returns:
        JSON-native collection manifest written below ``nifti_collection``.

    Raises:
        FileNotFoundError: If required canonical NIfTI inputs are absent.
        FileExistsError: If an existing collection is not owned by this tool.
        ValueError: If NIfTIs, sidecars, mask parameters, or grids are invalid.

    Side Effects:
        Replaces a prior tool-owned ``nifti_collection`` atomically after the
        complete replacement has been built in a sibling staging directory.
        Canonical reconstruction outputs are never modified.
    """
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reconstruction output root does not exist: {root}")
    mask_parameters = parameters or HeadMaskParameters()
    _validate_parameters(mask_parameters)

    case_sources = _discover_case_sources(root, require_retro=require_retro)
    normal_magnitude = _select_normal_magnitude(case_sources["normal"])
    normal_image, _ = _load_validated_nifti(normal_magnitude)
    head_mask, mask_details = create_whole_head_mask(normal_image, mask_parameters)

    collection = root / "nifti_collection"
    _validate_existing_collection(collection)
    staging = Path(tempfile.mkdtemp(prefix=".nifti_collection-", dir=root))
    try:
        manifest = _materialize_collection(
            staging,
            root,
            case_sources,
            normal_magnitude,
            head_mask,
            mask_parameters,
            mask_details,
        )
        _write_json(staging / "manifest.json", manifest)
        _replace_owned_collection(staging, collection)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def create_whole_head_mask(
    magnitude_image: nib.spatialimages.SpatialImage,
    parameters: HeadMaskParameters | None = None,
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Extract one conservative whole-head mask from a normal magnitude image.

    Args:
        magnitude_image: Canonical-RAS, finite, real-valued 3D normal MPRAGE
            magnitude image.
        parameters: Optional physical and intensity mask parameters.

    Returns:
        A uint8 NIfTI mask on the input grid and JSON-native threshold,
        morphology, and coverage details.

    Raises:
        ValueError: If the image or parameters are invalid, or if foreground
            extraction produces an implausible empty or near-global mask.
    """
    config = parameters or HeadMaskParameters()
    _validate_parameters(config)
    data = _validated_image_data(magnitude_image, label_name="Normal magnitude")
    if np.any(data < 0):
        raise ValueError("Normal magnitude contains negative values.")
    zooms = _validated_zooms(magnitude_image)

    # Work in physical units so native and anisotropic inputs use the same mask rule.
    sigma_voxels = tuple(config.smoothing_mm / value for value in zooms)
    smoothed = gaussian_filter(data.astype(np.float32, copy=False), sigma=sigma_voxels)
    positive = smoothed[smoothed > 0]
    if positive.size == 0:
        raise ValueError("Normal magnitude contains no positive signal.")
    robust_signal = float(np.percentile(positive, 99.0))
    if not np.isfinite(robust_signal) or robust_signal <= 0:
        raise ValueError("Normal magnitude has no finite positive robust scale.")

    boundary = smoothed[_boundary_shell(smoothed.shape, zooms, config.boundary_width_mm)]
    boundary_median = float(np.median(boundary))
    boundary_mad = float(np.median(np.abs(boundary - boundary_median)))
    robust_sigma = 1.4826 * boundary_mad
    threshold = max(
        config.relative_threshold * robust_signal,
        boundary_median + config.background_mad_multiplier * robust_sigma,
    )

    # A high-confidence core can bound how far dim connected artifacts may grow.
    candidate = smoothed >= threshold
    core_threshold = max(threshold, config.core_relative_threshold * robust_signal)
    if config.maximum_growth_distance_mm > 0:
        core = _largest_component(smoothed >= core_threshold)
        core = binary_fill_holes(core)
        distance_from_core = distance_transform_edt(~core, sampling=zooms)
        candidate &= distance_from_core <= config.maximum_growth_distance_mm

    # Opening removes thin artifact bridges before closing low-signal scalp gaps.
    candidate = binary_opening(
        candidate, structure=_ellipsoid_structure(zooms, config.opening_radius_mm)
    )
    candidate = binary_closing(
        candidate, structure=_ellipsoid_structure(zooms, config.closing_radius_mm)
    )
    foreground = _largest_component(candidate)
    foreground = binary_fill_holes(foreground)
    foreground = binary_dilation(
        foreground, structure=_ellipsoid_structure(zooms, config.dilation_radius_mm)
    )
    foreground = np.asarray(foreground, dtype=bool)
    fraction = float(np.mean(foreground))
    if not 0.005 <= fraction <= 0.85:
        raise ValueError(
            "Whole-head mask coverage is implausible: "
            f"{fraction:.6f}; expected 0.005 through 0.85."
        )

    header = magnitude_image.header.copy()
    header.set_data_dtype(np.uint8)
    mask_image = nib.Nifti1Image(
        foreground.astype(np.uint8), magnitude_image.affine, header=header
    )
    details = {
        "algorithm": (
            "Gaussian smoothing; maximum of relative and robust boundary-noise "
            "thresholds; physical opening and closing; largest 26-connected "
            "component; 3D hole filling; physical dilation"
        ),
        "threshold": float(threshold),
        "core_threshold": float(core_threshold),
        "robust_signal_positive_p99": robust_signal,
        "boundary_median": boundary_median,
        "boundary_mad": boundary_mad,
        "boundary_robust_sigma": robust_sigma,
        "mask_voxels": int(np.count_nonzero(foreground)),
        "mask_fraction": fraction,
        "voxel_size_mm": [float(value) for value in zooms],
    }
    return mask_image, details


def resample_head_mask(
    mask_image: nib.spatialimages.SpatialImage,
    target_image: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, str]:
    """Map a normal-grid whole-head mask to a target NIfTI grid.

    Args:
        mask_image: Binary whole-head mask in canonical-RAS physical space.
        target_image: Canonical-RAS image whose grid should receive the mask.

    Returns:
        Boolean target-grid mask and either ``same_grid`` or
        ``nearest_neighbor_physical_space`` as the mapping description.

    Raises:
        ValueError: If either grid is invalid or the mapped mask is empty.
    """
    _validated_image_data(mask_image, label_name="Head mask")
    _validated_image_data(target_image, label_name="Mask target")
    if mask_image.shape == target_image.shape and np.allclose(
        mask_image.affine, target_image.affine, rtol=1e-5, atol=1e-4
    ):
        mapped = np.asanyarray(mask_image.dataobj) > 0.5
        method = "same_grid"
    else:
        mapped_image = resample_from_to(mask_image, target_image, order=0, mode="constant", cval=0)
        mapped = np.asanyarray(mapped_image.dataobj) > 0.5
        method = "nearest_neighbor_physical_space"
    if not np.any(mapped):
        raise ValueError("Normal whole-head mask has no overlap with a target NIfTI grid.")
    return np.asarray(mapped, dtype=bool), method


def _validate_parameters(parameters: HeadMaskParameters) -> None:
    """Validate whole-head extraction parameters.

    Args:
        parameters: Parameter set to validate.

    Returns:
        None.

    Raises:
        ValueError: If a threshold or physical extent is outside its accepted
            nonnegative range.
    """
    if not 0 < parameters.relative_threshold < 1:
        raise ValueError("relative_threshold must be strictly between zero and one.")
    if not 0 < parameters.core_relative_threshold < 1:
        raise ValueError("core_relative_threshold must be strictly between zero and one.")
    if parameters.core_relative_threshold < parameters.relative_threshold:
        raise ValueError("core_relative_threshold must not be below relative_threshold.")
    if parameters.maximum_growth_distance_mm < 0:
        raise ValueError("maximum_growth_distance_mm must be nonnegative.")
    if parameters.background_mad_multiplier < 0:
        raise ValueError("background_mad_multiplier must be nonnegative.")
    for name in (
        "smoothing_mm",
        "boundary_width_mm",
        "opening_radius_mm",
        "closing_radius_mm",
        "dilation_radius_mm",
    ):
        if getattr(parameters, name) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if parameters.boundary_width_mm == 0:
        raise ValueError("boundary_width_mm must be positive.")


def _discover_case_sources(
    output_root: Path, *, require_retro: bool
) -> dict[str, list[tuple[Path, Path]]]:
    """Discover complete canonical NIfTI/JSON pairs for available cases.

    Args:
        output_root: Existing reconstruction output root.
        require_retro: Whether all four retrospective cases are mandatory.

    Returns:
        Mapping from case label to sorted NIfTI and JSON sidecar pairs.

    Raises:
        FileNotFoundError: If normal or required retrospective pairs are absent.
        ValueError: If a source directory contains an incomplete NIfTI pair.
    """
    sources = {"normal": _discover_nifti_pairs(output_root / "normal" / "nifti")}
    if not sources["normal"]:
        raise FileNotFoundError(
            f"No normal canonical NIfTI files found: {output_root / 'normal' / 'nifti'}"
        )
    for case in RETRO_CASES:
        directory = output_root / "retro" / case / "nifti"
        pairs = _discover_nifti_pairs(directory)
        if pairs:
            sources[case] = pairs
        elif require_retro:
            raise FileNotFoundError(f"No canonical NIfTI files found for {case}: {directory}")
    return sources


def _discover_nifti_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Find NIfTI files and require matching JSON sidecars.

    Args:
        directory: Canonical NIfTI directory, which may not yet exist.

    Returns:
        Sorted ``(nifti, json)`` pairs, or an empty list for a missing directory.

    Raises:
        ValueError: If any discovered NIfTI lacks its matching JSON sidecar or
            if two nested sources would flatten to the same filename.
    """
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"Canonical NIfTI path is not a directory: {directory}")
    pairs: list[tuple[Path, Path]] = []
    names: set[str] = set()
    for nifti in sorted(directory.rglob("*.nii.gz")):
        sidecar = nifti.with_name(nifti.name[: -len(".nii.gz")] + ".json")
        if not sidecar.is_file():
            raise ValueError(f"Canonical NIfTI has no JSON sidecar: {nifti}")
        if nifti.name in names:
            raise ValueError(f"Flattened canonical NIfTI filename is duplicated: {nifti.name}")
        names.add(nifti.name)
        pairs.append((nifti.resolve(), sidecar.resolve()))
    return pairs


def _select_normal_magnitude(pairs: list[tuple[Path, Path]]) -> Path:
    """Select the unique normal magnitude NIfTI used to estimate the mask.

    Args:
        pairs: Normal canonical NIfTI and JSON pairs.

    Returns:
        Unique magnitude NIfTI path.

    Raises:
        ValueError: If the normal directory does not contain exactly one
            ``_part-mag_`` NIfTI.
    """
    matches = [nifti for nifti, _ in pairs if "_part-mag_" in nifti.name]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one normal '_part-mag_' NIfTI for head masking; "
            f"found {len(matches)}."
        )
    return matches[0]


def _load_validated_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load one canonical-RAS, real, finite 3D NIfTI.

    Args:
        path: NIfTI path to load without modification.

    Returns:
        Loaded NIfTI image and its real-valued NumPy data.

    Raises:
        ValueError: If dimensionality, orientation, affine, or samples are invalid.
    """
    image = nib.load(str(path))
    data = _validated_image_data(image, label_name=str(path))
    return image, data


def _validated_image_data(
    image: nib.spatialimages.SpatialImage, *, label_name: str
) -> np.ndarray:
    """Validate a loaded NIfTI-like image and return its data.

    Args:
        image: Loaded spatial image to validate.
        label_name: Human-readable label used in validation errors.

    Returns:
        Finite, real-valued 3D NumPy array.

    Raises:
        ValueError: If the image is not nonempty 3D canonical RAS with a finite
            affine and finite real-valued samples.
    """
    if len(image.shape) != 3 or any(int(value) < 1 for value in image.shape):
        raise ValueError(f"{label_name} must be a nonempty 3D image: {image.shape}")
    if not np.isfinite(image.affine).all():
        raise ValueError(f"{label_name} has a non-finite affine.")
    axis_codes = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    if axis_codes != ("R", "A", "S"):
        raise ValueError(f"{label_name} is not canonical RAS: {axis_codes}")
    data = np.asanyarray(image.dataobj)
    if np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError(f"{label_name} must contain finite real-valued samples.")
    _validated_zooms(image)
    return data


def _validated_zooms(image: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    """Return validated positive spatial voxel sizes.

    Args:
        image: Spatial image whose first three zooms are inspected.

    Returns:
        Positive finite XYZ voxel sizes in millimetres.

    Raises:
        ValueError: If a spatial zoom is absent, non-finite, or nonpositive.
    """
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(zooms) != 3 or not np.isfinite(zooms).all() or any(value <= 0 for value in zooms):
        raise ValueError(f"Invalid spatial voxel sizes: {zooms}")
    return zooms


def _boundary_shell(
    shape: tuple[int, ...], zooms: tuple[float, float, float], width_mm: float
) -> np.ndarray:
    """Construct a physical-width boundary shell for noise estimation.

    Args:
        shape: Three-dimensional image matrix.
        zooms: XYZ voxel sizes in millimetres.
        width_mm: Requested boundary-shell width in millimetres.

    Returns:
        Boolean array marking voxels near any image boundary.
    """
    shell = np.zeros(shape, dtype=bool)
    for axis, (size, spacing) in enumerate(zip(shape, zooms, strict=True)):
        width = max(1, min(int(np.ceil(width_mm / spacing)), max(1, size // 4)))
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, width)
        upper[axis] = slice(size - width, size)
        shell[tuple(lower)] = True
        shell[tuple(upper)] = True
    return shell


def _ellipsoid_structure(
    zooms: tuple[float, float, float], radius_mm: float
) -> np.ndarray:
    """Build a binary ellipsoid approximating a physical-radius ball.

    Args:
        zooms: XYZ voxel sizes in millimetres.
        radius_mm: Requested nonnegative physical radius.

    Returns:
        Three-dimensional boolean morphology footprint containing its centre.
    """
    if radius_mm == 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = tuple(max(1, int(np.ceil(radius_mm / spacing))) for spacing in zooms)
    coordinates = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    distance = np.zeros(tuple(2 * radius + 1 for radius in radii), dtype=np.float64)
    for coordinate, spacing in zip(coordinates, zooms, strict=True):
        distance += (coordinate * spacing / radius_mm) ** 2
    return distance <= 1.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Retain the largest 26-connected component of a binary array.

    Args:
        mask: Three-dimensional candidate foreground array.

    Returns:
        Boolean mask containing only the largest connected component.

    Raises:
        ValueError: If the candidate contains no foreground voxels.
    """
    components, count = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError("Whole-head threshold produced no foreground component.")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _materialize_collection(
    staging: Path,
    output_root: Path,
    case_sources: dict[str, list[tuple[Path, Path]]],
    normal_magnitude: Path,
    head_mask: nib.Nifti1Image,
    parameters: HeadMaskParameters,
    mask_details: dict[str, Any],
) -> dict[str, Any]:
    """Write one complete collection into an empty staging directory.

    Args:
        staging: Empty temporary collection directory.
        output_root: Source reconstruction root used for relative provenance.
        case_sources: Available canonical NIfTI/JSON pairs by case.
        normal_magnitude: Normal magnitude path used to derive the head mask.
        head_mask: Binary mask on the normal grid.
        parameters: Mask extraction parameters.
        mask_details: Calculated mask threshold and coverage metadata.

    Returns:
        JSON-native manifest describing all copied and masked files.

    Side Effects:
        Writes NIfTIs, sidecars, one native-grid mask, and a manifest-ready
        record below ``staging``; source files remain read-only.
    """
    mask_dir = staging / "masks"
    mask_dir.mkdir(parents=True)
    mask_path = mask_dir / "head_mask_from_normal.nii.gz"
    nib.save(head_mask, str(mask_path))
    mask_record = {
        "source_nifti": str(normal_magnitude.relative_to(output_root)),
        "source_sha256": sha256_file(normal_magnitude),
        "collection_file": str(mask_path.relative_to(staging)),
        "collection_sha256": sha256_file(mask_path),
        "parameters": asdict(parameters),
        "details": mask_details,
        "intended_use": "presentation masking only; excluded from sweep evaluation",
        "bet_used": False,
    }
    mask_metadata = mask_dir / "head_mask_from_normal.json"
    _write_json(mask_metadata, mask_record)
    mask_record["metadata_file"] = str(mask_metadata.relative_to(staging))
    mask_record["metadata_sha256"] = sha256_file(mask_metadata)

    case_records: list[dict[str, Any]] = []
    for case, pairs in case_sources.items():
        relative_leaf = Path("normal") if case == "normal" else Path("retro") / case
        original_dir = staging / "original_nifti" / relative_leaf
        masked_dir = staging / "head_masked_nifti" / relative_leaf
        original_dir.mkdir(parents=True)
        masked_dir.mkdir(parents=True)
        files = [
            _materialize_pair(
                nifti,
                sidecar,
                output_root,
                staging,
                original_dir,
                masked_dir,
                head_mask,
                mask_record,
            )
            for nifti, sidecar in pairs
        ]
        case_records.append({"case": case, "file_count": len(files), "files": files})

    return {
        "format_version": 1,
        "builder": COLLECTION_BUILDER,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "canonical_reconstruction_outputs_modified": False,
            "original_niftis_copied_byte_for_byte": True,
            "whole_head_mask_source": "normal canonical magnitude",
            "whole_head_mask_backend": "SciPy morphology; no BET",
            "mask_resampling": "nearest-neighbor in NIfTI physical space when grids differ",
            "masked_outputs_for_presentation_only": True,
            "masked_outputs_excluded_from_regularization_evaluation": True,
        },
        "head_mask": mask_record,
        "cases": case_records,
    }


def _materialize_pair(
    source_nifti: Path,
    source_sidecar: Path,
    output_root: Path,
    staging: Path,
    original_dir: Path,
    masked_dir: Path,
    head_mask: nib.Nifti1Image,
    mask_record: dict[str, Any],
) -> dict[str, Any]:
    """Copy one canonical pair and write its masked derivative.

    Args:
        source_nifti: Canonical source NIfTI.
        source_sidecar: Matching canonical JSON sidecar.
        output_root: Reconstruction root used for relative provenance paths.
        staging: Collection staging directory.
        original_dir: Destination leaf for byte-identical source copies.
        masked_dir: Destination leaf for masked derivatives.
        head_mask: Normal-grid binary whole-head mask.
        mask_record: Native mask provenance record.

    Returns:
        JSON-native provenance and hash record for the copied and masked files.

    Side Effects:
        Copies the NIfTI and JSON source pair and writes one masked NIfTI/JSON pair.
    """
    image, data = _load_validated_nifti(source_nifti)
    source_metadata = _load_json(source_sidecar)
    original_nifti = original_dir / source_nifti.name
    original_sidecar = original_dir / source_sidecar.name
    shutil.copy2(source_nifti, original_nifti)
    shutil.copy2(source_sidecar, original_sidecar)
    if sha256_file(original_nifti) != sha256_file(source_nifti):
        raise RuntimeError(f"Canonical NIfTI copy hash mismatch: {source_nifti}")
    if sha256_file(original_sidecar) != sha256_file(source_sidecar):
        raise RuntimeError(f"Canonical JSON copy hash mismatch: {source_sidecar}")

    mapped_mask, mapping_method = resample_head_mask(head_mask, image)
    masked_data = np.where(mapped_mask, data, 0)
    header = image.header.copy()
    header.set_data_dtype(image.get_data_dtype())
    masked_image = nib.Nifti1Image(masked_data.astype(image.get_data_dtype()), image.affine, header)
    masked_nifti = masked_dir / source_nifti.name
    masked_sidecar = masked_dir / source_sidecar.name
    nib.save(masked_image, str(masked_nifti))
    masked_metadata = dict(source_metadata)
    masked_metadata.update(
        {
            "WholeHeadMaskApplied": True,
            "WholeHeadMaskPurpose": "presentation background suppression",
            "WholeHeadMaskSource": mask_record["source_nifti"],
            "WholeHeadMaskSHA256": mask_record["collection_sha256"],
            "WholeHeadMaskMapping": mapping_method,
            "WholeHeadMaskBETUsed": False,
            "CanonicalUnmaskedSource": str(source_nifti.relative_to(output_root)),
            "CanonicalUnmaskedSourceSHA256": sha256_file(source_nifti),
        }
    )
    _write_json(masked_sidecar, masked_metadata)
    return {
        "source_nifti": str(source_nifti.relative_to(output_root)),
        "source_nifti_sha256": sha256_file(source_nifti),
        "source_json": str(source_sidecar.relative_to(output_root)),
        "source_json_sha256": sha256_file(source_sidecar),
        "original_nifti": str(original_nifti.relative_to(staging)),
        "original_nifti_sha256": sha256_file(original_nifti),
        "original_json": str(original_sidecar.relative_to(staging)),
        "original_json_sha256": sha256_file(original_sidecar),
        "masked_nifti": str(masked_nifti.relative_to(staging)),
        "masked_nifti_sha256": sha256_file(masked_nifti),
        "masked_json": str(masked_sidecar.relative_to(staging)),
        "masked_json_sha256": sha256_file(masked_sidecar),
        "mask_mapping": mapping_method,
        "masked_nonzero_voxels": int(np.count_nonzero(masked_data)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: UTF-8 JSON file expected to contain an object.

    Returns:
        Parsed JSON mapping.

    Raises:
        ValueError: If the top-level JSON value is not an object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object with stable readable formatting.

    Args:
        path: Destination JSON path whose parent already exists.
        payload: JSON-native mapping to serialize.

    Returns:
        None.

    Side Effects:
        Creates or replaces ``path`` within the collection staging tree.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_existing_collection(collection: Path) -> None:
    """Reject replacement of a collection not owned by this utility.

    Args:
        collection: Intended final collection directory.

    Returns:
        None.

    Raises:
        FileExistsError: If the path is a symlink, non-directory, nonempty
            unmanifested directory, or a collection from another builder.
    """
    if not collection.exists():
        return
    if collection.is_symlink() or not collection.is_dir():
        raise FileExistsError(f"Refusing to replace non-directory collection: {collection}")
    manifest = collection / "manifest.json"
    if not manifest.is_file():
        if any(collection.iterdir()):
            raise FileExistsError(f"Existing collection is not tool-owned: {collection}")
        return
    payload = _load_json(manifest)
    if payload.get("builder") != COLLECTION_BUILDER:
        raise FileExistsError(f"Existing collection has a different builder: {collection}")
    expected = _manifest_owned_files(payload)
    actual = {
        str(path.relative_to(collection))
        for path in collection.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(expected):
        raise FileExistsError(
            "Existing tool-owned collection has added or missing files; "
            f"refusing replacement: {collection}"
        )
    for relative_path, expected_hash in expected.items():
        path = collection / relative_path
        if sha256_file(path) != expected_hash:
            raise FileExistsError(
                f"Existing collection file changed since its manifest: {path}"
            )


def _manifest_owned_files(payload: dict[str, Any]) -> dict[str, str]:
    """Extract the complete owned-file hash map from a collection manifest.

    Args:
        payload: Previously written collection manifest mapping.

    Returns:
        Relative collection paths mapped to their recorded SHA-256 digests.

    Raises:
        FileExistsError: If the manifest lacks a required owned path or digest.
    """
    try:
        mask = payload["head_mask"]
        owned = {
            str(mask["collection_file"]): str(mask["collection_sha256"]),
            str(mask["metadata_file"]): str(mask["metadata_sha256"]),
        }
        file_fields = (
            ("original_nifti", "original_nifti_sha256"),
            ("original_json", "original_json_sha256"),
            ("masked_nifti", "masked_nifti_sha256"),
            ("masked_json", "masked_json_sha256"),
        )
        for case in payload["cases"]:
            for record in case["files"]:
                for path_key, hash_key in file_fields:
                    owned[str(record[path_key])] = str(record[hash_key])
    except (KeyError, TypeError) as exc:
        raise FileExistsError("Existing collection manifest is incomplete.") from exc
    if any(not path or len(digest) != 64 for path, digest in owned.items()):
        raise FileExistsError("Existing collection manifest contains an invalid file hash.")
    return owned


def _replace_owned_collection(staging: Path, collection: Path) -> None:
    """Atomically install a staged collection while retaining rollback safety.

    Args:
        staging: Complete staged directory on the destination filesystem.
        collection: Final tool-owned collection path.

    Returns:
        None.

    Side Effects:
        Renames the old collection to a temporary backup, installs the staged
        directory, then removes the replaced tool-owned backup.
    """
    if not collection.exists():
        os.replace(staging, collection)
        return
    backup = collection.with_name(f".{collection.name}-backup-{uuid.uuid4().hex}")
    os.replace(collection, backup)
    try:
        os.replace(staging, collection)
    except Exception:
        os.replace(backup, collection)
        raise
    shutil.rmtree(backup)
