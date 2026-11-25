"""Native-grid ROVir nuisance-region geometry: boxes, spheres, and ellipsoids.

The MPRAGE ROVir workflow describes the nuisance (null) region as a union of
inclusive index boxes. A box that covers the face over every partition also
cuts into the frontal lobe, because the face lies anterior-inferior to the
brain. This module adds axis-aligned spheres and ellipsoids whose centres are
given in native array indices and whose radii are given in millimetres, so a
physically round region stays round on anisotropic grids. Boxes and
ellipsoids may be combined in one union.

The functions here are pure geometry: they read and write no files and never
launch BART. The union hash follows the MPRAGE candidate convention, so an
identical binary union yields the same identifier in both workflows.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

AXES = ("ro", "lin", "par")
NULL_BOX_PATTERN = re.compile(
    r"\s*ro\s*=\s*([^,]+)\s*,\s*lin\s*=\s*([^,]+)\s*,\s*par\s*=\s*([^,]+)\s*",
    re.IGNORECASE,
)
_ELLIPSOID_CENTRE_KEYS = AXES
_ELLIPSOID_RADIUS_KEYS = tuple(f"{axis}_radius_mm" for axis in AXES)
_ELLIPSOID_ALLOWED_KEYS = frozenset(
    (*_ELLIPSOID_CENTRE_KEYS, *_ELLIPSOID_RADIUS_KEYS, "radius_mm")
)


def _validated_geometry(
    shape: Sequence[int], voxel_mm: Sequence[float] | None = None
) -> tuple[tuple[int, int, int], tuple[float, float, float] | None]:
    """Validate the native grid and, optionally, its voxel size.

    Args:
        shape: Native ``(RO, LIN, PAR)`` dimensions.
        voxel_mm: Optional native ``(RO, LIN, PAR)`` voxel size in millimetres.

    Returns:
        Integer shape and float voxel size (``None`` when not supplied).

    Raises:
        ValueError: If either tuple has the wrong length or nonpositive values.
    """
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise ValueError("ROVir region geometry must contain three positive dimensions.")
    if voxel_mm is None:
        return dimensions, None  # type: ignore[return-value]
    spacing = tuple(float(value) for value in voxel_mm)
    if len(spacing) != 3 or not all(math.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("Voxel size must contain three positive finite millimetre values.")
    return dimensions, spacing  # type: ignore[return-value]


def parse_null_box(specification: str, shape: Sequence[int]) -> dict[str, list[int]]:
    """Parse one inclusive native-grid null box.

    Args:
        specification: Text in ``ro=a:b,lin=c:d,par=e:f`` form. ``all`` is
            accepted independently for each axis.
        shape: Native ``(RO, LIN, PAR)`` grid.

    Returns:
        Canonical mapping with inclusive integer bounds per axis.

    Raises:
        ValueError: If syntax or bounds are invalid.
    """
    dimensions, _ = _validated_geometry(shape)
    match = NULL_BOX_PATTERN.fullmatch(str(specification))
    if match is None:
        raise ValueError(
            "Null box must use ro=a:b,lin=c:d,par=e:f with inclusive bounds or 'all'."
        )
    result: dict[str, list[int]] = {}
    for axis, token, size in zip(AXES, match.groups(), dimensions, strict=True):
        value = token.strip().lower()
        if value == "all":
            start, stop = 0, size - 1
        else:
            parts = value.split(":")
            if len(parts) != 2 or any(not part.strip().isdigit() for part in parts):
                raise ValueError(f"Invalid inclusive {axis.upper()} bounds: {token!r}.")
            start, stop = (int(part.strip()) for part in parts)
        if start < 0 or stop < start or stop >= size:
            raise ValueError(
                f"{axis.upper()} bounds [{start}, {stop}] lie outside [0, {size - 1}]."
            )
        result[axis] = [start, stop]
    return result


def _finite_float(token: object, label: str) -> float:
    """Convert one numeric token and require a finite value.

    Args:
        token: Text or number supplied by the user.
        label: Field name used in validation errors.

    Returns:
        Finite float value.

    Raises:
        ValueError: If the token is not a finite number.
    """
    try:
        value = float(str(token).strip())
    except ValueError as error:
        raise ValueError(f"Ellipsoid {label} must be numeric: {token!r}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Ellipsoid {label} must be finite.")
    return value


def canonical_null_ellipsoid(
    fields: Mapping[str, object],
    shape: Sequence[int],
    voxel_mm: Sequence[float],
) -> dict[str, list[float]]:
    """Validate one ellipsoid given as centre and radius fields.

    Args:
        fields: Mapping with centre keys ``ro``, ``lin``, ``par`` (native
            array indices; fractional and outside-grid values are allowed)
            and either ``radius_mm`` (sphere) or all three of
            ``ro_radius_mm``, ``lin_radius_mm``, ``par_radius_mm``.
        shape: Native ``(RO, LIN, PAR)`` grid.
        voxel_mm: Native voxel size in millimetres.

    Returns:
        ``{"center_index": [ro, lin, par], "radii_mm": [r_ro, r_lin, r_par]}``.

    Raises:
        ValueError: If keys are missing, duplicated, or unknown, a radius is
            nonpositive, or the ellipsoid does not intersect the grid.
    """
    dimensions, spacing = _validated_geometry(shape, voxel_mm)
    keys = {str(key).strip().lower() for key in fields}
    unknown = keys - _ELLIPSOID_ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown ellipsoid fields: {sorted(unknown)}.")
    normalized = {str(key).strip().lower(): value for key, value in fields.items()}
    missing_centre = [axis for axis in _ELLIPSOID_CENTRE_KEYS if axis not in normalized]
    if missing_centre:
        raise ValueError(f"Ellipsoid centre lacks {missing_centre}.")
    per_axis = [key for key in _ELLIPSOID_RADIUS_KEYS if key in normalized]
    if "radius_mm" in normalized and per_axis:
        raise ValueError("Use either radius_mm or the three per-axis radii, not both.")
    if "radius_mm" in normalized:
        radius = _finite_float(normalized["radius_mm"], "radius_mm")
        radii = [radius, radius, radius]
    elif len(per_axis) == 3:
        radii = [_finite_float(normalized[key], key) for key in _ELLIPSOID_RADIUS_KEYS]
    else:
        raise ValueError(
            "Ellipsoid requires radius_mm or ro_radius_mm, lin_radius_mm, and par_radius_mm."
        )
    if any(value <= 0 for value in radii):
        raise ValueError("Ellipsoid radii must be positive.")
    centre = [_finite_float(normalized[axis], axis) for axis in _ELLIPSOID_CENTRE_KEYS]
    record = {"center_index": centre, "radii_mm": radii}
    if not np.any(ellipsoid_mask(dimensions, spacing, centre, radii)):  # type: ignore[arg-type]
        raise ValueError(f"Ellipsoid {record} contains no voxel centre of the native grid.")
    return record


def parse_null_ellipsoid(
    specification: str,
    shape: Sequence[int],
    voxel_mm: Sequence[float],
) -> dict[str, list[float]]:
    """Parse one sphere or axis-aligned ellipsoid specification.

    Args:
        specification: Comma-separated ``key=value`` text, for example
            ``ro=125,lin=-20,par=-10,radius_mm=90`` (sphere) or
            ``ro=125,lin=-20,par=-10,ro_radius_mm=80,lin_radius_mm=100,par_radius_mm=70``.
            Centres are native array indices; radii are millimetres.
        shape: Native ``(RO, LIN, PAR)`` grid.
        voxel_mm: Native voxel size in millimetres.

    Returns:
        Canonical centre/radius record (see :func:`canonical_null_ellipsoid`).

    Raises:
        ValueError: If the text is malformed or the geometry is invalid.
    """
    fields: dict[str, str] = {}
    for item in str(specification).split(","):
        if not item.strip():
            raise ValueError(f"Empty field in ellipsoid specification {specification!r}.")
        key, separator, value = item.partition("=")
        key = key.strip().lower()
        if not separator or not key or not value.strip():
            raise ValueError(f"Ellipsoid fields must use key=value: {item!r}.")
        if key in fields:
            raise ValueError(f"Duplicate ellipsoid field {key!r}.")
        fields[key] = value.strip()
    return canonical_null_ellipsoid(fields, shape, voxel_mm)


def ellipsoid_mask(
    shape: Sequence[int],
    voxel_mm: Sequence[float],
    center_index: Sequence[float],
    radii_mm: Sequence[float],
) -> np.ndarray:
    """Rasterize one axis-aligned ellipsoid on voxel centres.

    Args:
        shape: Native ``(RO, LIN, PAR)`` grid.
        voxel_mm: Native voxel size in millimetres.
        center_index: Ellipsoid centre in native array indices.
        radii_mm: Semi-axes along RO, LIN, and PAR in millimetres.

    Returns:
        Boolean mask that is true where the voxel centre lies inside or on
        the ellipsoid surface.
    """
    dimensions, spacing = _validated_geometry(shape, voxel_mm)
    assert spacing is not None
    centre = [float(value) for value in center_index]
    radii = [float(value) for value in radii_mm]
    # Normalized squared distance, separable along each axis.
    terms = [
        ((np.arange(size, dtype=np.float64) - c) * step / radius) ** 2
        for size, c, step, radius in zip(dimensions, centre, spacing, radii, strict=True)
    ]
    distance = (
        terms[0][:, np.newaxis, np.newaxis]
        + terms[1][np.newaxis, :, np.newaxis]
        + terms[2][np.newaxis, np.newaxis, :]
    )
    return distance <= 1.0


def box_mask(shape: Sequence[int], box: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Rasterize one canonical inclusive box.

    Args:
        shape: Native ``(RO, LIN, PAR)`` grid.
        box: Mapping with inclusive ``[start, stop]`` bounds per axis.

    Returns:
        Boolean mask of the box.
    """
    dimensions, _ = _validated_geometry(shape)
    mask = np.zeros(dimensions, dtype=bool)
    mask[tuple(slice(int(box[axis][0]), int(box[axis][1]) + 1) for axis in AXES)] = True
    return mask


def _canonical_box_from_mapping(
    raw: Mapping[str, object], shape: Sequence[int]
) -> dict[str, list[int]]:
    """Convert one JSON box mapping to the canonical parsed form.

    Args:
        raw: Mapping with ``ro``, ``lin``, ``par`` set to ``"all"`` or two
            inclusive integers.
        shape: Native ``(RO, LIN, PAR)`` grid.

    Returns:
        Canonical inclusive box.

    Raises:
        ValueError: If the mapping is malformed.
    """
    if set(raw) != set(AXES):
        raise ValueError("Each null box must define exactly ro, lin, and par.")
    tokens = []
    for axis in AXES:
        bounds = raw[axis]
        if isinstance(bounds, str) and bounds.lower() == "all":
            tokens.append(f"{axis}=all")
        elif isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)) and len(bounds) == 2:
            tokens.append(f"{axis}={int(bounds[0])}:{int(bounds[1])}")
        else:
            raise ValueError(f"JSON null-box {axis} must be 'all' or two inclusive integers.")
    return parse_null_box(",".join(tokens), shape)


def union_sha256(mask: np.ndarray) -> str:
    """Hash one binary region exactly as the MPRAGE candidate workflow does.

    Args:
        mask: Boolean native-grid region.

    Returns:
        Hex SHA-256 of the little-endian int64 shape followed by uint8 voxels.
    """
    values = np.asarray(mask, dtype=bool)
    hasher = hashlib.sha256()
    hasher.update(np.asarray(values.shape, dtype="<i8").tobytes())
    hasher.update(np.ascontiguousarray(values, dtype=np.uint8).tobytes())
    return hasher.hexdigest()


def null_region_union(
    shape: Sequence[int],
    voxel_mm: Sequence[float],
    *,
    boxes: Sequence[str | Mapping[str, object]] = (),
    ellipsoids: Sequence[str | Mapping[str, object]] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the order-independent union of null boxes and ellipsoids.

    Args:
        shape: Native ``(RO, LIN, PAR)`` grid.
        voxel_mm: Native voxel size in millimetres.
        boxes: Box text specifications or JSON mappings.
        ellipsoids: Ellipsoid text specifications or JSON mappings.

    Returns:
        Boolean null-region mask and a JSON-compatible record containing the
        canonical shapes, voxel counts, union hash, and region identifier.

    Raises:
        ValueError: If no shape is given, a shape is invalid, or the union or
            its complement is empty.
    """
    dimensions, spacing = _validated_geometry(shape, voxel_mm)
    if not boxes and not ellipsoids:
        raise ValueError("At least one null box or ellipsoid is required.")
    canonical_boxes = []
    for raw in boxes:
        canonical_boxes.append(
            parse_null_box(raw, dimensions)
            if isinstance(raw, str)
            else _canonical_box_from_mapping(raw, dimensions)
        )
    canonical_ellipsoids = []
    for raw in ellipsoids:
        if isinstance(raw, str):
            canonical_ellipsoids.append(parse_null_ellipsoid(raw, dimensions, spacing))  # type: ignore[arg-type]
        elif isinstance(raw, Mapping):
            if set(raw) == {"center_index", "radii_mm"}:
                centre, radii = raw["center_index"], raw["radii_mm"]
                fields = {
                    **{axis: value for axis, value in zip(AXES, centre, strict=True)},  # type: ignore[arg-type]
                    **{
                        key: value
                        for key, value in zip(_ELLIPSOID_RADIUS_KEYS, radii, strict=True)  # type: ignore[arg-type]
                    },
                }
            else:
                fields = dict(raw)
            canonical_ellipsoids.append(
                canonical_null_ellipsoid(fields, dimensions, spacing)  # type: ignore[arg-type]
            )
        else:
            raise ValueError("Each ellipsoid must be a text specification or a mapping.")

    # Sort and deduplicate so argument order never changes the identity.
    box_keys = sorted({tuple(v for axis in AXES for v in box[axis]) for box in canonical_boxes})
    unique_boxes = [
        {axis: [key[2 * i], key[2 * i + 1]] for i, axis in enumerate(AXES)} for key in box_keys
    ]
    ellipsoid_keys = sorted(
        {(*item["center_index"], *item["radii_mm"]) for item in canonical_ellipsoids}
    )
    unique_ellipsoids = [
        {"center_index": list(key[:3]), "radii_mm": list(key[3:])} for key in ellipsoid_keys
    ]

    union = np.zeros(dimensions, dtype=bool)
    box_counts = []
    for box in unique_boxes:
        region = box_mask(dimensions, box)
        box_counts.append(int(np.count_nonzero(region)))
        union |= region
    ellipsoid_counts = []
    for item in unique_ellipsoids:
        region = ellipsoid_mask(dimensions, spacing, item["center_index"], item["radii_mm"])  # type: ignore[arg-type]
        ellipsoid_counts.append(int(np.count_nonzero(region)))
        union |= region
    union_count = int(np.count_nonzero(union))
    if union_count == 0 or union_count == union.size:
        raise ValueError("Null-region union and its exact complement must both be nonempty.")
    digest = union_sha256(union)
    record = {
        "construction": "exact union of inclusive native-grid boxes and voxel-centre ellipsoids",
        "axis_order": ["RO", "LIN", "PAR"],
        "voxel_mm_ro_lin_par": list(spacing),  # type: ignore[arg-type]
        "ellipsoid_convention": (
            "voxel centre i is inside when sum(((i - centre) * voxel_mm / radius_mm)**2) <= 1; "
            "centres are native array indices and may lie outside the grid"
        ),
        "submitted_box_count": len(canonical_boxes),
        "submitted_ellipsoid_count": len(canonical_ellipsoids),
        "canonical_boxes": unique_boxes,
        "canonical_ellipsoids": unique_ellipsoids,
        "individual_box_voxel_counts": box_counts,
        "individual_ellipsoid_voxel_counts": ellipsoid_counts,
        "union_voxel_count": union_count,
        "grid_voxel_count": int(union.size),
        "positive_is_exact_complement": True,
        "negative_union_sha256": digest,
        "region_id": f"null_{digest[:12]}",
        "mprage_equivalent_candidate_id": f"negative_union_{digest[:16]}",
    }
    return union, record
