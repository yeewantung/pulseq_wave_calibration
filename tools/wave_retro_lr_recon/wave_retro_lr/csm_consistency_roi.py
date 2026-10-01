"""Five-label ROI contract and descriptive ROI statistics for CSM-consistency review.

Arrays are on the BART logical ``(RO, LIN, PAR)`` grid of the accepted image
and coil-sensitivity maps unless a docstring states otherwise. For sagittal
MPRAGE, RO is physical z, LIN is physical y (the R = 3 acceleration axis),
and PAR is physical x (R = 1).

The module provides:

* the five-label ROI volume contract and inclusive-box label construction;
* the lossless mapping between BART logical arrays and canonical-RAS NIfTI
  storage, mirroring the ROVir manual-annotation export and validation;
* exact interval-overlap mapping of masks onto another centered grid with
  the same field of view, such as the native ACS grid;
* the pre-registered R = 3 LIN alias-partner overlap statistic and its
  non-partner LIN-shift null, reported as descriptive numbers only;
* an edge-matched control region and a deterministic SNR-matched subsample;
* fixed display scales and windows, and NaN-aware ROI summaries.

Nothing here reads or writes files, launches external programs, selects a
winner, or states a mechanism verdict.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from fractions import Fraction
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

from .rovir_feasibility import parse_null_box

ROI_LABELS: dict[int, str] = {
    0: "unassigned",
    1: "metal_void",
    2: "metal_pileup",
    3: "fringe",
    4: "preserved_anatomy",
    5: "background_air",
}
ROI_LABEL_DESCRIPTIONS: dict[int, str] = {
    0: "Unassigned, mixed, or uncertain voxels.",
    1: "Metal-related signal void or null.",
    2: "Metal-related pile-up or bright displaced signal.",
    3: "Visible fringe.",
    4: "Preserved anatomy away from metal.",
    5: (
        "Noise-only background air, preferably superior to the scalp so that "
        "no head signal shares its RO rows."
    ),
}
LABEL_VALUES_BY_NAME: dict[str, int] = {name: value for value, name in ROI_LABELS.items()}
FIXED_DISPLAY_WINDOWS: dict[str, tuple[float, float]] = {
    "lambda": (0.0, 1.0),
    "rho": (0.0, 0.5),
    "log10_rnr": (-0.5, 2.0),
    "e1": (0.5, 1.0),
    "coherence": (0.9, 1.0),
}
DEFAULT_DISPLAY_PERCENTILE = 99.5
DEFAULT_ALIAS_ACCELERATION = 3
DEFAULT_ALIAS_TOLERANCE = 1
DEFAULT_RO_DILATIONS = (0, 16, 32, 64, None)
DEFAULT_EDGE_BAND_MM = 3.0
DEFAULT_EDGE_EXCLUSION_MM = 40.0

_BART_AXES = ("ro", "lin", "par")
_BART_AXIS_NAMES = ("RO", "LIN", "PAR")
_RO_AXIS = 0
_RAS_CODES = ("R", "A", "S")
_ASSIGNABLE_LABELS = (1, 2, 3, 4, 5)
_METAL_LABELS = (1, 2)
# Tolerance (mm) for internal agreement between a stored affine and the
# canonical-RAS affine derived from the source affine; it absorbs float32
# header rounding but rejects any physical geometry error.
_AFFINE_CONSISTENCY_ATOL = 1e-4
_NULL_INTERPRETATION = (
    "Descriptive only: exceedance_fraction is the fraction of non-partner "
    "LIN-shift overlaps that are greater than or equal to the observed "
    "alias-partner overlap. It is not a p-value, not a significance test, and "
    "not proof of an alias mechanism; the R = 3 partner location is a "
    "pre-registered, testable hypothesis."
)
_NULL_MASK_CONSTRUCTION = (
    "Each null value uses the source mask shifted by +s and -s along LIN with "
    "the same tolerance, circular wrap, and RO dilation as the observed "
    "partner mask, so s and -s give identical values."
)


def validate_label_volume(
    labels: np.ndarray,
    expected_shape: Sequence[int],
    *,
    required: Sequence[str] = ("fringe", "preserved_anatomy", "background_air"),
    require_metal: bool = True,
) -> dict[str, Any]:
    """Validate one five-label ROI volume on its expected grid.

    Args:
        labels: Label volume whose values must be finite integers from
            :data:`ROI_LABELS`; integer-valued floating arrays are accepted.
        expected_shape: Exact expected three-dimensional shape, normally the
            BART logical ``(RO, LIN, PAR)`` grid.
        required: Label names (or numbers 1-5) that must each be nonempty.
        require_metal: Whether the metal source, label 1 or label 2, must be
            nonempty.

    Returns:
        JSON-native record with ``shape``, ``label_counts`` (voxel count per
        label name for all six labels), the canonical ``required`` names, and
        ``require_metal``.

    Raises:
        ValueError: If the shape, dtype, values, a required label, or the
            metal source violates the contract.
    """
    shape = _shape(expected_shape, "expected_shape")
    values = _label_values(labels)
    if values.shape != shape:
        raise ValueError(f"ROI label shape {values.shape} differs from the expected {shape}.")
    if isinstance(required, str):
        raise ValueError("required must be a sequence of label names, not one string.")
    required_names = [ROI_LABELS[_assignable_label(token)] for token in required]
    frequencies = np.bincount(values.ravel(), minlength=len(ROI_LABELS))
    counts = {name: int(frequencies[value]) for value, name in ROI_LABELS.items()}
    for name in required_names:
        if counts[name] == 0:
            raise ValueError(
                f"Required ROI label {LABEL_VALUES_BY_NAME[name]} ({name}) is empty."
            )
    if require_metal and sum(counts[ROI_LABELS[value]] for value in _METAL_LABELS) == 0:
        raise ValueError(
            "The ROI metal source is empty; label 1 (metal_void) or label 2 "
            "(metal_pileup) must be nonempty."
        )
    return {
        "shape": list(shape),
        "label_counts": counts,
        "required": required_names,
        "require_metal": bool(require_metal),
    }


def label_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Split a validated five-label volume into boolean masks.

    Args:
        labels: Label volume whose values are finite integers from
            :data:`ROI_LABELS`.

    Returns:
        Mapping from each label name for labels 1-5, in label order, to a
        boolean mask, followed by ``"metal"`` (label 1 union label 2).

    Raises:
        ValueError: If the label values violate the contract.
    """
    values = _label_values(labels)
    masks = {ROI_LABELS[value]: values == value for value in _ASSIGNABLE_LABELS}
    masks["metal"] = masks[ROI_LABELS[1]] | masks[ROI_LABELS[2]]
    return masks


def parse_roi_box(
    specification: str, shape: Sequence[int]
) -> tuple[int, dict[str, list[int]]]:
    """Parse one labeled inclusive BART-index ROI box.

    Args:
        specification: Text ``LABEL=ro=a:b,lin=c:d,par=e:f`` where ``LABEL``
            is a label name (case-insensitive) or number 1-5 and each axis
            uses inclusive bounds or ``all``.
        shape: BART logical ``(RO, LIN, PAR)`` grid.

    Returns:
        ``(label_value, box)`` where ``box`` maps ``ro``, ``lin``, and ``par``
        to inclusive ``[start, stop]`` bounds, as parsed by
        :func:`wave_retro_lr.rovir_feasibility.parse_null_box`.

    Raises:
        ValueError: If the label, syntax, or bounds are invalid.
    """
    if not isinstance(specification, str):
        raise ValueError("An ROI box specification must be text.")
    label_token, separator, box_text = specification.partition("=")
    if not separator or not label_token.strip():
        raise ValueError(
            f"ROI box {specification!r} must use LABEL=ro=a:b,lin=c:d,par=e:f."
        )
    label = _assignable_label(label_token)
    try:
        box = parse_null_box(box_text, shape)
    except ValueError as exc:
        raise ValueError(f"Invalid ROI box {specification!r}: {exc}") from exc
    return label, box


def labels_from_boxes(
    specifications: Sequence[str], shape: Sequence[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an order-independent label volume from labeled inclusive boxes.

    Boxes with the same label may overlap and form a union. Boxes with
    different labels must be disjoint. Exact duplicate boxes are merged.

    Args:
        specifications: Nonempty sequence of ``LABEL=ro=a:b,lin=c:d,par=e:f``
            texts accepted by :func:`parse_roi_box`.
        shape: BART logical ``(RO, LIN, PAR)`` grid.

    Returns:
        ``(labels, record)`` with a uint8 label volume and a JSON-native record
        listing the canonical sorted boxes, per-label voxel counts,
        ``canonical_boxes_sha256`` (SHA-256 of the compact sorted-key JSON of
        ``{"axis_order", "shape", "boxes"}``, where each box holds ``label``,
        ``ro``, ``lin``, and ``par``), and ``label_volume_sha256`` (SHA-256 of
        the little-endian int64 shape followed by the uint8 C-order volume).
        The record does not depend on the submission order.

    Raises:
        ValueError: If the sequence is empty, a box is invalid, or boxes with
            different labels overlap.
    """
    dimensions = _shape(shape, "shape")
    if isinstance(specifications, str) or not isinstance(specifications, Sequence):
        raise ValueError(
            "ROI boxes must be a sequence of LABEL=ro=a:b,lin=c:d,par=e:f texts."
        )
    if not specifications:
        raise ValueError("At least one ROI box is required.")
    parsed = [parse_roi_box(text, dimensions) for text in specifications]
    keys = sorted(
        {
            (label, *(bound for axis in _BART_AXES for bound in box[axis]))
            for label, box in parsed
        }
    )
    canonical = [
        {
            "label": key[0],
            "name": ROI_LABELS[key[0]],
            "ro": [key[1], key[2]],
            "lin": [key[3], key[4]],
            "par": [key[5], key[6]],
        }
        for key in keys
    ]
    labels = np.zeros(dimensions, dtype=np.uint8)
    for box in canonical:
        region = tuple(slice(box[axis][0], box[axis][1] + 1) for axis in _BART_AXES)
        existing = labels[region]
        conflicts = np.unique(existing[(existing != 0) & (existing != box["label"])])
        if conflicts.size:
            others = ", ".join(f"{int(value)} ({ROI_LABELS[int(value)]})" for value in conflicts)
            bounds = ",".join(f"{axis}={box[axis][0]}:{box[axis][1]}" for axis in _BART_AXES)
            raise ValueError(
                f"ROI box {box['name']}={bounds} overlaps label {others}; boxes "
                "with different labels must be disjoint."
            )
        labels[region] = box["label"]

    hash_payload = {
        "axis_order": list(_BART_AXIS_NAMES),
        "shape": list(dimensions),
        "boxes": [{key: box[key] for key in ("label", *_BART_AXES)} for box in canonical],
    }
    canonical_sha256 = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    volume_hasher = hashlib.sha256()
    volume_hasher.update(np.asarray(dimensions, dtype="<i8").tobytes())
    volume_hasher.update(np.ascontiguousarray(labels).tobytes())
    frequencies = np.bincount(labels.ravel(), minlength=len(ROI_LABELS))
    record = {
        "construction": "order-independent union of inclusive BART-index boxes",
        "axis_order": list(_BART_AXIS_NAMES),
        "shape": list(dimensions),
        "submitted_box_count": len(parsed),
        "canonical_box_count": len(canonical),
        "duplicate_boxes_removed": len(parsed) - len(canonical),
        "canonical_boxes": canonical,
        "label_counts": {name: int(frequencies[value]) for value, name in ROI_LABELS.items()},
        "canonical_boxes_sha256": canonical_sha256,
        "label_volume_sha256": volume_hasher.hexdigest(),
    }
    return labels, record


def to_stored_orientation(
    bart_array: np.ndarray,
    array_flips: Sequence[bool],
    source_affine: np.ndarray,
    helpers: Any = None,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Map a BART logical array to canonical-RAS NIfTI storage without resampling.

    The array is reversed along the physical flip axes with the upstream
    ``apply_array_axis_flips`` and then canonicalized with the upstream
    ``canonicalize_arrays_to_ras``, as in the ROVir manual-annotation export.
    The result is verified against :func:`to_bart_orientation`.

    Args:
        bart_array: Three-dimensional array on the BART logical grid.
        array_flips: Three booleans; axis ``i`` is reversed before
            canonicalization when ``array_flips[i]`` is true.
        source_affine: Voxel-to-RAS affine describing the flipped array.
        helpers: Namespace providing ``apply_array_axis_flips`` and
            ``canonicalize_arrays_to_ras`` with upstream semantics; ``None``
            loads them with ``wave_retro_lr.mprage.load_wave_mprage_helpers``.

    Returns:
        ``(stored_array, canonical_affine, orientation_transform)``: a new
        C-contiguous array with the input dtype, the float64 canonical affine
        with axis codes R, A, S, and the JSON-native nibabel orientation
        transform from the source orientation to RAS.

    Raises:
        ValueError: If inputs are invalid, the helpers disagree with the
            canonical-RAS transform of ``source_affine``, or the documented
            inverse does not restore the input exactly.

    Side Effects:
        With ``helpers=None``, loading the pinned upstream helpers adds the
        upstream reconstruction directory to ``sys.path``.
    """
    array = np.asarray(bart_array)
    if array.ndim != 3:
        raise ValueError(f"BART arrays must be three-dimensional, not {array.shape}.")
    flips = _flips(array_flips)
    affine = _affine(source_affine, "source_affine")
    if helpers is None:
        helpers = _load_helpers()
    stored, canonical_affine, transform = _forward_orientation(array, flips, affine, helpers)
    expected_transform, expected_shape, expected_affine = _canonical_geometry(
        array.shape, affine
    )
    if (
        stored.shape != expected_shape
        or not np.array_equal(np.asarray(transform, dtype=np.float64), expected_transform)
        or not np.allclose(
            canonical_affine, expected_affine, rtol=0.0, atol=_AFFINE_CONSISTENCY_ATOL
        )
    ):
        raise ValueError(
            "Orientation helpers disagree with the canonical-RAS transform of source_affine."
        )
    if tuple(nib.aff2axcodes(canonical_affine)) != _RAS_CODES:
        raise ValueError("The canonical affine does not have axis codes R, A, S.")
    # Binding the export to the exact inverse guarantees that labels drawn on
    # the stored NIfTI return to the same BART voxels.
    if not _arrays_identical(to_bart_orientation(stored, flips, affine), array):
        raise ValueError(
            "The stored orientation is not inverted exactly; refusing a lossy mapping."
        )
    return stored, canonical_affine, transform


def to_bart_orientation(
    stored_array: np.ndarray,
    array_flips: Sequence[bool],
    source_affine: np.ndarray,
) -> np.ndarray:
    """Invert canonical-RAS NIfTI storage back to the BART logical grid.

    This mirrors the ROVir reviewed-label inverse: nibabel reorients the
    stored array from RAS to the source orientation, and the physical array
    flips are then undone. It never resamples.

    Args:
        stored_array: Three-dimensional array in canonical-RAS storage order.
        array_flips: The three booleans used by :func:`to_stored_orientation`.
        source_affine: Voxel-to-RAS affine describing the flipped array.

    Returns:
        New C-contiguous array on the BART logical grid with the input dtype.

    Raises:
        ValueError: If the array, flips, or affine are invalid.
    """
    array = np.asarray(stored_array)
    if array.ndim != 3:
        raise ValueError(f"Stored arrays must be three-dimensional, not {array.shape}.")
    flips = _flips(array_flips)
    affine = _affine(source_affine, "source_affine")
    reverse = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(_RAS_CODES),
        nib.orientations.io_orientation(affine),
    )
    corrected = nib.orientations.apply_orientation(array, reverse)
    for axis, should_flip in enumerate(flips):
        if should_flip:
            corrected = np.flip(corrected, axis=axis)
    return np.array(corrected, order="C", copy=True)


def orientation_round_trip(
    bart_shape: Sequence[int],
    array_flips: Sequence[bool],
    source_affine: np.ndarray,
    helpers: Any = None,
) -> dict[str, Any]:
    """Verify the forward and inverse orientation with a unique-ID volume.

    Args:
        bart_shape: BART logical ``(RO, LIN, PAR)`` grid.
        array_flips: Three booleans applied before canonicalization.
        source_affine: Voxel-to-RAS affine describing the flipped array.
        helpers: Upstream helper namespace, or ``None`` to load the pinned
            upstream helpers.

    Returns:
        JSON-native record with ``identity`` (whether the int32 unique-ID
        volume is restored exactly), ``bart_shape``, ``stored_shape``,
        ``canonical_affine``, ``stored_axis_codes``, ``orientation_transform``,
        ``array_flips``, and ``canonicalization_used_resampling`` (false).

    Raises:
        ValueError: If inputs are invalid or the grid exceeds int32 IDs.

    Side Effects:
        With ``helpers=None``, loading the pinned upstream helpers adds the
        upstream reconstruction directory to ``sys.path``.
    """
    shape = _shape(bart_shape, "bart_shape")
    flips = _flips(array_flips)
    affine = _affine(source_affine, "source_affine")
    size = int(np.prod(shape, dtype=np.int64))
    if size > np.iinfo(np.int32).max:
        raise ValueError(f"Grid {shape} is too large for int32 unique voxel IDs.")
    if helpers is None:
        helpers = _load_helpers()
    identifiers = np.arange(size, dtype=np.int32).reshape(shape)
    stored, canonical_affine, transform = _forward_orientation(
        identifiers, flips, affine, helpers
    )
    identity = stored.ndim == 3 and _arrays_identical(
        to_bart_orientation(stored, flips, affine), identifiers
    )
    return {
        "identity": bool(identity),
        "bart_shape": list(shape),
        "stored_shape": [int(value) for value in stored.shape],
        "canonical_affine": canonical_affine.tolist(),
        "stored_axis_codes": [str(code) for code in nib.aff2axcodes(canonical_affine)],
        "orientation_transform": transform,
        "array_flips": list(flips),
        "canonicalization_used_resampling": False,
    }


def roi_geometry_record(
    *,
    bart_shape: Sequence[int],
    stored_shape: Sequence[int],
    stored_affine: np.ndarray,
    source_affine: np.ndarray,
    array_flips: Sequence[bool],
    orientation_transform: Sequence[Sequence[float]],
    reference_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the JSON-native geometry binding of an ROI annotation template.

    Args:
        bart_shape: BART logical ``(RO, LIN, PAR)`` grid.
        stored_shape: Canonical-RAS stored NIfTI shape.
        stored_affine: Canonical-RAS stored affine.
        source_affine: Voxel-to-RAS affine describing the flipped BART array.
        array_flips: Physical array flips applied before canonicalization.
        orientation_transform: nibabel transform from the source orientation
            to RAS, as returned by :func:`to_stored_orientation`.
        reference_sha256: SHA-256 of the exported reference image.
        source_manifest_sha256: SHA-256 of the manifest that defines the
            source grid.

    Returns:
        JSON-native record whose keys include ``bart_shape``,
        ``stored_shape``, ``stored_affine``, ``stored_axis_codes``,
        ``source_affine``, ``array_flips``, ``orientation_transform``, and
        both SHA-256 values, so that ``array_flips`` and ``source_affine``
        can be passed directly to :func:`to_bart_orientation`.

    Raises:
        ValueError: If any value is malformed or the stored shape, affine, or
            transform disagrees with ``source_affine`` reoriented to RAS.
    """
    logical_shape = _shape(bart_shape, "bart_shape")
    stored = _shape(stored_shape, "stored_shape")
    stored_matrix = _affine(stored_affine, "stored_affine")
    source_matrix = _affine(source_affine, "source_affine")
    flips = _flips(array_flips)
    transform = _transform(orientation_transform)
    expected_transform, expected_shape, expected_affine = _canonical_geometry(
        logical_shape, source_matrix
    )
    if not np.array_equal(transform, expected_transform):
        raise ValueError(
            "orientation_transform differs from the canonical-RAS transform of source_affine."
        )
    if stored != expected_shape:
        raise ValueError(
            f"stored_shape {stored} differs from the reoriented BART shape {expected_shape}."
        )
    codes = tuple(nib.aff2axcodes(stored_matrix))
    if codes != _RAS_CODES:
        raise ValueError(f"stored_affine axis codes {codes} are not R, A, S.")
    if not np.allclose(
        stored_matrix, expected_affine, rtol=0.0, atol=_AFFINE_CONSISTENCY_ATOL
    ):
        raise ValueError("stored_affine differs from source_affine reoriented to canonical RAS.")
    return {
        "axis_order_bart": list(_BART_AXIS_NAMES),
        "bart_shape": list(logical_shape),
        "stored_shape": list(stored),
        "stored_affine": stored_matrix.tolist(),
        "stored_axis_codes": list(_RAS_CODES),
        "source_affine": source_matrix.tolist(),
        "array_flips": list(flips),
        "orientation_transform": transform.tolist(),
        "canonicalization_used_resampling": False,
        "reference_sha256": _sha256_text(reference_sha256, "reference_sha256"),
        "source_manifest_sha256": _sha256_text(
            source_manifest_sha256, "source_manifest_sha256"
        ),
    }


def validate_roi_geometry(
    image_shape: Sequence[int],
    image_affine: np.ndarray,
    record: Mapping[str, Any],
    *,
    atol: float = 1e-6,
) -> None:
    """Check that a reviewed ROI image keeps the recorded stored geometry.

    The affine comparison is made in float64 and, because NIfTI headers store
    the affine rows in float32, also after rounding both affines to float32;
    either agreement within ``atol`` is accepted.

    Args:
        image_shape: Shape of the reviewed NIfTI data array.
        image_affine: Affine of the reviewed NIfTI image.
        record: Geometry record from :func:`roi_geometry_record`.
        atol: Absolute affine tolerance in millimetres.

    Returns:
        None.

    Raises:
        ValueError: On a malformed record or tolerance, a shape mismatch,
            non-RAS axis codes in the record or image, or affine drift.
    """
    if not isinstance(record, Mapping):
        raise ValueError("The ROI geometry record must be a mapping.")
    tolerance = _finite_float(atol, "atol")
    if tolerance < 0:
        raise ValueError("atol must be nonnegative.")
    try:
        expected_shape = _shape(record["stored_shape"], "recorded stored_shape")
        expected_affine = _affine(record["stored_affine"], "recorded stored_affine")
        recorded_codes = tuple(str(code) for code in record["stored_axis_codes"])
    except KeyError as exc:
        raise ValueError(f"The ROI geometry record lacks {exc.args[0]!r}.") from exc
    shape = _shape(image_shape, "image_shape")
    if shape != expected_shape:
        raise ValueError(
            f"ROI image shape {shape} differs from the recorded stored shape {expected_shape}."
        )
    matrix = _affine(image_affine, "image_affine")
    affine_codes = tuple(nib.aff2axcodes(expected_affine))
    if recorded_codes != _RAS_CODES or affine_codes != _RAS_CODES:
        raise ValueError(
            f"Recorded ROI axis codes {recorded_codes} and recorded stored-affine axis "
            f"codes {affine_codes} must both be R, A, S."
        )
    image_codes = tuple(nib.aff2axcodes(matrix))
    if image_codes != _RAS_CODES:
        raise ValueError(f"ROI image axis codes {image_codes} are not R, A, S.")
    single = np.float32
    if not (
        np.allclose(matrix, expected_affine, rtol=0.0, atol=tolerance)
        or np.allclose(
            matrix.astype(single).astype(np.float64),
            expected_affine.astype(single).astype(np.float64),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        difference = float(np.max(np.abs(matrix - expected_affine)))
        raise ValueError(
            "ROI image affine differs from the recorded stored affine "
            f"(maximum absolute difference {difference:.6g} mm, tolerance {tolerance:g} mm)."
        )


def map_mask_to_grid(
    mask: np.ndarray,
    target_shape: Sequence[int],
    *,
    coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a boolean mask onto another centered grid with the same FOV.

    Along each axis, voxel ``i`` of an ``N``-voxel grid is centered at
    ``(i - N // 2) * FOV / N`` and extends half a voxel to each side. The
    covered fraction of a target voxel is the exact separable interval
    overlap with source mask voxels divided by the target voxel volume,
    evaluated in integer units of ``FOV / (2 * N_source * N_target)``.
    Target voxels that extend past the source grid can therefore never be
    fully covered. Equal grids map identically.

    Args:
        mask: Boolean (or 0/1) source mask.
        target_shape: Target grid with the same number of axes.
        coverage: Minimum covered fraction, in ``(0, 1]``, for a target voxel
            to enter the target mask.

    Returns:
        ``(target_mask, covered_fraction)``: a boolean mask and the float64
        covered fraction on the target grid.

    Raises:
        ValueError: If the mask, target shape, or coverage is invalid.
    """
    source = _boolean_mask(mask, "mask")
    target = _shape(target_shape, "target_shape", ndim=source.ndim)
    level = _finite_float(coverage, "coverage")
    if not 0.0 < level <= 1.0:
        raise ValueError("coverage must lie in (0, 1].")
    # Integer-valued float64 keeps every partial sum exact far below 2**53.
    numerator = source.astype(np.float64)
    denominator = 1
    for axis, (source_size, target_size) in enumerate(zip(source.shape, target)):
        if source_size == target_size:
            continue
        weights, scale = _axis_overlap_weights(source_size, target_size)
        numerator = np.moveaxis(
            np.tensordot(weights.astype(np.float64), numerator, axes=([1], [axis])),
            0,
            axis,
        )
        denominator *= scale
    fraction = np.ascontiguousarray(numerator / float(denominator))
    return fraction >= level, fraction


def alias_partner_shift(n_lin: int, acceleration: int = DEFAULT_ALIAS_ACCELERATION) -> int:
    """Return the pre-registered LIN alias-partner shift ``round(N_LIN / R)``.

    Args:
        n_lin: Number of LIN voxels on the analysis grid.
        acceleration: LIN acceleration factor ``R`` (at least 2).

    Returns:
        Integer shift, rounded half up (85 for 256 voxels and R = 3).

    Raises:
        ValueError: If ``n_lin`` or ``acceleration`` is invalid.
    """
    size = _integer(n_lin, "n_lin", minimum=1)
    factor = _integer(acceleration, "acceleration", minimum=2)
    if size < factor:
        raise ValueError(f"n_lin={size} must be at least the acceleration {factor}.")
    return (2 * size + factor) // (2 * factor)


def alias_partner_mask(
    source_mask: np.ndarray,
    *,
    acceleration: int = DEFAULT_ALIAS_ACCELERATION,
    tolerance: int = DEFAULT_ALIAS_TOLERANCE,
    lin_axis: int = 1,
) -> np.ndarray:
    """Shift a source mask to its hypothesized R-fold LIN alias partners.

    The partner mask is the union of the source shifted by ``+shift + d`` and
    ``-shift + d`` along LIN for every ``d`` in ``[-tolerance, tolerance]``,
    with circular wrap because aliasing wraps modulo ``N_LIN``. Other axes are
    unchanged. For R = 3 these two copies are the complete replica set; for
    R > 3 only the two nearest replicas are represented.

    Args:
        source_mask: Boolean (or 0/1) metal source mask.
        acceleration: LIN acceleration factor ``R``.
        tolerance: Nonnegative LIN tolerance, smaller than the shift.
        lin_axis: Axis index of LIN.

    Returns:
        Boolean partner mask with the source shape.

    Raises:
        ValueError: If the mask, axis, acceleration, or tolerance is invalid.
    """
    source = _boolean_mask(source_mask, "source_mask")
    axis = _axis(lin_axis, source.ndim, "lin_axis")
    shift = alias_partner_shift(source.shape[axis], acceleration)
    width = _integer(tolerance, "tolerance", minimum=0)
    if width >= shift:
        raise ValueError(f"tolerance={width} must be smaller than the partner shift {shift}.")
    partner = np.zeros(source.shape, dtype=bool)
    for sign in (1, -1):
        for offset in range(-width, width + 1):
            partner |= np.roll(source, sign * shift + offset, axis=axis)
    return partner


def dilate_along_axis(
    mask: np.ndarray,
    *,
    axis: int = 0,
    half_width: int | None = None,
) -> np.ndarray:
    """Dilate a boolean mask along one axis without circular wrap.

    Args:
        mask: Boolean (or 0/1) mask.
        axis: Axis to dilate along; RO is axis 0 on the BART grid.
        half_width: Nonnegative half-width in voxels, or ``None`` to fill the
            full extent of every line along ``axis`` that contains the mask.

    Returns:
        New boolean mask with the input shape.

    Raises:
        ValueError: If the mask, axis, or half-width is invalid.
    """
    array = _boolean_mask(mask, "mask")
    dimension = _axis(axis, array.ndim, "axis")
    if half_width is None:
        return np.broadcast_to(array.any(axis=dimension, keepdims=True), array.shape).copy()
    width = _integer(half_width, "half_width", minimum=0)
    if width == 0:
        return array.copy()
    return ndimage.maximum_filter1d(
        array.astype(np.uint8),
        size=2 * width + 1,
        axis=dimension,
        mode="constant",
        cval=0,
    ).astype(bool)


def partner_overlap(target_mask: np.ndarray, partner_mask: np.ndarray) -> float:
    """Return the fraction of target voxels inside a partner mask.

    Args:
        target_mask: Boolean (or 0/1) target mask, such as the fringe.
        partner_mask: Boolean (or 0/1) partner mask with the same shape.

    Returns:
        ``|target AND partner| / |target|``, or NaN when the target is empty.

    Raises:
        ValueError: If the masks are invalid or differ in shape.
    """
    target = _boolean_mask(target_mask, "target_mask")
    partner = _boolean_mask(partner_mask, "partner_mask", shape=target.shape)
    voxels = int(np.count_nonzero(target))
    if voxels == 0:
        return float("nan")
    return float(np.count_nonzero(target & partner)) / voxels


def non_partner_null(
    target_mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    acceleration: int = DEFAULT_ALIAS_ACCELERATION,
    tolerance: int = DEFAULT_ALIAS_TOLERANCE,
    guard: int = 1,
    lin_axis: int = 1,
    ro_half_width: int | None = 0,
) -> dict[str, Any]:
    """Compare the alias-partner overlap with non-partner LIN shifts.

    The observed statistic is :func:`partner_overlap` of the target with the
    alias-partner mask of the source dilated along RO (axis 0). The null
    repeats the same statistic with the partner shift replaced by every LIN
    shift ``s`` in ``[-N/2, N/2)`` whose circular distance from ``+shift``,
    ``-shift``, and 0 exceeds ``tolerance + guard``. All numbers are
    descriptive; none is a p-value.

    Args:
        target_mask: Boolean (or 0/1) target mask, normally the fringe.
        source_mask: Boolean (or 0/1) nonempty metal source mask.
        acceleration: LIN acceleration factor ``R``.
        tolerance: LIN tolerance of every shifted copy.
        guard: Additional LIN guard beyond the tolerance for null exclusion.
        lin_axis: LIN axis index; it must differ from the RO axis 0.
        ro_half_width: RO dilation half-width, or ``None`` for the full RO
            extent.

    Returns:
        JSON-native record with ``partner_shift``, ``observed_overlap``,
        ``null_shifts``, ``null_overlaps``, ``exceedance_fraction`` (fraction
        of null values greater than or equal to the observed value),
        ``ro_half_width``, ``interpretation``, and the parameters used.
        Overlaps and the exceedance fraction are ``None`` for an empty target.

    Raises:
        ValueError: If masks, axes, or parameters are invalid, the source is
            empty, or no non-partner shift remains.
    """
    target = _boolean_mask(target_mask, "target_mask")
    source = _boolean_mask(source_mask, "source_mask", shape=target.shape)
    if target.ndim < 2:
        raise ValueError("Alias-partner masks need separate RO and LIN axes.")
    axis = _axis(lin_axis, target.ndim, "lin_axis")
    if axis == _RO_AXIS:
        raise ValueError("lin_axis must differ from the RO axis 0.")
    size = target.shape[axis]
    shift = alias_partner_shift(size, acceleration)
    width = _integer(tolerance, "tolerance", minimum=0)
    margin = _integer(guard, "guard", minimum=0)
    ro_width = None if ro_half_width is None else _integer(
        ro_half_width, "ro_half_width", minimum=0
    )
    if not source.any():
        raise ValueError("source_mask must contain at least one voxel.")
    excluded = width + margin
    null_shifts = [
        value
        for value in range(-(size // 2), size - size // 2)
        if all(
            _circular_distance(value, center, size) > excluded
            for center in (shift, -shift, 0)
        )
    ]
    if not null_shifts:
        raise ValueError(
            f"No non-partner LIN shift remains for N_LIN={size}, tolerance={width}, guard={margin}."
        )
    partner = dilate_along_axis(
        alias_partner_mask(source, acceleration=acceleration, tolerance=width, lin_axis=axis),
        axis=_RO_AXIS,
        half_width=ro_width,
    )
    voxels = int(np.count_nonzero(target))
    observed: float | None = None
    null_overlaps: list[float | None] = [None] * len(null_shifts)
    exceedance: float | None = None
    if voxels:
        observed_count = int(np.count_nonzero(target & partner))
        null_counts = _symmetric_shift_counts(
            target,
            dilate_along_axis(source, axis=_RO_AXIS, half_width=ro_width),
            null_shifts,
            width,
            axis,
        )
        observed = observed_count / voxels
        null_overlaps = [count / voxels for count in null_counts]
        exceedance = sum(count >= observed_count for count in null_counts) / len(null_counts)
    return {
        "partner_shift": shift,
        "acceleration": int(acceleration),
        "tolerance": width,
        "guard": margin,
        "excluded_within_lin": excluded,
        "lin_axis": axis,
        "ro_axis": _RO_AXIS,
        "ro_half_width": ro_width,
        "target_voxels": voxels,
        "source_voxels": int(np.count_nonzero(source)),
        "observed_overlap": observed,
        "null_shifts": null_shifts,
        "null_overlaps": null_overlaps,
        "exceedance_fraction": exceedance,
        "null_mask_construction": _NULL_MASK_CONSTRUCTION,
        "interpretation": _NULL_INTERPRETATION,
    }


def partner_test(
    target_mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    ro_dilations: Sequence[int | None] = DEFAULT_RO_DILATIONS,
    acceleration: int = DEFAULT_ALIAS_ACCELERATION,
    tolerance: int = DEFAULT_ALIAS_TOLERANCE,
    guard: int = 1,
) -> list[dict[str, Any]]:
    """Run the descriptive alias-partner comparison for each RO dilation.

    Args:
        target_mask: Boolean (or 0/1) target mask, normally the fringe.
        source_mask: Boolean (or 0/1) metal source mask (labels 1 and 2).
        ro_dilations: RO half-widths; ``None`` means the full RO extent.
        acceleration: LIN acceleration factor ``R``.
        tolerance: LIN tolerance of every shifted copy.
        guard: Additional LIN guard for null exclusion.

    Returns:
        One :func:`non_partner_null` record per RO dilation, in input order,
        on the BART grid with LIN axis 1.

    Raises:
        ValueError: If no dilation is given or any input is invalid.
    """
    if isinstance(ro_dilations, str) or not isinstance(ro_dilations, Sequence):
        raise ValueError("ro_dilations must be a sequence of half-widths or None.")
    if not ro_dilations:
        raise ValueError("At least one RO dilation is required.")
    return [
        non_partner_null(
            target_mask,
            source_mask,
            acceleration=acceleration,
            tolerance=tolerance,
            guard=guard,
            lin_axis=1,
            ro_half_width=width,
        )
        for width in ro_dilations
    ]


def edge_matched_control(
    head_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    voxel_size_mm: Sequence[float],
    *,
    band_mm: float = DEFAULT_EDGE_BAND_MM,
    exclusion_mm: float = DEFAULT_EDGE_EXCLUSION_MM,
) -> np.ndarray:
    """Select head-boundary voxels far from the metal source.

    A voxel qualifies when it lies inside the head, the physical distance
    from its center to the nearest non-head voxel center is at most
    ``band_mm``, and the distance to the nearest exclusion voxel center is
    greater than ``exclusion_mm``. Distances use the physical voxel sizes.
    The array border is not treated as air, and an empty exclusion mask
    excludes nothing.

    Args:
        head_mask: Boolean (or 0/1) nonempty head mask.
        exclusion_mask: Boolean (or 0/1) metal source mask to stay away from.
        voxel_size_mm: Positive voxel size per axis in millimetres.
        band_mm: Positive boundary-band width in millimetres.
        exclusion_mm: Nonnegative exclusion distance in millimetres.

    Returns:
        Boolean control mask; it is empty when the head has no boundary
        inside the array.

    Raises:
        ValueError: If masks, voxel sizes, or distances are invalid.
    """
    head = _boolean_mask(head_mask, "head_mask")
    exclusion = _boolean_mask(exclusion_mask, "exclusion_mask", shape=head.shape)
    sampling = _voxel_sizes(voxel_size_mm, head.ndim)
    band = _finite_float(band_mm, "band_mm")
    distance_limit = _finite_float(exclusion_mm, "exclusion_mm")
    if band <= 0:
        raise ValueError("band_mm must be positive.")
    if distance_limit < 0:
        raise ValueError("exclusion_mm must be nonnegative.")
    if not head.any():
        raise ValueError("head_mask must contain at least one voxel.")
    if head.all():
        return np.zeros(head.shape, dtype=bool)
    depth = ndimage.distance_transform_edt(head, sampling=sampling)
    control = head & (depth <= band)
    if exclusion.any():
        distance = ndimage.distance_transform_edt(~exclusion, sampling=sampling)
        control &= distance > distance_limit
    return control


def snr_matched_mask(
    candidate_mask: np.ndarray,
    target_mask: np.ndarray,
    snr: np.ndarray,
    *,
    bins: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw a deterministic candidate subsample matching a target SNR histogram.

    Bin edges are the unique quantiles of the finite target SNR at
    ``linspace(0, 1, bins + 1)``; bins are half-open except the top bin.
    Each bin receives the target's voxel count when enough eligible
    candidates exist. Otherwise every bin is scaled by the same exact factor
    ``min(available / target)`` (rounded down) so the histogram shape is kept.
    Candidates overlapping the target, with non-finite SNR, or outside the
    target SNR range are ineligible. Within a bin, voxels are ordered by
    seeded uniform keys, so a smaller quota selects a subset of a larger one.

    Args:
        candidate_mask: Boolean (or 0/1) candidate control mask.
        target_mask: Boolean (or 0/1) mask whose SNR histogram is matched.
        snr: Real SNR array with the mask shape.
        bins: Requested number of quantile bins (at least 1).
        seed: Nonnegative seed for ``numpy.random.default_rng``.

    Returns:
        ``(selected_mask, record)`` where ``record`` is JSON-native and lists
        bin edges and target, available, and selected counts per bin.

    Raises:
        ValueError: If inputs are invalid or the target has no finite SNR.
    """
    values = np.asarray(snr)
    if values.dtype == np.bool_ or values.dtype.kind not in "iuf":
        raise ValueError("snr must be a real numeric array.")
    values = values.astype(np.float64, copy=False)
    candidate = _boolean_mask(candidate_mask, "candidate_mask", shape=values.shape)
    target = _boolean_mask(target_mask, "target_mask", shape=values.shape)
    bin_count = _integer(bins, "bins", minimum=1)
    seed_value = _integer(seed, "seed", minimum=0)
    finite = np.isfinite(values)
    target_values = values[target & finite]
    if target_values.size == 0:
        raise ValueError("target_mask must contain voxels with finite SNR.")
    edges = np.unique(np.quantile(target_values, np.linspace(0.0, 1.0, bin_count + 1)))
    effective_bins = max(edges.size - 1, 1)
    target_counts = np.bincount(
        _snr_bin_index(target_values, edges), minlength=effective_bins
    )
    candidate_flat = np.flatnonzero(candidate & ~target & finite)
    candidate_bins = _snr_bin_index(values.ravel()[candidate_flat], edges)
    in_range = candidate_bins >= 0
    available_counts = np.bincount(candidate_bins[in_range], minlength=effective_bins)
    ratio = min(
        [
            Fraction(int(available), int(wanted))
            for available, wanted in zip(available_counts, target_counts)
            if wanted > 0
        ]
        + [Fraction(1)]
    )
    selected_counts = [
        (int(wanted) * ratio.numerator) // ratio.denominator for wanted in target_counts
    ]
    generator = np.random.default_rng(seed_value)
    selected = np.zeros(values.size, dtype=bool)
    for index in range(effective_bins):
        pool = candidate_flat[candidate_bins == index]
        # Keys are drawn for every bin, independent of its quota.
        keys = generator.random(pool.size)
        selected[pool[np.argsort(keys, kind="stable")[: selected_counts[index]]]] = True
    mask = selected.reshape(values.shape)
    record = {
        "method": (
            "quantile bins of the finite target SNR; deterministic per-bin draw "
            "without replacement"
        ),
        "bins_requested": bin_count,
        "bins_effective": effective_bins,
        "bin_edges": [float(value) for value in edges],
        "target_counts": [int(value) for value in target_counts],
        "available_counts": [int(value) for value in available_counts],
        "selected_counts": [int(value) for value in selected_counts],
        "scale_factor": float(ratio),
        "complete_match": ratio == 1,
        "seed": seed_value,
        "target_voxels": int(target_values.size),
        "target_nonfinite_excluded": int(np.count_nonzero(target & ~finite)),
        "candidate_voxels": int(np.count_nonzero(candidate)),
        "candidate_target_overlap_excluded": int(np.count_nonzero(candidate & target)),
        "candidate_nonfinite_excluded": int(np.count_nonzero(candidate & ~target & ~finite)),
        "candidate_outside_target_range": int(np.count_nonzero(~in_range)),
        "selected_voxels": int(np.count_nonzero(mask)),
    }
    return mask, record


def display_scale(
    magnitude: np.ndarray,
    reference_mask: np.ndarray | None = None,
    *,
    percentile: float = DEFAULT_DISPLAY_PERCENTILE,
) -> dict[str, Any]:
    """Return the single per-subject magnitude display scale.

    The scale is the requested percentile of the finite magnitude inside the
    preserved-anatomy mask. When that mask is absent or holds no finite
    voxel, all finite positive voxels are used instead and the fallback is
    recorded. Complex input is converted to its absolute value.

    Args:
        magnitude: Real or complex image.
        reference_mask: Optional boolean (or 0/1) preserved-anatomy mask.
        percentile: Percentile in ``(0, 100]``.

    Returns:
        JSON-native ``{"scale", "percentile", "source", "voxels"}`` where
        ``source`` is ``"preserved_anatomy_mask"`` or ``"all_positive_voxels"``
        and ``voxels`` counts the values used.

    Raises:
        ValueError: If inputs are invalid or the scale is not positive.
    """
    values = np.asarray(magnitude)
    if np.iscomplexobj(values):
        values = np.abs(values)
    if values.dtype == np.bool_ or values.dtype.kind not in "iuf":
        raise ValueError("magnitude must be a real or complex numeric array.")
    level = _finite_float(percentile, "percentile")
    if not 0.0 < level <= 100.0:
        raise ValueError("percentile must lie in (0, 100].")
    selected = np.empty(0, dtype=np.float64)
    source = "preserved_anatomy_mask"
    if reference_mask is not None:
        mask = _boolean_mask(reference_mask, "reference_mask", shape=values.shape)
        selected = np.asarray(values[mask], dtype=np.float64)
        selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        source = "all_positive_voxels"
        selected = np.asarray(values[np.isfinite(values) & (values > 0)], dtype=np.float64)
        if selected.size == 0:
            raise ValueError("magnitude has no finite positive voxel for the display scale.")
    scale = float(np.percentile(selected, level))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"The {source} display scale {scale} is not positive.")
    return {
        "scale": scale,
        "percentile": level,
        "source": source,
        "voxels": int(selected.size),
    }


def window_for(metric_name: str) -> tuple[float, float]:
    """Return the fixed display window of one metric family.

    Args:
        metric_name: One of the keys of :data:`FIXED_DISPLAY_WINDOWS`.

    Returns:
        ``(low, high)`` display limits.

    Raises:
        ValueError: If the metric family has no fixed window.
    """
    try:
        return FIXED_DISPLAY_WINDOWS[metric_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"No fixed display window for {metric_name!r}; use one of "
            f"{sorted(FIXED_DISPLAY_WINDOWS)}."
        ) from exc


def apply_display_window(values: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Clip values to a fixed display window without autoscaling.

    Args:
        values: Real array.
        window: ``(low, high)`` finite limits with ``low < high``.

    Returns:
        New float32 array clipped to the window; NaN stays NaN.

    Raises:
        ValueError: If the values are not real or the window is invalid.
    """
    try:
        limits = tuple(window)
    except TypeError as exc:
        raise ValueError("window must contain exactly two finite limits.") from exc
    if len(limits) != 2:
        raise ValueError("window must contain exactly two finite limits.")
    low, high = (_finite_float(value, "window limit") for value in limits)
    if not low < high:
        raise ValueError(f"Display window ({low}, {high}) must satisfy low < high.")
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError("Display windows require real values.")
    output = np.array(array, dtype=np.float32, copy=True)
    np.clip(output, low, high, out=output)
    return output


def summarize_metric(
    metric: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    thresholds: Sequence[float] = (),
) -> dict[str, Any]:
    """Summarize one metric inside each mask with NaN-aware statistics.

    Args:
        metric: Real metric array; non-finite values are ignored.
        masks: Mapping from mask name to boolean (or 0/1) mask with the
            metric shape.
        quantiles: Distinct quantile levels in ``[0, 1]`` (linear method).
        thresholds: Distinct finite thresholds for ``fraction_ge``.

    Returns:
        JSON-native mapping, in sorted mask-name order, from mask name to
        ``{"voxels", "finite", "quantiles", "fraction_ge"}``. Keys of the two
        inner mappings are ``str(float(level))``; values are ``None`` when
        the mask has no finite value. ``fraction_ge`` divides by the finite
        count.

    Raises:
        ValueError: If the metric, masks, levels, or thresholds are invalid.
    """
    values = np.asarray(metric)
    if values.dtype.kind not in "biuf":
        raise ValueError("Metric summaries require a real numeric array.")
    levels = _distinct_floats(quantiles, "quantiles")
    if any(not 0.0 <= level <= 1.0 for level in levels):
        raise ValueError("quantiles must lie in [0, 1].")
    cuts = _distinct_floats(thresholds, "thresholds")
    if not isinstance(masks, Mapping):
        raise ValueError("masks must be a mapping from names to boolean masks.")
    if any(not isinstance(name, str) for name in masks):
        raise ValueError("Mask names must be strings.")
    summary: dict[str, Any] = {}
    for name in sorted(masks):
        mask = _boolean_mask(masks[name], f"mask {name!r}", shape=values.shape)
        selected = np.asarray(values[mask], dtype=np.float64)
        finite = selected[np.isfinite(selected)]
        if finite.size:
            quantile_values = np.quantile(finite, levels) if levels else []
            quantile_record = {
                str(level): float(value) for level, value in zip(levels, quantile_values)
            }
            fraction_record = {
                str(cut): float(np.count_nonzero(finite >= cut)) / finite.size for cut in cuts
            }
        else:
            quantile_record = {str(level): None for level in levels}
            fraction_record = {str(cut): None for cut in cuts}
        summary[name] = {
            "voxels": int(np.count_nonzero(mask)),
            "finite": int(finite.size),
            "quantiles": quantile_record,
            "fraction_ge": fraction_record,
        }
    return summary


def summary_rows(metric_name: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten one :func:`summarize_metric` result into CSV-ready rows.

    Args:
        metric_name: Metric label written to every row.
        summary: Mapping returned by :func:`summarize_metric`.

    Returns:
        One row per mask in sorted mask-name order with keys ``metric``,
        ``mask``, ``voxels``, ``finite``, ``quantile_<level>`` in ascending
        level order, and ``fraction_ge_<threshold>`` in ascending threshold
        order. Undefined values stay ``None``.

    Raises:
        ValueError: If the summary is malformed.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping from mask names to records.")
    rows: list[dict[str, Any]] = []
    for name in sorted(summary):
        record = summary[name]
        try:
            quantile_record = record["quantiles"]
            fraction_record = record["fraction_ge"]
            row: dict[str, Any] = {
                "metric": str(metric_name),
                "mask": str(name),
                "voxels": int(record["voxels"]),
                "finite": int(record["finite"]),
            }
            for key in sorted(quantile_record, key=float):
                row[f"quantile_{key}"] = quantile_record[key]
            for key in sorted(fraction_record, key=float):
                row[f"fraction_ge_{key}"] = fraction_record[key]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed summary record for mask {name!r}.") from exc
        rows.append(row)
    return rows


def representative_slices(mask: np.ndarray, axis: int, count: int = 5) -> list[int]:
    """Choose evenly spaced slice indices across the extent of a mask.

    The extent ``[low, high]`` along ``axis`` is split into ``count`` equal
    segments; each segment center is snapped to the nearest slice that
    contains the mask (ties go to the lower index). Duplicate indices are
    merged, so fewer than ``count`` indices may be returned.

    Args:
        mask: Boolean (or 0/1) mask.
        axis: Slice axis.
        count: Requested number of slices (at least 1).

    Returns:
        Sorted slice indices; ``[shape[axis] // 2]`` for an empty mask.

    Raises:
        ValueError: If the mask, axis, or count is invalid.
    """
    array = _boolean_mask(mask, "mask")
    dimension = _axis(axis, array.ndim, "axis")
    requested = _integer(count, "count", minimum=1)
    others = tuple(index for index in range(array.ndim) if index != dimension)
    occupied = np.flatnonzero(array.any(axis=others) if others else array)
    if occupied.size == 0:
        return [int(array.shape[dimension] // 2)]
    low, high = int(occupied[0]), int(occupied[-1])
    span = high - low + 1
    # Work in units of 1 / (2 * count) slice so that segment centers and
    # distances are exact integers.
    scaled = occupied.astype(np.int64) * (2 * requested)
    chosen = set()
    for segment in range(requested):
        center = (2 * low - 1) * requested + span * (2 * segment + 1)
        position = int(np.searchsorted(scaled, center))
        candidates = [
            (abs(int(scaled[index]) - center), index)
            for index in (position - 1, position)
            if 0 <= index < occupied.size
        ]
        chosen.add(int(occupied[min(candidates)[1]]))
    return sorted(chosen)


def _shape(shape: object, name: str, ndim: int | None = 3) -> tuple[int, ...]:
    """Validate a shape of positive integers.

    Args:
        shape: Candidate shape sequence.
        name: Name used in error messages.
        ndim: Required number of axes, or ``None`` for any positive number.

    Returns:
        Shape as a tuple of Python integers.

    Raises:
        ValueError: If the shape is malformed.
    """
    try:
        values = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of positive integers.") from exc
    if (
        not values
        or (ndim is not None and len(values) != ndim)
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
            or int(value) < 1
            for value in values
        )
    ):
        expected = "positive integers" if ndim is None else f"{ndim} positive integers"
        raise ValueError(f"{name} must contain {expected}, not {shape!r}.")
    return tuple(int(value) for value in values)


def _integer(value: object, name: str, *, minimum: int) -> int:
    """Validate one integer parameter.

    Args:
        value: Candidate integer (booleans are rejected).
        name: Name used in error messages.
        minimum: Smallest accepted value.

    Returns:
        Value as a Python integer.

    Raises:
        ValueError: If the value is not an integer of at least ``minimum``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {value}.")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    """Validate one finite real number.

    Args:
        value: Candidate number (booleans are rejected).
        name: Name used in error messages.

    Returns:
        Value as a Python float.

    Raises:
        ValueError: If the value is not a finite real number.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, not {value!r}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, not {value!r}.")
    return number


def _distinct_floats(values: Sequence[float], name: str) -> list[float]:
    """Validate a sequence of distinct finite numbers.

    Args:
        values: Candidate numbers.
        name: Name used in error messages.

    Returns:
        Values as Python floats in input order.

    Raises:
        ValueError: If a value is invalid or repeated.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numbers.")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of numbers.") from exc
    numbers_list = [_finite_float(value, name) for value in items]
    if len(set(numbers_list)) != len(numbers_list):
        raise ValueError(f"{name} must not repeat values.")
    return numbers_list


def _voxel_sizes(voxel_size_mm: object, ndim: int) -> tuple[float, ...]:
    """Validate one positive finite voxel size per axis.

    Args:
        voxel_size_mm: Candidate voxel sizes in millimetres.
        ndim: Number of array axes.

    Returns:
        Voxel sizes as Python floats.

    Raises:
        ValueError: If the count or any value is invalid.
    """
    try:
        values = tuple(voxel_size_mm)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("voxel_size_mm must be a sequence of positive sizes.") from exc
    if len(values) != ndim:
        raise ValueError(f"voxel_size_mm must contain {ndim} sizes, not {len(values)}.")
    sizes = tuple(_finite_float(value, "voxel_size_mm") for value in values)
    if any(size <= 0 for size in sizes):
        raise ValueError(f"voxel_size_mm must be positive, not {sizes}.")
    return sizes


def _axis(axis: object, ndim: int, name: str) -> int:
    """Normalize one axis index.

    Args:
        axis: Candidate axis; negative values count from the end.
        ndim: Number of array axes.
        name: Name used in error messages.

    Returns:
        Nonnegative axis index.

    Raises:
        ValueError: If the axis is not an integer inside the array rank.
    """
    index = _integer(axis, name, minimum=-ndim)
    if index >= ndim:
        raise ValueError(f"{name}={index} is outside an array with {ndim} axes.")
    return index % ndim


def _boolean_mask(
    mask: object, name: str, shape: Sequence[int] | None = None
) -> np.ndarray:
    """Validate one boolean or 0/1 numeric mask.

    Args:
        mask: Candidate mask.
        name: Name used in error messages.
        shape: Optional exact required shape.

    Returns:
        Boolean array; boolean inputs are returned without copying.

    Raises:
        ValueError: If the mask is not boolean or 0/1, is zero-dimensional,
            or has the wrong shape.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        if array.dtype.kind not in "iuf" or not ((array == 0) | (array == 1)).all():
            raise ValueError(f"{name} must be a boolean or 0/1 numeric mask.")
        array = array.astype(bool)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one axis.")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name} shape {array.shape} differs from {tuple(shape)}.")
    return array


def _label_values(labels: object) -> np.ndarray:
    """Validate ROI label values and return them as uint8.

    Args:
        labels: Candidate label array.

    Returns:
        uint8 label array with values from :data:`ROI_LABELS`.

    Raises:
        ValueError: If labels are empty, non-numeric, non-finite, non-integer,
            or outside the label contract.
    """
    values = np.asarray(labels)
    if values.dtype == np.bool_ or values.dtype.kind not in "iuf":
        raise ValueError("ROI labels must be a real integer-valued numeric array.")
    if values.ndim == 0 or values.size == 0:
        raise ValueError("ROI labels must be a nonempty array.")
    if values.dtype.kind == "f":
        if not np.isfinite(values).all():
            raise ValueError("ROI labels must be finite.")
        if not np.array_equal(values, np.rint(values)):
            raise ValueError("ROI labels must be integer values.")
    # Labels are the consecutive integers 0-5, so a range check is an exact
    # membership test.
    if values.min() < min(ROI_LABELS) or values.max() > max(ROI_LABELS):
        unsupported = sorted(
            {float(value) for value in np.unique(values)} - {float(key) for key in ROI_LABELS}
        )
        raise ValueError(f"ROI labels contain unsupported values: {unsupported}.")
    return values.astype(np.uint8)


def _assignable_label(token: object) -> int:
    """Resolve a label name or number to an assignable label value.

    Args:
        token: Label name (case-insensitive), numeric text, or integer.

    Returns:
        Label value from 1 to 5.

    Raises:
        ValueError: If the token is unknown or names label 0.
    """
    if isinstance(token, (bool, np.bool_)):
        raise ValueError(f"Unknown ROI label {token!r}.")
    if isinstance(token, numbers.Integral):
        value = int(token)
    else:
        text = str(token).strip().lower()
        if text.isascii() and text.isdigit():
            value = int(text)
        elif text in LABEL_VALUES_BY_NAME:
            value = LABEL_VALUES_BY_NAME[text]
        else:
            raise ValueError(
                f"Unknown ROI label {token!r}; use a name from "
                f"{[ROI_LABELS[key] for key in _ASSIGNABLE_LABELS]} or a number 1-5."
            )
    if value not in _ASSIGNABLE_LABELS:
        raise ValueError(f"ROI label {token!r} is not assignable; use labels 1-5.")
    return value


def _flips(array_flips: object) -> tuple[bool, bool, bool]:
    """Validate the three physical array-flip flags.

    Args:
        array_flips: Candidate sequence of three booleans.

    Returns:
        Flags as a tuple of Python booleans.

    Raises:
        ValueError: If the value is not exactly three booleans.
    """
    try:
        flags = tuple(array_flips)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("array_flips must contain exactly three booleans.") from exc
    if len(flags) != 3 or not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError(f"array_flips must contain exactly three booleans, not {array_flips!r}.")
    return tuple(bool(flag) for flag in flags)  # type: ignore[return-value]


def _affine(affine: object, name: str) -> np.ndarray:
    """Validate one finite nonsingular homogeneous 4x4 affine.

    Args:
        affine: Candidate affine.
        name: Name used in error messages.

    Returns:
        Float64 affine copy.

    Raises:
        ValueError: If the affine is malformed or singular.
    """
    try:
        matrix = np.array(affine, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 4x4 affine.") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 affine.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must have the homogeneous row (0, 0, 0, 1).")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError(f"{name} must be nonsingular.")
    return matrix


def _transform(orientation_transform: object) -> np.ndarray:
    """Validate a nibabel orientation transform.

    Args:
        orientation_transform: Candidate ``(3, 2)`` transform.

    Returns:
        Float64 ``(3, 2)`` array.

    Raises:
        ValueError: If the transform is malformed.
    """
    try:
        transform = np.array(orientation_transform, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation_transform must be a 3x2 nibabel transform.") from exc
    if (
        transform.shape != (3, 2)
        or sorted(transform[:, 0].tolist()) != [0.0, 1.0, 2.0]
        or not np.isin(transform[:, 1], (-1.0, 1.0)).all()
    ):
        raise ValueError("orientation_transform must be a 3x2 nibabel transform.")
    return transform


def _sha256_text(value: object, name: str) -> str:
    """Validate one hexadecimal SHA-256 digest.

    Args:
        value: Candidate digest.
        name: Name used in error messages.

    Returns:
        Lower-case digest.

    Raises:
        ValueError: If the value is not 64 hexadecimal characters.
    """
    text = str(value).strip().lower() if isinstance(value, str) else ""
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest.")
    return text


def _canonical_geometry(
    bart_shape: Sequence[int], source_affine: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
    """Derive the canonical-RAS transform, shape, and affine of a source grid.

    Args:
        bart_shape: Shape of the flipped BART array described by the affine.
        source_affine: Validated voxel-to-RAS source affine.

    Returns:
        ``(transform, stored_shape, canonical_affine)`` computed with nibabel
        exactly as the upstream canonicalization does.
    """
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(source_affine),
        nib.orientations.axcodes2ornt(_RAS_CODES),
    )
    stored_shape = [0, 0, 0]
    for axis, size in enumerate(bart_shape):
        stored_shape[int(transform[axis, 0])] = int(size)
    canonical_affine = source_affine @ nib.orientations.inv_ornt_aff(transform, tuple(bart_shape))
    return transform, tuple(stored_shape), canonical_affine  # type: ignore[return-value]


def _forward_orientation(
    array: np.ndarray,
    flips: tuple[bool, bool, bool],
    affine: np.ndarray,
    helpers: Any,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Apply the upstream flips and RAS canonicalization to one array.

    Args:
        array: Three-dimensional BART array.
        flips: Validated physical array flips.
        affine: Validated source affine.
        helpers: Namespace with the upstream helper callables.

    Returns:
        Copied stored array, float64 canonical affine, and JSON-native
        orientation transform, without contract checks.
    """
    corrected = helpers.apply_array_axis_flips((array,), flips)[0]
    arrays, canonical_affine, transform = helpers.canonicalize_arrays_to_ras(
        (np.asarray(corrected),), affine
    )
    stored = np.array(arrays[0], order="C", copy=True)
    matrix = np.asarray(canonical_affine, dtype=np.float64)
    transform_list = [
        [float(value) for value in row]
        for row in np.asarray(transform, dtype=np.float64).tolist()
    ]
    return stored, matrix, transform_list


def _load_helpers() -> Any:
    """Load the pinned upstream orientation helpers.

    Returns:
        Namespace from ``wave_retro_lr.mprage.load_wave_mprage_helpers``.

    Side Effects:
        Adds the pinned upstream reconstruction directory to ``sys.path``.
    """
    from .mprage import load_wave_mprage_helpers

    return load_wave_mprage_helpers()


def _arrays_identical(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two arrays share shape, dtype, and exact values.

    Args:
        left: Array to compare.
        right: Array to compare.

    Returns:
        True when shape, dtype, and every value match; NaN matches NaN.
    """
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(
        np.array_equal(left, right, equal_nan=bool(np.issubdtype(left.dtype, np.inexact)))
    )


def _axis_overlap_weights(source_size: int, target_size: int) -> tuple[np.ndarray, int]:
    """Compute exact 1D interval overlaps between two centered grids.

    Args:
        source_size: Number of source voxels spanning the FOV.
        target_size: Number of target voxels spanning the same FOV.

    Returns:
        ``(weights, denominator)`` where ``weights[j, i] / denominator`` is
        the fraction of target voxel ``j`` covered by source voxel ``i``,
        reduced by their common divisor.
    """
    # In units of FOV / (2 * Ns * Nt), source voxels are 2 * Nt wide and
    # target voxels 2 * Ns wide, so every center and edge is an integer.
    source_center = (np.arange(source_size, dtype=np.int64) - source_size // 2) * (
        2 * target_size
    )
    target_center = (np.arange(target_size, dtype=np.int64) - target_size // 2) * (
        2 * source_size
    )
    low = np.maximum(
        target_center[:, None] - source_size, source_center[None, :] - target_size
    )
    high = np.minimum(
        target_center[:, None] + source_size, source_center[None, :] + target_size
    )
    overlap = np.clip(high - low, 0, None)
    denominator = 2 * source_size
    divisor = int(np.gcd.reduce(np.append(overlap.ravel(), denominator)))
    return overlap // divisor, denominator // divisor


def _circular_distance(left: int, right: int, size: int) -> int:
    """Return the circular distance between two shifts modulo ``size``.

    Args:
        left: Shift value.
        right: Shift value.
        size: Period of the circular axis.

    Returns:
        Smallest nonnegative wrapped distance.
    """
    difference = (left - right) % size
    return min(difference, size - difference)


def _symmetric_shift_counts(
    target: np.ndarray,
    dilated_source: np.ndarray,
    shifts: Sequence[int],
    tolerance: int,
    lin_axis: int,
) -> list[int]:
    """Count target voxels inside the symmetric shifted source for each shift.

    For shift ``s`` the evaluated mask is the RO-dilated source rolled by
    ``+s + d`` and ``-s + d`` along LIN for all ``|d| <= tolerance``; it is
    evaluated only at target voxels, which is equivalent to building it.

    Args:
        target: Boolean target mask.
        dilated_source: Boolean RO-dilated source mask.
        shifts: LIN shifts to evaluate.
        tolerance: LIN tolerance of every shifted copy.
        lin_axis: LIN axis index.

    Returns:
        Target voxel count inside the mask for each shift, in input order.
    """
    window = np.zeros(dilated_source.shape, dtype=bool)
    for offset in range(-tolerance, tolerance + 1):
        window |= np.roll(dilated_source, offset, axis=lin_axis)
    moved = np.moveaxis(window, lin_axis, -1)
    coordinates = np.nonzero(np.moveaxis(target, lin_axis, -1))
    others, lin_index = coordinates[:-1], coordinates[-1]
    size = window.shape[lin_axis]
    counts = []
    for shift in shifts:
        # np.roll(window, s)[..., l] equals window[..., (l - s) mod N].
        hits = moved[others + (np.remainder(lin_index - shift, size),)] | moved[
            others + (np.remainder(lin_index + shift, size),)
        ]
        counts.append(int(np.count_nonzero(hits)))
    return counts


def _snr_bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign SNR values to quantile bins.

    Args:
        values: Finite SNR values.
        edges: Sorted unique bin edges.

    Returns:
        Bin index per value, or -1 outside ``[edges[0], edges[-1]]``. Bins are
        half-open except the top bin; a single edge forms one exact-value bin.
    """
    if edges.size == 1:
        return np.where(values == edges[0], 0, -1)
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = edges.size - 2
    index[(values < edges[0]) | (values > edges[-1])] = -1
    return index
