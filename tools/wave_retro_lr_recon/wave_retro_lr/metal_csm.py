"""Explicitly requested, bounded single-map CSM continuation for metal diagnostics.

This is not metal detection, anatomical segmentation, or ESPIRiT soft-SENSE.
All spatial settings are explicit and expressed on a verified reference grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from .bart_io import create_cfl, open_cfl

METHOD_VERSION = "bounded-metal-csm-v1"


@dataclass(frozen=True)
class MetalCSMParameters:
    """Explicit ROI, physical lengths and confidence gates for one reviewed experiment.

    ROI bounds are half-open reference-NIfTI voxel indices. ``envelope_axis``
    selects the axis normal to each 2-D convex-envelope plane. Global base
    extension is separately opt-in: when false, nothing outside the ROI changes.
    Numeric values are experiment settings, not population-validated defaults.
    """

    roi_voxels: tuple[tuple[int, int], ...]
    envelope_axis: int
    base_distance_mm: float
    max_distance_mm: float
    near_sigma_mm: float
    far_sigma_mm: float
    edge_ramp_mm: float
    global_base_extension: bool = False
    minimum_weight: float = 0.01
    minimum_coherence: float = 0.2

    @classmethod
    def from_dict(cls, values: Mapping) -> "MetalCSMParameters":
        """Parse a strict parameter mapping; reject unknown keys and unsafe numeric values.

        Args:
            values: JSON-compatible parameters (no paths or subject identifiers).
        Returns:
            Validated immutable parameters; grid bounds are checked separately.
        Raises:
            ValueError: Missing/unknown settings, invalid ROI, or nonfinite values.
        """
        try:
            result = cls(**values)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Invalid metal CSM parameters: {exc}") from exc
        roi = result.roi_voxels
        if not isinstance(roi, (list, tuple)) or len(roi) != 3 or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1]
            for pair in roi
        ):
            raise ValueError("roi_voxels must contain three nonempty half-open integer ranges")
        if type(result.envelope_axis) is not int or result.envelope_axis not in (0, 1, 2):
            raise ValueError("envelope_axis must be 0, 1 or 2")
        if type(result.global_base_extension) is not bool:
            raise ValueError("global_base_extension must be boolean")
        for name in ("base_distance_mm", "max_distance_mm", "near_sigma_mm", "far_sigma_mm",
                     "edge_ramp_mm", "minimum_weight", "minimum_coherence"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if result.max_distance_mm < result.base_distance_mm or result.far_sigma_mm < result.near_sigma_mm:
            raise ValueError("Far distance/sigma must not be smaller than base distance/near sigma")
        if result.minimum_weight > 1 or result.minimum_coherence > 1:
            raise ValueError("Confidence gates must be in (0, 1]")
        return cls(**dict(asdict(result), roi_voxels=tuple(tuple(p) for p in roi)))

    def to_dict(self) -> dict:
        """Return a stable JSON-native representation of these parameters."""
        return dict(asdict(self), roi_voxels=[list(p) for p in self.roi_voxels])


def orient(values: np.ndarray, flips: tuple[bool, ...]) -> np.ndarray:
    """Return a logical RO/LIN/PAR scalar array in verified NIfTI voxel order."""
    result = values.transpose(2, 1, 0)
    for axis, flip in enumerate(flips):
        if flip:
            result = np.flip(result, axis)
    return result


def inverse_orient(values: np.ndarray, flips: tuple[bool, ...]) -> np.ndarray:
    """Return NIfTI-order scalar values in logical RO/LIN/PAR order."""
    result = values
    for axis, flip in enumerate(flips):
        if flip:
            result = np.flip(result, axis)
    return result.transpose(2, 1, 0)


def smoothstep(values: np.ndarray) -> np.ndarray:
    """Return the cubic zero-to-one ramp, clipped outside its transition interval."""
    t = np.clip(values, 0, 1)
    return t*t*(3-2*t)


def convex_envelope(plane: np.ndarray) -> np.ndarray:
    """Return the 2-D convex hull of the largest connected support component.

    Empty or collinear components are returned unchanged. This geometric
    restriction is not an anatomical tissue mask.
    """
    from matplotlib.path import Path as Polygon

    labels, count = ndi.label(plane)
    if not count:
        return np.zeros_like(plane, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == sizes.argmax()
    boundary = np.argwhere(main & ~ndi.binary_erosion(main))
    if len(boundary) < 4 or np.linalg.matrix_rank(boundary-boundary[0]) < 2:
        return main
    vertices = boundary[ConvexHull(boundary).vertices]
    coordinates = np.indices(plane.shape).reshape(2, -1).T
    return Polygon(vertices).contains_points(coordinates, radius=1e-6).reshape(plane.shape)


def continuation_vectors(
    maps: np.ndarray, flips: tuple[bool, ...], spacing: tuple[float, ...],
    parameters: MetalCSMParameters, *, checkpoint: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute sparse added coil vectors on a verified reference grid.

    Args:
        maps: Logical (RO, LIN, PAR, coil) complex maps with binary unit-RSS support.
        flips: Verified flips following logical-to-NIfTI permutation (2, 1, 0).
        spacing: NIfTI voxel sizes in millimetres, allowing anisotropic grids.
        parameters: Explicit continuation settings.
        checkpoint: Optional resource guard called between memory-bounded stages.
    Returns:
        Native support, added support (both NIfTI order), added unit coil vectors,
        and validation statistics. Original vectors are not modified.
    Raises:
        ValueError: Invalid geometry, soft/multiple-map input, or unstable values.
    """
    parameters = MetalCSMParameters.from_dict(parameters.to_dict())
    if maps.ndim != 4 or min(maps.shape) < 1 or len(flips) != 3:
        raise ValueError("Expected one 3-D map per coil")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Voxel spacing must be three finite positive values")
    rss2 = np.zeros(maps.shape[:3], np.float32)
    for coil in range(maps.shape[-1]):
        if checkpoint:
            checkpoint()
        field = np.asarray(maps[..., coil])
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite original CSM")
        rss2 += abs(field)**2
    rss = np.sqrt(rss2)
    if not np.all((rss < 1e-8) | (abs(rss-1) < 1e-5)):
        raise ValueError("Only hard-cropped unit-RSS single-map CSM is supported; not soft-SENSE")
    native = orient(rss > .5, flips)
    if not native.any():
        raise ValueError("Empty native CSM support")
    roi = parameters.roi_voxels
    if any(b > n for (_, b), n in zip(roi, native.shape)):
        raise ValueError("ROI exceeds the reference grid")
    window = np.ones(native.shape, np.float32)
    for axis, (low, high) in enumerate(roi):
        coordinate = np.arange(native.shape[axis])
        ramp = smoothstep(np.minimum(coordinate-low, high-1-coordinate)*spacing[axis]/parameters.edge_ramp_mm)
        shape = [1, 1, 1]
        shape[axis] = len(ramp)
        window *= ramp.reshape(shape)
    if not np.any(window > 0):
        raise ValueError("ROI has no interior after edge taper")
    hull = np.zeros_like(native)
    axis = parameters.envelope_axis
    planes = range(native.shape[axis]) if parameters.global_base_extension else range(*roi[axis])
    for index in planes:
        selection = [slice(None)]*3
        selection[axis] = index
        hull[tuple(selection)] = convex_envelope(native[tuple(selection)])
    distance = ndi.distance_transform_edt(~native, sampling=spacing)
    base = parameters.base_distance_mm
    cap = (base + (parameters.max_distance_mm-base)*window if parameters.global_base_extension
           else parameters.max_distance_mm*window)
    proposal = hull & ~native & (distance <= cap)
    if not parameters.global_base_extension:
        proposal &= window > 0
    if not proposal.any():
        return native, np.zeros_like(native), np.empty((0, maps.shape[-1]), np.complex64), {"added_voxels": 0}
    sigmas = (parameters.near_sigma_mm, parameters.far_sigma_mm)
    weights, predictions = [], []
    for sigma in sigmas:
        weights.append(ndi.gaussian_filter(native.astype(np.float32), sigma/spacing, mode="constant", truncate=3)[proposal])
        predictions.append(np.empty((int(proposal.sum()), maps.shape[-1]), np.complex64))
    # Store only proposed vectors, not two full spatial-by-coil prediction arrays.
    for coil in range(maps.shape[-1]):
        if checkpoint:
            checkpoint()
        field = orient(np.asarray(maps[..., coil]), flips)
        for index, sigma in enumerate(sigmas):
            predictions[index][:, coil] = ndi.gaussian_filter(
                field, sigma/spacing, mode="constant", truncate=3
            )[proposal]/np.maximum(weights[index], 1e-12)
    coherence = [np.linalg.norm(v, axis=-1) for v in predictions]
    unit = [v/np.maximum(c[:, None], 1e-12) for v, c in zip(predictions, coherence)]
    blend = window[proposal]*smoothstep((distance[proposal]-base/2)/(base/2))
    mixture = (1-blend[:, None])*unit[0] + blend[:, None]*unit[1]
    norm = np.linalg.norm(mixture, axis=-1)
    near_ok = ((distance[proposal] <= base) & (weights[0] >= parameters.minimum_weight)
               & (coherence[0] >= parameters.minimum_coherence))
    local_ok = ((window[proposal] > 0) & (weights[1] >= parameters.minimum_weight)
                & (coherence[1] >= parameters.minimum_coherence))
    accepted = local_ok | (near_ok if parameters.global_base_extension else False)
    if not np.isfinite(mixture).all() or np.any(norm[accepted] < .2):
        raise ValueError("Unstable complex coil-vector continuation; inspect phase gauge")
    vectors = mixture[accepted]/np.maximum(norm[accepted, None], 1e-12)
    added = np.zeros_like(native)
    added[proposal] = accepted
    stats = {
        "added_voxels": int(added.sum()), "added_fraction_of_grid": float(added.mean()),
        "added_outside_roi": int(np.sum(added & (window == 0))),
        "native_support_voxels": int(native.sum()),
        "maximum_added_distance_mm": float(distance[added].max()) if added.any() else 0.,
        "minimum_accepted_mixture_norm": float(norm[accepted].min()) if accepted.any() else None,
    }
    return native, added, vectors, stats


def write_candidate(
    source: str | Path, destination: str | Path, flips: tuple[bool, ...],
    spacing: tuple[float, ...], parameters: MetalCSMParameters,
    *, checkpoint: Callable[[], None] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Create an immutable candidate CFL; preserve every original/non-added vector exactly.

    Args:
        source: Hash-validated native CFL basename, with singleton map/trailing axes.
        destination: New CFL basename; neither member may already exist.
        flips: Verified NIfTI orientation flips.
        spacing: NIfTI voxel sizes in mm.
        parameters: Explicit continuation settings.
        checkpoint: Optional memory guard.
    Returns:
        Validation statistics and native/added logical-grid masks.
    Raises:
        FileExistsError: Destination exists; ValueError: invalid/no eligible support.
    """
    destination = Path(destination)
    if any(destination.with_suffix(ext).exists() for ext in (".cfl", ".hdr")):
        raise FileExistsError("Refusing to overwrite candidate CSM")
    array = open_cfl(source)
    if len(array.shape) < 4 or any(n != 1 for n in array.shape[4:]):
        raise ValueError("Multiple ESPIRiT maps/echoes are not supported")
    maps = array.reshape(array.shape[:4], order="F")
    native, added, vectors, stats = continuation_vectors(maps, flips, spacing, parameters, checkpoint=checkpoint)
    if not added.any():
        raise ValueError("No eligible support addition; reconstruction was not requested for a no-op")
    logical_native, logical_added = inverse_orient(native, flips), inverse_orient(added, flips)
    candidate = create_cfl(destination, maps.shape)
    rss2 = np.zeros(maps.shape[:3], np.float32)
    max_change = 0.
    for coil in range(maps.shape[-1]):
        if checkpoint:
            checkpoint()
        field = np.asarray(maps[..., coil]).copy()
        orient(field, flips)[added] = vectors[:, coil]
        max_change = max(max_change, float(np.max(abs(field[~logical_added]-maps[..., coil][~logical_added]))))
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite candidate CSM")
        rss2 += abs(field)**2
        candidate[..., coil] = field
    candidate.flush()
    rss_error = float(np.max(abs(np.sqrt(rss2)-(logical_native | logical_added))))
    if max_change != 0 or rss_error > 1e-5:
        raise ValueError("Original-vector preservation or unit-RSS check failed")
    stats.update(unchanged_vectors_max_error=max_change, candidate_rss_max_error=rss_error)
    return stats, logical_native, logical_added
