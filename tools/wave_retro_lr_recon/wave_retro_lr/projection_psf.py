"""Select stable spatial support for integrated projection PSF calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

SPATIAL_SELECTION_VERSION = 2
FULL_CORE_MAXIMUM_PHASE_DIFFERENCE_RAD = 0.15
MAXIMUM_MEDIAN_WRAPPED_RMS_RAD = 0.45
MINIMUM_VALID_READOUT_FRACTION = 0.60


def centered_spatial_core_bounds(size: int) -> tuple[int, int]:
    """Return the half-open central 50% calibration-plane interval.

    Args:
        size: Full calibration-plane width in samples.

    Returns:
        Inclusive lower and exclusive upper bounds. A minimum width of 12
        samples is retained for small test and calibration matrices.

    Raises:
        ValueError: If ``size`` cannot contain the minimum central interval.
    """
    width = int(size)
    if width < 12:
        raise ValueError("Spatial calibration width must be at least 12 samples.")
    core_width = max(12, width // 2)
    lower = width // 2 - core_width // 2
    return lower, lower + core_width


class AutomaticSpatialRegionRejected(ValueError):
    """Report that no stable center-containing projection region was found."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any],
        plane: Mapping[str, object],
    ) -> None:
        """Initialize a rejection with serializable metrics and plot evidence.

        Args:
            message: Human-readable rejection reason.
            diagnostics: Candidate selection metrics.
            plane: Full-grid theoretical and measured plane evidence.

        Returns:
            None.
        """
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.plane = dict(plane)


def align_constant_phase_branch(
    coefficient_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a continuous representative of a constant phase coefficient.

    Only integer multiples of ``2*pi`` are added, so the complex phase is
    unchanged at every readout sample. The continuous branch is anchored to
    the original value at the center finite sample.

    Args:
        coefficient_c: One-dimensional constant phase coefficient in radians.

    Returns:
        Branch-aligned coefficient and the signed integer turn added at each
        sample. Non-finite samples are retained with a zero turn.

    Raises:
        ValueError: If the input is not a nonempty one-dimensional vector.
    """
    original = np.asarray(coefficient_c, dtype=np.float64)
    if original.ndim != 1 or original.size == 0:
        raise ValueError("Constant phase coefficient must be a nonempty 1D vector.")
    finite_indices = np.flatnonzero(np.isfinite(original))
    if finite_indices.size == 0:
        return original.copy(), np.zeros(original.size, dtype=np.int64)

    wrapped = np.angle(np.exp(1j * original[finite_indices]))
    continuous = np.unwrap(wrapped)
    center = original.size // 2
    anchor_position = int(np.argmin(np.abs(finite_indices - center)))
    global_turn = int(
        np.rint(
            (original[finite_indices[anchor_position]] - continuous[anchor_position])
            / (2.0 * np.pi)
        )
    )
    continuous += global_turn * 2.0 * np.pi

    turns = np.zeros(original.size, dtype=np.int64)
    turns[finite_indices] = np.rint(
        (continuous - original[finite_indices]) / (2.0 * np.pi)
    ).astype(np.int64)
    aligned = original.copy()
    aligned[finite_indices] = (
        original[finite_indices] + turns[finite_indices] * 2.0 * np.pi
    )
    return aligned, turns


def _quality_summary(result: Mapping[str, Any]) -> dict[str, float]:
    """Summarize valid readout support and wrapped residual RMS.

    Args:
        result: Wrapped-plane fitter output.

    Returns:
        Finite-readout fraction and median wrapped RMS.
    """
    rms = np.asarray(result["wrapped_rms"], dtype=np.float64).reshape(-1)
    finite = np.isfinite(rms)
    return {
        "valid_readout_fraction": float(np.mean(finite)),
        "median_wrapped_rms_rad": (
            float(np.median(rms[finite])) if finite.any() else float("inf")
        ),
    }


def _projection_coefficients(
    result: Mapping[str, Any], axis_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the varying-axis slope and constant coefficient.

    Args:
        result: Wrapped-plane fitter output.
        axis_name: ``y`` selects ``a`` and ``z`` selects ``b``.

    Returns:
        Slope and constant coefficient vectors.
    """
    slope_key = "a_fit_all" if axis_name == "y" else "b_fit_all"
    return (
        np.asarray(result[slope_key], dtype=np.float64).reshape(-1),
        np.asarray(result["c_fit_all"], dtype=np.float64).reshape(-1),
    )


def _model_difference(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    coordinates: np.ndarray,
    axis_name: str,
) -> float:
    """Measure sustained complex-phase disagreement between two plane fits.

    Args:
        first: First wrapped-plane fit result.
        second: Second wrapped-plane fit result.
        coordinates: Original full-FOV spatial coordinates used for comparison.
        axis_name: Varying projection axis, ``y`` or ``z``.

    Returns:
        Upper-quartile absolute wrapped phase difference in radians. This
        detects a sustained corrupted readout segment without allowing a few
        edge outliers to determine the decision.
    """
    slope_first, constant_first = _projection_coefficients(first, axis_name)
    slope_second, constant_second = _projection_coefficients(second, axis_name)
    valid = (
        np.isfinite(slope_first)
        & np.isfinite(constant_first)
        & np.isfinite(slope_second)
        & np.isfinite(constant_second)
    )
    if not valid.any():
        return float("inf")
    difference = (
        (slope_first[valid] - slope_second[valid])[:, None]
        * coordinates[None, :]
        + (constant_first[valid] - constant_second[valid])[:, None]
    )
    wrapped = np.angle(np.exp(1j * difference))
    per_readout = np.median(np.abs(wrapped), axis=1)
    return float(np.quantile(per_readout, 0.75))


def _candidate_bounds(size: int, core: tuple[int, int]) -> list[tuple[int, int]]:
    """Build a small deterministic set of center-containing fit intervals.

    Args:
        size: Full calibration projection width.
        core: Half-open central reference interval.

    Returns:
        Candidate intervals ordered from widest to narrowest.
    """
    step = max(2, size // 12)
    lowers = sorted(set(range(0, core[0] + 1, step)) | {core[0]})
    uppers = sorted(set(range(core[1], size + 1, step)) | {size})
    candidates = {
        (lower, upper)
        for lower in lowers
        for upper in uppers
        if lower <= core[0] and upper >= core[1]
    }
    return sorted(candidates, key=lambda value: (-(value[1] - value[0]), value))


def select_spatial_projection_region(
    psf_difference: Any,
    hybrid_nowave: Any,
    spatial_coordinates: np.ndarray,
    *,
    axis_name: str,
    fit_region: Callable[[tuple[int, int]], Mapping[str, Any]],
    manual_bounds: tuple[int, int] | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Select one global spatial fit interval for an entire projection plane.

    Every automatic call compares the full spatial support with a central-core
    fit in complex phase. Clean cases return the original full-support result
    without refitting it. When the full result is unstable, the widest fit
    consistent with the central core is selected.

    Args:
        psf_difference: Full wrapped phase-difference array; used for shape
            validation and rejection evidence.
        hybrid_nowave: Full no-Wave hybrid data; used for shape validation.
        spatial_coordinates: Original full-FOV coordinates. Subsets are never
            recentered or rescaled.
        axis_name: ``y`` for the sin projection or ``z`` for the cos projection.
        fit_region: Callable returning an upstream wrapped-plane fit for a
            half-open spatial interval.
        manual_bounds: Optional explicit half-open spatial interval.

    Returns:
        Selected fit result and JSON-compatible selection diagnostics.

    Raises:
        AutomaticSpatialRegionRejected: If the central model is not reliable.
        ValueError: If dimensions, axis name, or manual bounds are invalid.
    """
    if axis_name not in {"y", "z"}:
        raise ValueError("Projection axis must be 'y' or 'z'.")
    difference_shape = tuple(int(value) for value in psf_difference.shape)
    hybrid_shape = tuple(int(value) for value in hybrid_nowave.shape[:3])
    coordinates = np.asarray(spatial_coordinates, dtype=np.float64).reshape(-1)
    spatial_size = difference_shape[1] if axis_name == "y" else difference_shape[2]
    if difference_shape != hybrid_shape or coordinates.size != spatial_size:
        raise ValueError("Projection phase, hybrid data, and coordinates disagree.")

    full_bounds = (0, spatial_size)
    if manual_bounds is not None:
        lower, upper = (int(value) for value in manual_bounds)
        if not (0 <= lower < upper <= spatial_size):
            raise ValueError(
                f"Manual {axis_name} fit bounds must satisfy 0 <= min < max <= "
                f"{spatial_size}."
            )
        minimum_width = max(12, spatial_size // 3)
        if not (lower <= spatial_size // 2 < upper) or upper - lower < minimum_width:
            raise ValueError(
                f"Manual {axis_name} fit bounds must contain the projection center "
                f"and span at least {minimum_width} samples."
            )
        result = fit_region((lower, upper))
        return result, {
            "name": "global-projection-spatial-region",
            "version": SPATIAL_SELECTION_VERSION,
            "selection": "manual",
            "axis": axis_name,
            "full_bounds": list(full_bounds),
            "selected_bounds": [lower, upper],
            "coordinates": "original full-FOV coordinates; not recentered or rescaled",
            "selected_quality": _quality_summary(result),
        }

    full_result = fit_region(full_bounds)
    core = centered_spatial_core_bounds(spatial_size)
    core_result = fit_region(core)
    core_coordinates = coordinates[core[0] : core[1]]
    full_quality = _quality_summary(full_result)
    core_quality = _quality_summary(core_result)
    full_core_difference = _model_difference(
        full_result, core_result, core_coordinates, axis_name
    )
    base = {
        "name": "global-projection-spatial-region",
        "version": SPATIAL_SELECTION_VERSION,
        "selection": "automatic",
        "axis": axis_name,
        "full_bounds": list(full_bounds),
        "central_core_bounds": list(core),
        "coordinates": "original full-FOV coordinates; not recentered or rescaled",
        "full_quality": full_quality,
        "central_core_quality": core_quality,
        "full_vs_central_core_upper_quartile_wrapped_phase_rad": (
            full_core_difference
        ),
        "maximum_full_core_phase_difference_rad": (
            FULL_CORE_MAXIMUM_PHASE_DIFFERENCE_RAD
        ),
        "maximum_median_wrapped_rms_rad": MAXIMUM_MEDIAN_WRAPPED_RMS_RAD,
        "minimum_valid_readout_fraction": MINIMUM_VALID_READOUT_FRACTION,
    }
    full_is_stable = (
        full_quality["valid_readout_fraction"] >= MINIMUM_VALID_READOUT_FRACTION
        and full_quality["median_wrapped_rms_rad"]
        <= MAXIMUM_MEDIAN_WRAPPED_RMS_RAD
        and full_core_difference <= FULL_CORE_MAXIMUM_PHASE_DIFFERENCE_RAD
    )
    if full_is_stable:
        return full_result, {
            **base,
            "outcome": "full_region_preserved",
            "selected_bounds": list(full_bounds),
            "clean_case_no_op": True,
        }

    if (
        core_quality["valid_readout_fraction"] < MINIMUM_VALID_READOUT_FRACTION
        or core_quality["median_wrapped_rms_rad"]
        > MAXIMUM_MEDIAN_WRAPPED_RMS_RAD
    ):
        raise AutomaticSpatialRegionRejected(
            f"Automatic {axis_name}-projection selection found no reliable central region.",
            diagnostics={**base, "outcome": "rejected"},
            plane={"selected_bounds": full_bounds, "full_result": full_result},
        )

    candidate_records = []
    selected: tuple[Mapping[str, Any], tuple[int, int]] | None = None
    for bounds in _candidate_bounds(spatial_size, core):
        if bounds in {full_bounds, core}:
            result = full_result if bounds == full_bounds else core_result
        else:
            result = fit_region(bounds)
        quality = _quality_summary(result)
        difference = _model_difference(result, core_result, core_coordinates, axis_name)
        accepted = (
            quality["valid_readout_fraction"] >= MINIMUM_VALID_READOUT_FRACTION
            and quality["median_wrapped_rms_rad"]
            <= MAXIMUM_MEDIAN_WRAPPED_RMS_RAD
            and difference <= FULL_CORE_MAXIMUM_PHASE_DIFFERENCE_RAD
        )
        candidate_records.append(
            {
                "bounds": list(bounds),
                **quality,
                "vs_central_core_upper_quartile_wrapped_phase_rad": difference,
                "accepted": accepted,
            }
        )
        if accepted and bounds != full_bounds:
            selected = (result, bounds)
            break

    if selected is None:
        raise AutomaticSpatialRegionRejected(
            f"Automatic {axis_name}-projection selection found no stable inner region.",
            diagnostics={**base, "outcome": "rejected", "candidates": candidate_records},
            plane={"selected_bounds": full_bounds, "full_result": full_result},
        )
    result, bounds = selected
    return result, {
        **base,
        "outcome": "inner_region_selected",
        "selected_bounds": list(bounds),
        "clean_case_no_op": False,
        "candidates": candidate_records,
    }
