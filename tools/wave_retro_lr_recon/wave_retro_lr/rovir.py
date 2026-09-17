"""Validated ROVir integration primitives for measured Wave reconstruction.

The numerical ROVir eigensolver is BART v1.0's ``bart rovir`` command. This
module prepares and validates its inputs and outputs, constructs explicit BART
commands, and computes review diagnostics without launching external processes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .bart_io import bart_base


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize one public array-axis argument without NumPy private APIs.

    Args:
        axis: Possibly negative axis index.
        ndim: Number of dimensions in the target array.

    Returns:
        Nonnegative axis index.

    Raises:
        ValueError: If the axis is outside the target array dimensions.
    """
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Invalid axis {axis} for an array with {ndim} dimensions.")
    return normalized


def _real_mask(mask: np.ndarray, name: str, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Return one validated floating-point spatial mask.

    Args:
        mask: Candidate real-valued region weights.
        name: Human-readable mask name used in validation errors.
        spatial_shape: Exact required spatial dimensions.

    Returns:
        A float64 view or copy with the requested shape.

    Raises:
        ValueError: If the mask is complex, nonfinite, negative, empty, or has
            incompatible dimensions.
    """
    values = np.asarray(mask)
    if values.shape != spatial_shape:
        raise ValueError(f"{name} mask shape {values.shape} does not match {spatial_shape}.")
    if np.iscomplexobj(values):
        raise ValueError(f"{name} mask must be real-valued.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} mask contains nonfinite values.")
    if np.any(values < 0):
        raise ValueError(f"{name} mask contains negative weights.")
    if not np.any(values > 0):
        raise ValueError(f"{name} mask has no positive support.")
    return values


def validate_region_masks(
    signal_mask: np.ndarray,
    interference_mask: np.ndarray,
    spatial_shape: Sequence[int],
) -> dict[str, object]:
    """Validate the disjoint weighted regions used to estimate ROVir coils.

    Args:
        signal_mask: Nonnegative desired-signal-region weights.
        interference_mask: Nonnegative shoulder or nuisance-region weights.
        spatial_shape: Exact spatial dimensions shared by both masks.

    Returns:
        JSON-compatible support and weight diagnostics.

    Raises:
        ValueError: If dimensions or values are invalid, either support is
            empty, or the positive supports overlap.
    """
    shape = tuple(int(value) for value in spatial_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError(f"Spatial mask dimensions must be positive: {shape}.")
    signal = _real_mask(signal_mask, "Signal", shape)
    interference = _real_mask(interference_mask, "Interference", shape)
    signal_support = signal > 0
    interference_support = interference > 0
    overlap = signal_support & interference_support
    if np.any(overlap):
        raise ValueError(
            "Signal and interference masks must have disjoint positive support; "
            f"found {int(np.count_nonzero(overlap))} overlapping voxels."
        )
    covered = signal_support | interference_support
    return {
        "spatial_shape": list(shape),
        "signal_support_voxels": int(np.count_nonzero(signal_support)),
        "interference_support_voxels": int(np.count_nonzero(interference_support)),
        "unassigned_voxels": int(covered.size - np.count_nonzero(covered)),
        "signal_maximum_weight": float(np.max(signal)),
        "interference_maximum_weight": float(np.max(interference)),
    }


def masked_coil_images(
    coil_images: np.ndarray,
    signal_mask: np.ndarray,
    interference_mask: np.ndarray,
    *,
    coil_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply reviewed ROVir region weights to complex physical-coil images.

    Args:
        coil_images: Finite floating-point or complex images with one
            physical-coil axis.
        signal_mask: Nonnegative signal-region weights over spatial dimensions.
        interference_mask: Nonnegative nuisance-region weights over spatial
            dimensions.
        coil_axis: Axis containing physical receive coils.

    Returns:
        Signal-weighted and interference-weighted images in the original axis
        order and data type.

    Raises:
        ValueError: If the image geometry, values, coil axis, or masks are
            invalid.
    """
    images = np.asarray(coil_images)
    if images.ndim < 2:
        raise ValueError("ROVir coil images require spatial and coil dimensions.")
    if not (
        np.issubdtype(images.dtype, np.floating)
        or np.issubdtype(images.dtype, np.complexfloating)
    ):
        raise ValueError("ROVir coil images must be floating-point or complex.")
    normalized_axis = _normalize_axis(coil_axis, images.ndim)
    if images.shape[normalized_axis] < 2:
        raise ValueError("ROVir requires at least two physical coils.")
    if not np.isfinite(images).all():
        raise ValueError("Physical-coil images contain nonfinite values.")
    coil_last = np.moveaxis(images, normalized_axis, -1)
    spatial_shape = coil_last.shape[:-1]
    validate_region_masks(signal_mask, interference_mask, spatial_shape)
    signal = np.asarray(signal_mask, dtype=images.real.dtype)
    interference = np.asarray(interference_mask, dtype=images.real.dtype)
    positive = coil_last * signal[..., np.newaxis]
    negative = coil_last * interference[..., np.newaxis]
    return (
        np.moveaxis(positive, -1, normalized_axis),
        np.moveaxis(negative, -1, normalized_axis),
    )


def region_correlation_diagnostics(
    coil_images: np.ndarray,
    signal_mask: np.ndarray,
    interference_mask: np.ndarray,
    *,
    coil_axis: int = -1,
    voxel_chunk: int = 65536,
    interference_relative_eigenvalue_floor: float = 1e-8,
) -> dict[str, object]:
    """Validate the correlation matrices consumed by the BART eigensolver.

    Args:
        coil_images: Finite physical-coil calibration images.
        signal_mask: Reviewed signal-region weights.
        interference_mask: Reviewed nuisance-region weights.
        coil_axis: Axis containing physical receive coils.
        voxel_chunk: Maximum spatial samples accumulated per iteration.
        interference_relative_eigenvalue_floor: Smallest permitted eigenvalue
            of the interference matrix relative to its largest eigenvalue.

    Returns:
        JSON-compatible ranks, eigenvalue ranges, and condition diagnostics.

    Raises:
        ValueError: If inputs are invalid or the interference correlation
            matrix is too singular for an unregularized generalized solve.
    """
    images = np.asarray(coil_images)
    if images.ndim < 2 or not np.isfinite(images).all():
        raise ValueError("ROVir correlations require finite multicoil images.")
    normalized_axis = _normalize_axis(coil_axis, images.ndim)
    coil_last = np.moveaxis(images, normalized_axis, -1)
    if coil_last.shape[-1] < 2:
        raise ValueError("ROVir requires at least two physical coils.")
    spatial_shape = coil_last.shape[:-1]
    validate_region_masks(signal_mask, interference_mask, spatial_shape)
    if voxel_chunk < 1:
        raise ValueError("Voxel chunk must be positive.")
    if (
        not np.isfinite(interference_relative_eigenvalue_floor)
        or interference_relative_eigenvalue_floor <= 0
        or interference_relative_eigenvalue_floor >= 1
    ):
        raise ValueError("Interference relative eigenvalue floor must lie in (0, 1).")

    flattened = coil_last.reshape(-1, coil_last.shape[-1])
    signal_weights = np.asarray(signal_mask, dtype=np.float64).reshape(-1)
    interference_weights = np.asarray(interference_mask, dtype=np.float64).reshape(-1)
    coils = coil_last.shape[-1]
    signal_correlation = np.zeros((coils, coils), dtype=np.complex128)
    interference_correlation = np.zeros((coils, coils), dtype=np.complex128)
    for start in range(0, flattened.shape[0], voxel_chunk):
        stop = min(start + voxel_chunk, flattened.shape[0])
        block = flattened[start:stop].astype(np.complex128, copy=False)
        signal_block = block * signal_weights[start:stop, np.newaxis]
        interference_block = block * interference_weights[start:stop, np.newaxis]
        signal_correlation += signal_block.conj().T @ signal_block
        interference_correlation += (
            interference_block.conj().T @ interference_block
        )
    signal_eigenvalues = np.maximum(
        np.linalg.eigvalsh(signal_correlation).real, 0.0
    )
    interference_eigenvalues = np.maximum(
        np.linalg.eigvalsh(interference_correlation).real, 0.0
    )
    interference_maximum = float(interference_eigenvalues[-1])
    if interference_maximum <= 0:
        raise ValueError("Interference correlation matrix has no positive energy.")
    interference_relative_minimum = float(
        interference_eigenvalues[0] / interference_maximum
    )
    if interference_relative_minimum < interference_relative_eigenvalue_floor:
        raise ValueError(
            "Interference correlation matrix is too singular for the reviewed "
            "unregularized BART ROVir solve: relative minimum eigenvalue "
            f"{interference_relative_minimum:g} is below "
            f"{interference_relative_eigenvalue_floor:g}."
        )
    signal_maximum = float(signal_eigenvalues[-1])
    signal_relative = (
        signal_eigenvalues / signal_maximum
        if signal_maximum > 0
        else np.zeros_like(signal_eigenvalues)
    )
    interference_relative = interference_eigenvalues / interference_maximum
    return {
        "physical_coils": int(coils),
        "signal_rank_at_floor": int(
            np.count_nonzero(signal_relative >= interference_relative_eigenvalue_floor)
        ),
        "interference_rank_at_floor": int(
            np.count_nonzero(
                interference_relative >= interference_relative_eigenvalue_floor
            )
        ),
        "signal_minimum_eigenvalue": float(signal_eigenvalues[0]),
        "signal_maximum_eigenvalue": signal_maximum,
        "interference_minimum_eigenvalue": float(interference_eigenvalues[0]),
        "interference_maximum_eigenvalue": interference_maximum,
        "interference_relative_minimum_eigenvalue": (
            interference_relative_minimum
        ),
        "interference_condition_number": float(
            interference_maximum / interference_eigenvalues[0]
        ),
        "relative_eigenvalue_floor": float(
            interference_relative_eigenvalue_floor
        ),
    }


def rovir_matrix_view(transform: np.ndarray) -> np.ndarray:
    """Extract a square matrix from a NumPy or BART-shaped ROVir transform.

    Args:
        transform: Either a two-dimensional matrix or a BART array whose only
            nonsingleton dimensions are coil dimension 3 and maps dimension 4.

    Returns:
        A two-dimensional complex matrix with physical coils in rows and
        ordered virtual coils in columns.

    Raises:
        ValueError: If the transform is not finite, square, or compatible with
            BART coil/maps dimensions.
    """
    values = np.asarray(transform)
    if values.ndim == 2:
        matrix = values
    else:
        if values.ndim < 5:
            raise ValueError(
                "A BART ROVir transform requires coil and maps dimensions 3 and 4."
            )
        unexpected = [
            axis
            for axis, size in enumerate(values.shape)
            if axis not in (3, 4) and size != 1
        ]
        if unexpected:
            raise ValueError(
                "Unexpected nonsingleton BART transform dimensions: "
                f"{unexpected} in {values.shape}."
            )
        matrix = np.reshape(values, (values.shape[3], values.shape[4]), order="F")
    if matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ROVir transform must be square; found {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("ROVir transform contains nonfinite values.")
    return matrix


def validate_rovir_transform(
    transform: np.ndarray,
    *,
    orthogonality_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Validate the full orthonormal transform emitted by ``bart rovir``.

    Args:
        transform: NumPy or BART-shaped full ROVir transform.
        orthogonality_tolerance: Maximum allowed absolute Gram-matrix residual.

    Returns:
        JSON-compatible matrix dimensions and orthogonality diagnostics.

    Raises:
        ValueError: If the tolerance is invalid or the matrix is not a finite
            square orthonormal transform.
    """
    if not np.isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0:
        raise ValueError("Orthogonality tolerance must be finite and positive.")
    matrix = rovir_matrix_view(transform)
    identity = np.eye(matrix.shape[1], dtype=matrix.dtype)
    residual = matrix.conj().T @ matrix - identity
    maximum = float(np.max(np.abs(residual)))
    relative_frobenius = float(
        np.linalg.norm(residual) / np.linalg.norm(identity)
    )
    if maximum > orthogonality_tolerance:
        raise ValueError(
            "ROVir transform is not orthonormal: maximum Gram residual "
            f"{maximum:g} exceeds {orthogonality_tolerance:g}."
        )
    return {
        "physical_coils": int(matrix.shape[0]),
        "virtual_coils_available": int(matrix.shape[1]),
        "maximum_gram_residual": maximum,
        "relative_frobenius_gram_residual": relative_frobenius,
        "orthogonality_tolerance": float(orthogonality_tolerance),
    }


def region_energy_curve(
    coil_images: np.ndarray,
    transform: np.ndarray,
    signal_mask: np.ndarray,
    interference_mask: np.ndarray,
    channel_counts: Sequence[int],
    *,
    coil_axis: int = -1,
    voxel_chunk: int = 65536,
) -> dict[str, object]:
    """Measure cumulative signal retention and interference after projection.

    Args:
        coil_images: Finite physical-coil calibration images.
        transform: Full ordered orthonormal ROVir transform.
        signal_mask: Reviewed signal-region weights.
        interference_mask: Reviewed nuisance-region weights.
        channel_counts: Output dimensions to report without selecting a winner.
        coil_axis: Axis containing physical coils.
        voxel_chunk: Maximum spatial samples projected in one matrix multiply.

    Returns:
        JSON-compatible per-channel energies and cumulative metrics for every
        requested output dimension.

    Raises:
        ValueError: If inputs are invalid, dimensions disagree, or a requested
            channel count is outside the available transform.
    """
    images = np.asarray(coil_images)
    if images.ndim < 2 or not np.isfinite(images).all():
        raise ValueError("ROVir diagnostics require finite multicoil images.")
    normalized_axis = _normalize_axis(coil_axis, images.ndim)
    matrix = rovir_matrix_view(transform)
    coil_last = np.moveaxis(images, normalized_axis, -1)
    if coil_last.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"Image coil count {coil_last.shape[-1]} does not match transform "
            f"{matrix.shape[0]}."
        )
    spatial_shape = coil_last.shape[:-1]
    validate_region_masks(signal_mask, interference_mask, spatial_shape)
    counts = tuple(int(value) for value in channel_counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Channel counts must be a nonempty unique sequence.")
    if any(value < 1 or value > matrix.shape[1] for value in counts):
        raise ValueError(
            f"Channel counts must lie in [1, {matrix.shape[1]}]: {counts}."
        )
    if voxel_chunk < 1:
        raise ValueError("Voxel chunk must be positive.")

    flattened = coil_last.reshape(-1, coil_last.shape[-1])
    signal_weights = np.asarray(signal_mask, dtype=np.float64).reshape(-1)
    interference_weights = np.asarray(interference_mask, dtype=np.float64).reshape(-1)
    signal_by_virtual = np.zeros(matrix.shape[1], dtype=np.float64)
    interference_by_virtual = np.zeros(matrix.shape[1], dtype=np.float64)
    original_signal = 0.0
    original_interference = 0.0
    for start in range(0, flattened.shape[0], voxel_chunk):
        stop = min(start + voxel_chunk, flattened.shape[0])
        block = flattened[start:stop]
        # BART ccapply's forward mode conjugates the stored compression matrix.
        projected = block @ matrix.conj()
        signal = signal_weights[start:stop, np.newaxis]
        interference = interference_weights[start:stop, np.newaxis]
        original_signal += float(np.sum(np.abs(block * signal) ** 2))
        original_interference += float(np.sum(np.abs(block * interference) ** 2))
        signal_by_virtual += np.sum(np.abs(projected * signal) ** 2, axis=0)
        interference_by_virtual += np.sum(
            np.abs(projected * interference) ** 2, axis=0
        )
    if original_signal <= 0 or original_interference <= 0:
        raise ValueError("Region-weighted physical-coil energies must be positive.")

    cumulative_signal = np.cumsum(signal_by_virtual)
    cumulative_interference = np.cumsum(interference_by_virtual)
    entries = []
    for count in counts:
        signal_fraction = float(cumulative_signal[count - 1] / original_signal)
        interference_fraction = float(
            cumulative_interference[count - 1] / original_interference
        )
        entries.append(
            {
                "virtual_coils": count,
                "signal_retention_fraction": signal_fraction,
                "interference_remaining_fraction": interference_fraction,
                "relative_signal_to_interference": (
                    signal_fraction / interference_fraction
                    if interference_fraction > 0
                    else None
                ),
            }
        )
    return {
        "physical_coils": int(matrix.shape[0]),
        "original_signal_energy": original_signal,
        "original_interference_energy": original_interference,
        "signal_energy_by_ordered_virtual_coil": signal_by_virtual.tolist(),
        "interference_energy_by_ordered_virtual_coil": (
            interference_by_virtual.tolist()
        ),
        "channel_counts": entries,
    }


def bart_rovir_command(
    positive_images: str | Path,
    negative_images: str | Path,
    transform_output: str | Path,
    *,
    bart_executable: str = "bart",
) -> tuple[str, ...]:
    """Construct the explicit BART command that estimates a ROVir transform.

    Args:
        positive_images: BART basename of signal-masked physical-coil images.
        negative_images: BART basename of interference-masked coil images.
        transform_output: Destination basename for the full ordered transform.
        bart_executable: Runtime-resolved BART executable or command name.

    Returns:
        Argument tuple suitable for display, recording, or reviewed execution.
    """
    return (
        bart_executable,
        "rovir",
        os.fspath(bart_base(positive_images)),
        os.fspath(bart_base(negative_images)),
        os.fspath(bart_base(transform_output)),
    )


def bart_ccapply_command(
    coil_data: str | Path,
    transform: str | Path,
    output: str | Path,
    virtual_coils: int,
    *,
    bart_executable: str = "bart",
) -> tuple[str, ...]:
    """Construct one fixed-transform BART coil-compression command.

    Args:
        coil_data: Physical-coil image or k-space BART basename.
        transform: Full transform produced by ``bart rovir``.
        output: Destination basename for projected virtual-coil data.
        virtual_coils: Number of leading ordered ROVir channels to retain.
        bart_executable: Runtime-resolved BART executable or command name.

    Returns:
        Argument tuple suitable for display, recording, or reviewed execution.

    Raises:
        ValueError: If the requested output channel count is not positive.
    """
    if virtual_coils < 1:
        raise ValueError("Virtual-coil count must be positive.")
    return (
        bart_executable,
        "ccapply",
        "-p",
        str(int(virtual_coils)),
        os.fspath(bart_base(coil_data)),
        os.fspath(bart_base(transform)),
        os.fspath(bart_base(output)),
    )


def build_rovir_command_plan(
    positive_images: str | Path,
    negative_images: str | Path,
    transform_output: str | Path,
    image_kspace: str | Path,
    calibration_kspace: str | Path,
    projected_image_kspace: str | Path,
    projected_calibration_kspace: str | Path,
    virtual_coils: int,
    *,
    bart_executable: str = "bart",
) -> dict[str, tuple[str, ...]]:
    """Build a nonexecuting command plan that shares one ROVir transform.

    Args:
        positive_images: Signal-masked physical-coil calibration images.
        negative_images: Interference-masked calibration images.
        transform_output: Destination for the full ordered transform.
        image_kspace: Measured physical-coil Wave image k-space.
        calibration_kspace: Separate integrated set-4 physical-coil ACS.
        projected_image_kspace: Destination for projected image k-space.
        projected_calibration_kspace: Destination for projected ACS.
        virtual_coils: Leading ROVir channels retained in both outputs.
        bart_executable: Runtime-resolved BART executable or command name.

    Returns:
        Named explicit command tuples. This function never launches BART.
    """
    return {
        "estimate_transform": bart_rovir_command(
            positive_images,
            negative_images,
            transform_output,
            bart_executable=bart_executable,
        ),
        "project_image_kspace": bart_ccapply_command(
            image_kspace,
            transform_output,
            projected_image_kspace,
            virtual_coils,
            bart_executable=bart_executable,
        ),
        "project_calibration_kspace": bart_ccapply_command(
            calibration_kspace,
            transform_output,
            projected_calibration_kspace,
            virtual_coils,
            bart_executable=bart_executable,
        ),
    }
