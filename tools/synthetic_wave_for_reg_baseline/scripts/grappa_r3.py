"""Explicit shared-weight 2D R=3 GRAPPA calibration and application.

The geometry follows ``pygrappa.grappa(..., kernel_size=(5, 5))`` for a regular
R=3 mask. A five-point nominal PE1 window contains two acquired PE1 lines for
each missing offset. Readout offsets are ``[-2, -1, 0, 1, 2]``. The two source
PE1 offset pairs relative to the target are ``[-1, 2]`` and ``[-2, 1]``.

Arrays use canonical coil-last layout ``[RO, PE1, PE2, coil]``. Calibration
normal equations can be accumulated over PE2 chunks, allowing one pair of
weights to be trained from all partitions without retaining the design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg as la


READOUT_OFFSETS = (-2, -1, 0, 1, 2)
SOURCE_PE1_OFFSETS = {
    1: (-1, 2),
    2: (-2, 1),
}


@dataclass
class NormalEquations:
    """Pooled GRAPPA normal equations for both R=3 target offsets."""

    shs: dict[int, np.ndarray]
    sht: dict[int, np.ndarray]
    rows: dict[int, int]

    @classmethod
    def zeros(cls, ncoil: int) -> "NormalEquations":
        nfeatures = len(READOUT_OFFSETS) * 2 * ncoil
        return cls(
            shs={
                offset: np.zeros((nfeatures, nfeatures), dtype=np.complex128)
                for offset in SOURCE_PE1_OFFSETS
            },
            sht={
                offset: np.zeros((nfeatures, ncoil), dtype=np.complex128)
                for offset in SOURCE_PE1_OFFSETS
            },
            rows={offset: 0 for offset in SOURCE_PE1_OFFSETS},
        )

    def add(self, other: "NormalEquations") -> None:
        for offset in SOURCE_PE1_OFFSETS:
            self.shs[offset] += other.shs[offset]
            self.sht[offset] += other.sht[offset]
            self.rows[offset] += other.rows[offset]

    def subtract(self, other: "NormalEquations") -> "NormalEquations":
        result = NormalEquations.zeros(self.sht[1].shape[1])
        for offset in SOURCE_PE1_OFFSETS:
            result.shs[offset] = self.shs[offset] - other.shs[offset]
            result.sht[offset] = self.sht[offset] - other.sht[offset]
            result.rows[offset] = self.rows[offset] - other.rows[offset]
        return result


def calibration_matrices(
    calibration: np.ndarray,
    target_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pygrappa-compatible source and target matrices for one geometry.

    All original RO/PE1 target positions contribute calibration examples.
    Zero padding reproduces pygrappa's padded ACS windows. PE2 partitions are
    independent examples and never enter the kernel spatial support.
    """
    calibration = np.asarray(calibration, dtype=np.complex64)
    if calibration.ndim != 4:
        raise ValueError(
            f"Calibration must have [RO, PE1, PE2, coil] layout, got {calibration.shape}."
        )
    if target_offset not in SOURCE_PE1_OFFSETS:
        raise ValueError(f"Unsupported R=3 target offset: {target_offset}.")

    nro, npe1, _, ncoil = calibration.shape
    padded = np.pad(calibration, ((2, 2), (2, 2), (0, 0), (0, 0)))
    sources = []
    for dx in READOUT_OFFSETS:
        x_slice = slice(2 + dx, 2 + dx + nro)
        for dy in SOURCE_PE1_OFFSETS[target_offset]:
            y_slice = slice(2 + dy, 2 + dy + npe1)
            sources.append(padded[x_slice, y_slice, :, :])

    # Stacking source locations before the coil axis matches the C-order
    # flattening used by pygrappa's boolean patch mask.
    source_matrix = np.stack(sources, axis=-2).reshape(-1, len(sources) * ncoil)
    target_matrix = calibration.reshape(-1, ncoil)
    return source_matrix, target_matrix


def accumulate_normal_equations(calibration: np.ndarray) -> NormalEquations:
    """Accumulate complex128 normal equations from one calibration chunk."""
    calibration = np.asarray(calibration, dtype=np.complex64)
    ncoil = calibration.shape[-1]
    result = NormalEquations.zeros(ncoil)
    for offset in SOURCE_PE1_OFFSETS:
        source, target = calibration_matrices(calibration, offset)
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("GRAPPA calibration contains non-finite samples.")
        # BLAS operates on compact complex64 chunks; pooled sums are retained
        # in complex128 to reduce loss across many PE2 chunks.
        result.shs[offset] += source.conj().T @ source
        result.sht[offset] += source.conj().T @ target
        result.rows[offset] += int(source.shape[0])
    return result


def feature_indices(max_ncc: int, ncc: int) -> np.ndarray:
    """Select leading virtual coils from every source location."""
    if not 1 <= ncc <= max_ncc:
        raise ValueError(f"ncc must be between 1 and max_ncc={max_ncc}.")
    nsource = len(READOUT_OFFSETS) * 2
    return np.concatenate(
        [np.arange(source * max_ncc, source * max_ncc + ncc) for source in range(nsource)]
    )


def solve_weights(
    equations: NormalEquations,
    *,
    max_ncc: int,
    ncc: int,
    regularization: float = 0.01,
) -> dict[int, np.ndarray]:
    """Solve pygrappa-style Tikhonov systems for a requested nested basis."""
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative.")
    selected = feature_indices(max_ncc, ncc)
    weights: dict[int, np.ndarray] = {}
    for offset in SOURCE_PE1_OFFSETS:
        shs = equations.shs[offset][np.ix_(selected, selected)]
        sht = equations.sht[offset][selected, :ncc]
        lamda0 = regularization * np.linalg.norm(shs) / shs.shape[0]
        system = shs + lamda0 * np.eye(shs.shape[0], dtype=shs.dtype)
        weights[offset] = la.solve(system, sht, assume_a="her").astype(
            np.complex64, copy=False
        )
    return weights


def source_matrix_for_targets(
    undersampled: np.ndarray,
    target_lines: Sequence[int],
    target_offset: int,
) -> np.ndarray:
    """Collect fixed-geometry sources for selected PE1 targets in one plane."""
    undersampled = np.asarray(undersampled, dtype=np.complex64)
    if undersampled.ndim != 3:
        raise ValueError(
            f"Undersampled plane must be [RO, PE1, coil], got {undersampled.shape}."
        )
    if target_offset not in SOURCE_PE1_OFFSETS:
        raise ValueError(f"Unsupported R=3 target offset: {target_offset}.")
    targets = np.asarray(target_lines, dtype=int)
    if targets.ndim != 1 or np.any(targets < 0) or np.any(targets >= undersampled.shape[1]):
        raise ValueError("Target PE1 indices are out of range.")

    nro, npe1, ncoil = undersampled.shape
    padded = np.pad(undersampled, ((2, 2), (2, 2), (0, 0)))
    sources = []
    for dx in READOUT_OFFSETS:
        x_slice = slice(2 + dx, 2 + dx + nro)
        for dy in SOURCE_PE1_OFFSETS[target_offset]:
            sources.append(padded[x_slice, targets + 2 + dy, :])
    return np.stack(sources, axis=-2).reshape(nro * len(targets), -1)


def source_matrix_for_targets_volume(
    undersampled: np.ndarray,
    target_lines: Sequence[int],
    target_offset: int,
) -> np.ndarray:
    """Collect sources from a ``[RO, PE1, PE2, coil]`` partition batch."""
    undersampled = np.asarray(undersampled, dtype=np.complex64)
    if undersampled.ndim != 4:
        raise ValueError(
            f"Undersampled volume must be [RO, PE1, PE2, coil], got {undersampled.shape}."
        )
    if target_offset not in SOURCE_PE1_OFFSETS:
        raise ValueError(f"Unsupported R=3 target offset: {target_offset}.")
    targets = np.asarray(target_lines, dtype=int)
    if targets.ndim != 1 or np.any(targets < 0) or np.any(targets >= undersampled.shape[1]):
        raise ValueError("Target PE1 indices are out of range.")

    nro, npe1, npe2, ncoil = undersampled.shape
    padded = np.pad(undersampled, ((2, 2), (2, 2), (0, 0), (0, 0)))
    sources = []
    for dx in READOUT_OFFSETS:
        x_slice = slice(2 + dx, 2 + dx + nro)
        for dy in SOURCE_PE1_OFFSETS[target_offset]:
            sources.append(padded[x_slice, targets + 2 + dy, :, :])
    return np.stack(sources, axis=-2).reshape(nro * len(targets) * npe2, -1)


def apply_grappa_plane(
    undersampled: np.ndarray,
    acquired_mask: Sequence[bool],
    weights: Mapping[int, np.ndarray],
    *,
    acceleration: int = 3,
    acquired_residue: int = 1,
) -> np.ndarray:
    """Fill only missing PE1 lines in one ``[RO, PE1, coil]`` plane."""
    if acceleration != 3:
        raise ValueError("This explicit implementation supports acceleration=3 only.")
    undersampled = np.asarray(undersampled, dtype=np.complex64)
    mask = np.asarray(acquired_mask, dtype=bool)
    if undersampled.ndim != 3 or mask.shape != (undersampled.shape[1],):
        raise ValueError("Plane and acquired mask dimensions are inconsistent.")

    result = undersampled.copy()
    missing = np.flatnonzero(~mask)
    for offset in (1, 2):
        residue = (acquired_residue + offset) % acceleration
        targets = missing[missing % acceleration == residue]
        if targets.size == 0:
            continue
        source = source_matrix_for_targets(undersampled, targets, offset)
        predicted = source @ np.asarray(weights[offset], dtype=np.complex64)
        result[:, targets, :] = predicted.reshape(
            undersampled.shape[0], targets.size, undersampled.shape[2]
        )

    # Acquisition indices, not nonzero magnitude, are authoritative. Copying
    # them again makes the preservation invariant explicit and testable.
    result[:, mask, :] = undersampled[:, mask, :]
    return result


def apply_grappa_volume(
    undersampled: np.ndarray,
    acquired_mask: Sequence[bool],
    weights: Mapping[int, np.ndarray],
    *,
    acceleration: int = 3,
    acquired_residue: int = 1,
) -> np.ndarray:
    """Fill a PE2 batch using one matrix multiplication per target offset."""
    if acceleration != 3:
        raise ValueError("This explicit implementation supports acceleration=3 only.")
    undersampled = np.asarray(undersampled, dtype=np.complex64)
    mask = np.asarray(acquired_mask, dtype=bool)
    if undersampled.ndim != 4 or mask.shape != (undersampled.shape[1],):
        raise ValueError("Volume and acquired mask dimensions are inconsistent.")

    result = undersampled.copy()
    missing = np.flatnonzero(~mask)
    for offset in (1, 2):
        residue = (acquired_residue + offset) % acceleration
        targets = missing[missing % acceleration == residue]
        if targets.size == 0:
            continue
        source = source_matrix_for_targets_volume(undersampled, targets, offset)
        predicted = source @ np.asarray(weights[offset], dtype=np.complex64)
        result[:, targets, :, :] = predicted.reshape(
            undersampled.shape[0], targets.size, undersampled.shape[2], undersampled.shape[3]
        )

    result[:, mask, :, :] = undersampled[:, mask, :, :]
    return result


def nrmse(numerator: float, denominator: float) -> float:
    """Return norm-ratio NRMSE from accumulated squared norms."""
    if denominator <= 0:
        raise ValueError("NRMSE denominator must be positive.")
    return float(np.sqrt(numerator / denominator))
