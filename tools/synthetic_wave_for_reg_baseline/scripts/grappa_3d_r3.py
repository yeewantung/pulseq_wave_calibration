"""Local joint-coil 5×5×Kz GRAPPA primitives for regular R=3 PE1 data.

Arrays use ``[RO, PE1, PE2, coil]`` order. The kernel uses five readout
locations, the two acquired PE1 locations present in a nominal five-line
window, and an odd number of adjacent PE2 locations. All source coils jointly predict
every target coil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.linalg as la


READOUT_OFFSETS = (-2, -1, 0, 1, 2)
SOURCE_PE1_OFFSETS = {1: (-1, 2), 2: (-2, 1)}


def pe2_offsets(kernel_size: int) -> tuple[int, ...]:
    """Return centered PE2 offsets for a positive odd kernel extent."""
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("PE2 kernel size must be a positive odd integer.")
    half = kernel_size // 2
    return tuple(range(-half, half + 1))


@dataclass
class NormalEquations3D:
    """Store pooled complex128 normal equations for both missing residues."""

    shs: dict[int, np.ndarray]
    sht: dict[int, np.ndarray]
    rows: dict[int, int]
    pe2_kernel_size: int

    @classmethod
    def zeros(cls, ncoil: int, pe2_kernel_size: int = 3) -> "NormalEquations3D":
        """Allocate zero equations for a specified joint source-coil count."""
        offsets = pe2_offsets(pe2_kernel_size)
        nfeatures = len(READOUT_OFFSETS) * 2 * len(offsets) * ncoil
        return cls(
            {key: np.zeros((nfeatures, nfeatures), np.complex128) for key in (1, 2)},
            {key: np.zeros((nfeatures, ncoil), np.complex128) for key in (1, 2)},
            {1: 0, 2: 0},
            len(offsets),
        )

    def add(self, other: "NormalEquations3D") -> None:
        """Add independently accumulated target rows in place."""
        if self.pe2_kernel_size != other.pe2_kernel_size:
            raise ValueError("Cannot add normal equations with different PE2 kernels.")
        for offset in (1, 2):
            self.shs[offset] += other.shs[offset]
            self.sht[offset] += other.sht[offset]
            self.rows[offset] += other.rows[offset]


def calibration_matrices_3d(
    calibration_block: np.ndarray,
    target_partitions: Sequence[int],
    target_offset: int,
    *,
    pe2_kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build source/target matrices for core partitions inside a halo block."""
    calibration = np.asarray(calibration_block, dtype=np.complex64)
    core = np.asarray(target_partitions, dtype=int)
    if calibration.ndim != 4:
        raise ValueError("Calibration must use [RO, PE1, PE2, coil] layout.")
    if target_offset not in SOURCE_PE1_OFFSETS:
        raise ValueError(f"Unsupported target offset: {target_offset}.")
    if core.ndim != 1 or np.any(core < 0) or np.any(core >= calibration.shape[2]):
        raise ValueError("Core PE2 indices are outside the calibration block.")

    offsets = pe2_offsets(pe2_kernel_size)
    halo = pe2_kernel_size // 2
    nro, npe1, _, ncoil = calibration.shape
    padded = np.pad(calibration, ((2, 2), (2, 2), (halo, halo), (0, 0)))
    sources = []
    # Source order is spatial location first and coil last, matching the 2D
    # implementation and pygrappa's flattened boolean kernel convention.
    for dx in READOUT_OFFSETS:
        for dy in SOURCE_PE1_OFFSETS[target_offset]:
            for dz in offsets:
                source = padded[
                    2 + dx : 2 + dx + nro,
                    2 + dy : 2 + dy + npe1,
                    :,
                    :,
                ]
                sources.append(source[:, :, core + halo + dz, :])
    source_matrix = np.stack(sources, axis=-2).reshape(-1, len(sources) * ncoil)
    target_matrix = calibration[:, :, core, :].reshape(-1, ncoil)
    return source_matrix, target_matrix


def accumulate_normal_equations_3d(
    calibration_block: np.ndarray,
    target_partitions: Sequence[int],
    *,
    pe2_kernel_size: int = 3,
) -> NormalEquations3D:
    """Accumulate 5×5×Kz equations for selected core partitions of one block."""
    calibration = np.asarray(calibration_block, dtype=np.complex64)
    result = NormalEquations3D.zeros(calibration.shape[-1], pe2_kernel_size)
    for offset in (1, 2):
        source, target = calibration_matrices_3d(
            calibration,
            target_partitions,
            offset,
            pe2_kernel_size=pe2_kernel_size,
        )
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("3D GRAPPA calibration contains non-finite samples.")
        result.shs[offset] += source.conj().T @ source
        result.sht[offset] += source.conj().T @ target
        result.rows[offset] += int(source.shape[0])
    return result


def solve_weights_3d(
    equations: NormalEquations3D,
    *,
    regularization: float = 0.01,
) -> dict[int, np.ndarray]:
    """Solve pygrappa-scaled Tikhonov systems for both missing residues."""
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative.")
    weights = {}
    for offset in (1, 2):
        shs = equations.shs[offset]
        lamda0 = regularization * np.linalg.norm(shs) / shs.shape[0]
        system = shs + lamda0 * np.eye(shs.shape[0], dtype=shs.dtype)
        weights[offset] = la.solve(
            system, equations.sht[offset], assume_a="her"
        ).astype(np.complex64, copy=False)
    return weights


def source_matrix_for_targets_3d(
    undersampled_block: np.ndarray,
    target_lines: Sequence[int],
    target_partitions: Sequence[int],
    target_offset: int,
    *,
    pe2_kernel_size: int = 3,
) -> np.ndarray:
    """Collect 5×5×Kz sources for PE1 targets in core PE2 partitions."""
    undersampled = np.asarray(undersampled_block, dtype=np.complex64)
    targets = np.asarray(target_lines, dtype=int)
    core = np.asarray(target_partitions, dtype=int)
    if undersampled.ndim != 4:
        raise ValueError("Input must use [RO, PE1, PE2, coil] layout.")
    if target_offset not in SOURCE_PE1_OFFSETS:
        raise ValueError(f"Unsupported target offset: {target_offset}.")
    if np.any(targets < 0) or np.any(targets >= undersampled.shape[1]):
        raise ValueError("Target PE1 indices are outside the input block.")
    if np.any(core < 0) or np.any(core >= undersampled.shape[2]):
        raise ValueError("Target PE2 indices are outside the input block.")

    offsets = pe2_offsets(pe2_kernel_size)
    halo = pe2_kernel_size // 2
    nro, _, _, ncoil = undersampled.shape
    padded = np.pad(undersampled, ((2, 2), (2, 2), (halo, halo), (0, 0)))
    sources = []
    for dx in READOUT_OFFSETS:
        for dy in SOURCE_PE1_OFFSETS[target_offset]:
            for dz in offsets:
                source = padded[2 + dx : 2 + dx + nro, targets + 2 + dy, :, :]
                sources.append(source[:, :, core + halo + dz, :])
    return np.stack(sources, axis=-2).reshape(
        nro * targets.size * core.size, -1
    )


def apply_grappa_3d_block(
    undersampled_block: np.ndarray,
    target_partitions: Sequence[int],
    acquired_mask: Sequence[bool],
    weights: Mapping[int, np.ndarray],
    *,
    acquired_residue: int = 1,
    pe2_kernel_size: int = 3,
) -> np.ndarray:
    """Reconstruct core PE2 partitions using the configured centered kz halo."""
    undersampled = np.asarray(undersampled_block, dtype=np.complex64)
    core = np.asarray(target_partitions, dtype=int)
    mask = np.asarray(acquired_mask, dtype=bool)
    if undersampled.ndim != 4 or mask.shape != (undersampled.shape[1],):
        raise ValueError("Input block and acquired mask dimensions are inconsistent.")

    result = undersampled[:, :, core, :].copy()
    missing = np.flatnonzero(~mask)
    for offset in (1, 2):
        residue = (acquired_residue + offset) % 3
        targets = missing[missing % 3 == residue]
        source = source_matrix_for_targets_3d(
            undersampled,
            targets,
            core,
            offset,
            pe2_kernel_size=pe2_kernel_size,
        )
        predicted = source @ np.asarray(weights[offset], dtype=np.complex64)
        result[:, targets, :, :] = predicted.reshape(
            undersampled.shape[0], targets.size, core.size, undersampled.shape[-1]
        )
    measured_core = undersampled[:, mask, :, :][:, :, core, :]
    result[:, mask, :, :] = measured_core
    return result
