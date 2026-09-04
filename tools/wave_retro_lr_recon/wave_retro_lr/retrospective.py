"""Measured-Wave and explicitly no-Wave retrospective utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bart_io import create_cfl, open_cfl, read_shape
from .core import ResolvedCase, apply_wave_forward, centered_fftn


def _measured_target_mask(
    source_mask: np.ndarray,
    case: ResolvedCase,
    source_acceleration_lin_par: tuple[int, int],
) -> np.ndarray:
    """Preserve measured residues and add acceleration without inferring ACS.

    Args:
        source_mask: Boolean measured image-stream mask in LIN/PAR order.
        case: Resolved crop and target acceleration.
        source_acceleration_lin_par: Known measured LIN/PAR acceleration.

    Returns:
        Cropped target mask containing only measured image samples.
    """

    mask = np.asarray(source_mask, dtype=bool)
    source_lin, source_par = case.source_logical_matrix_ro_lin_par[1:]
    if mask.shape != (source_lin, source_par):
        raise ValueError(
            f"Source mask shape {mask.shape} does not match {(source_lin, source_par)}."
        )
    lin = slice(*case.crop_bounds_lin)
    par = slice(*case.crop_bounds_par)
    cropped = mask[lin, par]
    target_lin, target_par = cropped.shape
    source_lin_acceleration, source_par_acceleration = source_acceleration_lin_par
    target_lin_acceleration, target_par_acceleration = case.acceleration_ry_rz
    for source, target, name in (
        (source_lin_acceleration, target_lin_acceleration, "LIN"),
        (source_par_acceleration, target_par_acceleration, "PAR"),
    ):
        if source > 1 and target != source:
            raise ValueError(
                f"Target {name} acceleration {target} is incompatible with measured {source}."
            )
        if source == 1 and target < 1:
            raise ValueError(f"Target {name} acceleration must be positive.")
    requested = cropped.copy()
    if source_lin_acceleration == 1 and target_lin_acceleration > 1:
        lin_keep = (
            (np.arange(target_lin) - target_lin // 2) % target_lin_acceleration
        ) == 0
        requested &= lin_keep[:, None]
    if source_par_acceleration == 1 and target_par_acceleration > 1:
        par_keep = (
            (np.arange(target_par) - target_par // 2) % target_par_acceleration
        ) == 0
        requested &= par_keep[None, :]
    if not np.any(requested):
        raise ValueError("The requested measured-data sampling mask is empty.")
    return requested


def write_measured_wave_crop(
    source_base: str | Path,
    output_base: str | Path,
    case: ResolvedCase,
    source_mask: np.ndarray,
    source_acceleration_lin_par: tuple[int, int],
) -> dict[str, float | int]:
    """Center-crop measured Wave k-space and apply its target mask.

    Args:
        source_base: Source measured-Wave BART CFL basename.
        output_base: Destination target-grid BART CFL basename.
        case: Resolved PE crop and target acceleration.
        source_mask: Boolean measured image-stream mask.
        source_acceleration_lin_par: Known source LIN/PAR acceleration.

    Returns:
        K-space norm, sample count, fraction, and image-center status.
    """

    source = open_cfl(source_base)
    source_shape = read_shape(source_base)
    shape = source_shape + (1,) * max(0, 5 - len(source_shape))
    ro_os, source_lin, source_par, coils, maps = shape[:5]
    if maps != 1 or any(value != 1 for value in shape[5:]):
        raise ValueError("Measured Wave k-space must contain one echo/map entry.")
    if (source_lin, source_par) != source_mask.shape:
        raise ValueError("Measured Wave k-space and source mask dimensions disagree.")
    _, target_lin, target_par = case.target_logical_matrix_ro_lin_par
    target_mask = _measured_target_mask(
        source_mask, case, source_acceleration_lin_par
    )
    output = create_cfl(output_base, (ro_os, target_lin, target_par, coils, 1))
    lin = slice(*case.crop_bounds_lin)
    par = slice(*case.crop_bounds_par)
    squared_norm = 0.0
    for coil in range(coils):
        cropped = np.asarray(source[:, lin, par, coil, ...]).squeeze()
        if cropped.shape != (ro_os, target_lin, target_par):
            raise ValueError("Measured Wave crop did not reduce to one 3D coil volume.")
        cropped = np.array(cropped, dtype=np.complex64, copy=True)
        cropped *= target_mask[None, :, :]
        output[:, :, :, coil, 0] = cropped
        squared_norm += float(np.vdot(cropped, cropped).real)
    output.flush()
    del output
    return {
        "wave_kspace_norm": float(np.sqrt(squared_norm)),
        "sampled_coordinate_count": int(np.count_nonzero(target_mask)),
        "sampling_fraction": float(np.mean(target_mask)),
        "image_kspace_center_acquired": bool(
            target_mask[target_lin // 2, target_par // 2]
        ),
    }


def resample_sensitivity_maps(
    source_base: str | Path, output_base: str | Path, *, target_lin_par: tuple[int, int]
) -> None:
    """Fourier-resample CSM fields to a same-FOV target PE grid.

    Args:
        source_base: Native BART ESPIRiT-map CFL basename.
        output_base: Destination target-grid CSM basename.
        target_lin_par: Target logical LIN/PAR dimensions.

    Returns:
        None. A coil-RSS-normalized BART CFL pair is written to ``output_base``.
    """

    source = open_cfl(source_base)
    source_shape = read_shape(source_base)
    shape = source_shape + (1,) * max(0, 5 - len(source_shape))
    ro, source_lin, source_par, coils, maps = shape[:5]
    target_lin, target_par = (int(value) for value in target_lin_par)
    if maps != 1 or any(value != 1 for value in shape[5:]):
        raise ValueError("Measured Wave reconstruction accepts exactly one ESPIRiT map set.")
    if target_lin > source_lin or target_par > source_par:
        raise ValueError("Retrospective CSM grids cannot exceed the native PE grid.")
    output = create_cfl(output_base, (ro, target_lin, target_par, coils, 1))
    rss_squared = np.zeros((ro, target_lin, target_par), dtype=np.float32)
    lin_start = source_lin // 2 - target_lin // 2
    par_start = source_par // 2 - target_par // 2
    for coil in range(coils):
        field = np.asarray(source[:, :, :, coil, ...]).squeeze()
        if field.shape != (ro, source_lin, source_par):
            raise ValueError("Sensitivity map did not reduce to one 3D coil field.")
        field = np.asarray(field, dtype=np.complex64)
        spectrum = centered_fftn(field, axes=(1, 2))
        cropped = spectrum[
            :, lin_start : lin_start + target_lin, par_start : par_start + target_par
        ]
        resized = centered_fftn(cropped, axes=(1, 2), inverse=True)
        output[:, :, :, coil, 0] = resized
        rss_squared += np.abs(resized) ** 2
    rss = np.sqrt(rss_squared)
    support = rss > 1e-8
    for coil in range(coils):
        current = np.asarray(output[:, :, :, coil, 0])
        current[support] /= rss[support]
        current[~support] = 0
        output[:, :, :, coil, 0] = current
    output.flush()
    del output


def synthesize_wave_from_no_wave_crop(
    no_wave_kspace: np.ndarray,
    target_psf: np.ndarray,
    *,
    readout_oversampled: int,
    target_mask: np.ndarray | None = None,
    fft_workers: int = 1,
) -> np.ndarray:
    """Explicit synthetic-only crop-first no-Wave to Wave operation.

    This utility is retained for ``synthetic_wave_for_reg_baseline``. It is
    deliberately not used by measured-Wave MPRAGE launchers.

    Args:
        no_wave_kspace: One-coil no-Wave k-space on the target grid.
        target_psf: Wave PSF evaluated directly on that grid.
        readout_oversampled: Extended Wave readout dimension.
        target_mask: Optional explicit image sampling mask without ACS.
        fft_workers: Maximum SciPy FFT workers.

    Returns:
        Complex64 synthetic Wave k-space, optionally masked after encoding.
    """

    encoded = apply_wave_forward(
        no_wave_kspace,
        target_psf,
        readout_oversampled=readout_oversampled,
        fft_workers=fft_workers,
    )
    if target_mask is not None:
        mask = np.asarray(target_mask, dtype=bool)
        if encoded.shape[1:] != mask.shape:
            raise ValueError("Synthetic Wave output and target mask dimensions disagree.")
        encoded *= mask[None, :, :]
    return encoded
