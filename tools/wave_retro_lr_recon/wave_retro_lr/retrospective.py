"""Measured-Wave and explicitly no-Wave retrospective utilities."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version

from .bart_io import bart_base, cfl_record, create_cfl, open_cfl, read_shape, sha256_file
from .core import ResolvedCase, apply_wave_forward, centered_fftn


def link_bart_pair(source_base: str | Path, destination_base: str | Path) -> None:
    """Create relative links to one immutable BART CFL pair.

    Args:
        source_base: Existing source BART basename.
        destination_base: New case-local BART basename.

    Returns:
        None. The destination header and payload become relative symlinks.

    Raises:
        FileNotFoundError: If either source file is absent.
        FileExistsError: If either destination already exists.
    """

    source_root = Path(source_base)
    destination_root = Path(destination_base)
    for suffix in (".hdr", ".cfl"):
        source = source_root.with_suffix(suffix).resolve()
        destination = destination_root.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.symlink_to(os.path.relpath(source, destination.parent))


def validate_same_grid_masked_wave(
    source_base: str | Path,
    target_base: str | Path,
    target_mask: np.ndarray,
    *,
    readout_chunk: int = 8,
) -> dict[str, Any]:
    """Verify exact acquired samples and zeros after same-grid undersampling.

    Args:
        source_base: Native measured-Wave BART basename.
        target_base: Retrospectively masked BART basename.
        target_mask: Pure logical LIN/PAR target lattice.
        readout_chunk: Maximum oversampled-readout planes checked together.

    Returns:
        Exact mismatch, zero-exterior, and finite-value validation metrics.

    Raises:
        ValueError: If geometry, acquired values, zero exterior, or finiteness
            violates the same-grid retrospective-undersampling contract.
    """

    if readout_chunk < 1:
        raise ValueError("Readout validation chunk must be positive.")
    source_shape = read_shape(source_base)
    target_shape = read_shape(target_base)
    source_shape = source_shape + (1,) * max(0, 5 - len(source_shape))
    target_shape = target_shape + (1,) * max(0, 5 - len(target_shape))
    if source_shape[:5] != target_shape[:5] or any(
        value != 1 for value in (*source_shape[5:], *target_shape[5:])
    ):
        raise ValueError("Same-grid undersampling must preserve native Wave dimensions.")
    ro_os, nlin, npar, coils, maps = source_shape[:5]
    mask = np.asarray(target_mask)
    if maps != 1 or mask.dtype != np.bool_ or mask.shape != (nlin, npar):
        raise ValueError("Same-grid target mask or Wave map geometry is invalid.")
    source = open_cfl(source_base).reshape(
        (ro_os, nlin, npar, coils, -1), order="F"
    )
    target = open_cfl(target_base).reshape(
        (ro_os, nlin, npar, coils, -1), order="F"
    )
    acquired_mismatch = 0
    unacquired_nonzero = 0
    nonfinite = 0
    for start in range(0, ro_os, readout_chunk):
        stop = min(start + readout_chunk, ro_os)
        source_block = np.asarray(source[start:stop, ..., 0])
        target_block = np.asarray(target[start:stop, ..., 0])
        acquired_mismatch += int(
            np.count_nonzero(target_block[:, mask, :] != source_block[:, mask, :])
        )
        unacquired_nonzero += int(np.count_nonzero(target_block[:, ~mask, :]))
        nonfinite += int(np.count_nonzero(~np.isfinite(target_block)))
    if acquired_mismatch or unacquired_nonzero or nonfinite:
        raise ValueError(
            "Same-grid masked Wave data failed acquired-equality, "
            "zero-exterior, or finite-value validation."
        )
    return {
        "acquired_mismatch_count": acquired_mismatch,
        "unacquired_nonzero_count": unacquired_nonzero,
        "nonfinite_count": nonfinite,
        "acquired_samples_equal_source_bitwise": True,
        "unacquired_samples_are_exact_zero": True,
    }


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
    *,
    target_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Center-crop measured Wave k-space and apply its target mask.

    Args:
        source_base: Source measured-Wave BART CFL basename.
        output_base: Destination target-grid BART CFL basename.
        case: Resolved PE crop and target acceleration.
        source_mask: Boolean measured image-stream mask.
        source_acceleration_lin_par: Known source LIN/PAR acceleration.
        target_mask: Optional explicit pure target mask. It must be a subset of
            the cropped measured source coordinates.

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
    derived_mask = _measured_target_mask(
        source_mask, case, source_acceleration_lin_par
    )
    if target_mask is None:
        resolved_mask = derived_mask
    else:
        resolved_mask = np.asarray(target_mask)
        cropped_source_mask = np.asarray(source_mask, dtype=bool)[
            slice(*case.crop_bounds_lin), slice(*case.crop_bounds_par)
        ]
        if (
            resolved_mask.dtype != np.bool_
            or resolved_mask.shape != (target_lin, target_par)
            or not np.any(resolved_mask)
            or np.any(resolved_mask & ~cropped_source_mask)
        ):
            raise ValueError(
                "Explicit target mask must be a nonempty boolean subset of measured samples."
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
        cropped *= resolved_mask[None, :, :]
        output[:, :, :, coil, 0] = cropped
        squared_norm += float(np.vdot(cropped, cropped).real)
    output.flush()
    del output
    return {
        "wave_kspace_norm": float(np.sqrt(squared_norm)),
        "sampled_coordinate_count": int(np.count_nonzero(resolved_mask)),
        "sampling_fraction": float(np.mean(resolved_mask)),
        "image_kspace_center_acquired": bool(
            resolved_mask[target_lin // 2, target_par // 2]
        ),
    }


CSM_RESAMPLING_POLICY = "same_fov_native_support_and_rss_v1"


def native_pe_coordinates(source: int, target: int) -> np.ndarray:
    """Return native voxel-center coordinates of a centered same-FOV target PE axis.

    The origin is index floor(N/2), matching the centered FFT convention for
    even and odd sizes. Coordinates may reach a native half-voxel boundary;
    scalar interpolation clamps to the nearest native edge, never wraps.
    """
    return (np.arange(target, dtype=np.float64) - target//2)*(source/target) + source//2


def map_native_rss(rss: np.ndarray, target_lin_par: tuple[int, int]) -> tuple[np.ndarray, tuple]:
    """Map native support and scalar RSS independently of Fourier ringing.

    Args:
        rss: Finite nonnegative native RO/LIN/PAR coil RSS volume.
        target_lin_par: Positive same-FOV target PE dimensions.
    Returns:
        Target RSS and nearest native LIN/PAR indices. Support uses nearest
        native voxel centers (half ties towards the higher index). Within that
        support, RSS is bilinearly interpolated using only supported native
        neighbors. Thus unit hard support stays unit and soft gain is retained.
        There is no interpolation along RO, dilation, or tissue inference.
    """
    from scipy.ndimage import map_coordinates

    lin = native_pe_coordinates(rss.shape[1], target_lin_par[0])
    par = native_pe_coordinates(rss.shape[2], target_lin_par[1])
    nearest = (np.clip(np.floor(lin+.5).astype(int), 0, rss.shape[1]-1),
               np.clip(np.floor(par+.5).astype(int), 0, rss.shape[2]-1))
    coordinates = np.asarray(np.meshgrid(lin, par, indexing="ij"))
    result = np.zeros((rss.shape[0], *target_lin_par), dtype=np.float32)
    for ro in range(rss.shape[0]):
        plane = rss[ro]
        support = plane > 0
        target_support = support[np.ix_(*nearest)]
        numerator = map_coordinates(plane, coordinates, order=1, mode="nearest", prefilter=False)
        denominator = map_coordinates(support.astype(np.float32), coordinates,
                                      order=1, mode="nearest", prefilter=False)
        np.divide(numerator, denominator, out=result[ro], where=target_support & (denominator > 0))
    return result, nearest


def resample_sensitivity_maps(
    source_base: str | Path, output_base: str | Path, *, target_lin_par: tuple[int, int]
) -> dict[str, Any]:
    """Resample coil directions while preserving mapped native support and RSS gain.

    Args:
        source_base: Finite native single-map BART CSM; zeros define its support.
        output_base: New target CSM basename, or a completed hash-bound derivative.
        target_lin_par: Positive integer target PE sizes, no larger than native.

    Returns:
        Exact source/output records and versioned support/weight transfer policy.
        Writes CFL/HDR plus an exclusive ``*_resampling.json`` completion record.

    Raises:
        ValueError: Invalid input, partial output, legacy unbound maps, or reuse
            mismatch. Existing maps and reconstructions are never overwritten.
        BlockingIOError: Another writer holds this derivative's lock.
    """
    source_base, output_base = bart_base(source_base).resolve(), bart_base(output_base).absolute()
    if source_base == output_base.resolve():
        raise ValueError("Native CSM must not be overwritten by retrospective resampling")
    source_shape = read_shape(source_base)
    if len(source_shape) < 4:
        raise ValueError("Native CSM requires spatial and coil dimensions")
    shape = source_shape + (1,) * max(0, 5 - len(source_shape))
    ro, source_lin, source_par, coils, maps = shape[:5]
    if len(target_lin_par) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                                      or v < 1 for v in target_lin_par):
        raise ValueError("Target PE grid must contain two positive integers")
    target_lin, target_par = (int(value) for value in target_lin_par)
    if maps != 1 or any(value != 1 for value in shape[5:]):
        raise ValueError("Measured Wave reconstruction accepts exactly one ESPIRiT map set.")
    if target_lin > source_lin or target_par > source_par:
        raise ValueError("Retrospective CSM grids cannot exceed the native PE grid.")
    metadata = output_base.with_name(output_base.name+"_resampling.json")
    lock_path = output_base.with_name(output_base.name+"_resampling.lock")
    if any(path.is_symlink() for path in (metadata, lock_path, output_base.with_suffix(".hdr"),
                                         output_base.with_suffix(".cfl"))):
        raise ValueError("Derived LR CSM files must not be symlinks")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not metadata.exists() and any(output_base.with_suffix(s).exists() for s in (".hdr", ".cfl")):
            raise ValueError("Existing LR CSM is legacy/unbound or partial; it is not support-preserving. "
                             "Preserve it and use a fresh reconstruction output root: " + str(output_base))
        source = open_cfl(source_base).reshape((ro, source_lin, source_par, coils), order="F")
        native_squared = np.zeros((ro, source_lin, source_par), dtype=np.float64)
        for coil in range(coils):
            field = source[..., coil]
            if not np.isfinite(field).all():
                raise ValueError("Native CSM contains non-finite values")
            native_squared += np.square(field.real, dtype=np.float64) + np.square(field.imag, dtype=np.float64)
        native_rss = np.sqrt(native_squared).astype(np.float32)
        if not np.isfinite(native_rss).all():
            raise ValueError("Native CSM RSS overflows float32")
        del native_squared
        request = {"policy": CSM_RESAMPLING_POLICY, "source": cfl_record(source_base),
                   "target_shape": [ro, target_lin, target_par, coils, 1],
                   "implementation_sha256": sha256_file(Path(__file__)),
                   "fft_implementation_sha256": sha256_file(Path(__file__).with_name("core.py")),
                   "numpy_version": np.__version__, "scipy_version": scipy_version,
                   "coordinate_rule": "(target_index-floor(target_N/2))*native_N/target_N+floor(native_N/2)",
                   "support": "nearest native nonzero RSS; half ties up; no periodic wrap",
                   "gain": "support-normalized bilinear native RSS; unchanged RO",
                   "direction": "centered Fourier PE crop of native unit coil vectors",
                   "degenerate_direction": "nearest supported native unit coil vector at RSS <= 1e-8"}
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved.get("request") != request or saved.get("output") != cfl_record(output_base):
                raise ValueError("LR CSM support policy, source or output hash mismatch: " + str(output_base))
            return saved
        result = _write_supported_maps(source, native_rss, output_base, (target_lin, target_par))
        if cfl_record(source_base) != request["source"]:
            raise ValueError("Native CSM changed during LR preparation")
        saved = {"request": request, "output": cfl_record(output_base), **result}
        with metadata.open("x") as stream:
            json.dump(saved, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return saved


def _write_supported_maps(source: np.ndarray, native_rss: np.ndarray, output_base: Path,
                          target_lin_par: tuple[int, int]) -> dict[str, Any]:
    """Write one new LR map and return support/fallback counts; caller owns validation and lock.

    Fourier ringing is restricted by separately mapped native support. Normalize
    only the coil direction, then restore target RSS, including soft weights.
    A vanishing Fourier vector inside support falls back to the nearest native
    unit vector; no arbitrary phase or new support is invented. Same-grid input
    is copied exactly without normalization or interpolation.
    """
    ro, source_lin, source_par, coils = source.shape
    target_lin, target_par = target_lin_par
    output = create_cfl(output_base, (ro, target_lin, target_par, coils, 1))
    if target_lin_par == (source_lin, source_par):
        for coil in range(coils):
            output[..., coil, 0] = source[..., coil]
        output.flush()
        return {"operation": "exact_same_grid_copy", "fallback_voxels": 0,
                "support_voxels": int(np.count_nonzero(native_rss))}
    target_rss, nearest = map_native_rss(native_rss, target_lin_par)
    rss_squared = np.zeros((ro, target_lin, target_par), dtype=np.float32)
    lin_start = source_lin // 2 - target_lin // 2
    par_start = source_par // 2 - target_par // 2
    for coil in range(coils):
        field = np.zeros(native_rss.shape, dtype=np.complex64)
        np.divide(source[..., coil], native_rss, out=field, where=native_rss > 0)
        spectrum = centered_fftn(field, axes=(1, 2))
        cropped = spectrum[
            :, lin_start : lin_start + target_lin, par_start : par_start + target_par
        ]
        resized = centered_fftn(cropped, axes=(1, 2), inverse=True)
        output[:, :, :, coil, 0] = resized
        rss_squared += np.abs(resized) ** 2
    rss = np.sqrt(rss_squared)
    support = target_rss > 0
    regular = support & (rss > 1e-8)
    fallback = support & ~regular
    nearest_rss = native_rss[:, nearest[0], :][:, :, nearest[1]]
    for coil in range(coils):
        current = np.asarray(output[:, :, :, coil, 0])
        current[regular] *= target_rss[regular]/rss[regular]
        if np.any(fallback):
            native_values = source[..., coil][:, nearest[0], :][:, :, nearest[1]]
            current[fallback] = native_values[fallback]/nearest_rss[fallback]*target_rss[fallback]
        current[~support] = 0
        if not np.isfinite(current).all():
            raise ValueError("Non-finite LR CSM after direction/support transfer")
        output[:, :, :, coil, 0] = current
    output.flush()
    del output
    return {"operation": CSM_RESAMPLING_POLICY, "fallback_voxels": int(np.count_nonzero(fallback)),
            "support_voxels": int(np.count_nonzero(support)),
            "source_support_voxels": int(np.count_nonzero(native_rss))}


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
