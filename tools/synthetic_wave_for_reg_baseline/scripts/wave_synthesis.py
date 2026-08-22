"""Centered FFT, extended-readout, and theoretical Wave-PSF helpers."""

from __future__ import annotations

import hashlib
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import fft


SPATIAL_AXES = (0, 1, 2)


def centered_fftn(
    array: np.ndarray,
    *,
    axes: Sequence[int],
    inverse: bool = False,
    workers: int = 1,
) -> np.ndarray:
    """Apply a centered orthonormal FFT over named axes, preserving complex64."""
    axes = tuple(int(axis) for axis in axes)
    shifted = np.fft.ifftshift(np.asarray(array), axes=axes)
    transform = fft.ifftn if inverse else fft.fftn
    result = transform(shifted, axes=axes, norm="ortho", workers=workers)
    return np.fft.fftshift(result, axes=axes).astype(np.complex64, copy=False)


def center_embed_readout(image: np.ndarray, nx_extended: int) -> tuple[np.ndarray, slice]:
    """Center a coil image in a zero-filled extended readout FOV on axis 0."""
    image = np.asarray(image, dtype=np.complex64)
    if image.ndim != 3:
        raise ValueError(f"Expected [RO, PE1, PE2] coil image, got {image.shape}.")
    nx = int(image.shape[0])
    if nx_extended < nx or (nx_extended - nx) % 2:
        raise ValueError("Extended readout must be larger by an even number of samples.")
    start = (nx_extended - nx) // 2
    support = slice(start, start + nx)
    extended = np.zeros((nx_extended, *image.shape[1:]), dtype=np.complex64)
    extended[support] = image
    if not np.array_equal(extended[support], image):
        raise RuntimeError("Extended-FOV center embedding did not preserve the input image.")
    if np.any(extended[:start]) or np.any(extended[start + nx :]):
        raise RuntimeError("Extended-FOV exterior must remain exact zero.")
    return extended, support


def build_theoretical_psf(
    delta_ky_idx: np.ndarray,
    delta_kz_idx: np.ndarray,
    *,
    ny: int,
    nz: int,
    yflip: int = -1,
    zflip: int = -1,
) -> np.ndarray:
    """Build the reference Wave-MPRAGE unit-magnitude hybrid-space PSF."""
    delta_ky_idx = np.asarray(delta_ky_idx, dtype=np.float64)
    delta_kz_idx = np.asarray(delta_kz_idx, dtype=np.float64)
    if delta_ky_idx.ndim != 1 or delta_kz_idx.shape != delta_ky_idx.shape:
        raise ValueError("Theoretical ky/kz trajectory offsets must be equal-length vectors.")
    if yflip not in (-1, 1) or zflip not in (-1, 1):
        raise ValueError("yflip and zflip must each be -1 or 1.")

    y_norm = (np.arange(ny, dtype=np.float64) - ny / 2.0) / ny
    z_norm = (np.arange(nz, dtype=np.float64) - nz / 2.0) / nz
    psf_y = np.exp(
        -1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm[None, :]
    ).astype(np.complex64)
    psf_z = np.exp(
        -1j * zflip * 2.0 * np.pi * delta_kz_idx[:, None] * z_norm[None, :]
    ).astype(np.complex64)
    psf = psf_y[:, :, None] * psf_z[:, None, :]
    if not np.isfinite(psf).all():
        raise ValueError("Theoretical PSF contains non-finite values.")
    return psf


def generate_theoretical_wave_trajectory(
    sequence_path: str | Path,
    *,
    nx_os: int,
    ncalib: int,
    nacs: int,
    orientation: str = "SAG",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Adapt the Wave-MPRAGE reference trajectory extraction for one sequence."""
    import pypulseq as pp

    sequence_path = Path(sequence_path).expanduser().resolve()
    sequence = pp.Sequence()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sequence.read(str(sequence_path), remove_duplicates=False)
        calculation = sequence.calculate_kspace()
    definitions = sequence.definitions
    definition_nx = int(float(definitions.get("Calibration_ReadoutSamples", nx_os)))
    definition_ncalib = int(float(definitions.get("Calibration_Ncalib1", ncalib)))
    definition_nacs = int(float(definitions.get("Calibration_Nacs", nacs)))
    definition_orientation = str(definitions.get("OrientationMapping", orientation)).upper()
    definition_os = int(float(definitions.get("ReadoutOversamplingFactor", nx_os // 256)))
    if definition_nx != nx_os:
        raise ValueError(
            f"Sequence Calibration_ReadoutSamples={definition_nx}, expected nx_os={nx_os}."
        )
    if (definition_ncalib, definition_nacs) != (ncalib, nacs):
        raise ValueError(
            "Sequence calibration definitions do not match the requested tail layout: "
            f"{(definition_ncalib, definition_nacs)} versus {(ncalib, nacs)}."
        )

    ktraj_adc = np.asarray(calculation[0], dtype=np.float64)
    calibration_tail = int(nx_os * (4 * ncalib + nacs * nacs))
    imaging_samples = int(ktraj_adc.shape[1] - calibration_tail)
    if imaging_samples <= 0 or imaging_samples % nx_os:
        raise ValueError(
            f"Imaging ADC samples {imaging_samples} are not a positive multiple of {nx_os}."
        )
    imaging = ktraj_adc[:, :imaging_samples].reshape(3, -1, nx_os)

    # This component mapping exactly follows the established MPRAGE reference.
    ky_lines = imaging[1]
    kz_lines = imaging[0]
    ky_line = int(np.argmin(np.abs(ky_lines[:, 0])))
    kz_line = int(np.argmin(np.abs(kz_lines[:, 0])))
    fov = np.asarray(definitions.get("FOV", [0.224, 0.224, 0.224]), dtype=float)
    orientation = str(orientation).upper()
    if definition_orientation != orientation:
        raise ValueError(
            f"Sequence OrientationMapping={definition_orientation}, requested {orientation}."
        )
    if orientation == "SAG":
        fov_y, fov_z = float(fov[1]), float(fov[0])
    elif orientation == "TRA":
        fov_y, fov_z = float(fov[1]), float(fov[2])
    else:
        raise ValueError(f"Unsupported sequence orientation: {orientation}.")
    delta_ky_idx = ky_lines[ky_line] * fov_y
    delta_kz_idx = kz_lines[kz_line] * fov_z

    info = {
        "pypulseq_version": metadata.version("pypulseq"),
        "warnings": [str(item.message) for item in caught],
        "total_adc_samples": int(ktraj_adc.shape[1]),
        "calibration_tail_samples": calibration_tail,
        "imaging_adc_samples": imaging_samples,
        "imaging_readout_lines": int(imaging_samples // nx_os),
        "readout_oversampling_factor": definition_os,
        "calibration_readout_samples": definition_nx,
        "calibration_ncalib1": definition_ncalib,
        "calibration_nacs": definition_nacs,
        "orientation_mapping": definition_orientation,
        "ky_center_line_index": ky_line,
        "kz_center_line_index": kz_line,
        "fov_y_m": fov_y,
        "fov_z_m": fov_z,
        "delta_ky_idx_minmax": [float(delta_ky_idx.min()), float(delta_ky_idx.max())],
        "delta_kz_idx_minmax": [float(delta_kz_idx.min()), float(delta_kz_idx.max())],
    }
    return delta_ky_idx, delta_kz_idx, info


def apply_wave_forward(
    extended_image: np.ndarray,
    psf: np.ndarray,
    *,
    workers: int = 1,
) -> np.ndarray:
    """Apply ``F_RO -> PSF -> F_PE1,PE2`` to one extended-FOV coil image."""
    extended_image = np.asarray(extended_image, dtype=np.complex64)
    psf = np.asarray(psf, dtype=np.complex64)
    if extended_image.shape != psf.shape or extended_image.ndim != 3:
        raise ValueError(
            f"Extended image and PSF must share a 3D shape; got {extended_image.shape} and {psf.shape}."
        )
    hybrid = centered_fftn(extended_image, axes=(0,), workers=workers)
    hybrid *= psf
    return centered_fftn(hybrid, axes=(1, 2), workers=workers)


def apply_wave_adjoint(
    wave_kspace: np.ndarray,
    psf: np.ndarray,
    *,
    workers: int = 1,
) -> np.ndarray:
    """Apply the adjoint ``F_PE^-1 -> PSF* -> F_RO^-1`` Wave operator.

    For the unit-magnitude theoretical PSF used by this workflow, the adjoint
    is also the exact full-sampling inverse apart from floating-point error.
    """
    wave_kspace = np.asarray(wave_kspace, dtype=np.complex64)
    psf = np.asarray(psf, dtype=np.complex64)
    if wave_kspace.shape != psf.shape or wave_kspace.ndim != 3:
        raise ValueError(
            "Wave k-space and PSF must share a 3D shape; "
            f"got {wave_kspace.shape} and {psf.shape}."
        )
    hybrid = centered_fftn(
        wave_kspace, axes=(1, 2), inverse=True, workers=workers
    )
    hybrid *= np.conjugate(psf)
    return centered_fftn(hybrid, axes=(0,), inverse=True, workers=workers)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def logical_array_sha256(array: np.ndarray, x_chunk: int = 16) -> str:
    """Hash logical C-order values without requiring one contiguous full copy."""
    array = np.asarray(array)
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], x_chunk):
        block = np.ascontiguousarray(array[start : start + x_chunk])
        digest.update(block.view(np.uint8))
    return digest.hexdigest()


def logical_bart_cfl_sha256(
    base_path: str | Path,
    shape: Sequence[int],
    x_chunk: int = 16,
) -> str:
    """Hash a Fortran-ordered BART CFL as logical C-order array values."""
    base = Path(base_path).with_suffix("")
    shape = tuple(int(value) for value in shape)
    raw = np.memmap(base.with_suffix(".cfl"), mode="r", dtype=np.complex64)
    expected = int(np.prod(shape, dtype=np.int64))
    if raw.size != expected:
        raise ValueError(f"BART CFL contains {raw.size} values, expected {expected}.")
    array = raw.reshape(shape, order="F")
    return logical_array_sha256(array, x_chunk=x_chunk)
