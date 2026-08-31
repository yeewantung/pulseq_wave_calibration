"""Geometry, sampling, PSF, and Wave-operator primitives.

The physical-axis contract is sagittal MPRAGE: logical ``(RO, LIN, PAR)`` is
physical ``(Z, Y, X)``. Retrospective resolution changes therefore crop only
LIN/PAR; the readout matrix and physical-Z resolution are invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import fft

PE_MATRIX_MULTIPLE = 4


@dataclass(frozen=True)
class Geometry:
    physical_fov_mm_xyz: tuple[float, float, float]
    logical_matrix_ro_lin_par: tuple[int, int, int]

    @property
    def physical_matrix_xyz(self) -> tuple[int, int, int]:
        """Map logical MPRAGE dimensions to physical XYZ matrix order.

        Returns:
            Physical ``(X, Y, Z)`` matrix dimensions.
        """
        ro, lin, par = self.logical_matrix_ro_lin_par
        return par, lin, ro

    @property
    def physical_resolution_mm_xyz(self) -> tuple[float, float, float]:
        """Calculate voxel spacing from physical FOV and matrix dimensions.

        Returns:
            Physical ``(X, Y, Z)`` voxel spacing in millimeters.
        """
        return tuple(
            fov / matrix
            for fov, matrix in zip(
                self.physical_fov_mm_xyz, self.physical_matrix_xyz, strict=True
            )
        )


@dataclass(frozen=True)
class CaseSpec:
    resolution_mm_xyz: tuple[float, float, float]
    acceleration_ry_rz: tuple[int, int]
    label: str | None = None


@dataclass(frozen=True)
class ResolvedCase:
    requested_resolution_mm_xyz: tuple[float, float, float]
    achieved_resolution_mm_xyz: tuple[float, float, float]
    source_logical_matrix_ro_lin_par: tuple[int, int, int]
    target_logical_matrix_ro_lin_par: tuple[int, int, int]
    target_physical_matrix_xyz: tuple[int, int, int]
    crop_bounds_lin: tuple[int, int]
    crop_bounds_par: tuple[int, int]
    acceleration_ry_rz: tuple[int, int]
    case_name: str
    label: str | None = None

    def to_json(self) -> dict[str, object]:
        """Convert a resolved case to stable JSON-native values.

        Returns:
            Manifest mapping used for provenance and resume comparison.
        """
        return {
            "requested_resolution_mm_xyz": list(self.requested_resolution_mm_xyz),
            "achieved_resolution_mm_xyz": list(self.achieved_resolution_mm_xyz),
            "source_logical_matrix_ro_lin_par": list(
                self.source_logical_matrix_ro_lin_par
            ),
            "target_logical_matrix_ro_lin_par": list(
                self.target_logical_matrix_ro_lin_par
            ),
            "target_physical_matrix_xyz": list(self.target_physical_matrix_xyz),
            "crop_bounds_lin": list(self.crop_bounds_lin),
            "crop_bounds_par": list(self.crop_bounds_par),
            "acceleration_ry_rz": list(self.acceleration_ry_rz),
            "pe_matrix_multiple": PE_MATRIX_MULTIPLE,
            "case_name": self.case_name,
            "label": self.label,
        }


def center_crop_bounds(source_size: int, target_size: int) -> tuple[int, int]:
    """Calculate a half-open crop preserving the Python center index.

    Args:
        source_size: Size of the source axis.
        target_size: Requested size not exceeding the source.

    Returns:
        Half-open ``(start, stop)`` crop bounds.
    """
    if not 1 <= target_size <= source_size:
        raise ValueError(
            f"Target size must satisfy 1 <= target <= source; got {target_size} and {source_size}."
        )
    start = source_size // 2 - target_size // 2
    stop = start + target_size
    if start + target_size // 2 != source_size // 2:
        raise AssertionError("Centered crop did not preserve the Python center index.")
    return start, stop


def _rounded_matrix(fov_mm: float, resolution_mm: float) -> int:
    """Round an ideal PE matrix to the required multiple.

    Args:
        fov_mm: Physical field of view in millimeters.
        resolution_mm: Requested voxel spacing in millimeters.

    Returns:
        Nearest positive matrix size divisible by ``PE_MATRIX_MULTIPLE``.
    """
    if not np.isfinite(resolution_mm) or resolution_mm <= 0:
        raise ValueError(f"Resolution must be positive and finite; got {resolution_mm}.")
    ideal_matrix = fov_mm / resolution_mm
    return PE_MATRIX_MULTIPLE * int(
        math.floor(ideal_matrix / PE_MATRIX_MULTIPLE + 0.5)
    )


def _format_number(value: float) -> str:
    """Format a numeric case-name component without trailing zeros.

    Args:
        value: Numeric value to round to two decimal places.

    Returns:
        Compact decimal string.
    """
    return f"{round(float(value), 2):.2f}".rstrip("0").rstrip(".")


def format_case_name(
    achieved_resolution_mm_xyz: Sequence[float], acceleration_ry_rz: tuple[int, int]
) -> str:
    """Build a stable result name from resolution and acceleration.

    Args:
        achieved_resolution_mm_xyz: Achieved physical XYZ spacing in millimeters.
        acceleration_ry_rz: Logical LIN/PAR acceleration factors.

    Returns:
        Dataset-independent case directory name.
    """
    resolution = "x".join(_format_number(value) for value in achieved_resolution_mm_xyz)
    return f"res{resolution}mm_R{acceleration_ry_rz[0]}x{acceleration_ry_rz[1]}"


def resolve_case(spec: CaseSpec, geometry: Geometry) -> ResolvedCase:
    """Map a physical-XYZ request onto a sagittal logical PE crop.

    Args:
        spec: Requested spacing, acceleration, and optional label.
        geometry: Source physical FOV and logical matrix contract.

    Returns:
        Fully resolved crop bounds, matrices, and achieved resolution.
    """
    fov_x, fov_y, fov_z = geometry.physical_fov_mm_xyz
    source_ro, source_lin, source_par = geometry.logical_matrix_ro_lin_par
    source_res_z = fov_z / source_ro
    request_x, request_y, request_z = spec.resolution_mm_xyz
    if not math.isclose(request_z, source_res_z, rel_tol=0.0, abs_tol=max(1e-4, source_res_z * 1e-3)):
        raise ValueError(
            "Retrospective resolution may change only phase encoding. "
            f"Requested physical-Z/readout resolution {request_z:g} mm differs from "
            f"the source {source_res_z:g} mm."
        )
    target_par = _rounded_matrix(fov_x, request_x)
    target_lin = _rounded_matrix(fov_y, request_y)
    if target_lin > source_lin or target_par > source_par:
        raise ValueError(
            "Requested PE resolution is finer than the source acquisition: "
            f"source={geometry.physical_resolution_mm_xyz}, requested={spec.resolution_mm_xyz}."
        )
    if any(value < 1 for value in spec.acceleration_ry_rz):
        raise ValueError("Ry and Rz must be positive integers.")
    achieved = (fov_x / target_par, fov_y / target_lin, fov_z / source_ro)
    target_logical = (source_ro, target_lin, target_par)
    return ResolvedCase(
        requested_resolution_mm_xyz=tuple(float(value) for value in spec.resolution_mm_xyz),
        achieved_resolution_mm_xyz=tuple(float(value) for value in achieved),
        source_logical_matrix_ro_lin_par=geometry.logical_matrix_ro_lin_par,
        target_logical_matrix_ro_lin_par=target_logical,
        target_physical_matrix_xyz=(target_par, target_lin, source_ro),
        crop_bounds_lin=center_crop_bounds(source_lin, target_lin),
        crop_bounds_par=center_crop_bounds(source_par, target_par),
        acceleration_ry_rz=spec.acceleration_ry_rz,
        case_name=format_case_name(achieved, spec.acceleration_ry_rz),
        label=spec.label,
    )


def build_case_mask(
    source_mask: np.ndarray,
    case: ResolvedCase,
    source_acceleration_ry_rz: tuple[int, int],
) -> np.ndarray:
    """Crop a legacy mask and add a uniform centered lattice on full axes.

    This compatibility helper never guesses which fully sampled rows are ACS.
    Calibration must remain a separate input rather than being merged into a
    Wave reconstruction mask.

    Args:
        source_mask: Boolean source sampling mask in logical LIN/PAR order.
        case: Resolved target crop and acceleration request.
        source_acceleration_ry_rz: Known source LIN/PAR acceleration.

    Returns:
        Cropped boolean reconstruction mask without inferred calibration rows.
    """
    mask = np.asarray(source_mask, dtype=bool)
    source_lin, source_par = case.source_logical_matrix_ro_lin_par[1:]
    if mask.shape != (source_lin, source_par):
        raise ValueError(f"Source mask shape {mask.shape} does not match {(source_lin, source_par)}.")
    lin = slice(*case.crop_bounds_lin)
    par = slice(*case.crop_bounds_par)
    cropped = mask[lin, par].copy()
    source_ry, source_rz = source_acceleration_ry_rz
    target_ry, target_rz = case.acceleration_ry_rz
    for source, target, name in (
        (source_ry, target_ry, "Ry"),
        (source_rz, target_rz, "Rz"),
    ):
        if source > 1 and target != source:
            raise ValueError(
                f"Source {name}={source} is already accelerated and must remain unchanged; "
                f"target {name}={target} was requested."
            )
    target_lin, target_par = cropped.shape
    if source_ry == 1 and target_ry > 1:
        lin_keep = ((np.arange(target_lin) - target_lin // 2) % target_ry) == 0
        cropped &= lin_keep[:, None]
    if source_rz == 1 and target_rz > 1:
        par_keep = ((np.arange(target_par) - target_par // 2) % target_rz) == 0
        cropped &= par_keep[None, :]
    if not np.any(cropped):
        raise ValueError("The final retrospective sampling mask is empty.")
    return cropped


def extract_psf_phase_planes(
    psf: np.ndarray, *, readout_chunk: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract phase-plane slopes from a final complex Wave PSF.

    Adjacent complex ratios avoid direct wrapped phase-plane fitting. The PSF
    must be unit magnitude because BART Wave calibration is represented here as
    a pure phase modulation.

    Args:
        psf: Complex PSF with shape ``(RO_os, LIN, PAR)``.
        readout_chunk: Number of readout samples processed per block.

    Returns:
        Per-readout LIN slope, PAR slope, and constant phase vectors.
    """
    if psf.ndim != 3:
        raise ValueError(f"PSF must have shape (RO_os, LIN, PAR); got {psf.shape}.")
    nro, nlin, npar = psf.shape
    alpha = np.empty(nro, dtype=np.float64)
    beta = np.empty(nro, dtype=np.float64)
    gamma = np.empty(nro, dtype=np.float64)
    lin_norm = (np.arange(nlin, dtype=np.float64) - nlin / 2.0) / nlin
    par_norm = (np.arange(npar, dtype=np.float64) - npar / 2.0) / npar
    for start in range(0, nro, readout_chunk):
        stop = min(start + readout_chunk, nro)
        # Accumulate the many nominally identical adjacent ratios in complex128.
        # Complex64 reduction over a full 256x256 plane measurably biases the
        # recovered slopes even when the stored PSF itself is valid complex64.
        block = np.asarray(psf[start:stop], dtype=np.complex128)
        magnitude = np.abs(block)
        if not np.isfinite(block).all() or np.max(np.abs(magnitude - 1.0)) > 2e-5:
            raise ValueError("Source PSF must be finite and unit magnitude within 2e-5.")
        unit = block / np.maximum(magnitude, 1e-12)
        lin_ratio = unit[:, 1:, :] * np.conj(unit[:, :-1, :])
        par_ratio = unit[:, :, 1:] * np.conj(unit[:, :, :-1])
        current_alpha = np.angle(
            np.mean(lin_ratio, axis=(1, 2), dtype=np.complex128)
        ) * nlin
        current_beta = np.angle(
            np.mean(par_ratio, axis=(1, 2), dtype=np.complex128)
        ) * npar
        residual = unit * np.exp(
            -1j
            * (
                current_alpha[:, None, None] * lin_norm[None, :, None]
                + current_beta[:, None, None] * par_norm[None, None, :]
            )
        )
        alpha[start:stop] = current_alpha
        beta[start:stop] = current_beta
        gamma[start:stop] = np.angle(
            np.mean(residual, axis=(1, 2), dtype=np.complex128)
        )
    return alpha, beta, gamma


def evaluate_psf_phase_planes(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    nlin: int,
    npar: int,
) -> np.ndarray:
    """Evaluate extracted Wave phase planes on a requested PE grid.

    Args:
        alpha: Per-readout normalized LIN phase slopes.
        beta: Per-readout normalized PAR phase slopes.
        gamma: Per-readout constant phase offsets.
        nlin: Target logical LIN dimension.
        npar: Target logical PAR dimension.

    Returns:
        Unit-magnitude complex64 PSF with shape ``(RO_os, LIN, PAR)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if alpha.shape != beta.shape or alpha.shape != gamma.shape:
        raise ValueError("PSF phase-plane coefficient vectors must have matching lengths.")
    lin_norm = (np.arange(nlin, dtype=np.float64) - nlin / 2.0) / nlin
    par_norm = (np.arange(npar, dtype=np.float64) - npar / 2.0) / npar
    phase = (
        alpha[:, None, None] * lin_norm[None, :, None]
        + beta[:, None, None] * par_norm[None, None, :]
        + gamma[:, None, None]
    )
    return np.exp(1j * phase).astype(np.complex64)


def psf_identity_metrics(
    source_psf: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    *,
    readout_chunk: int = 8,
) -> dict[str, float]:
    """Measure phase-plane regeneration accuracy on the source grid.

    Args:
        source_psf: Reference complex PSF.
        alpha: Extracted LIN phase slopes.
        beta: Extracted PAR phase slopes.
        gamma: Extracted constant phase offsets.
        readout_chunk: Number of readout samples evaluated per block.

    Returns:
        Relative complex error, maximum error, and phase RMS metrics.
    """
    nro, nlin, npar = source_psf.shape
    error_squared = 0.0
    source_squared = 0.0
    maximum_error = 0.0
    phase_squared = 0.0
    count = 0
    for start in range(0, nro, readout_chunk):
        stop = min(start + readout_chunk, nro)
        reference = np.asarray(source_psf[start:stop], dtype=np.complex64)
        regenerated = evaluate_psf_phase_planes(
            alpha[start:stop], beta[start:stop], gamma[start:stop], nlin, npar
        )
        difference = regenerated - reference
        error_squared += float(np.vdot(difference, difference).real)
        source_squared += float(np.vdot(reference, reference).real)
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
        phase_residual = np.angle(regenerated * np.conj(reference))
        phase_squared += float(np.sum(phase_residual**2))
        count += int(phase_residual.size)
    return {
        "relative_complex_l2": float(np.sqrt(error_squared / source_squared)),
        "maximum_complex_error": maximum_error,
        "phase_residual_rms_rad": float(np.sqrt(phase_squared / count)),
    }


def centered_fftn(
    array: np.ndarray, *, axes: Sequence[int], inverse: bool = False, workers: int = 1
) -> np.ndarray:
    """Apply the project-standard centered orthonormal FFT.

    Args:
        array: Input array to transform.
        axes: Axes transformed in one operation.
        inverse: Use inverse transforms when true.
        workers: Maximum SciPy FFT workers.

    Returns:
        Centered complex64 transformed array.
    """
    axes = tuple(int(axis) for axis in axes)
    shifted = np.fft.ifftshift(np.asarray(array), axes=axes)
    transform = fft.ifftn if inverse else fft.fftn
    result = transform(shifted, axes=axes, norm="ortho", workers=workers)
    return np.fft.fftshift(result, axes=axes).astype(np.complex64, copy=False)


def apply_wave_forward(
    no_wave_kspace: np.ndarray,
    psf: np.ndarray,
    *,
    readout_oversampled: int,
    fft_workers: int = 1,
) -> np.ndarray:
    """Apply crop-first Wave synthesis for one coil at fixed FOV.

    Args:
        no_wave_kspace: Logical no-Wave k-space with shape ``(RO, LIN, PAR)``.
        psf: Target-grid Wave PSF with oversampled readout.
        readout_oversampled: Extended Wave readout dimension.
        fft_workers: Maximum SciPy FFT workers.

    Returns:
        Complex64 Wave-encoded k-space with shape ``(RO_os, LIN, PAR)``.
    """
    kspace = np.asarray(no_wave_kspace, dtype=np.complex64)
    if kspace.ndim != 3 or psf.shape[1:] != kspace.shape[1:]:
        raise ValueError(
            f"No-wave k-space {kspace.shape} and PSF {psf.shape} have incompatible PE grids."
        )
    image = centered_fftn(kspace, axes=(0, 1, 2), inverse=True, workers=fft_workers)
    if readout_oversampled < image.shape[0] or (readout_oversampled - image.shape[0]) % 2:
        raise ValueError("Oversampled readout must preserve a centered integer embedding.")
    extended = np.zeros((readout_oversampled, *image.shape[1:]), dtype=np.complex64)
    start = readout_oversampled // 2 - image.shape[0] // 2
    extended[start : start + image.shape[0]] = image
    hybrid = centered_fftn(extended, axes=(0,), workers=fft_workers)
    hybrid *= np.asarray(psf, dtype=np.complex64)
    return centered_fftn(hybrid, axes=(1, 2), workers=fft_workers)


def build_wave_options(
    regularizer: str,
    lambda_value: float | None,
    *,
    block_size: int,
    iterations: int,
    tolerance: float,
    maximum_eigenvalue: float | None,
) -> list[str]:
    """Build validated BART Wave options with mandatory GPU execution.

    Args:
        regularizer: One of ``none``, ``wavelet``, or ``llr``.
        lambda_value: Regularization weight, or zero/``None`` for no penalty.
        block_size: LLR block size when that regularizer is selected.
        iterations: Positive FISTA iteration count.
        tolerance: Positive stopping tolerance.
        maximum_eigenvalue: Optional positive operator eigenvalue bound.

    Returns:
        Command arguments suitable for appending to ``bart wave``.
    """
    if regularizer not in {"none", "wavelet", "llr"}:
        raise ValueError(f"Unsupported regularizer: {regularizer}.")
    if iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Iterations and tolerance must be positive.")
    options: list[str] = []
    if regularizer == "none":
        if lambda_value not in (None, 0, 0.0):
            raise ValueError("Lambda must be absent or zero without regularization.")
    else:
        if lambda_value is None or not np.isfinite(lambda_value) or lambda_value < 0:
            raise ValueError("BART Wave FISTA requires a nonnegative finite lambda.")
        if regularizer == "wavelet":
            options.append("-w")
        else:
            if block_size < 1:
                raise ValueError("LLR block size must be positive.")
            options.extend(["-l", "-v", "-b", str(block_size)])
        options.extend(["-f", "-r", f"{lambda_value:.12g}"])
    options.extend(["-i", str(iterations), "-t", f"{tolerance:.12g}"])
    if maximum_eigenvalue is not None:
        if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0:
            raise ValueError("Maximum eigenvalue must be positive and finite.")
        options.extend(["-e", f"{maximum_eigenvalue:.12g}"])
    options.append("-g")
    return options
