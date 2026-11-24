"""Model-based consistency navigator with a spatially linear fluctuating field.

This extends the consistency navigator by placing the fluctuation inside the
Wave forward model instead of approximating it by per-coil phases. For line
``l`` acquired at time ``t_l`` the field is

    dB0(r, t_l) = f0(t_l) + g(t_l) . (r - r_ref),

so the line is encoded from ``x(r) exp(i [phi0_l + beta_l . (r - r_ref)])`` with
``phi0 = 2 pi TE f0`` and ``beta = 2 pi TE g``. The uniform term is an exact
line phase. The gradient term is linearized,

    y_l ~ exp(i phi0_l) [A_l x + i sum_p beta_lp A_l(r_p x)],

which is accurate while ``|beta . r|`` stays well below one radian (a 1 Hz
change across the head at TE = 20 ms is 0.13 rad).

Each cycle reconstructs ``x`` from the currently corrected data, then solves
for parameter increments per line by Gauss-Newton: the real normal equations
``Re(S^H S) d = Im(S^H r)`` with ``S = [A_l x, A_l(r_1 x), ...]`` and the
residual ``r``, accumulated over all coils and Wave readout samples and then
smoothed along acquisition time. The corrected data handed to BART are

    y_corr = exp(-i Phi0) y - i sum_p B_p A(r_p x_hat),

where the second term removes the modelled gradient contribution using the
final image. Its error is second order (gradient times image error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .acquisition import LineTiming
from .navigator import gaussian_time_kernel
from .recon import cg_sense
from .wave_operator import WaveOperator

AXES = ("x", "y", "z")  # RO, LIN, PAR


@dataclass
class FieldModelSettings:
    """Parameters of the linear-field navigator.

    Attributes:
        axes: Gradient axes (``x`` = RO, ``y`` = LIN, ``z`` = PAR); empty for a
            uniform field only.
        outer_iterations: Maximum cycles.
        cg_iterations: CG iterations per cycle (warm started).
        tikhonov: Tikhonov weight of the CG-SENSE reconstruction.
        temporal_fwhm_s: Gaussian smoothing FWHM along acquisition time.
        tolerance_rad: Stop when the RMS uniform-phase update falls below this.
        relaxation: Factor applied to each Gauss-Newton increment.
        echo_time_s: Echo time of the estimated echo.
        voxel_mm: Image voxel size in millimetres.
        ridge: Relative Tikhonov term of the per-line normal equations.
        gradient_fwhm_s: Extra Gaussian smoothing FWHM applied to the
            gradient increments, which have lower SNR than the uniform term;
            ``None`` applies no extra smoothing.
    """

    axes: Sequence[str] = ("x", "y", "z")
    outer_iterations: int = 20
    cg_iterations: int = 30
    tikhonov: float = 0.0
    temporal_fwhm_s: float = 0.3
    tolerance_rad: float = 5e-4
    relaxation: float = 2.0
    echo_time_s: float = 0.02
    voxel_mm: Sequence[float] = (1.0, 1.0, 1.0)
    ridge: float = 1e-6
    gradient_fwhm_s: float | None = 0.6

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If any value is out of range.
        """
        if any(axis not in AXES for axis in self.axes) or len(set(self.axes)) != len(self.axes):
            raise ValueError(f"axes must be distinct members of {AXES}.")
        if self.outer_iterations < 1 or self.cg_iterations < 1:
            raise ValueError("Iteration counts must be positive.")
        if self.temporal_fwhm_s < 0 or self.tikhonov < 0 or self.echo_time_s <= 0:
            raise ValueError("FWHM and Tikhonov must be nonnegative and TE positive.")
        if not 0 < self.relaxation <= 4:
            raise ValueError("relaxation must lie in (0, 4].")
        if len(self.voxel_mm) != 3 or any(value <= 0 for value in self.voxel_mm):
            raise ValueError("voxel_mm must contain three positive values.")
        if self.gradient_fwhm_s is not None and self.gradient_fwhm_s < 0:
            raise ValueError("gradient_fwhm_s must be nonnegative.")


@dataclass
class FieldModelResult:
    """Output of :func:`run_linear_field_navigator`.

    Attributes:
        parameters: ``(P, T)`` totals: row 0 uniform phase [rad], rows 1.. the
            gradient phases [rad/mm] for ``axes``.
        axes: Gradient axes of rows 1...
        reference_mm: ``(3,)`` reference position relative to the grid centre.
        line_energy: ``(T,)`` synthesized energy per line (final cycle).
        image: Reconstruction from the final corrected data.
        corrected: Corrected k-space ``(C, Xe, Y, Z)`` on the device.
        history: Per-cycle records.
    """

    parameters: np.ndarray
    axes: tuple[str, ...]
    reference_mm: np.ndarray
    line_energy: np.ndarray
    image: torch.Tensor
    corrected: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)

    def traces(self, echo_time_s: float) -> dict[str, np.ndarray]:
        """Convert the parameters to Hz and Hz/mm traces.

        Args:
            echo_time_s: Echo time of the estimated echo.

        Returns:
            ``uniform_hz`` and ``gradient_<axis>_hz_per_mm`` arrays ``(T,)``.
        """
        scale = 1.0 / (2 * math.pi * echo_time_s)
        traces = {"uniform_hz": self.parameters[0] * scale}
        for index, axis in enumerate(self.axes):
            traces[f"gradient_{axis}_hz_per_mm"] = self.parameters[index + 1] * scale
        return traces


def coordinate_images(
    shape: Sequence[int], voxel_mm: Sequence[float], reference_mm: Sequence[float], axes: Sequence[str], device: torch.device
) -> list[torch.Tensor]:
    """Centred coordinate maps in millimetres for the selected axes.

    Args:
        shape: Image dimensions ``(X, Y, Z)``.
        voxel_mm: Voxel size.
        reference_mm: Reference position relative to the grid centre.
        axes: Axes to return.
        device: Target device.

    Returns:
        One real ``(X, Y, Z)`` tensor per axis.
    """
    maps = []
    for axis in axes:
        index = AXES.index(axis)
        coordinate = (torch.arange(shape[index], device=device, dtype=torch.float32) - shape[index] // 2) * float(
            voxel_mm[index]
        ) - float(reference_mm[index])
        view = [1, 1, 1]
        view[index] = shape[index]
        maps.append(coordinate.reshape(view).expand(*shape))
    return maps


def signal_centre_mm(image: torch.Tensor, voxel_mm: Sequence[float]) -> np.ndarray:
    """Energy-weighted centre of the image relative to the grid centre.

    Args:
        image: ``(X, Y, Z)`` image.
        voxel_mm: Voxel size.

    Returns:
        ``(3,)`` position in millimetres.
    """
    energy = torch.abs(image) ** 2
    total = energy.sum()
    centre = []
    for index, step in enumerate(voxel_mm):
        n = image.shape[index]
        coordinate = (torch.arange(n, device=image.device, dtype=torch.float32) - n // 2) * float(step)
        others = tuple(d for d in range(3) if d != index)
        centre.append(float((energy.sum(dim=others) * coordinate).sum() / total))
    return np.asarray(centre)


def modelled_gradient_kspace(
    operator: WaveOperator, image: torch.Tensor, coordinates: Sequence[torch.Tensor], gradients: torch.Tensor,
    lin: torch.Tensor, par: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``i sum_p B_p A(r_p x)`` on the acquired lines.

    Args:
        operator: Wave operator.
        image: ``(X, Y, Z)`` image.
        coordinates: Coordinate maps ``r_p`` [mm].
        gradients: ``(P-1, T)`` gradient phases [rad/mm] per line.
        lin: LIN index per line.
        par: PAR index per line.

    Returns:
        ``(C, Xe, Y, Z)`` k-space contribution (zero on unacquired lines).
    """
    total = torch.zeros(operator.kspace_shape, dtype=image.dtype, device=image.device)
    for coordinate, gradient in zip(coordinates, gradients, strict=True):
        grid = torch.zeros(operator.line_mask.shape, dtype=image.real.dtype, device=image.device)
        grid[lin, par] = gradient.to(grid.dtype)
        total += 1j * grid[None, None] * operator.forward(image * coordinate)
    return total


def line_normal_equations(
    operator: WaveOperator,
    corrected: torch.Tensor,
    image: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    lin: torch.Tensor,
    par: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Accumulate per-line Gauss-Newton normal equations over coils and readout.

    Args:
        operator: Wave operator.
        corrected: Currently corrected k-space ``(C, Xe, Y, Z)``.
        image: Current image.
        coordinates: Coordinate maps for the gradient terms.
        lin: LIN index per line.
        par: PAR index per line.

    Returns:
        ``(P, P, T)`` matrices ``Re(S^H S)``, ``(P, T)`` vectors
        ``Im(S^H r)``, and the squared residual norm.
    """
    basis = [torch.ones_like(image.real)] + list(coordinates)
    count = len(basis)
    gram = torch.zeros((count, count, lin.numel()), dtype=torch.float64, device=image.device)
    rhs = torch.zeros((count, lin.numel()), dtype=torch.float64, device=image.device)
    residual_energy = 0.0
    for start in range(0, operator.coils, operator.coil_chunk):
        stop = min(start + operator.coil_chunk, operator.coils)
        coil_images = operator.maps[start:stop] * image
        synthesized = [operator.coil_forward(coil_images * b) * operator.mask_float for b in basis]
        residual = corrected[start:stop] - synthesized[0]
        residual_energy += float(torch.sum(residual.real**2 + residual.imag**2))
        for p in range(count):
            rhs[p] += torch.sum(synthesized[p].conj() * residual, dim=(0, 1))[lin, par].imag.to(torch.float64)
            for q in range(p, count):
                value = torch.sum(synthesized[p].conj() * synthesized[q], dim=(0, 1))[lin, par].real.to(torch.float64)
                gram[p, q] += value
                if q != p:
                    gram[q, p] += value
        del synthesized, residual
    return gram, rhs, residual_energy


def solve_smoothed(gram: torch.Tensor, rhs: torch.Tensor, kernel: torch.Tensor, ridge: float) -> torch.Tensor:
    """Smooth the normal equations along time and solve them per line.

    Args:
        gram: ``(P, P, T)`` matrices.
        rhs: ``(P, T)`` vectors.
        kernel: ``(T, T)`` smoothing kernel.
        ridge: Relative diagonal loading.

    Returns:
        ``(P, T)`` increments.
    """
    kernel64 = kernel.to(torch.float64)
    count, _, lines = gram.shape
    smooth_gram = (gram.reshape(count * count, lines) @ kernel64).reshape(count, count, lines).permute(2, 0, 1)
    smooth_rhs = (rhs @ kernel64).T
    scale = smooth_gram.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
    # Jacobi scaling equalizes the units of the uniform and gradient columns.
    inverse_root = 1.0 / torch.sqrt(scale)
    scaled = smooth_gram * inverse_root[:, :, None] * inverse_root[:, None, :]
    scaled = scaled + ridge * torch.eye(count, dtype=scaled.dtype, device=scaled.device)
    solution = torch.linalg.solve(scaled, (smooth_rhs * inverse_root)[..., None])[..., 0]
    return (solution * inverse_root).T


def run_linear_field_navigator(
    operator: WaveOperator,
    measured: torch.Tensor,
    timing: LineTiming,
    settings: FieldModelSettings,
    *,
    log: Callable[[str], None] = print,
) -> FieldModelResult:
    """Estimate uniform and linear-gradient B0 fluctuations per line.

    Args:
        operator: Wave operator whose line mask matches the data.
        measured: Measured k-space ``(C, Xe, Y, Z)`` on the device.
        timing: Acquisition order of the acquired lines.
        settings: Navigator parameters.
        log: Progress callback.

    Returns:
        Parameter traces, final image, corrected k-space, and history.

    Raises:
        ValueError: If the timing does not match the line mask.
    """
    timing.check_mask(operator.line_mask.cpu().numpy())
    device = measured.device
    lin = torch.as_tensor(timing.lin, device=device)
    par = torch.as_tensor(timing.par, device=device)
    kernel = gaussian_time_kernel(timing.time_s, settings.temporal_fwhm_s, device)
    gradient_kernel = (
        gaussian_time_kernel(timing.time_s, settings.gradient_fwhm_s, device).to(torch.float64)
        if settings.gradient_fwhm_s
        else None
    )
    measured_norm = float(torch.linalg.vector_norm(measured))
    axes = tuple(settings.axes)
    image, _ = cg_sense(operator, measured, iterations=settings.cg_iterations, tikhonov=settings.tikhonov)
    reference = signal_centre_mm(image, settings.voxel_mm)
    coordinates = coordinate_images(image.shape, settings.voxel_mm, reference, axes, device)
    parameters = torch.zeros((1 + len(axes), timing.count), dtype=torch.float64, device=device)
    corrected = measured.clone()
    history: list[dict[str, float]] = []

    def build_corrected(current_image: torch.Tensor) -> torch.Tensor:
        """Apply the current uniform phase and subtract modelled gradient terms."""
        grid = torch.zeros(operator.line_mask.shape, dtype=torch.float64, device=device)
        grid[lin, par] = parameters[0]
        data = measured * torch.exp(-1j * grid).to(measured.dtype)[None, None]
        if axes:
            data -= modelled_gradient_kspace(operator, current_image, coordinates, parameters[1:], lin, par)
        return data

    line_energy = torch.zeros(timing.count, dtype=torch.float64, device=device)
    for cycle in range(settings.outer_iterations):
        if cycle > 0:
            del corrected
            corrected = build_corrected(image)
            image, _ = cg_sense(
                operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov, initial=image
            )
        gram, rhs, residual_energy = line_normal_equations(operator, corrected, image, coordinates, lin, par)
        line_energy = gram[0, 0]
        increment = solve_smoothed(gram, rhs, kernel, settings.ridge) * settings.relaxation
        if gradient_kernel is not None and axes:
            # Energy-weighted extra smoothing of the noisier gradient increments.
            energy = gram[0, 0]
            increment[1:] = ((increment[1:] * energy) @ gradient_kernel) / (energy @ gradient_kernel).clamp_min(1e-30)
        # Time-constant terms are degenerate with the image: a constant uniform
        # phase rotates it and a constant gradient adds a linear phase ramp.
        weights = line_energy / line_energy.sum()
        increment -= torch.sum(weights * increment, dim=1, keepdim=True)
        parameters += increment
        update_rms = float(torch.sqrt(torch.sum(weights * increment[0] ** 2)))
        record = {
            "cycle": cycle + 1,
            "data_residual_before_update": math.sqrt(residual_energy) / measured_norm,
            "rms_uniform_update_rad": update_rms,
        }
        for index, axis in enumerate(axes):
            record[f"rms_gradient_{axis}_update_rad_per_mm"] = float(
                torch.sqrt(torch.sum(weights * increment[index + 1] ** 2))
            )
        history.append(record)
        log(
            f"cycle {cycle + 1}: data residual {record['data_residual_before_update']:.5f}, "
            f"RMS uniform update {update_rms:.2e} rad"
        )
        if update_rms < settings.tolerance_rad:
            break
    del corrected
    corrected = build_corrected(image)
    image, _ = cg_sense(operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov, initial=image)
    # Refresh the modelled gradient term with the final image.
    if axes:
        del corrected
        corrected = build_corrected(image)
        image, _ = cg_sense(operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov, initial=image)
    final_residual = float(torch.linalg.vector_norm(corrected - operator.forward(image))) / measured_norm
    history.append({"cycle": len(history) + 1, "data_residual_before_update": final_residual})
    log(f"final data residual {final_residual:.5f}")
    return FieldModelResult(
        parameters=parameters.cpu().numpy(),
        axes=axes,
        reference_mm=reference,
        line_energy=line_energy.cpu().numpy(),
        image=image,
        corrected=corrected,
        history=history,
    )
