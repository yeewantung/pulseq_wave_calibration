"""Consistency navigator for B0 fluctuations in Wave-CAIPI GRE.

Implements the data-consistency method of Meineke and Nielsen (Magn Reson
Med 2019;81:3046-3055) with the Wave-CAIPI forward model:

1. reconstruct an image from the current multi-coil data (CG-SENSE with the
   Wave operator);
2. synthesize every acquired line from that image (``A x``);
3. compute, per line and coil, the least-squares complex ratio between
   measured and synthesized data over the whole Wave readout (the paper's
   readout averaging, weighted by the synthesized energy);
4. smooth numerator and denominator with a Gaussian along acquisition time;
5. take the phase of the smoothed ratio and remove it from the data;
6. iterate; the total correction is the sum of the per-cycle phases.

A B0 fluctuation that is constant during one readout multiplies the whole
Wave line by one phase, so the correction itself is model independent and the
corrected k-space can be reconstructed by BART ``wave`` unchanged.

Estimation modes:
    ``global``: one phase per line shared by all coils (spatially uniform
        fluctuation; most robust).
    ``coil_affine``: per-coil phases constrained per line to an affine
        function of each coil's signal centre of mass (uniform term plus
        linear gradients along the selected axes). This uses the spatial
        interpretation of the paper's Fig. 2 as a constraint.
    ``per_coil``: one free phase per coil and line, as in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .acquisition import LineTiming
from .recon import cg_sense
from .wave_operator import WaveOperator

MODES = ("global", "coil_affine", "per_coil")
AXES = ("x", "y", "z")  # RO, LIN, PAR
MAX_DENSE_KERNEL_LINES = 40000


@dataclass
class NavigatorSettings:
    """Parameters of the consistency navigator.

    Attributes:
        mode: One of :data:`MODES`.
        outer_iterations: Maximum reconstruct/synthesize/correct cycles.
        cg_iterations: CG iterations per cycle (warm started).
        tikhonov: Tikhonov weight of the CG-SENSE reconstruction.
        temporal_fwhm_s: Full width at half maximum of the Gaussian smoothing
            along acquisition time.
        tolerance_rad: Stop when the energy-weighted RMS phase update falls
            below this value.
        echo_time_s: Echo time of the estimated echo (Hz conversion only).
        relaxation: Factor applied to each phase update. Values above one
            compensate for the part of the inconsistency that the
            least-squares image absorbs in every cycle. Keep 1 for
            ``per_coil``, whose free per-coil phases are noisy.
        affine_axes: Gradient axes of ``coil_affine`` (``x`` = RO, ``y`` =
            LIN, ``z`` = PAR).
        voxel_mm: Image voxel size used for coil positions in millimetres.
    """

    mode: str = "coil_affine"
    outer_iterations: int = 20
    cg_iterations: int = 30
    tikhonov: float = 0.0
    temporal_fwhm_s: float = 0.3
    tolerance_rad: float = 5e-4
    echo_time_s: float = 0.02
    relaxation: float = 2.0
    affine_axes: Sequence[str] = ("z",)
    voxel_mm: Sequence[float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If any value is out of range.
        """
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}.")
        if self.outer_iterations < 1 or self.cg_iterations < 1:
            raise ValueError("Iteration counts must be positive.")
        if self.temporal_fwhm_s < 0 or self.tikhonov < 0 or self.echo_time_s <= 0:
            raise ValueError("FWHM and Tikhonov must be nonnegative and TE positive.")
        if not 0 < self.relaxation <= 4:
            raise ValueError("relaxation must lie in (0, 4].")
        if any(axis not in AXES for axis in self.affine_axes) or len(set(self.affine_axes)) != len(self.affine_axes):
            raise ValueError(f"affine_axes must be distinct members of {AXES}.")
        if len(self.voxel_mm) != 3 or any(value <= 0 for value in self.voxel_mm):
            raise ValueError("voxel_mm must contain three positive values.")


@dataclass
class NavigatorResult:
    """Output of :func:`run_consistency_navigator`.

    Attributes:
        phase: Total phase removed from each coil and line ``(C, T)`` [rad].
        weights: Synthesized line energy ``(C, T)`` of the final cycle.
        coil_positions_mm: Coil signal centres of mass ``(C, 3)`` relative to
            the grid centre (RO, LIN, PAR).
        image: Reconstruction from the corrected data.
        corrected: Corrected k-space ``(C, Xe, Y, Z)`` on the device.
        history: One record per cycle.
    """

    phase: np.ndarray
    weights: np.ndarray
    coil_positions_mm: np.ndarray
    image: torch.Tensor
    corrected: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)

    def global_hz(self, echo_time_s: float) -> np.ndarray:
        """Energy-weighted coil average of the total phase per line, in Hz.

        Args:
            echo_time_s: Echo time of the estimated echo.

        Returns:
            ``(T,)`` frequency offset.
        """
        combined = np.sum(self.weights * np.exp(1j * self.phase), axis=0)
        return np.angle(combined) / (2 * math.pi * echo_time_s)

    def affine_traces(self, echo_time_s: float, axes: Sequence[str] = AXES) -> dict[str, np.ndarray]:
        """Fit the total per-coil phase to ``uniform + gradient . r_coil`` per line.

        Args:
            echo_time_s: Echo time of the estimated echo.
            axes: Gradient axes to fit.

        Returns:
            ``uniform_hz`` at the energy-weighted coil centre and
            ``gradient_<axis>_hz_per_mm`` traces, each ``(T,)``.
        """
        design, centre = _affine_design(self.coil_positions_mm, self.weights.sum(1), axes)
        coefficients = _weighted_affine_fit(
            torch.as_tensor(self.phase, dtype=torch.float64),
            torch.as_tensor(self.weights, dtype=torch.float64),
            torch.as_tensor(design, dtype=torch.float64),
        ).numpy()
        scale = 1.0 / (2 * math.pi * echo_time_s)
        traces = {"uniform_hz": coefficients[:, 0] * scale}
        for index, axis in enumerate(axes):
            traces[f"gradient_{axis}_hz_per_mm"] = coefficients[:, index + 1] * scale
        traces["reference_position_mm"] = centre
        return traces


def coil_centres_of_mass(maps: torch.Tensor, image: torch.Tensor, voxel_mm: Sequence[float]) -> np.ndarray:
    """Signal-weighted centre of mass of every coil (paper Eq. for r_j^com).

    Args:
        maps: ``(C, X, Y, Z)`` coil sensitivities.
        image: ``(X, Y, Z)`` reconstruction.
        voxel_mm: Voxel size in millimetres.

    Returns:
        ``(C, 3)`` positions in millimetres relative to the grid centre.
    """
    shape = image.shape
    axes = [
        (torch.arange(n, device=image.device, dtype=torch.float32) - n // 2) * float(step)
        for n, step in zip(shape, voxel_mm, strict=True)
    ]
    positions = []
    for coil in range(maps.shape[0]):
        energy = torch.abs(maps[coil] * image) ** 2
        total = energy.sum()
        positions.append(
            [
                float((energy.sum(dim=(1, 2)) * axes[0]).sum() / total),
                float((energy.sum(dim=(0, 2)) * axes[1]).sum() / total),
                float((energy.sum(dim=(0, 1)) * axes[2]).sum() / total),
            ]
        )
    return np.asarray(positions)


def _affine_design(
    positions_mm: np.ndarray, coil_energy: np.ndarray, axes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix ``[1, r - r_ref]`` for the selected axes.

    Args:
        positions_mm: ``(C, 3)`` coil centres.
        coil_energy: ``(C,)`` weights defining the reference position.
        axes: Gradient axes.

    Returns:
        ``(C, 1 + len(axes))`` design and the ``(3,)`` reference position.
    """
    weights = np.asarray(coil_energy, dtype=np.float64)
    centre = (positions_mm * weights[:, None]).sum(0) / weights.sum()
    columns = [np.ones(positions_mm.shape[0])]
    for axis in axes:
        index = AXES.index(axis)
        columns.append(positions_mm[:, index] - centre[index])
    return np.stack(columns, axis=1), centre


def _weighted_affine_fit(phase: torch.Tensor, weights: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Per-line weighted least squares ``phase[:, t] ~ design @ beta_t``.

    Args:
        phase: ``(C, T)`` phases (small, unwrapped relative to a reference).
        weights: ``(C, T)`` nonnegative weights.
        design: ``(C, P)`` design matrix.

    Returns:
        ``(T, P)`` coefficients.
    """
    weighted = weights.T[:, :, None] * design[None]  # (T, C, P)
    normal = torch.einsum("tcp,cq->tpq", weighted, design)
    rhs = torch.einsum("tcp,ct->tp", weighted, phase)
    ridge = 1e-9 * torch.eye(design.shape[1], dtype=design.dtype, device=design.device)
    trace = normal.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-30)
    return torch.linalg.solve(normal + ridge * trace[:, None, None], rhs)


def gaussian_time_kernel(times_s: np.ndarray, fwhm_s: float, device: torch.device) -> torch.Tensor:
    """Build a symmetric Gaussian smoothing matrix over acquisition times.

    Args:
        times_s: Sorted line times ``(T,)``.
        fwhm_s: Full width at half maximum; zero returns the identity.
        device: Target device.

    Returns:
        Dense ``(T, T)`` float32 kernel (unnormalized; normalization happens
        through the smoothed weights).

    Raises:
        ValueError: If the dense kernel would be too large.
    """
    count = int(times_s.size)
    if count > MAX_DENSE_KERNEL_LINES:
        raise ValueError(f"{count} lines exceed the dense-kernel limit {MAX_DENSE_KERNEL_LINES}.")
    times = torch.as_tensor(times_s, dtype=torch.float64, device=device)
    if fwhm_s == 0:
        return torch.eye(count, dtype=torch.float32, device=device)
    sigma = fwhm_s / (2 * math.sqrt(2 * math.log(2)))
    difference = times[:, None] - times[None, :]
    kernel = torch.exp(-0.5 * (difference / sigma) ** 2)
    kernel[torch.abs(difference) > 4 * sigma] = 0
    return kernel.to(torch.float32)


def line_statistics(
    measured: torch.Tensor,
    synthesized: torch.Tensor,
    lin: torch.Tensor,
    par: torch.Tensor,
    coil_chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-line, per-coil least-squares ratio terms over the Wave readout.

    Args:
        measured: Measured k-space ``(C, Xe, Y, Z)``.
        synthesized: Model k-space ``(C, Xe, Y, Z)``.
        lin: LIN index per line ``(T,)``.
        par: PAR index per line ``(T,)``.
        coil_chunk: Coils processed per batch.

    Returns:
        Numerator ``sum_x s conj(s')`` and denominator ``sum_x |s'|^2``,
        both ``(C, T)``.
    """
    numerator = []
    denominator = []
    for start in range(0, measured.shape[0], coil_chunk):
        stop = min(start + coil_chunk, measured.shape[0])
        model = synthesized[start:stop]
        numerator.append(torch.sum(measured[start:stop] * model.conj(), dim=1)[:, lin, par])
        denominator.append(torch.sum(model.real**2 + model.imag**2, dim=1)[:, lin, par])
    return torch.cat(numerator), torch.cat(denominator)


def estimate_line_phase(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    kernel: torch.Tensor,
    mode: str,
    design: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase of the time-smoothed measured/synthesized ratio.

    Args:
        numerator: ``(C, T)`` complex ratio numerators.
        denominator: ``(C, T)`` real ratio denominators (line energies).
        kernel: ``(T, T)`` smoothing kernel.
        mode: One of :data:`MODES`.
        design: ``(C, P)`` affine design, required for ``coil_affine``.

    Returns:
        ``(C, T)`` phase in radians.

    Raises:
        ValueError: If ``coil_affine`` is requested without a design.
    """
    smooth_numerator = torch.complex(numerator.real @ kernel, numerator.imag @ kernel)
    smooth_denominator = (denominator @ kernel).clamp_min(1e-30)
    shared = torch.angle(smooth_numerator.sum(0) / smooth_denominator.sum(0))
    if mode == "global":
        return shared[None].expand_as(denominator).clone()
    per_coil = torch.angle(smooth_numerator / smooth_denominator)
    if mode == "per_coil":
        return per_coil
    if design is None:
        raise ValueError("coil_affine needs a design matrix.")
    # Coil phases relative to the shared phase are small and need no unwrapping.
    relative = torch.angle(torch.exp(1j * (per_coil - shared[None]).to(torch.float64)))
    coefficients = _weighted_affine_fit(relative, smooth_denominator.to(torch.float64), design)
    return (shared[None].to(torch.float64) + design @ coefficients.T).to(numerator.real.dtype)


def apply_line_phase(
    kspace: torch.Tensor, phase: torch.Tensor, lin: torch.Tensor, par: torch.Tensor
) -> torch.Tensor:
    """Multiply each acquired line by ``exp(-i phase)`` in place.

    Args:
        kspace: ``(C, Xe, Y, Z)`` k-space, modified in place.
        phase: ``(C, T)`` phase in radians.
        lin: LIN index per line ``(T,)``.
        par: PAR index per line ``(T,)``.

    Returns:
        The modified k-space tensor.
    """
    grid = torch.zeros((kspace.shape[0], kspace.shape[2], kspace.shape[3]), dtype=phase.dtype, device=phase.device)
    grid[:, lin, par] = phase
    kspace *= torch.exp(-1j * grid.to(kspace.real.dtype))[:, None]
    return kspace


def run_consistency_navigator(
    operator: WaveOperator,
    measured: torch.Tensor,
    timing: LineTiming,
    settings: NavigatorSettings,
    *,
    log: Callable[[str], None] = print,
) -> NavigatorResult:
    """Estimate and remove line-wise B0-fluctuation phase by data consistency.

    Args:
        operator: Wave operator whose line mask matches the data.
        measured: Measured k-space ``(C, Xe, Y, Z)`` on the operator's device.
        timing: Acquisition order of the acquired lines.
        settings: Navigator parameters.
        log: Progress callback.

    Returns:
        Total phase, weights, coil positions, final image, corrected k-space,
        and the per-cycle history.

    Raises:
        ValueError: If the timing does not match the operator's line mask.
    """
    timing.check_mask(operator.line_mask.cpu().numpy())
    device = measured.device
    lin = torch.as_tensor(timing.lin, device=device)
    par = torch.as_tensor(timing.par, device=device)
    kernel = gaussian_time_kernel(timing.time_s, settings.temporal_fwhm_s, device)
    corrected = measured.clone()
    total = torch.zeros((operator.coils, timing.count), dtype=torch.float32, device=device)
    measured_norm = torch.linalg.vector_norm(measured)
    image, _ = cg_sense(operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov)
    positions = coil_centres_of_mass(operator.maps, image, settings.voxel_mm)
    design = None
    if settings.mode == "coil_affine":
        coil_energy = torch.sum(torch.abs(operator.maps * image) ** 2, dim=(1, 2, 3)).cpu().numpy()
        design_np, _ = _affine_design(positions, coil_energy, settings.affine_axes)
        design = torch.as_tensor(design_np, dtype=torch.float64, device=device)
    weights = torch.ones_like(total)
    history: list[dict[str, float]] = []
    for cycle in range(settings.outer_iterations):
        if cycle > 0:
            image, _ = cg_sense(
                operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov, initial=image
            )
        synthesized = operator.forward(image)
        residual = float(torch.linalg.vector_norm(corrected - synthesized) / measured_norm)
        numerator, weights = line_statistics(corrected, synthesized, lin, par)
        del synthesized
        update = estimate_line_phase(numerator, weights, kernel, settings.mode, design)
        # A phase common to all coils and constant in time only rotates the image.
        common = torch.angle(torch.sum(weights * torch.exp(1j * update.to(torch.float64))))
        update = (update - common.to(update.dtype)) * settings.relaxation
        update_rms = float(torch.sqrt(torch.sum(weights * update**2) / torch.sum(weights)))
        apply_line_phase(corrected, update, lin, par)
        total += update
        record = {
            "cycle": cycle + 1,
            "data_residual_before_update": residual,
            "weighted_rms_update_rad": update_rms,
            "max_abs_update_rad": float(torch.max(torch.abs(update))),
        }
        history.append(record)
        log(
            f"cycle {cycle + 1}: data residual {residual:.5f}, RMS update {update_rms:.2e} rad, "
            f"max {record['max_abs_update_rad']:.3f} rad"
        )
        if update_rms < settings.tolerance_rad:
            break
    image, _ = cg_sense(operator, corrected, iterations=settings.cg_iterations, tikhonov=settings.tikhonov, initial=image)
    final_residual = float(torch.linalg.vector_norm(corrected - operator.forward(image)) / measured_norm)
    history.append({"cycle": len(history) + 1, "data_residual_before_update": final_residual})
    log(f"final data residual {final_residual:.5f}")
    return NavigatorResult(
        phase=total.cpu().numpy(),
        weights=weights.cpu().numpy(),
        coil_positions_mm=positions,
        image=image,
        corrected=corrected,
        history=history,
    )
