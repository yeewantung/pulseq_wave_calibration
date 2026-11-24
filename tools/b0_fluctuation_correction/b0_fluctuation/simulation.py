"""Inject known B0 fluctuations into Wave k-space for validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .acquisition import LineTiming
from .wave_operator import WaveOperator


@dataclass(frozen=True)
class FluctuationModel:
    """Respiration-like field ``dB0(r, t) = theta(t) * (uniform_hz + gradient_hz_per_mm * z)``.

    Attributes:
        uniform_hz: Spatially uniform amplitude.
        gradient_hz_per_mm: Linear amplitude along PAR (``z`` from the grid centre).
        frequency_hz: Fundamental frequency of ``theta``.
        harmonic_fraction: Amplitude of an added second harmonic.
        drift_hz: Linear drift over the scan added to the uniform term.
    """

    uniform_hz: float = 1.5
    gradient_hz_per_mm: float = 0.01
    frequency_hz: float = 0.35
    harmonic_fraction: float = 0.3
    drift_hz: float = 0.0

    def theta(self, time_s: np.ndarray) -> np.ndarray:
        """Unit-amplitude temporal waveform.

        Args:
            time_s: Line times.

        Returns:
            ``theta(t)`` per line.
        """
        fundamental = np.sin(2 * math.pi * self.frequency_hz * time_s)
        harmonic = self.harmonic_fraction * np.sin(4 * math.pi * self.frequency_hz * time_s + 1.0)
        return fundamental + harmonic

    def uniform_trace_hz(self, time_s: np.ndarray) -> np.ndarray:
        """Uniform component in Hz per line (includes drift).

        Args:
            time_s: Line times.

        Returns:
            Hz per line.
        """
        span = max(float(time_s[-1] - time_s[0]), 1e-9)
        return self.uniform_hz * self.theta(time_s) + self.drift_hz * (time_s - time_s[0]) / span

    def gradient_trace_hz_per_mm(self, time_s: np.ndarray) -> np.ndarray:
        """Gradient component in Hz/mm per line.

        Args:
            time_s: Line times.

        Returns:
            Hz/mm per line.
        """
        return self.gradient_hz_per_mm * self.theta(time_s)


def inject_fluctuation(
    operator: WaveOperator,
    image: torch.Tensor,
    timing: LineTiming,
    model: FluctuationModel,
    *,
    echo_time_s: float,
    voxel_mm_par: float,
    gradient_levels: int = 41,
) -> torch.Tensor:
    """Synthesize Wave k-space whose lines carry a known B0 fluctuation.

    The uniform term is an exact per-line phase. The PAR-gradient term
    ``exp(i 2 pi TE g(t) z)`` changes the encoded image per line; lines are
    grouped into ``gradient_levels`` quantized gradient values and each group
    is synthesized exactly from the modulated image.

    Args:
        operator: Wave operator (its line mask defines the acquired lines).
        image: Ground-truth image ``(X, Y, Z)``.
        timing: Acquisition order of the lines.
        model: Fluctuation model.
        echo_time_s: Echo time.
        voxel_mm_par: PAR voxel size in millimetres.
        gradient_levels: Number of quantized gradient values.

    Returns:
        Corrupted k-space ``(C, Xe, Y, Z)``.
    """
    device = image.device
    lin = torch.as_tensor(timing.lin, device=device)
    par = torch.as_tensor(timing.par, device=device)
    gradient = model.gradient_trace_hz_per_mm(timing.time_s)
    nz = image.shape[2]
    z_mm = (torch.arange(nz, device=device, dtype=torch.float32) - nz // 2) * voxel_mm_par
    output = torch.zeros(operator.kspace_shape, dtype=image.dtype, device=device)
    if model.gradient_hz_per_mm == 0:
        levels = np.zeros(1)
        assignment = np.zeros(timing.count, dtype=np.int64)
    else:
        levels = np.linspace(gradient.min(), gradient.max(), gradient_levels)
        assignment = np.abs(gradient[:, None] - levels[None, :]).argmin(axis=1)
    # Each quantized gradient level is encoded exactly from the modulated image.
    for level_index, level in enumerate(levels):
        selected = np.flatnonzero(assignment == level_index)
        if selected.size == 0:
            continue
        modulation = torch.exp(1j * 2 * math.pi * echo_time_s * float(level) * z_mm).to(image.dtype)
        encoded = operator.forward(image * modulation[None, None, :])
        rows = lin[selected]
        cols = par[selected]
        output[:, :, rows, cols] = encoded[:, :, rows, cols]
        del encoded
    uniform = torch.as_tensor(2 * math.pi * echo_time_s * model.uniform_trace_hz(timing.time_s), device=device)
    grid = torch.zeros(output.shape[2:], dtype=torch.float64, device=device)
    grid[lin, par] = uniform
    output *= torch.exp(1j * grid).to(output.dtype)[None, None]
    return output


def add_noise(kspace: torch.Tensor, sigma: float, line_mask: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Add circular complex Gaussian noise on acquired lines.

    Args:
        kspace: ``(C, Xe, Y, Z)`` k-space, modified in place.
        sigma: Per-component standard deviation.
        line_mask: ``(Y, Z)`` acquired lines.
        seed: Random seed.

    Returns:
        The modified k-space.
    """
    generator = torch.Generator(device=kspace.device).manual_seed(seed)
    noise = torch.randn(kspace.shape, dtype=kspace.dtype, device=kspace.device, generator=generator)
    kspace += sigma * math.sqrt(2) * noise * line_mask.to(kspace.real.dtype)
    return kspace
