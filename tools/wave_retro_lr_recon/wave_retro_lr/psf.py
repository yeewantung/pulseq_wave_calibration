"""Calibrated Wave PSF evaluation on native and retrospective PE grids."""

from __future__ import annotations

import numpy as np


def evaluate_calibrated_psf(
    delta_k_lin: np.ndarray,
    delta_k_par: np.ndarray,
    coefficient_a: np.ndarray,
    coefficient_b: np.ndarray,
    coefficient_c: np.ndarray,
    *,
    nlin: int,
    npar: int,
    lin_sign: int = -1,
    par_sign: int = -1,
) -> np.ndarray:
    """Evaluate trajectory plus calibrated ``a,b,c`` phase on one PE grid.

    ``delta_k_lin`` and ``delta_k_par`` are the sequence-derived Wave
    displacements in index units. The calibrated plane coefficients use the
    same normalized LIN/PAR coordinates as the pinned Wave-MPRAGE code.

    Args:
        delta_k_lin: Sequence-derived per-readout LIN displacement.
        delta_k_par: Sequence-derived per-readout PAR displacement.
        coefficient_a: Calibrated per-readout LIN phase correction.
        coefficient_b: Calibrated per-readout PAR phase correction.
        coefficient_c: Calibrated per-readout constant phase correction.
        nlin: Target logical LIN dimension.
        npar: Target logical PAR dimension.
        lin_sign: Upstream Wave-MPRAGE LIN trajectory sign.
        par_sign: Upstream Wave-MPRAGE PAR trajectory sign.

    Returns:
        Unit-magnitude complex64 PSF with shape ``(RO_os, LIN, PAR)``.
    """

    vectors = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (delta_k_lin, delta_k_par, coefficient_a, coefficient_b, coefficient_c)
    ]
    if len({value.size for value in vectors}) != 1 or vectors[0].size < 1:
        raise ValueError(
            "Wave trajectory and calibrated coefficient vectors must have one common length."
        )
    if lin_sign not in (-1, 1) or par_sign not in (-1, 1):
        raise ValueError("Wave trajectory signs must be -1 or +1.")
    if nlin < 1 or npar < 1:
        raise ValueError("PSF PE dimensions must be positive.")
    if not all(np.isfinite(value).all() for value in vectors):
        raise ValueError("Wave trajectory and calibrated coefficients must be finite.")

    delta_lin, delta_par, a_fit, b_fit, c_fit = vectors
    lin_norm = (np.arange(nlin, dtype=np.float64) - nlin / 2.0) / nlin
    par_norm = (np.arange(npar, dtype=np.float64) - npar / 2.0) / npar
    phase = (
        (-lin_sign * 2.0 * np.pi * delta_lin + a_fit)[:, None, None]
        * lin_norm[None, :, None]
        + (-par_sign * 2.0 * np.pi * delta_par + b_fit)[:, None, None]
        * par_norm[None, None, :]
        + c_fit[:, None, None]
    )
    return np.exp(1j * phase).astype(np.complex64)
