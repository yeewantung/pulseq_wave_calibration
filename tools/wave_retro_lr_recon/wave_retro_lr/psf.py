"""Calibrated Wave PSF evaluation on native and retrospective PE grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PSF_COEFFICIENT_PLOT_NAME = "PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png"


def write_psf_coefficient_plot(
    coefficient_a: np.ndarray,
    coefficient_b: np.ndarray,
    coefficient_c: np.ndarray,
    output_path: str | Path,
    *,
    processing: str,
    fit_kx_range: tuple[int, int] | None = None,
) -> Path:
    """Plot the processed PSF phase coefficients against readout index.

    Args:
        coefficient_a: Processed per-readout LIN phase coefficient.
        coefficient_b: Processed per-readout PAR phase coefficient.
        coefficient_c: Processed per-readout constant phase coefficient.
        output_path: Destination PNG path.
        processing: Coefficient-processing mode recorded in the figure title.
        fit_kx_range: Optional half-open sine-line fit interval ``[min, max)``.

    Returns:
        The resolved path of the written PNG diagnostic.

    Raises:
        ValueError: If coefficient vectors or the optional fit interval are
            inconsistent with the readout grid.
    """
    vectors = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (coefficient_a, coefficient_b, coefficient_c)
    )
    if not vectors[0].size or len({value.size for value in vectors}) != 1:
        raise ValueError("PSF coefficient vectors must have one common nonzero length.")
    if not all(np.isfinite(value).all() for value in vectors):
        raise ValueError("PSF coefficient vectors must contain only finite values.")
    if fit_kx_range is not None:
        lower, upper = (int(value) for value in fit_kx_range)
        if not (0 <= lower < upper <= vectors[0].size):
            raise ValueError("PSF plot fit range must lie within the readout grid.")

    # Use the noninteractive canvas directly so preparation remains safe on
    # headless reconstruction hosts without changing the global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    kx = np.arange(vectors[0].size)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for axis, name, vector, color in zip(axes, ("a", "b", "c"), vectors, colors):
        axis.plot(kx, vector, color=color, linewidth=1.5)
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axvline(vectors[0].size // 2, color="black", linewidth=0.8, linestyle=":")
        if fit_kx_range is not None:
            axis.axvspan(lower, upper, color="tab:purple", alpha=0.12)
        axis.set_ylabel(f"{name} (rad)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("kx (oversampled readout sample index)")
    mode = str(processing).strip().lower()
    interval = "" if fit_kx_range is None else f"; fit range [{lower}, {upper})"
    figure.suptitle(
        "Native R3x1 PSF coefficients used for reconstruction\n"
        f"processing: {mode}{interval}"
    )
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    figure.savefig(temporary, dpi=160, format="png")
    temporary.replace(destination)
    return destination


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
