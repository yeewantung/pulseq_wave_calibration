"""Calibrated Wave PSF evaluation on native and retrospective PE grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PSF_COEFFICIENT_PLOT_NAME = "PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png"
PSF_COEFFICIENT_REJECTED_PLOT_NAME = "PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED.png"
PSF_COEFFICIENT_REJECTED_DIAGNOSTICS_NAME = (
    "PSF_COEFFICIENTS_AUTOMATIC_FIT_REJECTED.json"
)


def write_psf_coefficient_plot(
    coefficient_a: np.ndarray | None,
    coefficient_b: np.ndarray | None,
    coefficient_c: np.ndarray | None,
    output_path: str | Path,
    *,
    processing: str,
    fit_kx_range: tuple[int, int] | None = None,
    fit_range_selection: str | None = None,
    raw_coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    accepted_for_reconstruction: bool = True,
    curve_labels: tuple[str, str, str] | None = None,
    comparison_coefficients: tuple[
        np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]
    | None = None,
    comparison_labels: tuple[str, str, str] | None = None,
) -> Path:
    """Plot raw and processed PSF phase coefficients against readout index.

    Args:
        coefficient_a: Processed per-readout LIN phase coefficient, or
            ``None`` for a rejected candidate unavailable for plotting.
        coefficient_b: Processed per-readout PAR phase coefficient, following
            the same optional convention as ``coefficient_a``.
        coefficient_c: Processed per-readout constant phase coefficient,
            following the same optional convention as ``coefficient_a``.
        output_path: Destination PNG path.
        processing: Coefficient-processing mode recorded in the figure title.
        fit_kx_range: Optional half-open sine-line fit interval ``[min, max)``.
        fit_range_selection: Optional ``automatic`` or ``manual`` selection
            label recorded in the figure title.
        raw_coefficients: Optional original ``a``, ``b``, and ``c`` samples
            shown as scatter points beneath the processed curves.
        accepted_for_reconstruction: Whether the processed curves were
            accepted for PSF construction. Rejected plots are labeled
            explicitly and never claim that their candidate was used.
        curve_labels: Optional per-coefficient labels for accepted curves.
        comparison_coefficients: Optional rejected or alternative curves to
            overlay without identifying them as reconstruction inputs.
        comparison_labels: Optional per-coefficient comparison-curve labels.

    Returns:
        The resolved path of the written PNG diagnostic.

    Raises:
        ValueError: If coefficient vectors or the optional fit interval are
            inconsistent with the readout grid.
    """
    supplied = tuple(
        value is not None for value in (coefficient_a, coefficient_b, coefficient_c)
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "PSF coefficient curves must be supplied together or all omitted."
        )
    vectors = (
        None
        if not any(supplied)
        else tuple(
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in (coefficient_a, coefficient_b, coefficient_c)
        )
    )
    if vectors is not None:
        if not vectors[0].size or len({value.size for value in vectors}) != 1:
            raise ValueError(
                "PSF coefficient vectors must have one common nonzero length."
            )
        if not all(np.isfinite(value).all() for value in vectors):
            raise ValueError("PSF coefficient vectors must contain only finite values.")
    raw_vectors = None
    if raw_coefficients is not None:
        raw_vectors = tuple(
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in raw_coefficients
        )
        if len(raw_vectors) != 3:
            raise ValueError(
                "Raw PSF coefficient vectors must match the processed readout grid."
            )
        expected_size = raw_vectors[0].size if vectors is None else vectors[0].size
        if (
            not expected_size
            or any(value.size != expected_size for value in raw_vectors)
        ):
            raise ValueError(
                "Raw PSF coefficient vectors must match the processed readout grid."
            )
    if vectors is None and raw_vectors is None:
        raise ValueError(
            "A PSF plot requires processed curves or raw coefficient samples."
        )
    readout_size = raw_vectors[0].size if vectors is None else vectors[0].size
    if curve_labels is not None and len(curve_labels) != 3:
        raise ValueError("PSF curve labels must contain a, b, and c entries.")
    comparison_vectors = None
    if comparison_coefficients is not None:
        if len(comparison_coefficients) != 3:
            raise ValueError("PSF comparison curves must contain a, b, and c entries.")
        comparison_vectors = tuple(
            None
            if value is None
            else np.asarray(value, dtype=np.float64).reshape(-1)
            for value in comparison_coefficients
        )
        if any(
            value is not None
            and (value.size != readout_size or not np.isfinite(value).all())
            for value in comparison_vectors
        ):
            raise ValueError(
                "PSF comparison curves must match the finite readout grid."
            )
    if comparison_labels is not None and len(comparison_labels) != 3:
        raise ValueError("PSF comparison labels must contain a, b, and c entries.")
    if fit_kx_range is not None:
        lower, upper = (int(value) for value in fit_kx_range)
        if not (0 <= lower < upper <= readout_size):
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
    kx = np.arange(readout_size)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, (axis, name, color) in enumerate(
        zip(axes, ("a", "b", "c"), colors, strict=True)
    ):
        if raw_vectors is not None:
            axis.scatter(
                kx,
                raw_vectors[index],
                color=color,
                marker="o",
                s=9,
                alpha=0.45,
                linewidths=0,
                label="raw samples",
                zorder=2,
            )
        if comparison_vectors is not None and comparison_vectors[index] is not None:
            comparison_label = (
                comparison_labels[index]
                if comparison_labels is not None
                else "rejected candidate"
            )
            axis.plot(
                kx,
                comparison_vectors[index],
                color="0.25",
                linewidth=1.2,
                linestyle="--",
                label=comparison_label,
                zorder=2,
            )
        if vectors is not None:
            curve_label = (
                curve_labels[index]
                if curve_labels is not None
                else (
                    "sine-line fit"
                    if str(processing).strip().lower() == "sine-line"
                    else "processed"
                )
            )
            if not accepted_for_reconstruction:
                curve_label = f"rejected {curve_label}"
            axis.plot(
                kx,
                vectors[index],
                color=color,
                linewidth=1.5,
                label=curve_label,
                zorder=3,
            )
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.axvline(readout_size // 2, color="black", linewidth=0.8, linestyle=":")
        if fit_kx_range is not None:
            axis.axvspan(lower, upper, color="tab:purple", alpha=0.12)
        axis.set_ylabel(f"{name} (rad)")
        axis.grid(alpha=0.2)
        if raw_vectors is not None or vectors is not None:
            axis.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("kx (oversampled readout sample index)")
    mode = str(processing).strip().lower().replace("_", " ")
    selection = "" if fit_range_selection is None else f" ({fit_range_selection})"
    interval = "" if fit_kx_range is None else f"; fit range [{lower}, {upper})"
    status = (
        "Native R3x1 PSF coefficients used for reconstruction"
        if accepted_for_reconstruction
        else "REJECTED automatic PSF fit — not used for reconstruction"
    )
    figure.suptitle(f"{status}\nprocessing: {mode}{selection}{interval}")
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
