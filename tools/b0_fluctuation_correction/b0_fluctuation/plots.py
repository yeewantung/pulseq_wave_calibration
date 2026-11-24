"""Diagnostic figures for B0-fluctuation estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _pyplot():
    """Import matplotlib with a non-interactive backend.

    Returns:
        ``matplotlib.pyplot`` module.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_traces(
    time_s: np.ndarray,
    traces: Mapping[str, np.ndarray],
    output: Path,
    *,
    title: str,
    lin: np.ndarray | None = None,
) -> Path:
    """Plot fluctuation traces against time and their amplitude spectra.

    Args:
        time_s: Line times.
        traces: Named traces (Hz or Hz/mm) sharing ``time_s``.
        output: PNG path.
        title: Figure title.
        lin: Optional LIN index per line; LIN changes are marked.

    Returns:
        The written path.
    """
    plt = _pyplot()
    names = list(traces)
    fig, axes = plt.subplots(len(names), 2, figsize=(15, 2.8 * len(names)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    step = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    for row, name in enumerate(names):
        values = np.asarray(traces[name])
        axes[row, 0].plot(time_s, values, lw=0.7)
        if lin is not None:
            for change in np.flatnonzero(np.diff(lin) != 0):
                axes[row, 0].axvline(time_s[change + 1], color="0.85", lw=0.4, zorder=0)
        axes[row, 0].set_ylabel(name)
        axes[row, 0].grid(alpha=0.3)
        centred = values - values.mean()
        spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size)))
        frequency = np.fft.rfftfreq(centred.size, d=step)
        keep = frequency <= 3.0
        axes[row, 1].plot(frequency[keep], spectrum[keep], lw=0.8)
        axes[row, 1].set_xlabel("frequency [Hz]")
        axes[row, 1].grid(alpha=0.3)
    axes[-1, 0].set_xlabel("acquisition time [s] (grey lines: LIN changes)" if lin is not None else "acquisition time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=90)
    plt.close(fig)
    return output


def plot_kspace_phase(lin: np.ndarray, par: np.ndarray, values: np.ndarray, shape: Sequence[int], output: Path, title: str) -> Path:
    """Show a per-line quantity on the (LIN, PAR) grid.

    Args:
        lin: LIN index per line.
        par: PAR index per line.
        values: Quantity per line.
        shape: ``(LIN, PAR)`` grid size.
        output: PNG path.
        title: Figure title.

    Returns:
        The written path.
    """
    plt = _pyplot()
    grid = np.full(tuple(shape), np.nan)
    grid[lin, par] = values
    fig, ax = plt.subplots(figsize=(7, 5))
    limit = np.nanpercentile(np.abs(grid), 99) or 1.0
    shown = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    fig.colorbar(shown, ax=ax)
    ax.set_xlabel("LIN (outer loop)")
    ax.set_ylabel("PAR (inner loop)")
    ax.set_title(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=90)
    plt.close(fig)
    return output


def plot_image_comparison(
    images: Mapping[str, np.ndarray], output: Path, *, title: str, planes: Sequence[tuple[str, int]] | None = None,
    voxel_mm: Sequence[float] = (1.0, 1.0, 1.0),
) -> Path:
    """Compare magnitude images in native orientation, plus differences to the first.

    Args:
        images: Named complex or magnitude volumes ``(X, Y, Z)``; the first is
            the reference.
        output: PNG path.
        title: Figure title.
        planes: ``("axial", PAR)`` or ``("sagittal", RO)`` planes.
        voxel_mm: Voxel size for aspect ratios.

    Returns:
        The written path.
    """
    plt = _pyplot()
    names = list(images)
    magnitudes = [np.abs(images[name]) for name in names]
    shape = magnitudes[0].shape
    if planes is None:
        planes = [("axial", int(f * shape[2])) for f in (0.35, 0.5, 0.65, 0.8)] + [("sagittal", shape[0] // 2)]
    reference = magnitudes[0]
    support = reference > 0.1 * np.percentile(reference, 99.5)
    # Match each image to the reference by a least-squares magnitude scale in the support.
    scaled = [m * (np.sum(m[support] * reference[support]) / max(np.sum(m[support] ** 2), 1e-30)) for m in magnitudes]
    vmax = float(np.percentile(reference[support], 99.5))
    rows = len(names) + len(names) - 1
    fig, axes = plt.subplots(rows, len(planes), figsize=(3.8 * len(planes), 3.6 * rows), squeeze=False)

    def show(ax, volume, plane, index, low, high, cmap):
        """Draw one plane (axial: LIN 0 = anterior at top)."""
        if plane == "axial":
            ax.imshow(volume[:, :, index].T, cmap=cmap, vmin=low, vmax=high, origin="upper",
                      aspect=voxel_mm[1] / voxel_mm[0])
        else:
            ax.imshow(volume[index].T, cmap=cmap, vmin=low, vmax=high, origin="lower",
                      aspect=voxel_mm[2] / voxel_mm[1])
        ax.set_xticks([])
        ax.set_yticks([])

    for column, (plane, index) in enumerate(planes):
        for row, name in enumerate(names):
            show(axes[row, column], scaled[row], plane, index, 0, vmax, "gray")
            axes[row, column].set_title(f"{name}: {plane} {index}", fontsize=9)
        for offset, name in enumerate(names[1:]):
            row = len(names) + offset
            show(axes[row, column], scaled[offset + 1] - reference, plane, index, -0.2 * vmax, 0.2 * vmax, "RdBu_r")
            axes[row, column].set_title(f"{name} - {names[0]} (+-20%)", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=75)
    plt.close(fig)
    return output
