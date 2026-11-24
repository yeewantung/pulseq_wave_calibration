"""Acquisition timing of phase-encoding lines (the navigator's time axis)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TWIX_TIMESTAMP_SECONDS = 2.5e-3


@dataclass(frozen=True)
class LineTiming:
    """Phase-encoding lines of one echo in acquisition order.

    Attributes:
        lin: LIN index of each line (BART array index).
        par: PAR index of each line (BART array index).
        time_s: Acquisition time of each line relative to the first line.
        source: Human-readable origin of the timing.
    """

    lin: np.ndarray
    par: np.ndarray
    time_s: np.ndarray
    source: str

    def __post_init__(self) -> None:
        """Validate one-to-one, time-ordered line records.

        Raises:
            ValueError: If arrays disagree in length, times decrease, or a
                line appears twice.
        """
        if not (self.lin.shape == self.par.shape == self.time_s.shape) or self.lin.ndim != 1:
            raise ValueError("LIN, PAR, and time arrays must be one-dimensional and equal in length.")
        if np.any(np.diff(self.time_s) < 0):
            raise ValueError("Line times must be nondecreasing.")
        pairs = np.stack([self.lin, self.par], axis=1)
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("Each (LIN, PAR) line must appear once.")

    @property
    def count(self) -> int:
        """Number of lines.

        Returns:
            Line count.
        """
        return int(self.lin.size)

    def check_mask(self, line_mask: np.ndarray) -> None:
        """Require the timing to cover exactly the acquired lines of a mask.

        Args:
            line_mask: Boolean ``(LIN, PAR)`` acquired-line pattern.

        Raises:
            ValueError: If the sets of lines differ.
        """
        timed = np.zeros(line_mask.shape, dtype=bool)
        timed[self.lin, self.par] = True
        if not np.array_equal(timed, np.asarray(line_mask, dtype=bool)):
            missing = int(np.count_nonzero(line_mask & ~timed))
            extra = int(np.count_nonzero(timed & ~line_mask))
            raise ValueError(f"Line timing disagrees with the k-space mask ({missing} missing, {extra} extra).")

    def subset(self, line_mask: np.ndarray) -> "LineTiming":
        """Keep only the lines acquired in a (retrospective) mask.

        Args:
            line_mask: Boolean ``(LIN, PAR)`` pattern, a subset of these lines.

        Returns:
            Time-ordered timing of the retained lines.

        Raises:
            ValueError: If the mask contains lines absent from this timing.
        """
        mask = np.asarray(line_mask, dtype=bool)
        keep = mask[self.lin, self.par]
        if int(keep.sum()) != int(mask.sum()):
            raise ValueError("The mask contains lines that this timing does not cover.")
        return LineTiming(self.lin[keep], self.par[keep], self.time_s[keep], f"{self.source}; subset of {int(keep.sum())} lines")

    def to_json(self) -> dict[str, object]:
        """Summarize the timing for provenance records.

        Returns:
            JSON-compatible summary.
        """
        steps = np.diff(self.time_s)
        return {
            "source": self.source,
            "line_count": self.count,
            "duration_s": float(self.time_s[-1] - self.time_s[0]) if self.count else 0.0,
            "median_line_interval_ms": float(np.median(steps) * 1e3) if steps.size else None,
        }


def timing_from_twix(twix_path: str | Path, *, echo: int = 0) -> LineTiming:
    """Read per-line acquisition times from TWIX image-stream MDH timestamps.

    Args:
        twix_path: Siemens TWIX file.
        echo: Zero-based echo whose lines are returned.

    Returns:
        Lines of the requested echo sorted by MDH timestamp.

    Raises:
        ValueError: If the image stream has no lines for the echo.

    Side Effects:
        Parses the TWIX headers and MDHs (no k-space samples are read).
    """
    import mapvbvd

    parsed = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement = parsed[-1] if isinstance(parsed, list) else parsed
    stream = measurement.image
    lin = np.asarray(stream.Lin).astype(np.int64)
    par = np.asarray(stream.Par).astype(np.int64)
    eco = np.asarray(stream.Eco).astype(np.int64)
    stamp = np.asarray(stream.timestamp, dtype=np.float64) * TWIX_TIMESTAMP_SECONDS
    keep = eco == int(echo)
    if not np.any(keep):
        raise ValueError(f"TWIX image stream has no lines for echo {echo}.")
    order = np.argsort(stamp[keep], kind="stable")
    times = stamp[keep][order]
    return LineTiming(lin[keep][order], par[keep][order], times - times[0], f"TWIX MDH timestamps ({twix_path})")


def timing_from_loop_order(
    line_mask: np.ndarray, repetition_time_s: float, *, outer: str = "lin"
) -> LineTiming:
    """Construct line times from a nested phase-encoding loop.

    Args:
        line_mask: Boolean ``(LIN, PAR)`` acquired-line pattern.
        repetition_time_s: Time between consecutive lines.
        outer: ``lin`` when LIN is the outer loop (the Wave-GRE default) or
            ``par`` when PAR is the outer loop.

    Returns:
        Lines in loop order with uniform spacing.

    Raises:
        ValueError: If ``outer`` is unknown.
    """
    lin, par = np.nonzero(np.asarray(line_mask, dtype=bool))
    if outer == "lin":
        order = np.lexsort((par, lin))
    elif outer == "par":
        order = np.lexsort((lin, par))
    else:
        raise ValueError("outer must be 'lin' or 'par'.")
    times = np.arange(order.size, dtype=np.float64) * float(repetition_time_s)
    return LineTiming(lin[order], par[order], times, f"nested loop order (outer {outer}, TR {repetition_time_s} s)")
