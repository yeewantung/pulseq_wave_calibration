"""Select Wave calibration SET/LIN/PAR labels before reading REFSCAN data."""

from pathlib import Path
from typing import Any

import numpy as np

from .bart_io import sha256_file


def refscan_reading_identity() -> dict[str, Any]:
    """Return the selective REFSCAN policy and its exact implementation hash.

    Returns:
        JSON-compatible provenance for newly prepared calibration artifacts.
    """
    return {
        "method": "explicit-set-lin-par-selection",
        "version": 1,
        "sets_per_read": 1,
        "partitions_per_read": 1,
        "dense_all_sets_materialized": False,
        "source_sha256": sha256_file(Path(__file__)),
    }


def open_refscan(filename: str | Path) -> Any:
    """Open the pinned measurement's REFSCAN metadata without loading samples.

    Args:
        filename: Siemens TWIX input path.

    Returns:
        mapVBVD REFSCAN stream from measurement 1 for multi-RAID files, or 0
        for a single measurement, matching the pinned Wave loader convention.

    Raises:
        ValueError: If the selected measurement has no REFSCAN stream.
    """
    import mapvbvd

    parsed = mapvbvd.mapVBVD(str(filename), quiet=True)
    measurements = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    measurement = measurements[1 if len(measurements) > 1 else 0]
    stream = measurement.get("refscan") if hasattr(measurement, "get") else getattr(measurement, "refscan", None)
    if stream is None:
        raise ValueError("Selected TWIX measurement has no REFSCAN stream.")
    return stream


def _dimensions(stream: Any) -> tuple[list[str], dict[str, int]]:
    """Validate stored REFSCAN dimensions and return their named sizes.

    Args:
        stream: mapVBVD REFSCAN stream with its original calibration labels.

    Returns:
        Axis names and positive integer raw sizes, without reading samples.

    Raises:
        ValueError: For unsupported axes, averaging, offsets, or invalid sizes.
    """
    dims = list(stream.dataDims)
    raw = np.asarray(stream.fullSize)
    required = {"Col", "Cha", "Lin", "Par", "Set"}
    if (len(set(dims)) != len(dims) or not required.issubset(dims)
            or raw.shape != (len(dims),) or not np.isfinite(raw).all()
            or np.any(raw < 1) or np.any(raw != np.floor(raw))):
        raise ValueError("REFSCAN has invalid or missing named dimensions.")
    sizes = {name: int(value) for name, value in zip(dims, raw, strict=True)}
    if any(size != 1 for name, size in sizes.items() if name not in required):
        raise ValueError("REFSCAN has unsupported non-singleton dimensions.")
    averaged = np.asarray(getattr(stream, "average_dim", np.zeros(len(dims), bool)))
    if averaged.shape != raw.shape or np.any(averaged & (raw > 1)):
        raise ValueError("REFSCAN selection does not support averaging non-singleton dimensions.")
    # Wave projection coordinates and leading ACS indices are raw labels.
    # A shifted origin must not silently change their Fourier/geometry meaning.
    if stream.skipLin != 0 or stream.skipPar != 0:
        raise ValueError("Integrated REFSCAN requires zero LIN/PAR origins.")
    return dims, sizes


def refscan_shape(stream: Any, *, ncalib: int, nacs: int, nsets: int = 5) -> tuple[int, ...]:
    """Check the complete integrated calibration layout using metadata only.

    Args:
        stream: mapVBVD REFSCAN stream.
        ncalib: Required projection width in both PE dimensions.
        nacs: Required square ACS width.
        nsets: Minimum number of integrated calibration SETs.

    Returns:
        Logical ``(RO, LIN, PAR, SET, physical_coil)`` stored shape.

    Raises:
        ValueError: If the layout cannot contain the requested calibration.
    """
    _, sizes = _dimensions(stream)
    for value in (ncalib, nacs, nsets):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError("REFSCAN calibration dimensions must be positive integers.")
    if min(sizes["Lin"], sizes["Par"]) < max(ncalib, nacs) or sizes["Set"] < nsets:
        raise ValueError("Integrated REFSCAN dimensions are smaller than the required projection/ACS/SET layout.")
    return tuple(sizes[name] for name in ("Col", "Lin", "Par", "Set", "Cha"))


def read_refscan_set(stream: Any, set_index: int, *, lin_count: int, par_count: int) -> Any:
    """Read only one SET's leading LIN/PAR region, one PAR at a time.

    Args:
        stream: Validated integrated mapVBVD REFSCAN stream.
        set_index: Raw SET label; -1 selects the final SET for native MPRAGE ACS.
        lin_count: Number of leading raw LIN indices to retain.
        par_count: Number of leading raw PAR indices to retain.

    Returns:
        Finite CPU complex64 tensor shaped ``(RO, lin_count, par_count, coil)``.
        Singleton axes are retained. PAR remains Fourier encoded, not raw SLI.

    Raises:
        ValueError: For invalid labels, dimensions, or nonfinite samples.
        RuntimeError: For an unexpected backend response shape.

    Side Effects:
        Reads the selected payload only. Temporarily disables readout OS removal
        and squeezing, restoring both flags on success or failure.
    """
    import torch

    dims, sizes = _dimensions(stream)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
           for value in (set_index, lin_count, par_count)):
        raise ValueError("REFSCAN SET and region bounds must be integers.")
    selected_set = set_index + sizes["Set"] if set_index < 0 else set_index
    if (not 0 <= selected_set < sizes["Set"] or not 1 <= lin_count <= sizes["Lin"]
            or not 1 <= par_count <= sizes["Par"]):
        raise ValueError("Requested REFSCAN SET/LIN/PAR region is outside stored bounds.")
    if hasattr(stream, "Set") and not np.any(np.asarray(stream.Set) == selected_set):
        raise ValueError("Requested REFSCAN SET has no acquired records.")
    output = torch.empty((sizes["Col"], lin_count, par_count, sizes["Cha"]), dtype=torch.complex64)
    old_remove_os, old_squeeze = stream.flagRemoveOS, stream.squeeze
    try:
        stream.flagRemoveOS = False
        stream.squeeze = False
        for par in range(par_count):
            selections = {"Col": slice(None), "Cha": slice(None),
                          "Lin": slice(0, lin_count), "Par": par, "Set": int(selected_set)}
            raw = np.asarray(stream[tuple(selections.get(name, 0) for name in dims)])
            expected = tuple(sizes[name] if name in {"Col", "Cha"} else lin_count if name == "Lin" else 1
                             for name in dims)
            if raw.shape != expected:
                raise RuntimeError(f"Unexpected selected REFSCAN shape {raw.shape}; expected {expected}.")
            axes = [dims.index(name) for name in ("Col", "Lin", "Cha")]
            axes += [axis for axis in range(len(dims)) if axis not in axes]
            block = np.ascontiguousarray(raw.transpose(axes).reshape(sizes["Col"], lin_count, sizes["Cha"]),
                                         dtype=np.complex64)
            if not np.isfinite(block).all():
                raise ValueError("Selected REFSCAN samples contain nonfinite values.")
            output[:, :, par, :] = torch.from_numpy(block)
            del block, raw
    finally:
        stream.flagRemoveOS, stream.squeeze = old_remove_os, old_squeeze
    return output
