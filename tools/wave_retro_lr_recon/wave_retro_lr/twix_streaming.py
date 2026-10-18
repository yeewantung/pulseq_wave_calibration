"""Read validated TWIX IMAGE streams one partition and echo at a time."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bart_io import sha256_file


def image_reading_identity() -> dict[str, Any]:
    """Return the IMAGE reader policy and source hash for new manifests.

    Returns:
        JSON-compatible provenance; this does not identify REFSCAN/PSF code.
    """
    return {
        "method": "partition-streaming-coil-compression",
        "version": 1,
        "partitions_per_read": 1,
        "echoes_per_read": 1,
        "physical_image_materialized": False,
        "physical_support_checked_before_compression": True,
        "source_sha256": sha256_file(Path(__file__)),
    }


def _positive_shape(values: Sequence[int]) -> tuple[int, ...]:
    """Validate IMAGE target dimensions without silently rounding values.

    Args:
        values: RO/LIN/PAR dimensions and an optional ECHO dimension.

    Returns:
        Positive integer dimensions.

    Raises:
        ValueError: If rank, type, or sizes are invalid.
    """
    shape = tuple(values)
    if len(shape) not in (3, 4) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
        for value in shape
    ):
        raise ValueError("IMAGE output_shape must contain three or four positive integers.")
    return tuple(int(value) for value in shape)


def iter_compressed_partitions(
    stream: Any,
    compression_matrix: np.ndarray,
    *,
    output_shape: Sequence[int],
    sampling_mask: np.ndarray,
) -> Iterator[tuple[int, slice, int, Any]]:
    """Validate and compress one physical-coil PAR/ECHO block per read.

    Args:
        stream: IMAGE object from the already validated mapVBVD measurement.
        compression_matrix: Physical-to-output coil matrix, including whitening
            when enabled. The matrix is applied directly, without conjugation.
        output_shape: Logical RO/LIN/PAR grid, optionally followed by ECHO.
        sampling_mask: Validated boolean LIN/PAR mask shared across echoes.

    Yields:
        Echo index, logical LIN slice, logical PAR index, and a CPU complex64
        tensor shaped RO/LIN/output-coil. Only acquired partitions are read.

    Raises:
        ValueError: For unsupported dimensions, geometry, mask, basis, nonfinite
            values, or physical samples outside the validated sampling mask.
        RuntimeError: If explicit mapVBVD selection returns a different layout.

    Side Effects:
        Reads IMAGE payloads. Temporarily disables squeezing and readout OS
        removal; restores both flags when exhausted, closed, or rejected.
        Never requests the full IMAGE payload or alters the source file.
    """
    import torch

    target = _positive_shape(output_shape)
    ro, nlin, npar = target[:3]
    echoes = target[3] if len(target) == 4 else 1
    mask = np.asarray(sampling_mask)
    if mask.shape != (nlin, npar) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("IMAGE sampling mask must be a nonempty boolean LIN/PAR grid.")
    basis = np.asarray(compression_matrix, dtype=np.complex64)
    if basis.ndim != 2 or min(basis.shape) < 1 or not np.isfinite(basis).all():
        raise ValueError("IMAGE compression matrix must be finite and two-dimensional.")
    dims = list(stream.dataDims)
    sizes_array = np.asarray(stream.fullSize)
    if (
        len(set(dims)) != len(dims)
        or sizes_array.shape != (len(dims),)
        or not np.isfinite(sizes_array).all()
        or np.any(sizes_array < 1)
        or np.any(sizes_array != np.floor(sizes_array))
        or not {"Col", "Cha", "Lin", "Par", "Eco"}.issubset(dims)
    ):
        raise ValueError("IMAGE stream has invalid or missing named dimensions.")
    sizes = dict(zip(dims, sizes_array.astype(int), strict=True))
    # mapVBVD expands selections along averaged axes. Reject that mode rather
    # than accidentally reading multiple partitions or echoes in one request.
    averaged = np.asarray(getattr(stream, "average_dim", np.zeros(len(dims), bool)))
    if averaged.shape != sizes_array.shape or np.any(averaged & (sizes_array > 1)):
        raise ValueError("IMAGE streaming does not support averaging non-singleton dimensions.")
    # Extra acquisitions must not disappear through selecting index zero.
    if any(size != 1 for name, size in sizes.items()
           if name not in {"Col", "Cha", "Lin", "Par", "Eco"}):
        raise ValueError("IMAGE stream has unsupported non-singleton dimensions.")
    if sizes["Col"] != ro or sizes["Cha"] != basis.shape[0] or sizes["Eco"] != echoes:
        raise ValueError("IMAGE readout, physical-coil count, or echo count disagrees with the target.")
    # Preserve mapVBVD's compact-grid origin, including a nonzero LIN residue.
    offsets = (stream.skipLin, stream.skipPar)
    if any(not np.isfinite(value) or value < 0 or value != int(value) for value in offsets):
        raise ValueError("IMAGE skipLin/skipPar must be nonnegative integers.")
    lin_start, par_start = (int(value) for value in offsets)
    lin_stop, par_stop = lin_start + sizes["Lin"], par_start + sizes["Par"]
    if lin_stop > nlin or par_stop > npar:
        raise ValueError("IMAGE stored extents and offsets exceed the logical grid.")
    covered = np.zeros_like(mask)
    covered[lin_start:lin_stop, par_start:par_stop] = True
    if np.any(mask & ~covered):
        raise ValueError("IMAGE stored extents do not cover the validated sampling mask.")
    lin_slice = slice(lin_start, lin_stop)
    partitions = np.flatnonzero(mask.any(axis=0))
    basis_tensor = torch.from_numpy(np.ascontiguousarray(basis))
    old_remove_os, old_squeeze = stream.flagRemoveOS, stream.squeeze
    try:
        stream.flagRemoveOS = False
        stream.squeeze = False
        for echo in range(echoes):
            for par in partitions:
                selections = {"Col": slice(None), "Cha": slice(None),
                              "Lin": slice(None), "Par": int(par) - par_start,
                              "Eco": echo}
                raw = np.asarray(stream[tuple(selections.get(name, 0) for name in dims)])
                expected = tuple(sizes[name] if name in {"Col", "Cha", "Lin"} else 1
                                 for name in dims)
                if raw.shape != expected:
                    raise RuntimeError(f"Unexpected IMAGE partition shape {raw.shape}; expected {expected}.")
                axes = [dims.index(name) for name in ("Col", "Lin", "Cha")]
                axes += [axis for axis in range(len(dims)) if axis not in axes]
                physical = np.ascontiguousarray(
                    raw.transpose(axes).reshape(ro, sizes["Lin"], sizes["Cha"]),
                    dtype=np.complex64,
                )
                if not np.isfinite(physical).all():
                    raise ValueError("TWIX physical IMAGE partition contains nonfinite values.")
                if np.any(physical[:, ~mask[lin_slice, par], :] != 0):
                    raise ValueError("TWIX physical IMAGE contains nonzero samples outside the sampling mask.")
                # Check before projection so invalid physical samples cannot
                # vanish in the nullspace of a PCA or ROVir basis.
                projected = torch.from_numpy(physical) @ basis_tensor
                if not torch.isfinite(projected).all():
                    raise ValueError("Compressed IMAGE partition contains nonfinite values.")
                del physical, raw
                yield echo, lin_slice, int(par), projected
                del projected
    finally:
        stream.flagRemoveOS, stream.squeeze = old_remove_os, old_squeeze


def load_compressed_image(
    stream: Any,
    compression_matrix: np.ndarray,
    *,
    output_shape: Sequence[int],
    sampling_mask: np.ndarray,
) -> Any:
    """Assemble a compressed IMAGE grid without a full physical-coil volume.

    Args:
        stream: Validated mapVBVD IMAGE stream.
        compression_matrix: Physical-to-output coil matrix, optionally whitened.
        output_shape: RO/LIN/PAR dimensions, optionally followed by ECHO.
        sampling_mask: Validated boolean LIN/PAR mask.

    Returns:
        CPU complex64 tensor with output coils appended to output_shape.

    Raises:
        ValueError, RuntimeError: Propagated partition validation failures.

    Side Effects:
        Reads the source and prints partition progress; writes no files.
    """
    import torch

    target = _positive_shape(output_shape)
    blocks = iter_compressed_partitions(
        stream, compression_matrix, output_shape=target, sampling_mask=sampling_mask
    )
    output = None
    try:
        for position, (echo, lines, par, block) in enumerate(blocks, start=1):
            if output is None:
                # All metadata and the first physical block are validated
                # before allocating the final virtual-coil grid.
                output = torch.zeros((*target, block.shape[-1]), dtype=torch.complex64)
            if len(target) == 3:
                output[:, lines, par, :] = block
            else:
                output[:, lines, par, echo, :] = block
            if position == 1 or position % 12 == 0:
                print(f"Read/compressed IMAGE partition {position} (echo {echo + 1}, PAR {par}).", flush=True)
            del block
    finally:
        blocks.close()
    return output
