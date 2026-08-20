"""Authoritative product sampling-mask parsing and BART CFL export helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def retrospective_cartesian_mask(
    shape: tuple[int, int],
    *,
    accelerations: tuple[int, int],
    residues: tuple[int, int],
    fully_sampled_pe1_lines: list[int] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a PE1×PE2 lattice plus a PE1 ACS band spanning every partition."""
    if len(shape) != 2 or any(int(value) < 1 for value in shape):
        raise ValueError("Sampling-mask shape must contain two positive dimensions.")
    npe1, npe2 = (int(value) for value in shape)
    if len(accelerations) != 2 or any(int(value) < 1 for value in accelerations):
        raise ValueError("PE accelerations must contain two positive integers.")
    pe1_acceleration, pe2_acceleration = (int(value) for value in accelerations)
    if len(residues) != 2:
        raise ValueError("Sampling residues must contain one value per PE axis.")
    pe1_residue, pe2_residue = (int(value) for value in residues)
    if not 0 <= pe1_residue < pe1_acceleration:
        raise ValueError("PE1 residue must lie within its acceleration range.")
    if not 0 <= pe2_residue < pe2_acceleration:
        raise ValueError("PE2 residue must lie within its acceleration range.")

    acs_lines = np.asarray(fully_sampled_pe1_lines, dtype=np.int64)
    if acs_lines.ndim != 1 or acs_lines.size < 1:
        raise ValueError("The fully sampled PE1 ACS band must contain at least one line.")
    if np.unique(acs_lines).size != acs_lines.size:
        raise ValueError("The fully sampled PE1 ACS band contains duplicate lines.")
    if np.any((acs_lines < 0) | (acs_lines >= npe1)):
        raise ValueError("The fully sampled PE1 ACS band contains an out-of-range line.")
    acs_lines = np.sort(acs_lines)

    pe1_indices = np.arange(npe1, dtype=np.int64)
    pe2_indices = np.arange(npe2, dtype=np.int64)
    image_pe1 = pe1_indices[pe1_indices % pe1_acceleration == pe1_residue]
    image_pe2 = pe2_indices[pe2_indices % pe2_acceleration == pe2_residue]
    image_mask = np.zeros((npe1, npe2), dtype=bool)
    image_mask[np.ix_(image_pe1, image_pe2)] = True
    acs_mask = np.zeros_like(image_mask)
    acs_mask[acs_lines, :] = True
    mask = image_mask | acs_mask

    acquired_count = int(mask.sum())
    image_count = int(image_mask.sum())
    acs_count = int(acs_mask.sum())
    overlap_count = int((image_mask & acs_mask).sum())
    metadata = {
        "shape": [npe1, npe2],
        "mask_kind": "Cartesian image lattice union fully sampled PE1 ACS band",
        "pe1_acceleration": pe1_acceleration,
        "pe2_acceleration": pe2_acceleration,
        "nominal_acceleration": pe1_acceleration * pe2_acceleration,
        "pe1_residue": pe1_residue,
        "pe2_residue": pe2_residue,
        "image_pe1_lines": image_pe1.tolist(),
        "image_pe2_partitions": image_pe2.tolist(),
        "fully_sampled_pe1_lines": acs_lines.tolist(),
        "acs_covers_full_pe2": True,
        "image_coordinate_count": image_count,
        "acs_coordinate_count": acs_count,
        "image_acs_overlap_coordinate_count": overlap_count,
        "acquired_coordinate_count": acquired_count,
        "full_grid_coordinate_count": int(mask.size),
        "sampling_fraction": float(mask.mean()),
        "effective_acceleration_including_acs": float(mask.size / acquired_count),
        "unacquired_coordinate_count": int(mask.size - acquired_count),
    }
    return mask, metadata


def product_mask_from_report(report: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the exact PE1/PE2 union mask recorded from TWIX MDH counters."""
    try:
        sampling = report["twix"]["selected_measurement_sampling"]
        npe1 = int(sampling["matrix_pe1"])
        npe2 = int(sampling["matrix_pe2"])
        groups = sampling["merged_patterns_by_pe2"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Sampling report is missing the selected TWIX mask metadata.") from exc
    if npe1 < 1 or npe2 < 1 or not isinstance(groups, list) or not groups:
        raise ValueError("Sampling report contains invalid matrix dimensions or mask groups.")

    mask = np.zeros((npe1, npe2), dtype=bool)
    partition_seen = np.zeros(npe2, dtype=bool)
    for group in groups:
        try:
            partitions = np.asarray(group["partitions"], dtype=np.int64)
            lines = np.asarray(group["pe1_lines"], dtype=np.int64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A merged sampling-pattern group is malformed.") from exc
        if partitions.size < 1 or lines.size < 1:
            raise ValueError("Sampling-pattern groups must contain partitions and PE1 lines.")
        if np.unique(partitions).size != partitions.size or np.unique(lines).size != lines.size:
            raise ValueError("Sampling-pattern groups contain duplicate indices.")
        if np.any((partitions < 0) | (partitions >= npe2)):
            raise ValueError("Sampling report contains an out-of-range PE2 partition.")
        if np.any((lines < 0) | (lines >= npe1)):
            raise ValueError("Sampling report contains an out-of-range PE1 line.")
        if np.any(partition_seen[partitions]):
            raise ValueError("A PE2 partition appears in more than one sampling-pattern group.")
        partition_seen[partitions] = True
        mask[np.ix_(lines, partitions)] = True

    if not np.all(partition_seen):
        missing = np.flatnonzero(~partition_seen).tolist()
        raise ValueError(f"Sampling report omits PE2 partitions: {missing}.")
    expected_count = int(sampling["union_unique_coordinate_count"])
    if int(mask.sum()) != expected_count:
        raise ValueError(
            f"Mask has {int(mask.sum())} acquired coordinates; report records {expected_count}."
        )

    image_stride = int(sampling["image_inferred_pe1_stride"])
    image_residues = [int(value) for value in sampling["image_pe1_residues_for_inferred_stride"]]
    refscan_lines = [int(value) for value in sampling["refscan_unique_pe1_lines"]]
    metadata = {
        "shape": [npe1, npe2],
        "acquired_coordinate_count": expected_count,
        "full_grid_coordinate_count": int(mask.size),
        "sampling_fraction": float(mask.mean()),
        "acquired_pe1_lines_per_partition": [int(mask[:, index].sum()) for index in range(npe2)],
        "image_pe1_stride": image_stride,
        "image_pe1_residues": image_residues,
        "refscan_pe1_lines": refscan_lines,
        "refscan_covers_full_pe2": bool(sampling["refscan_covers_full_pe2"]),
        "source": "union of image and refscan TWIX MDH coordinates",
    }
    return mask, metadata


def load_product_mask(report_path: str | Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load a dataset-inspection JSON report and return its exact product mask."""
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    mask, metadata = product_mask_from_report(report)
    return mask, metadata, report


def write_masked_bart_kspace(
    full_wave_path: str | Path,
    mask: np.ndarray,
    output_base: str | Path,
    *,
    pe2_chunk: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply a PE mask while streaming a full Wave NPY into BART column-major CFL."""
    full_wave_path = Path(full_wave_path).expanduser().resolve()
    output_base = Path(output_base).expanduser().resolve().with_suffix("")
    if pe2_chunk < 1:
        raise ValueError("pe2_chunk must be positive.")
    source = np.load(full_wave_path, mmap_mode="r")
    if source.ndim != 4 or source.dtype != np.complex64:
        raise ValueError(f"Expected complex64 [RO,PE1,PE2,Ncc] input, got {source.shape} {source.dtype}.")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != source.shape[1:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match PE grid {source.shape[1:3]}.")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    header_path = output_base.with_suffix(".hdr")
    cfl_path = output_base.with_suffix(".cfl")
    partial_path = Path(str(cfl_path) + ".partial")
    existing = [path for path in (header_path, cfl_path, partial_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"BART output already exists: {', '.join(str(path) for path in existing)}")
    if overwrite:
        for path in existing:
            path.unlink()

    bart_shape = (*source.shape, 1)
    target = np.memmap(
        partial_path,
        mode="w+",
        dtype=np.complex64,
        shape=bart_shape,
        order="F",
    )
    squared_norm = 0.0
    all_finite = True
    zero = np.complex64(0.0)
    for coil in range(source.shape[3]):
        for start in range(0, source.shape[2], pe2_chunk):
            stop = min(start + pe2_chunk, source.shape[2])
            source_block = np.asarray(source[:, :, start:stop, coil])
            mask_block = mask[:, start:stop]
            # np.where copies acquired samples bitwise and creates exact complex zeros elsewhere.
            masked_block = np.where(mask_block[None, :, :], source_block, zero)
            all_finite &= bool(np.isfinite(masked_block).all())
            squared_norm += float(np.vdot(masked_block, masked_block).real)
            target[:, :, start:stop, coil, 0] = masked_block
    target.flush()
    del target
    os.replace(partial_path, cfl_path)
    header_path.write_text(
        "# Dimensions\n" + " ".join(str(int(value)) for value in bart_shape) + "\n",
        encoding="utf-8",
    )

    expected_bytes = int(np.prod(bart_shape, dtype=np.int64) * np.dtype(np.complex64).itemsize)
    if cfl_path.stat().st_size != expected_bytes:
        raise RuntimeError("BART Wave k-space byte count does not match its header dimensions.")
    if not all_finite:
        raise ValueError("Masked Wave k-space contains non-finite values.")

    acquired_coordinates = int(mask.sum())
    validation = validate_masked_bart_kspace(
        full_wave_path,
        mask,
        output_base,
        pe2_chunk=pe2_chunk,
    )
    return {
        "base": str(output_base),
        "header": str(header_path),
        "cfl": str(cfl_path),
        "shape": list(bart_shape),
        "dtype": "complex64",
        "order": "Fortran/BART column-major",
        "size_bytes": expected_bytes,
        "norm": float(np.sqrt(squared_norm)),
        "all_samples_finite": all_finite,
        "acquired_pe_coordinates": acquired_coordinates,
        "unacquired_pe_coordinates": int(mask.size - acquired_coordinates),
        "acquired_complex_samples": int(source.shape[0] * source.shape[3] * acquired_coordinates),
        "unacquired_complex_samples": int(
            source.shape[0] * source.shape[3] * (mask.size - acquired_coordinates)
        ),
        "unacquired_samples_are_exact_zero": validation["unacquired_nonzero_count"] == 0,
        "acquired_samples_equal_full_wave_bitwise": validation["acquired_mismatch_count"] == 0,
        "sampling_mask_applied": True,
        "full_chunked_readback_validation": validation,
    }


def validate_masked_bart_kspace(
    full_wave_path: str | Path,
    mask: np.ndarray,
    output_base: str | Path,
    *,
    pe2_chunk: int = 8,
) -> dict[str, Any]:
    """Read back every BART sample and verify masking against the full Wave source."""
    source = np.load(Path(full_wave_path).expanduser().resolve(), mmap_mode="r")
    mask = np.asarray(mask, dtype=bool)
    if source.ndim != 4 or source.dtype != np.complex64 or mask.shape != source.shape[1:3]:
        raise ValueError("Source k-space or sampling-mask layout is invalid for validation.")
    cfl_path = Path(output_base).expanduser().resolve().with_suffix(".cfl")
    bart_shape = (*source.shape, 1)
    output = np.memmap(
        cfl_path,
        mode="r",
        dtype=np.complex64,
        shape=bart_shape,
        order="F",
    )[..., 0]

    mismatch_count = 0
    unacquired_nonzero_count = 0
    nonfinite_count = 0
    for coil in range(source.shape[3]):
        for start in range(0, source.shape[2], pe2_chunk):
            stop = min(start + pe2_chunk, source.shape[2])
            mask_block = mask[:, start:stop]
            source_block = np.asarray(source[:, :, start:stop, coil])
            output_block = np.asarray(output[:, :, start:stop, coil])
            mismatch_count += int(
                np.count_nonzero(output_block[:, mask_block] != source_block[:, mask_block])
            )
            unacquired_nonzero_count += int(np.count_nonzero(output_block[:, ~mask_block]))
            nonfinite_count += int(np.count_nonzero(~np.isfinite(output_block)))

    validation = {
        "mode": "full sample-by-sample chunked readback",
        "acquired_mismatch_count": mismatch_count,
        "unacquired_nonzero_count": unacquired_nonzero_count,
        "nonfinite_count": nonfinite_count,
    }
    if any((mismatch_count, unacquired_nonzero_count, nonfinite_count)):
        raise ValueError(f"Masked BART k-space readback validation failed: {validation}")
    return validation
