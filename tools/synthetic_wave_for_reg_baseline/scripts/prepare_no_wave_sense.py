#!/usr/bin/env python3
"""Prepare measured 12-channel no-wave k-space for SENSE reconstruction.

The product image stream is placed on its native 256^3 Cartesian grid, and
the measured PAT refscan overwrites the 24-line ACS support. Both streams use
the same previously estimated physical-to-virtual coil basis. Missing samples
remain exact zeros; no GRAPPA-derived samples enter this pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from estimate_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    select_product_measurement,
)
from export_bart_calibration_acs import validate_refscan_rectangle
from bart_cfl import bart_base, open_bart_memmap, sha256_file, write_bart_header


GRID_SHAPE = (256, 256, 256)
EXPECTED_IMAGE_LINES = tuple(range(1, 254, 3))
EXPECTED_ACS_LINES = tuple(range(115, 139))


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for measured k-space preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--coil-basis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--ncc", type=int, default=12)
    parser.add_argument("--pe2-chunk", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_basis(path: Path, ncc: int) -> np.ndarray:
    """Load the leading virtual-coil columns from the saved nested basis."""
    with np.load(path) as archive:
        saved = np.asarray(archive["basis"], dtype=np.complex64)
    if saved.ndim != 2 or not 1 <= ncc <= saved.shape[1]:
        raise ValueError(f"Invalid coil basis {saved.shape} or Ncc={ncc}.")
    return np.ascontiguousarray(saved[:, :ncc])


def _integer_coordinates(values: Any, name: str) -> list[int]:
    """Convert integral mapVBVD acquisition counters to Python integers."""
    coordinates = [int(value) for value in values]
    if any(float(raw) != value for raw, value in zip(values, coordinates)):
        raise ValueError(f"{name} contains non-integral coordinates.")
    return coordinates


def build_pe1_mask(image_lines: Sequence[int], acs_lines: Sequence[int]) -> np.ndarray:
    """Return the exact PE1 union mask used by the no-wave SENSE operator."""
    mask = np.zeros(GRID_SHAPE[1], dtype=bool)
    for coordinate in set(image_lines) | set(acs_lines):
        if not 0 <= int(coordinate) < mask.size:
            raise ValueError(f"PE1 coordinate is outside the 256-line grid: {coordinate}.")
        mask[int(coordinate)] = True
    return mask


def _read_stream_chunk(stream: Any, start: int, stop: int, name: str) -> np.ndarray:
    """Read one mapVBVD PE2 chunk in native ``[RO, coil, PE1, PE2]`` order."""
    raw = np.asarray(stream[:, :, :, start:stop], dtype=np.complex64)
    expected = (
        int(stream.sqzSize[0]),
        int(stream.sqzSize[1]),
        int(stream.sqzSize[2]),
        stop - start,
    )
    if raw.shape != expected:
        raise ValueError(f"{name} chunk {start}:{stop} has shape {raw.shape}, expected {expected}.")
    return raw


def _compress_native_chunk(raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Compress a native mapVBVD chunk and return ``[RO, PE1, PE2, virtual coil]``."""
    coillast = np.transpose(raw, (0, 2, 3, 1))
    return apply_coil_compression_coillast(coillast, basis)


def _prepare_output_paths(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    """Validate output ownership and return BART and partial payload paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = bart_base(output_dir / "no_wave_kspace_measured")
    partial = Path(str(base.with_suffix(".cfl")) + ".partial")
    owned = [
        base.with_suffix(".hdr"),
        base.with_suffix(".cfl"),
        partial,
        output_dir / "pe1_sampling_mask.npy",
        output_dir / "prepare_manifest.json",
    ]
    existing = [path for path in owned if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Preparation outputs already exist: {existing}")
    if overwrite:
        for path in existing:
            path.unlink()
    return base, partial


def _validate_payload(base: Path, mask: np.ndarray, pe2_chunk: int) -> dict[str, Any]:
    """Check finiteness, exact-zero missing lines, and the completed CFL norm."""
    kspace = open_bart_memmap(base)
    if kspace.shape != (*GRID_SHAPE, 12):
        raise ValueError(f"Unexpected prepared k-space shape: {kspace.shape}.")
    squared_norm = 0.0
    nonfinite_count = 0
    missing_nonzero_count = 0
    for start in range(0, GRID_SHAPE[2], pe2_chunk):
        stop = min(start + pe2_chunk, GRID_SHAPE[2])
        block = np.asarray(kspace[:, :, start:stop, :])
        nonfinite_count += int(np.count_nonzero(~np.isfinite(block)))
        missing_nonzero_count += int(np.count_nonzero(block[:, ~mask, :, :]))
        squared_norm += float(np.vdot(block, block).real)
    if nonfinite_count or missing_nonzero_count:
        raise ValueError("Prepared k-space contains non-finite or nonzero unacquired samples.")
    return {
        "all_samples_finite": nonfinite_count == 0,
        "missing_samples_are_exact_zero": missing_nonzero_count == 0,
        "norm": float(np.sqrt(squared_norm)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Stream, compress, merge, validate, and record measured no-wave k-space."""
    import mapvbvd

    started = time.perf_counter()
    twix_path = args.twix.expanduser().resolve()
    basis_path = args.coil_basis.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (twix_path, basis_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.ncc != 12:
        raise ValueError("The no-wave SENSE diagnostic requires exactly 12 virtual coils.")
    if args.pe2_chunk < 1:
        raise ValueError("--pe2-chunk must be positive.")

    basis = _load_basis(basis_path, args.ncc)
    twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_product_measurement(twix_root)
    if measurement_index != args.measurement_index:
        raise ValueError(
            f"Selected product measurement {measurement_index}, expected {args.measurement_index}."
        )
    image = configure_stream(measurement["image"])
    refscan = configure_stream(measurement["refscan"])
    image_lines = sorted(set(_integer_coordinates(image.Lin, "image.Lin")))
    ref_lines, ref_partitions = validate_refscan_rectangle(
        _integer_coordinates(refscan.Lin, "refscan.Lin"),
        _integer_coordinates(refscan.Par, "refscan.Par"),
        npe1=GRID_SHAPE[1],
        npe2=GRID_SHAPE[2],
    )
    if tuple(image_lines) != EXPECTED_IMAGE_LINES:
        raise ValueError("Product imaging PE1 coordinates differ from the validated R3 pattern.")
    if tuple(ref_lines) != EXPECTED_ACS_LINES:
        raise ValueError("Product refscan support differs from the validated 24-line ACS.")
    if tuple(ref_partitions) != tuple(range(GRID_SHAPE[2])):
        raise ValueError("Product refscan does not cover every PE2 partition.")
    if tuple(int(value) for value in image.sqzSize) != (256, basis.shape[0], 254, 256):
        raise ValueError(f"Unexpected product image shape: {tuple(image.sqzSize)}.")
    if tuple(int(value) for value in refscan.sqzSize) != (256, basis.shape[0], 24, 256):
        raise ValueError(f"Unexpected product refscan shape: {tuple(refscan.sqzSize)}.")

    mask = build_pe1_mask(image_lines, ref_lines)
    base, partial = _prepare_output_paths(output_dir, args.overwrite)
    output_shape = (*GRID_SHAPE, args.ncc)
    target = np.memmap(partial, mode="w+", dtype=np.complex64, shape=output_shape, order="F")
    target.fill(np.complex64(0.0))
    image_support_hash = hashlib.sha256()
    refscan_support_hash = hashlib.sha256()
    ref_slice = slice(ref_lines[0], ref_lines[-1] + 1)

    for start in range(0, GRID_SHAPE[2], args.pe2_chunk):
        stop = min(start + args.pe2_chunk, GRID_SHAPE[2])
        image_chunk = _compress_native_chunk(
            _read_stream_chunk(image, start, stop, "image"), basis
        )
        refscan_chunk = _compress_native_chunk(
            _read_stream_chunk(refscan, start, stop, "refscan"), basis
        )
        target[:, : image_chunk.shape[1], start:stop, :] = image_chunk
        # The separate measured ACS is authoritative where it overlaps image data.
        target[:, ref_slice, start:stop, :] = refscan_chunk
        image_support_hash.update(np.ascontiguousarray(image_chunk).view(np.uint8))
        refscan_support_hash.update(np.ascontiguousarray(refscan_chunk).view(np.uint8))
    target.flush()
    del target
    os.replace(partial, base.with_suffix(".cfl"))
    write_bart_header(base, output_shape)
    np.save(output_dir / "pe1_sampling_mask.npy", mask)

    validation = _validate_payload(base, mask, args.pe2_chunk)
    manifest = {
        "format_version": 1,
        "status": "measured_no_wave_kspace_ready_for_bart_ecalib_and_sense",
        "source_twix": str(twix_path),
        "measurement_index": measurement_index,
        "coil_basis": str(basis_path),
        "coil_basis_file_sha256": sha256_file(basis_path),
        "basis_columns_half_open": [0, args.ncc],
        "compression_reference": (
            "external/wave-mprage/recon/utils/coil_compression_kspace.py"
        ),
        "kspace_base": str(base),
        "kspace_layout": ["READ", "PHS1", "PHS2", "virtual coil"],
        "kspace_shape": list(output_shape),
        "image_pe1_lines": image_lines,
        "acs_pe1_lines": ref_lines,
        "merged_pe1_lines": np.flatnonzero(mask).tolist(),
        "merged_pe1_line_count": int(mask.sum()),
        "pe1_sampling_mask": str(output_dir / "pe1_sampling_mask.npy"),
        "image_compressed_payload_sha256": image_support_hash.hexdigest(),
        "refscan_compressed_payload_sha256": refscan_support_hash.hexdigest(),
        "acs_overwrites_overlapping_image_samples": True,
        "contains_grappa_samples": False,
        "validation": validation,
        "runtime_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared k-space: {base}")
    print(f"Merged PE1 lines: {manifest['merged_pe1_line_count']}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and map expected failures to status 2."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
