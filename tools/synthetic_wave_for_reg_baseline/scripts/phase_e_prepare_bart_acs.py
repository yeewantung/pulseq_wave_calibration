#!/usr/bin/env python3
"""Export measured, compressed product refscan ACS for BART ecalib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phase_b_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    iter_refscan_coillast_chunks,
    select_product_measurement,
)
from phase_e_utils import bart_base, open_bart_memmap, sha256_file, write_bart_header


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--coil-basis", required=True, type=Path)
    parser.add_argument("--bart-input-dir", required=True, type=Path)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--ncc", type=int, default=12)
    parser.add_argument("--pe2-chunk", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _integer_values(values: Any, name: str) -> list[int]:
    result = [int(value) for value in values]
    if any(float(raw) != value for raw, value in zip(values, result)):
        raise ValueError(f"{name} contains non-integral coordinates.")
    return result


def validate_refscan_rectangle(
    lines: Sequence[int], partitions: Sequence[int], *, npe1: int, npe2: int
) -> tuple[list[int], list[int]]:
    """Validate that MDH counters describe one full rectangular ACS support."""
    if len(lines) != len(partitions):
        raise ValueError("Refscan PE1 and PE2 counter counts differ.")
    unique_lines = sorted(set(int(value) for value in lines))
    unique_partitions = sorted(set(int(value) for value in partitions))
    if unique_lines != list(range(unique_lines[0], unique_lines[-1] + 1)):
        raise ValueError("Refscan PE1 support is not consecutive.")
    if unique_partitions != list(range(npe2)):
        raise ValueError("Refscan does not cover every PE2 partition.")
    pairs = set(zip(lines, partitions))
    if len(pairs) != len(unique_lines) * len(unique_partitions):
        raise ValueError("Refscan MDH coordinates do not form a complete rectangle.")
    if unique_lines[0] < 0 or unique_lines[-1] >= npe1:
        raise ValueError("Refscan contains out-of-range PE1 coordinates.")
    return unique_lines, unique_partitions


def _update_bart_manifest(path: Path, calibration: dict[str, Any]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "calibration_kspace_ready_for_ecalib"
    manifest["kspace_calib"] = calibration
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mapvbvd

    started = time.perf_counter()
    twix_path = args.twix.expanduser().resolve()
    basis_path = args.coil_basis.expanduser().resolve()
    bart_dir = args.bart_input_dir.expanduser().resolve()
    bart_manifest_path = bart_dir / "manifest.json"
    for path in (twix_path, basis_path, bart_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(basis_path) as archive:
        saved_basis = np.asarray(archive["basis"], dtype=np.complex64)
    if saved_basis.ndim != 2 or not 1 <= args.ncc <= saved_basis.shape[1]:
        raise ValueError(f"Invalid saved basis {saved_basis.shape} or Ncc={args.ncc}.")
    basis = saved_basis[:, : args.ncc]

    twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_product_measurement(twix_root)
    if measurement_index != args.measurement_index:
        raise ValueError(
            f"Selected product measurement {measurement_index}, expected {args.measurement_index}."
        )
    refscan = configure_stream(measurement["refscan"])
    lines = _integer_values(refscan.Lin, "refscan.Lin")
    partitions = _integer_values(refscan.Par, "refscan.Par")
    unique_lines, unique_partitions = validate_refscan_rectangle(
        lines, partitions, npe1=256, npe2=256
    )
    expected_lines = list(range(115, 139))
    if unique_lines != expected_lines:
        raise ValueError(f"Expected product ACS PE1 lines {expected_lines}, got {unique_lines}.")

    compact_shape = tuple(int(value) for value in refscan.sqzSize)
    expected_compact = (256, saved_basis.shape[0], len(unique_lines), len(unique_partitions))
    if compact_shape != expected_compact:
        raise ValueError(f"Refscan compact shape {compact_shape}, expected {expected_compact}.")

    output_base = bart_base(bart_dir / "kspace_calib")
    header_path = output_base.with_suffix(".hdr")
    cfl_path = output_base.with_suffix(".cfl")
    partial_path = Path(str(cfl_path) + ".partial")
    existing = [path for path in (header_path, cfl_path, partial_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Calibration output already exists: {existing}")
    if args.overwrite:
        for path in existing:
            path.unlink()

    output_shape = (256, 256, 256, args.ncc)
    target = np.memmap(
        partial_path, mode="w+", dtype=np.complex64, shape=output_shape, order="F"
    )
    target.fill(np.complex64(0.0))
    acquired_digest = hashlib.sha256()
    acquired_norm_squared = 0.0
    all_finite = True
    line_slice = slice(unique_lines[0], unique_lines[-1] + 1)
    for start, stop, physical in iter_refscan_coillast_chunks(
        refscan, pe2_chunk=args.pe2_chunk
    ):
        compressed = apply_coil_compression_coillast(physical, basis)
        all_finite &= bool(np.isfinite(compressed).all())
        acquired_norm_squared += float(np.vdot(compressed, compressed).real)
        acquired_digest.update(np.ascontiguousarray(compressed).view(np.uint8))
        target[:, line_slice, start:stop, :] = compressed
    target.flush()
    del target
    os.replace(partial_path, cfl_path)
    write_bart_header(output_base, output_shape)

    # Reopen the finished CFL, hash all acquired blocks, and inspect every zero exterior sample.
    readback = open_bart_memmap(output_base)
    readback_digest = hashlib.sha256()
    nonfinite_count = 0
    exterior_nonzero_count = 0
    for start in range(0, output_shape[2], args.pe2_chunk):
        stop = min(start + args.pe2_chunk, output_shape[2])
        acquired = np.asarray(readback[:, line_slice, start:stop, :])
        readback_digest.update(np.ascontiguousarray(acquired).view(np.uint8))
        nonfinite_count += int(np.count_nonzero(~np.isfinite(acquired)))
        for exterior in (
            np.asarray(readback[:, : line_slice.start, start:stop, :]),
            np.asarray(readback[:, line_slice.stop :, start:stop, :]),
        ):
            exterior_nonzero_count += int(np.count_nonzero(exterior))
            nonfinite_count += int(np.count_nonzero(~np.isfinite(exterior)))
    if not all_finite or nonfinite_count or exterior_nonzero_count:
        raise ValueError("Calibration CFL finiteness or exact-zero exterior validation failed.")
    if acquired_digest.hexdigest() != readback_digest.hexdigest():
        raise ValueError("Calibration CFL acquired payload differs from compressed refscan ACS.")

    calibration = {
        "source": "measured no-wave product TWIX refscan ACS",
        "source_twix": str(twix_path),
        "measurement_index": measurement_index,
        "coil_basis": str(basis_path),
        "coil_basis_file_sha256": sha256_file(basis_path),
        "basis_columns_half_open": [0, args.ncc],
        "compact_refscan_layout": ["READ", "physical_coil", "PE1", "PE2"],
        "compact_refscan_shape": list(compact_shape),
        "bart_base": str(output_base),
        "bart_layout": ["READ", "PHS1", "PHS2", "COIL"],
        "bart_shape": list(output_shape),
        "bart_cfl_bytes": cfl_path.stat().st_size,
        "pe1_lines_half_open": [unique_lines[0], unique_lines[-1] + 1],
        "pe2_partitions_half_open": [unique_partitions[0], unique_partitions[-1] + 1],
        "acquired_pe_coordinate_count": len(unique_lines) * len(unique_partitions),
        "acquired_payload_sha256": readback_digest.hexdigest(),
        "acquired_norm": float(np.sqrt(acquired_norm_squared)),
        "acquired_payload_matches_compressed_refscan": True,
        "exterior_nonzero_count": exterior_nonzero_count,
        "all_samples_finite": nonfinite_count == 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    calibration_manifest = bart_dir / "kspace_calib_manifest.json"
    calibration_manifest.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    _update_bart_manifest(bart_manifest_path, calibration)
    print(f"Calibration manifest: {calibration_manifest}")
    return calibration


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
