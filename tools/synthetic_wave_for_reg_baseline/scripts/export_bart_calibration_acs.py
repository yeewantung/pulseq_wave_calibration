#!/usr/bin/env python3
"""Export measured, compressed ACS for BART ESPIRiT calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from estimate_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    iter_refscan_coillast_chunks,
    select_product_measurement,
)
from bart_cfl import bart_base, open_bart_memmap, sha256_file, write_bart_header
from checkpoint_io import write_json_atomic
from dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the measured-ACS export command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--twix", type=Path)
    parser.add_argument("--coil-basis", type=Path)
    parser.add_argument("--bart-input-dir", type=Path)
    parser.add_argument("--measurement-index", type=int)
    parser.add_argument("--ncc", type=int)
    parser.add_argument("--pe2-chunk", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse or recover only an exact manifest-backed calibration export.",
    )
    return parser


def _integer_values(values: Any, name: str) -> list[int]:
    """Convert integral-valued MDH coordinates to Python integers."""
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
    if not unique_lines or not unique_partitions:
        raise ValueError("Refscan ACS counters are empty.")
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
    """Atomically attach calibration provenance to the BART input manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "calibration_kspace_ready_for_ecalib"
    manifest["kspace_calib"] = calibration
    write_json_atomic(path, manifest)


def write_calibration_cfl(
    chunks: Iterator[tuple[int, int, np.ndarray]],
    output_base: Path,
    *,
    output_shape: tuple[int, int, int, int],
    line_slice: slice,
    pe2_chunk: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Write measured ACS on its full grid and validate payload and zero exterior."""
    if pe2_chunk < 1:
        raise ValueError("pe2_chunk must be positive.")
    output_base = bart_base(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    header_path = output_base.with_suffix(".hdr")
    cfl_path = output_base.with_suffix(".cfl")
    partial_path = Path(str(cfl_path) + ".partial")
    existing = [path for path in (header_path, cfl_path, partial_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Calibration output already exists: {existing}")
    if overwrite:
        for path in existing:
            path.unlink()

    target = np.memmap(
        partial_path, mode="w+", dtype=np.complex64, shape=output_shape, order="F"
    )
    target.fill(np.complex64(0.0))
    acquired_digest = hashlib.sha256()
    acquired_norm_squared = 0.0
    all_finite = True
    expected_next_partition = 0
    for start, stop, compressed in chunks:
        expected_chunk_shape = (
            output_shape[0],
            line_slice.stop - line_slice.start,
            stop - start,
            output_shape[3],
        )
        if start != expected_next_partition or compressed.shape != expected_chunk_shape:
            raise ValueError(
                "ACS chunks must cover consecutive PE2 partitions with shape "
                f"{expected_chunk_shape}; received {start}:{stop} {compressed.shape}."
            )
        all_finite &= bool(np.isfinite(compressed).all())
        acquired_norm_squared += float(np.vdot(compressed, compressed).real)
        acquired_digest.update(np.ascontiguousarray(compressed).view(np.uint8))
        target[:, line_slice, start:stop, :] = compressed
        expected_next_partition = stop
    if expected_next_partition != output_shape[2]:
        raise ValueError("ACS chunks do not cover every PE2 partition.")
    target.flush()
    del target
    os.replace(partial_path, cfl_path)
    write_bart_header(output_base, output_shape)

    # Reopen every sample so exact-zero support is a persisted-file invariant.
    readback = open_bart_memmap(output_base)
    readback_digest = hashlib.sha256()
    nonfinite_count = 0
    exterior_nonzero_count = 0
    for start in range(0, output_shape[2], pe2_chunk):
        stop = min(start + pe2_chunk, output_shape[2])
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
        raise ValueError("Calibration CFL acquired payload differs from its measured source.")

    return {
        "bart_base": str(output_base),
        "bart_layout": ["READ", "PHS1", "PHS2", "COIL"],
        "bart_shape": list(output_shape),
        "bart_cfl_bytes": cfl_path.stat().st_size,
        "bart_cfl_sha256": sha256_file(cfl_path),
        "acquired_payload_sha256": readback_digest.hexdigest(),
        "acquired_norm": float(np.sqrt(acquired_norm_squared)),
        "acquired_payload_matches_measured_source": True,
        "exterior_nonzero_count": exterior_nonzero_count,
        "all_samples_finite": nonfinite_count == 0,
    }


def _source_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated source payload from either supported preparation path."""
    record = report.get("assembly", report.get("reconstruction"))
    if not isinstance(record, Mapping):
        raise ValueError("Source report has neither direct assembly nor reconstruction.")
    return record


def _file_identity(path: Path) -> dict[str, Any]:
    """Record a large source file without forcing an additional full-file read."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _image_acs_chunks(
    source: np.ndarray,
    *,
    line_slice: slice,
    pe2_chunk: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield measured ACS directly from compressed, fully sampled image k-space."""
    for start in range(0, source.shape[2], pe2_chunk):
        stop = min(start + pe2_chunk, source.shape[2])
        yield start, stop, np.asarray(
            source[:, line_slice, start:stop, :], dtype=np.complex64
        )


def _refscan_acs_chunks(
    refscan: Any,
    basis: np.ndarray,
    *,
    pe2_chunk: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield refscan ACS after applying the dataset's accepted coil basis."""
    for start, stop, physical in iter_refscan_coillast_chunks(
        refscan, pe2_chunk=pe2_chunk
    ):
        yield start, stop, apply_coil_compression_coillast(physical, basis)


def _completed_manifest_export_reusable(
    manifest_path: Path,
    *,
    dataset_sha256: str,
    bart_export_sha256: str,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
) -> bool:
    """Require exact upstream provenance and an intact completed calibration CFL."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        calibration = manifest["calibration"]
        base = Path(calibration["bart_base"])
        return (
            manifest.get("status") == "calibration_kspace_ready_for_ecalib"
            and manifest.get("dataset_manifest", {}).get("sha256") == dataset_sha256
            and manifest.get("bart_export_manifest", {}).get("sha256")
            == bart_export_sha256
            and manifest.get("config") == dict(config)
            and manifest.get("source") == dict(source)
            and base.with_suffix(".hdr").is_file()
            and base.with_suffix(".cfl").is_file()
            and tuple(open_bart_memmap(base).shape) == tuple(config["matrix_rolinpar_ncc"])
            and sha256_file(base.with_suffix(".cfl"))
            == calibration["bart_cfl_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_manifest_bart_tree(
    dataset: DatasetManifest,
) -> tuple[Path, Path, dict[str, Any], str]:
    """Validate the separate target-sampling tree before adding calibration."""
    export_dir = dataset.output_path("bart_export_dir")
    export_manifest_path = export_dir / "manifest.json"
    bart_dir = export_dir / "bart_inputs"
    bart_manifest_path = bart_dir / "manifest.json"
    for path in (export_manifest_path, bart_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    bart_manifest = json.loads(bart_manifest_path.read_text(encoding="utf-8"))
    if export_manifest.get("status") != "manifest_bart_inputs_ready":
        raise ValueError("Manifest-backed target BART export is not complete.")
    if export_manifest.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Target BART export uses a stale dataset manifest.")
    if bart_manifest.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Nested BART input manifest uses a stale dataset manifest.")
    allowed_status = {
        "masked_wave_inputs_ready_for_map_estimation_and_reconstruction",
        "calibration_kspace_ready_for_ecalib",
    }
    if bart_manifest.get("status") not in allowed_status:
        raise ValueError("Nested BART input manifest is not ready for calibration.")
    if bart_manifest.get("source_synthesis_manifest") != export_manifest.get(
        "source_synthesis_manifest"
    ):
        raise ValueError("Target export manifests disagree about full-Wave provenance.")
    return bart_dir, bart_manifest_path, bart_manifest, sha256_file(export_manifest_path)


def _run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Export manifest-selected image or refscan ACS on the logical image grid."""
    explicit = (
        args.twix,
        args.coil_basis,
        args.bart_input_dir,
        args.measurement_index,
        args.ncc,
    )
    if any(value is not None for value in explicit) or args.overwrite:
        raise ValueError(
            "--dataset-manifest cannot be combined with explicit dataset options "
            "or --overwrite"
        )
    dataset = load_dataset_manifest(args.dataset_manifest)
    inspection = load_passed_inspection(dataset)
    bart_dir, bart_manifest_path, _, bart_export_sha256 = _load_manifest_bart_tree(
        dataset
    )
    contract = dataset.payload
    matrix = tuple(int(value) for value in contract["geometry"]["matrix"])
    reconstruction = contract["reconstruction"]
    ncc = int(reconstruction["virtual_coils"])
    calibration_source = str(reconstruction["bart"]["calibration_source"])
    acs_start = int(contract["sampling"]["synthetic_wave_acs_pe1_start"])
    acs_stop = int(
        contract["sampling"]["synthetic_wave_acs_pe1_stop_exclusive"]
    )
    line_slice = slice(acs_start, acs_stop)
    output_shape = (*matrix, ncc)
    config = {
        "calibration_source": calibration_source,
        "matrix_rolinpar_ncc": list(output_shape),
        "pe1_lines_half_open": [acs_start, acs_stop],
        "pe2_partitions_half_open": [0, matrix[2]],
    }

    chunks: Iterator[tuple[int, int, np.ndarray]]
    large_source_path: Path
    large_source_identity: dict[str, Any]
    hashed_provenance_files: list[tuple[Path, str]]
    if calibration_source == "image":
        if contract["sampling"]["source_acceleration_pe1_pe2"] != [1, 1]:
            raise ValueError("Image-derived calibration requires a fully sampled R1 source.")
        prefix = dataset.output_path("source_reconstruction_prefix")
        source_path = prefix.with_name(prefix.name + f"_full_ncc{ncc}.npy")
        source_report_path = prefix.with_name(prefix.name + "_report.json")
        if not source_report_path.is_file():
            raise FileNotFoundError(source_report_path)
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        record = _source_record(source_report)
        if source_report.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
            raise ValueError("No-Wave source report uses a stale dataset manifest.")
        if (
            Path(record.get("output", "")).resolve() != source_path
            or record.get("shape") != list(output_shape)
            or record.get("finite") is not True
            or record.get("grappa_applied") is not False
            or record.get("interpolation") != "none"
        ):
            raise ValueError("Image calibration requires validated direct measured k-space.")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_array = np.load(source_path, mmap_mode="r")
        if source_array.shape != output_shape or source_array.dtype != np.complex64:
            raise ValueError("Direct source k-space shape or dtype is invalid.")
        large_source_path = source_path
        large_source_identity = _file_identity(source_path)
        source_report_sha256 = sha256_file(source_report_path)
        hashed_provenance_files = [(source_report_path, source_report_sha256)]
        source = {
            "kind": "fully sampled compressed image k-space",
            "file": large_source_identity,
            "report": str(source_report_path),
            "report_sha256": source_report_sha256,
            "interpolation": "none",
        }
        chunks = _image_acs_chunks(
            source_array, line_slice=line_slice, pe2_chunk=args.pe2_chunk
        )
    else:
        twix_path = dataset.input_path("twix")
        basis_prefix = dataset.output_path("coil_compression_prefix")
        basis_path = basis_prefix.with_suffix(".npz")
        basis_report_path = basis_prefix.with_suffix(".json")
        for path in (twix_path, basis_path, basis_report_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        basis_report = json.loads(basis_report_path.read_text(encoding="utf-8"))
        if basis_report.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
            raise ValueError("Coil-compression report uses a stale dataset manifest.")
        with np.load(basis_path) as archive:
            saved_basis = np.asarray(archive["basis"], dtype=np.complex64)
        physical_coils = int(reconstruction["physical_coils"])
        if saved_basis.shape != (physical_coils, ncc):
            raise ValueError(
                f"Expected coil basis {(physical_coils, ncc)}, got {saved_basis.shape}."
            )
        try:
            import mapvbvd
        except ImportError as exc:
            raise RuntimeError("pymapvbvd>=0.6.1 is required for refscan ACS.") from exc
        twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
        measurement_index, measurement = select_product_measurement(twix_root)
        expected_measurement = int(inspection["twix"]["selected_measurement_index"])
        if measurement_index != expected_measurement:
            raise ValueError(
                f"Selected measurement {measurement_index}, expected {expected_measurement}."
            )
        refscan = configure_stream(measurement["refscan"])
        lines = _integer_values(refscan.Lin, "refscan.Lin")
        partitions = _integer_values(refscan.Par, "refscan.Par")
        unique_lines, unique_partitions = validate_refscan_rectangle(
            lines, partitions, npe1=matrix[1], npe2=matrix[2]
        )
        if unique_lines != list(range(acs_start, acs_stop)):
            raise ValueError(
                f"Manifest ACS lines {acs_start}:{acs_stop} differ from {unique_lines}."
            )
        compact_shape = tuple(int(value) for value in refscan.sqzSize)
        expected_compact = (matrix[0], physical_coils, acs_stop - acs_start, matrix[2])
        if compact_shape != expected_compact:
            raise ValueError(
                f"Refscan compact shape {compact_shape}, expected {expected_compact}."
            )
        large_source_path = twix_path
        large_source_identity = _file_identity(twix_path)
        basis_sha256 = sha256_file(basis_path)
        basis_report_sha256 = sha256_file(basis_report_path)
        hashed_provenance_files = [
            (basis_path, basis_sha256),
            (basis_report_path, basis_report_sha256),
        ]
        source = {
            "kind": "compressed TWIX refscan k-space",
            "twix": large_source_identity,
            "measurement_index": measurement_index,
            "coil_basis": str(basis_path),
            "coil_basis_sha256": basis_sha256,
            "coil_compression_report": str(basis_report_path),
            "coil_compression_report_sha256": basis_report_sha256,
            "compact_refscan_shape": list(compact_shape),
            "pe1_lines": unique_lines,
            "pe2_partitions": unique_partitions,
        }
        chunks = _refscan_acs_chunks(
            refscan, saved_basis, pe2_chunk=args.pe2_chunk
        )

    calibration_manifest_path = bart_dir / "kspace_calib_manifest.json"
    if args.resume and _completed_manifest_export_reusable(
        calibration_manifest_path,
        dataset_sha256=dataset.sha256,
        bart_export_sha256=bart_export_sha256,
        config=config,
        source=source,
    ):
        completed = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
        _update_bart_manifest(bart_manifest_path, completed["calibration"])
        print(f"Reusing validated calibration: {calibration_manifest_path}")
        return completed["calibration"]

    output_base = bart_base(bart_dir / "kspace_calib")
    artifacts = (
        output_base.with_suffix(".hdr"),
        output_base.with_suffix(".cfl"),
        Path(str(output_base.with_suffix(".cfl")) + ".partial"),
        calibration_manifest_path,
    )
    recover_incomplete = False
    existing = [path for path in artifacts if path.exists()]
    if existing:
        if args.resume and calibration_manifest_path.is_file():
            prior = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
            recover_incomplete = (
                prior.get("status") == "exporting_calibration_kspace"
                and prior.get("dataset_manifest", {}).get("sha256") == dataset.sha256
                and prior.get("bart_export_manifest", {}).get("sha256")
                == bart_export_sha256
                and prior.get("config") == config
                and prior.get("source") == source
            )
        if not recover_incomplete:
            raise FileExistsError(
                "Calibration artifacts are not safely reusable: "
                + ", ".join(str(path) for path in existing)
            )

    started_at = time.perf_counter()
    started = {
        "format_version": 1,
        "status": "exporting_calibration_kspace",
        "dataset_manifest": dataset.provenance(),
        "bart_export_manifest": {
            "path": str(dataset.output_path("bart_export_dir") / "manifest.json"),
            "sha256": bart_export_sha256,
        },
        "config": config,
        "source": source,
    }
    write_json_atomic(calibration_manifest_path, started)
    payload = write_calibration_cfl(
        chunks,
        output_base,
        output_shape=output_shape,
        line_slice=line_slice,
        pe2_chunk=args.pe2_chunk,
        overwrite=recover_incomplete,
    )
    if _file_identity(large_source_path) != large_source_identity or any(
        sha256_file(path) != expected_hash
        for path, expected_hash in hashed_provenance_files
    ):
        raise ValueError("Measured calibration source changed during ACS export.")
    calibration = {
        "source": source["kind"],
        "calibration_source": calibration_source,
        **payload,
        "pe1_lines_half_open": [acs_start, acs_stop],
        "pe2_partitions_half_open": [0, matrix[2]],
        "acquired_pe_coordinate_count": (acs_stop - acs_start) * matrix[2],
        "runtime_seconds": time.perf_counter() - started_at,
    }
    completed = {
        **started,
        "status": "calibration_kspace_ready_for_ecalib",
        "calibration": calibration,
    }
    write_json_atomic(calibration_manifest_path, completed)
    _update_bart_manifest(bart_manifest_path, calibration)
    print(f"Calibration manifest: {calibration_manifest_path}")
    return calibration


def _run_explicit(args: argparse.Namespace) -> dict[str, Any]:
    """Compress measured refscan ACS and write it on the full BART grid."""
    import mapvbvd

    started = time.perf_counter()
    if args.twix is None or args.coil_basis is None or args.bart_input_dir is None:
        raise ValueError(
            "Use --dataset-manifest, or provide --twix, --coil-basis, and --bart-input-dir"
        )
    if args.resume:
        raise ValueError("--resume requires --dataset-manifest")
    twix_path = args.twix.expanduser().resolve()
    basis_path = args.coil_basis.expanduser().resolve()
    bart_dir = args.bart_input_dir.expanduser().resolve()
    bart_manifest_path = bart_dir / "manifest.json"
    for path in (twix_path, basis_path, bart_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(basis_path) as archive:
        saved_basis = np.asarray(archive["basis"], dtype=np.complex64)
    ncc = 12 if args.ncc is None else args.ncc
    measurement_expected = 1 if args.measurement_index is None else args.measurement_index
    if saved_basis.ndim != 2 or not 1 <= ncc <= saved_basis.shape[1]:
        raise ValueError(f"Invalid saved basis {saved_basis.shape} or Ncc={ncc}.")
    basis = saved_basis[:, :ncc]

    twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_product_measurement(twix_root)
    if measurement_index != measurement_expected:
        raise ValueError(
            f"Selected product measurement {measurement_index}, expected {measurement_expected}."
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
    output_shape = (256, 256, 256, ncc)
    line_slice = slice(unique_lines[0], unique_lines[-1] + 1)
    payload = write_calibration_cfl(
        _refscan_acs_chunks(refscan, basis, pe2_chunk=args.pe2_chunk),
        output_base,
        output_shape=output_shape,
        line_slice=line_slice,
        pe2_chunk=args.pe2_chunk,
        overwrite=args.overwrite,
    )

    calibration = {
        "source": "measured no-wave product TWIX refscan ACS",
        "source_twix": str(twix_path),
        "measurement_index": measurement_index,
        "coil_basis": str(basis_path),
        "coil_basis_file_sha256": sha256_file(basis_path),
        "basis_columns_half_open": [0, ncc],
        "compact_refscan_layout": ["READ", "physical_coil", "PE1", "PE2"],
        "compact_refscan_shape": list(compact_shape),
        **payload,
        "pe1_lines_half_open": [unique_lines[0], unique_lines[-1] + 1],
        "pe2_partitions_half_open": [unique_partitions[0], unique_partitions[-1] + 1],
        "acquired_pe_coordinate_count": len(unique_lines) * len(unique_partitions),
        "acquired_payload_matches_compressed_refscan": True,
        "runtime_seconds": time.perf_counter() - started,
    }
    calibration_manifest = bart_dir / "kspace_calib_manifest.json"
    write_json_atomic(calibration_manifest, calibration)
    _update_bart_manifest(bart_manifest_path, calibration)
    print(f"Calibration manifest: {calibration_manifest}")
    return calibration


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch to the portable manifest route or compatible product route."""
    if args.dataset_manifest is not None:
        return _run_manifest(args)
    return _run_explicit(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ACS exporter from command-line arguments."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        DatasetManifestError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
