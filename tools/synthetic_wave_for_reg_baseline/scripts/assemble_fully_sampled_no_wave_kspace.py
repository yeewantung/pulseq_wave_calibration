#!/usr/bin/env python3
"""Assemble fully sampled no-Wave TWIX data without GRAPPA interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from checkpoint_io import (
    load_coil_basis,
    open_or_create_complex64_npy,
    validate_resume_pair,
    write_json_atomic,
)
from dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
    sha256_file,
)
from estimate_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    select_imaging_measurement,
)


@dataclass(frozen=True)
class FullySampledInputs:
    """Resolved contract for direct no-Wave k-space assembly."""

    manifest: DatasetManifest
    inspection: Mapping[str, Any]
    twix: Path
    coil_basis: Path
    output_prefix: Path
    matrix_rolinpar: tuple[int, int, int]
    physical_coils: int
    virtual_coils: int

    @property
    def output_path(self) -> Path:
        return self.output_prefix.with_name(
            self.output_prefix.name + f"_full_ncc{self.virtual_coils}.npy"
        )

    @property
    def progress_path(self) -> Path:
        return self.output_prefix.with_name(self.output_prefix.name + "_recon_progress.json")

    @property
    def report_path(self) -> Path:
        return self.output_prefix.with_name(self.output_prefix.name + "_report.json")


def _build_parser() -> argparse.ArgumentParser:
    """Build the fully sampled source-assembly command interface.

    Returns:
        Parser for validation and resumable source materialization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        type=Path,
        help="Passed R1 dataset contract containing geometry, paths, and coil counts.",
    )
    parser.add_argument(
        "--pe2-chunk",
        type=int,
        default=4,
        help="Number of PE2 partitions read, compressed, and flushed per checkpoint.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Optional destination prefix outside the immutable dataset tree.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from a matching output/progress pair.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest, inspection, basis, and TWIX layout without reading payloads.",
    )
    return parser


def resolve_fully_sampled_inputs(manifest_path: Path) -> FullySampledInputs:
    """Resolve and gate an R1 contract before TWIX payload access."""
    manifest = load_dataset_manifest(manifest_path)
    inspection = load_passed_inspection(manifest)
    contract = manifest.payload
    sampling_contract = contract["sampling"]
    acceleration = list(sampling_contract["source_acceleration_pe1_pe2"])
    if acceleration != [1, 1] or sampling_contract["require_complete_source_grid"] is not True:
        raise ValueError(
            "Direct source assembly requires source acceleration [1, 1] and "
            "require_complete_source_grid=true."
        )

    matrix = tuple(int(value) for value in contract["geometry"]["matrix"])
    measured = inspection["twix"]["selected_measurement_sampling"]
    expected_coordinates = matrix[1] * matrix[2]
    if (
        measured.get("image_unique_coordinate_count") != expected_coordinates
        or measured.get("image_duplicate_coordinate_count") != 0
        or measured.get("image_inferred_pe1_stride") != 1
        or measured.get("out_of_range_coordinates") != []
    ):
        raise ValueError(
            "Measured TWIX image sampling is not one duplicate-free, complete PE1xPE2 grid."
        )

    checks = {
        check["name"]: check for check in inspection["contract_checks"]["checks"]
    }
    readout_check = checks.get("complete_centered_readout")
    if not isinstance(readout_check, Mapping) or readout_check.get("passed") is not True:
        raise ValueError(
            "A passed complete_centered_readout inspection check is required."
        )

    reconstruction = contract["reconstruction"]
    return FullySampledInputs(
        manifest=manifest,
        inspection=inspection,
        twix=manifest.input_path("twix"),
        coil_basis=manifest.output_path("coil_compression_prefix").with_suffix(".npz"),
        output_prefix=manifest.output_path("source_reconstruction_prefix"),
        matrix_rolinpar=matrix,
        physical_coils=int(reconstruction["physical_coils"]),
        virtual_coils=int(reconstruction["virtual_coils"]),
    )


def validate_image_stream(
    image: Any,
    *,
    matrix_rolinpar: tuple[int, int, int],
    physical_coils: int,
) -> dict[str, Any]:
    """Require a compact mapVBVD image stream that exactly spans the logical grid."""
    configure_stream(image)
    expected_shape = (
        matrix_rolinpar[0],
        physical_coils,
        matrix_rolinpar[1],
        matrix_rolinpar[2],
    )
    observed_shape = tuple(int(value) for value in image.sqzSize)
    if observed_shape != expected_shape:
        raise ValueError(
            "Fully sampled image stream has shape "
            f"{observed_shape}; expected [RO,coil,PE1,PE2] {expected_shape}."
        )
    if int(image.skipLin) != 0 or int(image.skipPar) != 0:
        raise ValueError(
            f"Fully sampled image support must start at PE1/PE2 zero, got "
            f"{int(image.skipLin)}/{int(image.skipPar)}."
        )
    lines = sorted({int(value) for value in image.Lin})
    partitions = sorted({int(value) for value in image.Par})
    if lines != list(range(matrix_rolinpar[1])) or partitions != list(
        range(matrix_rolinpar[2])
    ):
        raise ValueError("Runtime TWIX PE counters do not span the declared complete grid.")
    return {
        "mapvbvd_shape_ro_coil_pe1_pe2": list(observed_shape),
        "skip_pe1_pe2": [int(image.skipLin), int(image.skipPar)],
        "unique_pe1_lines": len(lines),
        "unique_pe2_partitions": len(partitions),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_run_signature(inputs: FullySampledInputs) -> tuple[str, dict[str, Any]]:
    """Bind resume state to dataset, raw-file identity, basis, and output geometry."""
    payload = {
        "method": "direct fully sampled no-Wave assembly with coil compression",
        "dataset_manifest_sha256": inputs.manifest.sha256,
        "dataset_inspection_sha256": sha256_file(inputs.manifest.inspection_report),
        "twix": _file_identity(inputs.twix),
        "coil_basis": {
            "path": str(inputs.coil_basis),
            "sha256": sha256_file(inputs.coil_basis),
        },
        "matrix_rolinpar": list(inputs.matrix_rolinpar),
        "physical_coils": inputs.physical_coils,
        "virtual_coils": inputs.virtual_coils,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _resume_start(
    output_path: Path,
    progress_path: Path,
    *,
    resume: bool,
    signature_sha256: str,
    npe2: int,
) -> int:
    validate_resume_pair(output_path, progress_path, resume=resume)
    if not resume or not progress_path.is_file():
        return 0
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("run_signature_sha256") != signature_sha256:
        raise ValueError("Resume checkpoint was created from different inputs or settings.")
    start = int(progress.get("next_partition", -1))
    if not 0 <= start <= npe2:
        raise ValueError(f"Invalid resume partition {start} for PE2 size {npe2}.")
    return start


def assemble_fully_sampled_kspace(
    image: Any,
    basis: np.ndarray,
    output_path: Path,
    progress_path: Path,
    *,
    matrix_rolinpar: tuple[int, int, int],
    pe2_chunk: int,
    resume: bool,
    run_signature_sha256: str,
) -> dict[str, Any]:
    """Stream, compress, and checkpoint direct measured k-space by PE2 partition."""
    if pe2_chunk < 1:
        raise ValueError("pe2_chunk must be positive.")
    stream = validate_image_stream(
        image,
        matrix_rolinpar=matrix_rolinpar,
        physical_coils=basis.shape[0],
    )
    output_shape = (*matrix_rolinpar, basis.shape[1])
    start_partition = _resume_start(
        output_path,
        progress_path,
        resume=resume,
        signature_sha256=run_signature_sha256,
        npe2=matrix_rolinpar[2],
    )
    output = open_or_create_complex64_npy(
        output_path, output_shape, resume=resume
    )
    started = time.perf_counter()
    input_energy_this_invocation = 0.0
    output_energy_this_invocation = 0.0

    for start in range(start_partition, matrix_rolinpar[2], pe2_chunk):
        stop = min(start + pe2_chunk, matrix_rolinpar[2])
        raw = np.asarray(image[:, :, :, start:stop], dtype=np.complex64)
        expected = (
            matrix_rolinpar[0],
            basis.shape[0],
            matrix_rolinpar[1],
            stop - start,
        )
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(
                f"TWIX image chunk {start}:{stop} has invalid shape/data: "
                f"{raw.shape}, expected {expected}."
            )
        coillast = np.transpose(raw, (0, 2, 3, 1))
        compressed = apply_coil_compression_coillast(coillast, basis)
        if not np.isfinite(compressed).all():
            raise ValueError(f"Compressed image chunk {start}:{stop} is non-finite.")
        output[:, :, start:stop, :] = compressed
        output.flush()
        input_energy_this_invocation += float(np.vdot(raw, raw).real)
        output_energy_this_invocation += float(np.vdot(compressed, compressed).real)
        write_json_atomic(
            progress_path,
            {
                "format_version": 1,
                "pipeline_step": "direct fully sampled no-Wave k-space assembly",
                "run_signature_sha256": run_signature_sha256,
                "next_partition": stop,
                "complete": stop == matrix_rolinpar[2],
            },
        )
        print(f"Direct source checkpoint: PE2 {stop}/{matrix_rolinpar[2]}", flush=True)

    final_energy = 0.0
    for start in range(0, matrix_rolinpar[2], pe2_chunk):
        block = np.asarray(output[:, :, start : start + pe2_chunk, :])
        if not np.isfinite(block).all():
            raise ValueError("Completed direct source checkpoint contains non-finite samples.")
        final_energy += float(np.vdot(block, block).real)
    if final_energy <= 0:
        raise ValueError("Completed direct source checkpoint contains no signal energy.")
    return {
        **stream,
        "output": str(output_path),
        "shape": list(output_shape),
        "dtype": "complex64",
        "finite": True,
        "interpolation": "none",
        "grappa_applied": False,
        "input_energy_this_invocation": input_energy_this_invocation,
        "output_energy_this_invocation": output_energy_this_invocation,
        "final_output_energy": final_energy,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate or create a compressed full-grid no-Wave checkpoint.

    Args:
        args: Parsed manifest, output override, validation, and resume settings.

    Returns:
        Completed assembly report, or ``None`` for validation-only execution.
    """
    inputs = resolve_fully_sampled_inputs(args.dataset_manifest)
    output_prefix = getattr(args, "output_prefix", None)
    if output_prefix is not None:
        inputs = replace(inputs, output_prefix=output_prefix.expanduser().resolve())
    if not inputs.twix.is_file():
        raise FileNotFoundError(f"TWIX file not found: {inputs.twix}")
    if not inputs.coil_basis.is_file():
        raise FileNotFoundError(f"Coil-compression basis not found: {inputs.coil_basis}")
    basis = load_coil_basis(inputs.coil_basis, inputs.virtual_coils)
    if basis.shape[0] != inputs.physical_coils:
        raise ValueError(
            f"Coil basis has {basis.shape[0]} physical coils; manifest expects "
            f"{inputs.physical_coils}."
        )

    try:
        import mapvbvd
    except ImportError as exc:
        raise RuntimeError("pymapvbvd>=0.6.1 is required.") from exc
    root = mapvbvd.mapVBVD(str(inputs.twix), quiet=True)
    measurement_index, measurement = select_imaging_measurement(root)
    image = configure_stream(measurement["image"])
    stream_validation = validate_image_stream(
        image,
        matrix_rolinpar=inputs.matrix_rolinpar,
        physical_coils=inputs.physical_coils,
    )
    if args.validate_only:
        print("Fully sampled no-Wave source validation: PASS")
        print(json.dumps(stream_validation, indent=2))
        return None

    signature_sha256, signature = build_run_signature(inputs)
    artifacts = (inputs.output_path, inputs.progress_path, inputs.report_path)
    if not args.resume:
        existing = [path for path in artifacts if path.exists()]
        if existing:
            raise FileExistsError(
                "Direct source artifacts already exist; use --resume or a new output root: "
                + ", ".join(str(path) for path in existing)
            )
    elif inputs.report_path.is_file():
        report = json.loads(inputs.report_path.read_text(encoding="utf-8"))
        progress = (
            json.loads(inputs.progress_path.read_text(encoding="utf-8"))
            if inputs.progress_path.is_file()
            else {}
        )
        expected_shape = (*inputs.matrix_rolinpar, inputs.virtual_coils)
        output = (
            np.load(inputs.output_path, mmap_mode="r")
            if inputs.output_path.is_file()
            else None
        )
        if (
            report.get("run_signature_sha256") == signature_sha256
            and report.get("assembly", {}).get("finite") is True
            and progress.get("run_signature_sha256") == signature_sha256
            and progress.get("complete") is True
            and progress.get("next_partition") == inputs.matrix_rolinpar[2]
            and output is not None
            and output.shape == expected_shape
            and output.dtype == np.complex64
        ):
            print(f"Completed direct source run already exists: {inputs.report_path}")
            return report
        raise ValueError("Existing direct source report is incomplete or has stale provenance.")

    inputs.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    assembly = assemble_fully_sampled_kspace(
        image,
        basis,
        inputs.output_path,
        inputs.progress_path,
        matrix_rolinpar=inputs.matrix_rolinpar,
        pe2_chunk=args.pe2_chunk,
        resume=args.resume,
        run_signature_sha256=signature_sha256,
    )
    report = {
        "format_version": 1,
        "pipeline_step": "direct fully sampled no-Wave source preparation",
        "method": "measured k-space assembly and coil compression; no interpolation",
        "dataset_manifest": inputs.manifest.provenance(),
        "dataset_inspection": {
            "path": str(inputs.manifest.inspection_report),
            "sha256": sha256_file(inputs.manifest.inspection_report),
        },
        "run_signature_sha256": signature_sha256,
        "run_signature": signature,
        "measurement_index": measurement_index,
        "assembly": assembly,
    }
    write_json_atomic(inputs.report_path, report)
    print(f"Direct source k-space: {inputs.output_path}")
    print(f"Report: {inputs.report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except (
        DatasetManifestError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
