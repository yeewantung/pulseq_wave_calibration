#!/usr/bin/env python3
"""Run resumable local 5×5×Kz R=3 GRAPPA on compatible MPRAGE data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from checkpoint_io import (
    load_coil_basis,
    open_or_create_complex64_npy,
    validate_resume_pair,
    write_json_atomic,
)
from grappa_3d_r3 import (
    NormalEquations3D,
    accumulate_normal_equations_3d,
    apply_grappa_3d_block,
    pe2_offsets,
    solve_weights_3d,
)
from estimate_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    select_imaging_measurement,
)
from dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
    sha256_file,
)


@dataclass(frozen=True)
class GrappaRunInputs:
    """Resolved acquisition-specific inputs for the fixed R3 GRAPPA algorithm."""

    twix: Path
    coil_basis: Path
    output_prefix: Path
    ncc: int
    regularization: float
    pe2_kernel_size: int
    matrix_rolinpar: tuple[int, int, int]
    manifest: DatasetManifest | None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse dataset, checkpoint, kernel, and chunk-size options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Use a passed dataset contract for geometry, sampling, settings, and paths.",
    )
    parser.add_argument("--twix", type=Path)
    parser.add_argument("--coil-basis", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--ncc", type=int)
    parser.add_argument("--regularization", type=float)
    parser.add_argument(
        "--pe2-kernel-size",
        type=int,
        help="Positive odd PE2 kernel extent; use 5 for a 5x5x5 kernel.",
    )
    parser.add_argument(
        "--matrix-rolinpar",
        nargs=3,
        type=int,
        metavar=("RO", "LIN", "PAR"),
        help="Logical output matrix for the explicit-path interface.",
    )
    parser.add_argument("--acs-chunk", type=int, default=8)
    parser.add_argument("--calibration-chunk", type=int, default=4)
    parser.add_argument("--reconstruction-chunk", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume validated ACS, equation, and reconstruction checkpoints.",
    )
    return parser.parse_args(argv)


def resolve_grappa_run_inputs(args: argparse.Namespace) -> GrappaRunInputs:
    """Resolve the manifest or the backward-compatible explicit-path interface."""
    explicit_fields = (
        args.twix,
        args.coil_basis,
        args.output_prefix,
        args.ncc,
        args.regularization,
        args.pe2_kernel_size,
        args.matrix_rolinpar,
    )
    if args.dataset_manifest is not None:
        if any(value is not None for value in explicit_fields):
            raise ValueError(
                "--dataset-manifest cannot be combined with explicit dataset or GRAPPA options"
            )
        manifest = load_dataset_manifest(args.dataset_manifest)
        inspection = load_passed_inspection(manifest)
        sampling = list(manifest.payload["sampling"]["source_acceleration_pe1_pe2"])
        if sampling != [3, 1]:
            raise ValueError(
                "This GRAPPA implementation only supports source acceleration [3, 1]; "
                f"the manifest declares {sampling}. Fully sampled R1 data require the "
                "separate direct source-reconstruction path."
            )
        measured_sampling = inspection["twix"]["selected_measurement_sampling"]
        if (
            measured_sampling.get("image_inferred_pe1_stride") != 3
            or measured_sampling.get("image_pe1_residues_for_inferred_stride") != [1]
            or measured_sampling.get("refscan_covers_full_pe2") is not True
        ):
            raise ValueError(
                "Measured sampling is incompatible with this R3 GRAPPA operator: "
                "expected PE1 stride/residue 3/[1] and refscan coverage of every PE2."
            )
        reconstruction = manifest.payload["reconstruction"]
        kernel = list(reconstruction["grappa"]["kernel"])
        if kernel[:2] != [5, 5]:
            raise ValueError(
                f"This GRAPPA implementation requires a 5x5 RO/PE1 kernel, got {kernel}."
            )
        return GrappaRunInputs(
            twix=manifest.input_path("twix"),
            coil_basis=manifest.output_path("coil_compression_prefix").with_suffix(".npz"),
            output_prefix=manifest.output_path("source_reconstruction_prefix"),
            ncc=int(reconstruction["virtual_coils"]),
            regularization=float(reconstruction["grappa"]["regularization"]),
            pe2_kernel_size=int(kernel[2]),
            matrix_rolinpar=tuple(int(value) for value in manifest.payload["geometry"]["matrix"]),
            manifest=manifest,
        )

    if (
        args.twix is None
        or args.coil_basis is None
        or args.output_prefix is None
        or args.matrix_rolinpar is None
    ):
        raise ValueError(
            "Use --dataset-manifest, or provide --twix, --coil-basis, "
            "--output-prefix, and --matrix-rolinpar"
        )
    matrix = tuple(args.matrix_rolinpar)
    if any(value < 1 for value in matrix):
        raise ValueError("--matrix-rolinpar values must be positive")
    return GrappaRunInputs(
        twix=args.twix.expanduser().resolve(),
        coil_basis=args.coil_basis.expanduser().resolve(),
        output_prefix=args.output_prefix.expanduser().resolve(),
        ncc=12 if args.ncc is None else args.ncc,
        regularization=0.01 if args.regularization is None else args.regularization,
        pe2_kernel_size=3 if args.pe2_kernel_size is None else args.pe2_kernel_size,
        matrix_rolinpar=matrix,
        manifest=None,
    )


# Private aliases preserve compatibility with earlier tests and local imports.
_write_json = write_json_atomic
_load_basis = load_coil_basis
_open_or_create_memmap = open_or_create_complex64_npy
_validate_resume_pair = validate_resume_pair


def _progress_start(path: Path, *, resume: bool) -> int:
    """Return the next safe partition from a progress file when resuming."""
    if resume and path.is_file():
        return int(json.loads(path.read_text(encoding="utf-8"))["next_partition"])
    return 0


def build_compressed_acs(
    refscan: Any,
    basis: np.ndarray,
    path: Path,
    progress_path: Path,
    *,
    chunk_size: int,
    resume: bool,
) -> np.ndarray:
    """Checkpoint the compact fully sampled refscan as [RO,24,PE2,Ncc]."""
    configure_stream(refscan)
    shape = (
        int(refscan.sqzSize[0]),
        int(refscan.sqzSize[2]),
        int(refscan.sqzSize[3]),
        basis.shape[1],
    )
    _validate_resume_pair(path, progress_path, resume=resume)
    output = _open_or_create_memmap(path, shape, resume=resume)
    start_partition = _progress_start(progress_path, resume=resume)
    for start in range(start_partition, shape[2], chunk_size):
        stop = min(start + chunk_size, shape[2])
        raw = np.asarray(refscan[:, :, :, start:stop], dtype=np.complex64)
        coillast = np.transpose(raw, (0, 2, 3, 1))
        output[:, :, start:stop, :] = apply_coil_compression_coillast(
            coillast, basis
        )
        output.flush()
        _write_json(progress_path, {"next_partition": stop, "complete": stop == shape[2]})
        print(f"ACS checkpoint: PE2 {stop}/{shape[2]}", flush=True)
    if not np.isfinite(output).all():
        raise ValueError("Compressed ACS checkpoint contains non-finite samples.")
    return output


def _save_equations(
    path: Path, equations: NormalEquations3D, next_partition: int
) -> None:
    """Atomically save pooled equations and their next calibration partition."""
    payload: dict[str, np.ndarray] = {
        "next_partition": np.asarray(next_partition),
        "pe2_kernel_size": np.asarray(equations.pe2_kernel_size),
    }
    for offset in (1, 2):
        payload[f"shs_offset{offset}"] = equations.shs[offset]
        payload[f"sht_offset{offset}"] = equations.sht[offset]
        payload[f"rows_offset{offset}"] = np.asarray(equations.rows[offset])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(path)


def _load_equations(
    path: Path, ncoil: int, pe2_kernel_size: int
) -> tuple[NormalEquations3D, int]:
    """Restore a compatible 3D normal-equation calibration checkpoint."""
    with np.load(path) as archive:
        saved_kernel = int(archive["pe2_kernel_size"]) if "pe2_kernel_size" in archive else 3
        if saved_kernel != pe2_kernel_size:
            raise ValueError(
                f"Equation checkpoint PE2 kernel {saved_kernel} does not match {pe2_kernel_size}."
            )
        equations = NormalEquations3D.zeros(ncoil, pe2_kernel_size)
        for offset in (1, 2):
            equations.shs[offset] = np.asarray(archive[f"shs_offset{offset}"])
            equations.sht[offset] = np.asarray(archive[f"sht_offset{offset}"])
            equations.rows[offset] = int(archive[f"rows_offset{offset}"])
        next_partition = int(archive["next_partition"])
    return equations, next_partition


def calibrate_weights(
    acs: np.ndarray,
    equations_path: Path,
    weights_path: Path,
    *,
    chunk_size: int,
    regularization: float,
    pe2_kernel_size: int,
    resume: bool,
) -> dict[int, np.ndarray]:
    """Accumulate halo-aware 5×5×Kz equations and checkpoint every PE2 chunk."""
    halo = pe2_kernel_size // 2
    if resume and weights_path.is_file():
        with np.load(weights_path) as archive:
            saved_regularization = float(archive["regularization"])
            saved_kernel = int(archive["pe2_kernel_size"]) if "pe2_kernel_size" in archive else 3
            weights = {
                offset: np.asarray(archive[f"offset{offset}"]) for offset in (1, 2)
            }
        expected_shape = (10 * pe2_kernel_size * acs.shape[-1], acs.shape[-1])
        if saved_regularization != regularization or saved_kernel != pe2_kernel_size or any(
            weight.shape != expected_shape for weight in weights.values()
        ):
            raise ValueError("Saved 3D weights do not match kernel, Ncc, or regularization.")
        return weights
    if resume and equations_path.is_file():
        equations, start_partition = _load_equations(
            equations_path, acs.shape[-1], pe2_kernel_size
        )
    else:
        equations = NormalEquations3D.zeros(acs.shape[-1], pe2_kernel_size)
        start_partition = 0

    for start in range(start_partition, acs.shape[2], chunk_size):
        stop = min(start + chunk_size, acs.shape[2])
        halo_start = max(0, start - halo)
        halo_stop = min(acs.shape[2], stop + halo)
        block = np.asarray(acs[:, :, halo_start:halo_stop, :])
        core = np.arange(start - halo_start, stop - halo_start)
        equations.add(
            accumulate_normal_equations_3d(
                block, core, pe2_kernel_size=pe2_kernel_size
            )
        )
        _save_equations(equations_path, equations, stop)
        print(f"Equation checkpoint: PE2 {stop}/{acs.shape[2]}", flush=True)

    weights = solve_weights_3d(equations, regularization=regularization)
    np.savez(
        weights_path,
        offset1=weights[1],
        offset2=weights[2],
        regularization=np.asarray(regularization),
        pe2_kernel_size=np.asarray(pe2_kernel_size),
        rows_offset1=np.asarray(equations.rows[1]),
        rows_offset2=np.asarray(equations.rows[2]),
    )
    return weights


def reconstruct(
    image: Any,
    refscan: Any,
    basis: np.ndarray,
    weights: dict[int, np.ndarray],
    output_path: Path,
    progress_path: Path,
    *,
    chunk_size: int,
    pe2_kernel_size: int,
    matrix_rolinpar: tuple[int, int, int],
    resume: bool,
) -> dict[str, Any]:
    """Reconstruct haloed PE2 chunks and resume only after flushed boundaries."""
    configure_stream(image)
    configure_stream(refscan)
    nro, npe1, npe2 = matrix_rolinpar
    shape = (nro, npe1, npe2, basis.shape[1])
    for label, stream in (("image", image), ("refscan", refscan)):
        stream_shape = tuple(int(value) for value in stream.sqzSize)
        if len(stream_shape) != 4:
            raise ValueError(f"Expected {label} [RO,coil,PE1,PE2], got {stream_shape}.")
        if stream_shape[0] != nro or stream_shape[3] != npe2:
            raise ValueError(
                f"{label} RO/PE2 support {stream_shape[0]}/{stream_shape[3]} "
                f"does not match manifest matrix {nro}/{npe2}."
            )
        if int(stream.skipPar) != 0:
            raise ValueError(f"{label} PE2 support does not start at partition zero.")
    _validate_resume_pair(output_path, progress_path, resume=resume)
    start_partition = _progress_start(progress_path, resume=resume)
    if chunk_size < 1 or not 0 <= start_partition <= npe2:
        raise ValueError("Invalid reconstruction chunk size or resume partition.")
    image_lines = np.asarray(sorted({int(value) for value in image.Lin}))
    ref_lines = np.asarray(sorted({int(value) for value in refscan.Lin}))
    if (
        image_lines.size == 0
        or ref_lines.size == 0
        or image_lines.min() < 0
        or ref_lines.min() < 0
        or image_lines.max() >= npe1
        or ref_lines.max() >= npe1
    ):
        raise ValueError("Image/refscan PE1 counters are empty or outside the logical matrix.")
    acquired = np.zeros(npe1, dtype=bool)
    acquired[image_lines] = True
    acquired[ref_lines] = True
    ref_start = int(refscan.skipLin)
    ref_count = int(refscan.sqzSize[2])
    image_count = int(image.sqzSize[2])
    if image_count > npe1 or ref_start < 0 or ref_start + ref_count > npe1:
        raise ValueError("Compact image/refscan PE1 support exceeds the logical matrix.")
    output = _open_or_create_memmap(output_path, shape, resume=resume)
    started = time.perf_counter()
    halo = pe2_kernel_size // 2

    for start in range(start_partition, npe2, chunk_size):
        stop = min(start + chunk_size, npe2)
        halo_start = max(0, start - halo)
        halo_stop = min(npe2, stop + halo)
        image_raw = np.asarray(image[:, :, :, halo_start:halo_stop], np.complex64)
        ref_raw = np.asarray(refscan[:, :, :, halo_start:halo_stop], np.complex64)
        block = np.zeros(
            (nro, npe1, halo_stop - halo_start, basis.shape[1]), np.complex64
        )
        for local in range(halo_stop - halo_start):
            image_plane = np.transpose(image_raw[:, :, :, local], (0, 2, 1))
            ref_plane = np.transpose(ref_raw[:, :, :, local], (0, 2, 1))
            block[:, :image_count, local, :] = apply_coil_compression_coillast(
                image_plane, basis
            )
            block[:, ref_start : ref_start + ref_count, local, :] = (
                apply_coil_compression_coillast(ref_plane, basis)
            )
        core = np.arange(start - halo_start, stop - halo_start)
        reconstructed = apply_grappa_3d_block(
            block,
            core,
            acquired,
            weights,
            acquired_residue=1,
            pe2_kernel_size=pe2_kernel_size,
        )
        output[:, :, start:stop, :] = reconstructed
        output.flush()
        _write_json(progress_path, {"next_partition": stop, "complete": stop == npe2})
        print(f"Reconstruction checkpoint: PE2 {stop}/{npe2}", flush=True)

    finite = bool(np.isfinite(output).all())
    return {
        "output": str(output_path),
        "shape": list(shape),
        "dtype": "complex64",
        "finite": finite,
        "measured_pe1_lines": np.flatnonzero(acquired).tolist(),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Orchestrate resumable ACS caching, calibration, and reconstruction."""
    inputs = resolve_grappa_run_inputs(args)
    try:
        import mapvbvd
    except ImportError as exc:
        raise RuntimeError("pymapvbvd>=0.6.1 is required.") from exc

    twix_path = inputs.twix
    basis_path = inputs.coil_basis
    prefix = inputs.output_prefix
    if not twix_path.is_file():
        raise FileNotFoundError(f"TWIX file not found: {twix_path}")
    if not basis_path.is_file():
        raise FileNotFoundError(f"Coil-compression basis not found: {basis_path}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    basis = _load_basis(basis_path, inputs.ncc)
    pe2_offsets(inputs.pe2_kernel_size)
    root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_imaging_measurement(root)
    image = configure_stream(measurement["image"])
    refscan = configure_stream(measurement["refscan"])

    acs_path = prefix.with_name(prefix.name + "_compressed_acs.npy")
    acs_progress = prefix.with_name(prefix.name + "_acs_progress.json")
    equations_path = prefix.with_name(prefix.name + "_normal_equations.npz")
    weights_path = prefix.with_name(prefix.name + "_weights.npz")
    output_path = prefix.with_name(prefix.name + f"_full_ncc{inputs.ncc}.npy")
    recon_progress = prefix.with_name(prefix.name + "_recon_progress.json")
    report_path = prefix.with_name(prefix.name + "_report.json")
    run_paths = (
        acs_path,
        acs_progress,
        equations_path,
        weights_path,
        output_path,
        recon_progress,
        report_path,
    )
    if not args.resume:
        existing = [path for path in run_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "GRAPPA run artifacts already exist; use --resume or a new output prefix: "
                + ", ".join(str(path) for path in existing)
            )
    started = time.perf_counter()

    acs = build_compressed_acs(
        refscan,
        basis,
        acs_path,
        acs_progress,
        chunk_size=args.acs_chunk,
        resume=args.resume,
    )
    weights = calibrate_weights(
        acs,
        equations_path,
        weights_path,
        chunk_size=args.calibration_chunk,
        regularization=inputs.regularization,
        pe2_kernel_size=inputs.pe2_kernel_size,
        resume=args.resume,
    )
    reconstruction = reconstruct(
        image,
        refscan,
        basis,
        weights,
        output_path,
        recon_progress,
        chunk_size=args.reconstruction_chunk,
        pe2_kernel_size=inputs.pe2_kernel_size,
        matrix_rolinpar=inputs.matrix_rolinpar,
        resume=args.resume,
    )
    report = {
        "format_version": 1,
        "kernel_ro_pe1_pe2": [5, 5, inputs.pe2_kernel_size],
        "logical_matrix_rolinpar": list(inputs.matrix_rolinpar),
        "ncc": inputs.ncc,
        "regularization": inputs.regularization,
        "twix": str(twix_path),
        "measurement_index": measurement_index,
        "coil_basis": str(basis_path),
        "compressed_acs": str(acs_path),
        "normal_equations": str(equations_path),
        "weights": str(weights_path),
        "reconstruction": reconstruction,
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    if inputs.manifest is not None:
        report["dataset_manifest"] = inputs.manifest.provenance()
        report["dataset_inspection"] = {
            "path": str(inputs.manifest.inspection_report),
            "sha256": sha256_file(inputs.manifest.inspection_report),
        }
    _write_json(report_path, report)
    print(f"Report: {report_path}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the resumable command and convert expected failures to exit code 2."""
    try:
        run(_parse_args(argv))
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
