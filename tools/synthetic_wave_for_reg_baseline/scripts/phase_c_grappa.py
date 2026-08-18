#!/usr/bin/env python3
"""Calibrate, validate, and apply shared 2D R=3 GRAPPA weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from grappa_r3 import (
    NormalEquations,
    SOURCE_PE1_OFFSETS,
    accumulate_normal_equations,
    apply_grappa_plane,
    apply_grappa_volume,
    nrmse,
    solve_weights,
)
from phase_b_coil_compression import (
    apply_coil_compression_coillast,
    configure_stream,
    iter_refscan_coillast_chunks,
    select_product_measurement,
)


PYGRAPPA_REFERENCE = "pygrappa.grappa 0.26.3, kernel_size=(5,5)"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run held-out ACS validation and full shared-weight R=3 GRAPPA."
    )
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--coil-basis", required=True, type=Path, help="Phase B .npz file.")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--ncc", nargs="+", type=int, default=[12, 16, 24])
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--calibration-pe2-chunk", type=int, default=2)
    parser.add_argument("--reconstruction-pe2-chunk", type=int, default=4)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument(
        "--reuse-normal-equations",
        action="store_true",
        help="Reuse OUTPUT_PREFIX_normal_equations.npz after verifying its expected fields.",
    )
    parser.add_argument(
        "--skip-full-reconstruction",
        action="store_true",
        help="Stop after held-out NRMSE and weight export.",
    )
    return parser


def _load_basis(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "basis" not in archive:
            raise ValueError(f"Coil basis archive lacks 'basis': {path}")
        basis = np.asarray(archive["basis"], dtype=np.complex64)
    if basis.ndim != 2 or not np.isfinite(basis).all():
        raise ValueError(f"Invalid coil basis shape/data: {basis.shape}.")
    return basis


def _equations_to_arrays(
    all_equations: NormalEquations,
    holdout_equations: NormalEquations,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for label, equations in (("all", all_equations), ("holdout", holdout_equations)):
        for offset in (1, 2):
            arrays[f"{label}_shs_offset{offset}"] = equations.shs[offset]
            arrays[f"{label}_sht_offset{offset}"] = equations.sht[offset]
            arrays[f"{label}_rows_offset{offset}"] = np.asarray(equations.rows[offset])
    return arrays


def _equations_from_archive(path: Path) -> tuple[NormalEquations, NormalEquations]:
    """Load the pooled equations saved before validation/reconstruction."""
    with np.load(path) as archive:
        loaded = []
        for label in ("all", "holdout"):
            ncoil = int(archive[f"{label}_sht_offset1"].shape[1])
            equations = NormalEquations.zeros(ncoil)
            for offset in (1, 2):
                equations.shs[offset] = np.asarray(
                    archive[f"{label}_shs_offset{offset}"], dtype=np.complex128
                )
                equations.sht[offset] = np.asarray(
                    archive[f"{label}_sht_offset{offset}"], dtype=np.complex128
                )
                equations.rows[offset] = int(archive[f"{label}_rows_offset{offset}"])
            loaded.append(equations)
    return loaded[0], loaded[1]


def _weights_to_arrays(
    validation_weights: dict[int, dict[int, np.ndarray]],
    final_weights: dict[int, dict[int, np.ndarray]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for ncc, weights in validation_weights.items():
        for offset in (1, 2):
            arrays[f"validation_ncc{ncc}_offset{offset}"] = weights[offset]
    for ncc, weights in final_weights.items():
        for offset in (1, 2):
            arrays[f"final_ncc{ncc}_offset{offset}"] = weights[offset]
    return arrays


def _reference_source_hash() -> str:
    try:
        grappa_module = importlib.import_module("pygrappa.grappa")
    except ImportError:
        return "unavailable"
    path = Path(grappa_module.__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate_equations(
    refscan: Any,
    basis: np.ndarray,
    *,
    pe2_chunk: int,
    holdout_stride: int,
    holdout_remainder: int,
) -> tuple[NormalEquations, NormalEquations, list[int], dict[str, Any]]:
    """Accumulate all-partition and held-out-partition normal equations once."""
    max_ncc = basis.shape[1]
    all_equations = NormalEquations.zeros(max_ncc)
    holdout_equations = NormalEquations.zeros(max_ncc)
    holdout_partitions: list[int] = []
    chunks = 0
    started = time.perf_counter()

    for start, stop, physical in iter_refscan_coillast_chunks(
        refscan, pe2_chunk=pe2_chunk
    ):
        compressed = apply_coil_compression_coillast(physical, basis)
        all_equations.add(accumulate_normal_equations(compressed))

        local_holdout = [
            index
            for index, partition in enumerate(range(start, stop))
            if partition % holdout_stride == holdout_remainder
        ]
        if local_holdout:
            holdout_partitions.extend(start + index for index in local_holdout)
            holdout_equations.add(
                accumulate_normal_equations(compressed[:, :, local_holdout, :])
            )
        chunks += 1

    return all_equations, holdout_equations, holdout_partitions, {
        "chunks": chunks,
        "runtime_seconds": time.perf_counter() - started,
    }


def validate_weights(
    refscan: Any,
    basis: np.ndarray,
    validation_weights: dict[int, dict[int, np.ndarray]],
    holdout_partitions: Sequence[int],
    *,
    acquired_residue: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Artificially undersample held-out PE2 partitions and compute ACS NRMSE."""
    configure_stream(refscan)
    raw_lines = np.arange(int(refscan.skipLin), int(refscan.skipLin) + int(refscan.sqzSize[2]))
    acquired_mask = raw_lines % 3 == acquired_residue
    local_acquired_residue = int((acquired_residue - raw_lines[0]) % 3)
    target_offsets = (raw_lines - acquired_residue) % 3
    missing_masks = {offset: target_offsets == offset for offset in (1, 2)}

    accumulators = {
        ncc: {
            "error": 0.0,
            "truth": 0.0,
            "offset1_error": 0.0,
            "offset1_truth": 0.0,
            "offset2_error": 0.0,
            "offset2_truth": 0.0,
            "measured_unchanged": True,
        }
        for ncc in validation_weights
    }
    started = time.perf_counter()

    for partition in holdout_partitions:
        raw = np.asarray(refscan[:, :, :, partition], dtype=np.complex64)
        truth_max = apply_coil_compression_coillast(
            np.transpose(raw, (0, 2, 1)), basis
        )
        for ncc, weights in validation_weights.items():
            truth = truth_max[..., :ncc]
            undersampled = np.zeros_like(truth)
            undersampled[:, acquired_mask, :] = truth[:, acquired_mask, :]
            predicted = apply_grappa_plane(
                undersampled,
                acquired_mask,
                weights,
                acquired_residue=local_acquired_residue,
            )
            accumulators[ncc]["measured_unchanged"] &= bool(
                np.array_equal(predicted[:, acquired_mask, :], undersampled[:, acquired_mask, :])
            )
            for offset in (1, 2):
                mask = missing_masks[offset]
                difference = predicted[:, mask, :] - truth[:, mask, :]
                error = float(np.vdot(difference, difference).real)
                denominator = float(np.vdot(truth[:, mask, :], truth[:, mask, :]).real)
                accumulators[ncc]["error"] += error
                accumulators[ncc]["truth"] += denominator
                accumulators[ncc][f"offset{offset}_error"] += error
                accumulators[ncc][f"offset{offset}_truth"] += denominator

    metrics: dict[str, Any] = {}
    for ncc, values in accumulators.items():
        metrics[str(ncc)] = {
            "nrmse": nrmse(values["error"], values["truth"]),
            "offset1_nrmse": nrmse(values["offset1_error"], values["offset1_truth"]),
            "offset2_nrmse": nrmse(values["offset2_error"], values["offset2_truth"]),
            "measured_samples_bitwise_unchanged": values["measured_unchanged"],
        }
    return metrics, {
        "runtime_seconds": time.perf_counter() - started,
        "heldout_partition_count": len(holdout_partitions),
        "heldout_partitions": list(holdout_partitions),
        "raw_refscan_pe1_lines": raw_lines.tolist(),
        "local_acquired_residue": local_acquired_residue,
        "artificially_acquired_pe1_lines": raw_lines[acquired_mask].tolist(),
        "evaluated_missing_pe1_lines": raw_lines[~acquired_mask].tolist(),
    }


def _read_imaging_chunk(stream: Any, start: int, stop: int) -> np.ndarray:
    raw = np.asarray(stream[:, :, :, start:stop], dtype=np.complex64)
    expected = (
        int(stream.sqzSize[0]),
        int(stream.sqzSize[1]),
        int(stream.sqzSize[2]),
        stop - start,
    )
    if raw.shape != expected:
        raise ValueError(f"Image chunk {start}:{stop} shape {raw.shape}, expected {expected}.")
    return raw


def reconstruct_full_volume(
    image: Any,
    refscan: Any,
    basis: np.ndarray,
    weights: dict[int, np.ndarray],
    output_path: Path,
    *,
    pe2_chunk: int,
    acquired_residue: int,
) -> dict[str, Any]:
    """Merge product image/refscan data and fill every missing PE1 line."""
    configure_stream(image)
    configure_stream(refscan)
    ncc = basis.shape[1]
    nro = int(image.sqzSize[0])
    npe1 = 256
    npe2 = int(image.sqzSize[3])
    image_npe1 = int(image.sqzSize[2])
    ref_npe1 = int(refscan.sqzSize[2])
    ref_start = int(refscan.skipLin)

    image_lines = np.asarray(sorted({int(value) for value in image.Lin}), dtype=int)
    ref_lines = np.asarray(sorted({int(value) for value in refscan.Lin}), dtype=int)
    acquired_mask = np.zeros(npe1, dtype=bool)
    acquired_mask[image_lines] = True
    acquired_mask[ref_lines] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.complex64, shape=(nro, npe1, npe2, ncc)
    )
    measured_unchanged = True
    finite = True
    reconstructed_missing_sample_count = 0
    started = time.perf_counter()

    for start in range(0, npe2, pe2_chunk):
        stop = min(start + pe2_chunk, npe2)
        image_raw = _read_imaging_chunk(image, start, stop)
        ref_raw = np.asarray(refscan[:, :, :, start:stop], dtype=np.complex64)
        expected_ref = (nro, int(refscan.sqzSize[1]), ref_npe1, stop - start)
        if ref_raw.shape != expected_ref:
            raise ValueError(
                f"Refscan chunk {start}:{stop} shape {ref_raw.shape}, expected {expected_ref}."
            )

        planes = np.zeros((nro, npe1, stop - start, ncc), dtype=np.complex64)
        for local in range(stop - start):
            image_plane = np.transpose(image_raw[:, :, :, local], (0, 2, 1))
            ref_plane = np.transpose(ref_raw[:, :, :, local], (0, 2, 1))
            planes[:, :image_npe1, local, :] = apply_coil_compression_coillast(
                image_plane, basis
            )
            # The PAT refscan is authoritative in its 24-line support, matching
            # the established Wave loader convention of overwriting ACS lines.
            planes[:, ref_start : ref_start + ref_npe1, local, :] = apply_coil_compression_coillast(
                ref_plane, basis
            )
        reconstructed = apply_grappa_volume(
            planes,
            acquired_mask,
            weights,
            acquired_residue=acquired_residue,
        )
        measured_unchanged &= bool(
            np.array_equal(
                reconstructed[:, acquired_mask, :, :], planes[:, acquired_mask, :, :]
            )
        )
        finite &= bool(np.isfinite(reconstructed).all())
        reconstructed_missing_sample_count += int(
            np.count_nonzero(reconstructed[:, ~acquired_mask, :, :])
        )
        output[:, :, start:stop, :] = reconstructed
        # Do not flush the full multi-gigabyte memmap after every small chunk;
        # on network storage that repeatedly syncs the entire dirty mapping.

    output.flush()

    central_partition = npe2 // 2
    central_kspace = np.asarray(output[:, :, central_partition, :])
    coil_images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(central_kspace, axes=(0, 1)), axes=(0, 1), norm="ortho"),
        axes=(0, 1),
    )
    central_rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=-1)).astype(np.float32)
    diagnostics_path = output_path.with_name(output_path.stem + "_central_rss.npz")
    np.savez(
        diagnostics_path,
        rss=central_rss,
        partition=np.asarray(central_partition),
        acquired_pe1_mask=acquired_mask,
    )
    del output
    return {
        "output_path": str(output_path),
        "shape": [nro, npe1, npe2, ncc],
        "dtype": "complex64",
        "size_bytes": output_path.stat().st_size,
        "acquired_pe1_lines": np.flatnonzero(acquired_mask).tolist(),
        "missing_pe1_line_count_before_grappa": int(np.count_nonzero(~acquired_mask)),
        "measured_samples_bitwise_unchanged": measured_unchanged,
        "all_output_samples_finite": finite,
        "nonzero_reconstructed_missing_sample_count": reconstructed_missing_sample_count,
        "expected_missing_sample_count": int(nro * np.count_nonzero(~acquired_mask) * npe2 * ncc),
        "central_rss_diagnostics": str(diagnostics_path),
        "runtime_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mapvbvd
    except ImportError as exc:
        raise RuntimeError("Phase C requires pymapvbvd>=0.6.1.") from exc

    twix_path = args.twix.expanduser().resolve()
    basis_path = args.coil_basis.expanduser().resolve()
    prefix = args.output_prefix.expanduser().resolve()
    if not twix_path.is_file() or not basis_path.is_file():
        raise FileNotFoundError("TWIX file or coil-basis archive does not exist.")
    if args.holdout_stride < 2 or not 0 <= args.holdout_remainder < args.holdout_stride:
        raise ValueError("Invalid holdout stride/remainder.")

    basis_max = _load_basis(basis_path)
    ncc_values = sorted(set(int(value) for value in args.ncc))
    if not ncc_values or ncc_values[-1] > basis_max.shape[1]:
        raise ValueError("Requested Ncc exceeds the saved Phase B basis.")

    twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_product_measurement(twix_root)
    image = configure_stream(measurement["image"])
    refscan = configure_stream(measurement["refscan"])

    equations_path = prefix.with_name(prefix.name + "_normal_equations.npz")
    weights_path = prefix.with_name(prefix.name + "_weights.npz")
    report_path = prefix.with_name(prefix.name + "_report.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    heldout = [
        partition
        for partition in range(int(refscan.sqzSize[3]))
        if partition % args.holdout_stride == args.holdout_remainder
    ]
    if args.reuse_normal_equations:
        if not equations_path.is_file():
            raise FileNotFoundError(f"Normal-equation cache not found: {equations_path}")
        all_equations, holdout_equations = _equations_from_archive(equations_path)
        calibration_info = {
            "reused": True,
            "cache": str(equations_path),
            "runtime_seconds": 0.0,
        }
    else:
        all_equations, holdout_equations, calibrated_holdout, calibration_info = calibrate_equations(
            refscan,
            basis_max,
            pe2_chunk=args.calibration_pe2_chunk,
            holdout_stride=args.holdout_stride,
            holdout_remainder=args.holdout_remainder,
        )
        if calibrated_holdout != heldout:
            raise ValueError("Calibrated holdout partitions do not match the configured split.")
        # Persist the expensive pooled calibration before solving, validation,
        # or full-volume reconstruction.
        np.savez(equations_path, **_equations_to_arrays(all_equations, holdout_equations))
    training_equations = all_equations.subtract(holdout_equations)
    validation_weights = {
        ncc: solve_weights(
            training_equations,
            max_ncc=basis_max.shape[1],
            ncc=ncc,
            regularization=args.regularization,
        )
        for ncc in ncc_values
    }
    final_weights = {
        ncc: solve_weights(
            all_equations,
            max_ncc=basis_max.shape[1],
            ncc=ncc,
            regularization=args.regularization,
        )
        for ncc in ncc_values
    }
    metrics, validation_info = validate_weights(
        refscan,
        basis_max,
        validation_weights,
        heldout,
        acquired_residue=1,
    )
    selected_ncc = min(ncc_values, key=lambda value: metrics[str(value)]["nrmse"])

    np.savez(weights_path, **_weights_to_arrays(validation_weights, final_weights))

    report: dict[str, Any] = {
        "format_version": 1,
        "phase": "C - shared 2D R=3 GRAPPA",
        "twix": str(twix_path),
        "measurement_index": measurement_index,
        "coil_basis": str(basis_path),
        "tested_ncc": ncc_values,
        "selected_ncc": selected_ncc,
        "selection_rule": "lowest held-out ACS NRMSE",
        "kernel_nominal_ro_pe1": [5, 5],
        "readout_offsets": [-2, -1, 0, 1, 2],
        "source_pe1_offsets_by_target_type": {
            str(key): list(value) for key, value in SOURCE_PE1_OFFSETS.items()
        },
        "pe2_kernel_extent": 1,
        "regularization": args.regularization,
        "regularization_scaling": "lambda * Frobenius_norm(S^H S) / feature_count",
        "reference": PYGRAPPA_REFERENCE,
        "pygrappa_version": metadata.version("pygrappa"),
        "pygrappa_grappa_source_sha256": _reference_source_hash(),
        "calibration": calibration_info,
        "normal_equation_rows_all": all_equations.rows,
        "normal_equation_rows_holdout": holdout_equations.rows,
        "normal_equations_file": str(equations_path),
        "weights_file": str(weights_path),
        "heldout_validation": validation_info,
        "heldout_metrics": metrics,
        "full_reconstruction": None,
    }
    report["report_file"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.skip_full_reconstruction:
        output_path = prefix.with_name(prefix.name + f"_full_ncc{selected_ncc}.npy")
        report["full_reconstruction"] = reconstruct_full_volume(
            image,
            refscan,
            basis_max[:, :selected_ncc],
            final_weights[selected_ncc],
            output_path,
            pe2_chunk=args.reconstruction_pe2_chunk,
            acquired_residue=1,
        )

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run(args)
    for ncc in report["tested_ncc"]:
        value = report["heldout_metrics"][str(ncc)]["nrmse"]
        print(f"Ncc={ncc}: held-out ACS NRMSE={value:.8f}")
    print(f"Selected Ncc={report['selected_ncc']} by lowest held-out NRMSE")
    if report["full_reconstruction"]:
        print(f"Full k-space: {report['full_reconstruction']['output_path']}")
    print(f"Report: {report['report_file']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
