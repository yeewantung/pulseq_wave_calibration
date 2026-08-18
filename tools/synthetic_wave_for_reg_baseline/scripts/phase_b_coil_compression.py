#!/usr/bin/env python3
"""Estimate one virtual-coil basis from a product TWIX PAT refscan.

This follows the covariance/eigendecomposition convention used by the verified
Wave-MPRAGE and Wave-GRE ``coil_compression_kspace.py`` utility. The adaptation
here is data access: the 256 x 24 x 256 product refscan is read in PE2 chunks so
the full roughly 0.8 GiB complex array is never retained in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import scipy.linalg as la


REFERENCE_UTILITY = (
    "sources/published_code/wave-mprage/recon/utils/"
    "coil_compression_kspace.py"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate and validate coil compression from product TWIX refscan data."
    )
    parser.add_argument("--twix", required=True, type=Path, help="Product Siemens TWIX file.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output prefix; writes .npz matrix data and .json metadata.",
    )
    parser.add_argument(
        "--ncc",
        nargs="+",
        type=int,
        default=[12, 16, 24],
        help="Virtual-coil counts to report; the largest basis is saved.",
    )
    parser.add_argument(
        "--pe2-chunk", type=int, default=8, help="Number of refscan PE2 partitions read at once."
    )
    parser.add_argument(
        "--readout-step",
        type=int,
        default=4,
        help="Readout stride used for covariance estimation, matching the reference utility.",
    )
    return parser


def select_product_measurement(twix_root: Any) -> tuple[int, Any]:
    """Select the measurement with the largest populated image stream."""
    measurements = list(twix_root) if isinstance(twix_root, (list, tuple)) else [twix_root]
    candidates = []
    for index, measurement in enumerate(measurements):
        image = measurement.get("image") if hasattr(measurement, "get") else None
        if image is not None and int(getattr(image, "NAcq", 0)) > 0:
            candidates.append((int(image.NAcq), index, measurement))
    if not candidates:
        raise ValueError("No TWIX measurement contains a populated image stream.")
    _, index, measurement = max(candidates, key=lambda item: (item[0], item[1]))
    return index, measurement


def configure_stream(stream: Any) -> Any:
    """Configure mapVBVD for oversampling removal and compact NumPy output."""
    stream.flagRemoveOS = True
    stream.squeeze = True
    return stream


def iter_refscan_coillast_chunks(
    refscan: Any,
    *,
    pe2_chunk: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield refscan chunks in canonical ``[RO, PE1, PE2, coil]`` layout."""
    if pe2_chunk < 1:
        raise ValueError("pe2_chunk must be positive.")
    configure_stream(refscan)
    size = tuple(int(value) for value in refscan.sqzSize)
    if len(size) != 4:
        raise ValueError(f"Expected refscan [RO, coil, PE1, PE2], received {size}.")
    nro, ncoil, npe1, npe2 = size

    for start in range(0, npe2, pe2_chunk):
        stop = min(start + pe2_chunk, npe2)
        raw = np.asarray(refscan[:, :, :, start:stop], dtype=np.complex64)
        expected = (nro, ncoil, npe1, stop - start)
        if raw.shape != expected:
            raise ValueError(
                f"Refscan chunk {start}:{stop} has shape {raw.shape}; expected {expected}."
            )
        yield start, stop, np.transpose(raw, (0, 2, 3, 1))


def accumulate_coil_covariance(
    chunks: Iterator[tuple[int, int, np.ndarray]],
    *,
    ncoil: int,
    readout_step: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accumulate ``XᴴX`` from coil-last chunks without retaining the design matrix."""
    if readout_step < 1:
        raise ValueError("readout_step must be positive.")
    covariance = np.zeros((ncoil, ncoil), dtype=np.complex128)
    rows = 0
    nonzero_rows = 0
    chunk_count = 0

    for _, _, chunk in chunks:
        if chunk.ndim != 4 or chunk.shape[-1] != ncoil:
            raise ValueError(
                f"Expected chunk [RO, PE1, PE2, {ncoil}], received {chunk.shape}."
            )
        design = np.ascontiguousarray(chunk[::readout_step]).reshape(-1, ncoil)
        finite = np.isfinite(design).all(axis=1)
        if not finite.all():
            raise ValueError("Refscan contains non-finite calibration samples.")
        nonzero = np.any(design != 0, axis=1)
        usable = design[nonzero]
        covariance += usable.conj().T @ usable
        rows += int(design.shape[0])
        nonzero_rows += int(usable.shape[0])
        chunk_count += 1

    if nonzero_rows == 0:
        raise ValueError("Refscan contains no nonzero calibration samples.")
    covariance = 0.5 * (covariance + covariance.conj().T)
    return covariance, {
        "chunk_count": chunk_count,
        "sample_rows_considered": rows,
        "nonzero_sample_rows": nonzero_rows,
        "zero_sample_rows_removed": rows - nonzero_rows,
    }


def coil_basis_from_covariance(
    covariance: np.ndarray,
    *,
    max_ncc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the leading eigenvectors, singular values, and retained energy."""
    covariance = np.asarray(covariance, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Coil covariance must be square.")
    ncoil = covariance.shape[0]
    if not 1 <= max_ncc <= ncoil:
        raise ValueError(f"max_ncc must be between 1 and {ncoil}.")

    eigenvalues, eigenvectors = la.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order].real, 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("Coil covariance has no positive energy.")

    basis = eigenvectors[:, :max_ncc].astype(np.complex64, copy=False)
    singular_values = np.sqrt(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues) / total
    return basis, singular_values, cumulative_energy


def apply_coil_compression_coillast(data: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply a shared ``[physical coil, virtual coil]`` basis to coil-last data."""
    data = np.asarray(data, dtype=np.complex64)
    basis = np.asarray(basis, dtype=np.complex64)
    if data.ndim < 2 or basis.ndim != 2 or data.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"Incompatible coil-last data {data.shape} and basis {basis.shape}."
        )
    return np.matmul(data, basis).astype(np.complex64, copy=False)


def _read_probe(stream: Any, raw_line: int, raw_partition: int) -> np.ndarray:
    """Read one acquired block and return canonical ``[RO, coil]`` data."""
    configure_stream(stream)
    local_line = raw_line - int(stream.skipLin)
    local_partition = raw_partition - int(stream.skipPar)
    block = np.asarray(stream[:, :, local_line, local_partition], dtype=np.complex64)
    if block.ndim != 2:
        raise ValueError(f"Expected probe [RO, coil], received {block.shape}.")
    return block


def _probe_validation(data: np.ndarray, basis: np.ndarray, ncc_values: Sequence[int]) -> dict[str, Any]:
    input_energy = float(np.vdot(data, data).real)
    results: dict[str, Any] = {
        "input_shape": list(data.shape),
        "input_energy": input_energy,
        "ncc": {},
    }
    for ncc in ncc_values:
        compressed = apply_coil_compression_coillast(data, basis[:, :ncc])
        output_energy = float(np.vdot(compressed, compressed).real)
        results["ncc"][str(ncc)] = {
            "output_shape": list(compressed.shape),
            "all_finite": bool(np.isfinite(compressed).all()),
            "retained_probe_energy": output_energy / input_energy,
        }
    return results


def run_phase_b(
    twix_path: Path,
    *,
    ncc_values: Sequence[int],
    pe2_chunk: int,
    readout_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run chunked covariance estimation and validate one common coil basis."""
    try:
        import mapvbvd
    except ImportError as exc:
        raise RuntimeError("Phase B requires pymapvbvd>=0.6.1.") from exc

    ncc_values = sorted(set(int(value) for value in ncc_values))
    if not ncc_values or ncc_values[0] < 1:
        raise ValueError("At least one positive --ncc value is required.")

    started = time.perf_counter()
    twix_root = mapvbvd.mapVBVD(str(twix_path), quiet=True)
    measurement_index, measurement = select_product_measurement(twix_root)
    if "refscan" not in measurement:
        raise ValueError("Selected product measurement has no refscan stream.")
    image = configure_stream(measurement["image"])
    refscan = configure_stream(measurement["refscan"])
    ncoil = int(refscan.NCha)
    if int(image.NCha) != ncoil:
        raise ValueError(f"Image/refscan coil mismatch: {int(image.NCha)} vs {ncoil}.")
    if ncc_values[-1] > ncoil:
        raise ValueError(f"Requested Ncc={ncc_values[-1]} but only {ncoil} coils exist.")

    refscan_shape = tuple(int(value) for value in refscan.sqzSize)
    covariance, accumulation = accumulate_coil_covariance(
        iter_refscan_coillast_chunks(refscan, pe2_chunk=pe2_chunk),
        ncoil=ncoil,
        readout_step=readout_step,
    )
    basis, singular_values, cumulative_energy = coil_basis_from_covariance(
        covariance, max_ncc=ncc_values[-1]
    )

    orthogonality_error = float(
        np.linalg.norm(basis.conj().T @ basis - np.eye(basis.shape[1]), ord="fro")
    )
    image_probe = _read_probe(image, raw_line=min(int(v) for v in image.Lin), raw_partition=0)
    refscan_probe = _read_probe(
        refscan, raw_line=min(int(v) for v in refscan.Lin), raw_partition=0
    )
    report = {
        "format_version": 1,
        "phase": "B - coil compression",
        "twix": str(twix_path.resolve()),
        "measurement_index": measurement_index,
        "measurement_selection_rule": "largest image-stream acquisition count",
        "reference_utility": REFERENCE_UTILITY,
        "algorithm": "coil covariance eigendecomposition",
        "physical_coil_count": ncoil,
        "reported_virtual_coil_counts": ncc_values,
        "saved_basis_shape": list(basis.shape),
        "refscan_mapvbvd_shape_ro_coil_pe1_pe2": list(refscan_shape),
        "refscan_canonical_shape_ro_pe1_pe2_coil": [
            refscan_shape[0], refscan_shape[2], refscan_shape[3], refscan_shape[1]
        ],
        "pe2_chunk": pe2_chunk,
        "readout_step": readout_step,
        "accumulation": accumulation,
        "cumulative_energy": {
            str(ncc): float(cumulative_energy[ncc - 1]) for ncc in ncc_values
        },
        "basis_orthogonality_frobenius_error": orthogonality_error,
        "image_probe": _probe_validation(image_probe, basis, ncc_values),
        "refscan_probe": _probe_validation(refscan_probe, basis, ncc_values),
        "runtime_seconds": time.perf_counter() - started,
        "heldout_acs_grappa_nrmse": "deferred to Phase C because it requires GRAPPA",
    }
    arrays = {
        "basis": basis,
        "singular_values": singular_values,
        "cumulative_energy": cumulative_energy,
        "covariance": covariance,
    }
    return report, arrays


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    twix_path = args.twix.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    if not twix_path.is_file():
        raise FileNotFoundError(f"TWIX file not found: {twix_path}")

    report, arrays = run_phase_b(
        twix_path,
        ncc_values=args.ncc,
        pe2_chunk=args.pe2_chunk,
        readout_step=args.readout_step,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    np.savez(npz_path, **arrays)
    report["matrix_file"] = str(npz_path)
    report["report_file"] = str(json_path)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Physical coils: {report['physical_coil_count']}")
    for ncc, energy in report["cumulative_energy"].items():
        print(f"Ncc={ncc}: covariance energy retained={energy:.8%}")
    print(f"Basis: {npz_path}")
    print(f"Report: {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
