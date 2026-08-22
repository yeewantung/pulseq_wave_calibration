#!/usr/bin/env python3
"""Validate PSF=1 identity and full-sampling Wave inversion on real source data."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from checkpoint_io import write_json_atomic
from dataset_manifest import load_dataset_manifest, sha256_file
from wave_synthesis import (
    SPATIAL_AXES,
    apply_wave_adjoint,
    apply_wave_forward,
    center_embed_readout,
    centered_fftn,
    logical_array_sha256,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def complex_error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    x_chunk: int = 16,
) -> dict[str, float]:
    """Calculate bounded-memory complex error relative to one reference."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim < 1:
        raise ValueError(
            f"Error arrays must share a nonempty shape: {reference.shape}, {candidate.shape}."
        )
    reference_energy = 0.0
    error_energy = 0.0
    maximum_reference = 0.0
    maximum_error = 0.0
    for start in range(0, reference.shape[0], x_chunk):
        stop = min(start + x_chunk, reference.shape[0])
        reference_block = np.asarray(reference[start:stop], dtype=np.complex64)
        candidate_block = np.asarray(candidate[start:stop], dtype=np.complex64)
        difference = candidate_block - reference_block
        reference_energy += float(np.vdot(reference_block, reference_block).real)
        error_energy += float(np.vdot(difference, difference).real)
        maximum_reference = max(
            maximum_reference, float(np.max(np.abs(reference_block), initial=0.0))
        )
        maximum_error = max(
            maximum_error, float(np.max(np.abs(difference), initial=0.0))
        )
    if reference_energy <= 0 or maximum_reference <= 0:
        raise ValueError("Operator reference has no signal energy.")
    return {
        "relative_complex_l2": math.sqrt(error_energy / reference_energy),
        "maximum_complex_error": maximum_error,
        "relative_maximum_complex_error": maximum_error / maximum_reference,
    }


def validate_coil_operator(
    source_kspace: np.ndarray,
    full_wave_kspace: np.ndarray,
    psf: np.ndarray,
    *,
    workers: int,
) -> dict[str, Any]:
    """Run both operator gates for one coil without presentation transforms."""
    source_kspace = np.asarray(source_kspace, dtype=np.complex64)
    full_wave_kspace = np.asarray(full_wave_kspace, dtype=np.complex64)
    psf = np.asarray(psf, dtype=np.complex64)
    if source_kspace.ndim != 3 or full_wave_kspace.ndim != 3 or psf.ndim != 3:
        raise ValueError("Source, full-Wave, and PSF inputs must each be 3D.")
    if full_wave_kspace.shape != psf.shape:
        raise ValueError("Full-Wave k-space and PSF shapes differ.")
    if source_kspace.shape[1:] != psf.shape[1:]:
        raise ValueError("Source and Wave phase-encoding shapes differ.")

    source_image = centered_fftn(
        source_kspace, axes=SPATIAL_AXES, inverse=True, workers=workers
    )
    unity_psf = np.broadcast_to(
        np.asarray(1.0 + 0.0j, dtype=np.complex64), source_image.shape
    )
    unity_reencoded = apply_wave_forward(source_image, unity_psf, workers=workers)
    unity_metrics = complex_error_metrics(source_kspace, unity_reencoded)
    del unity_reencoded

    extended_image, support = center_embed_readout(source_image, psf.shape[0])
    recovered = apply_wave_adjoint(full_wave_kspace, psf, workers=workers)
    full_sampling_metrics = complex_error_metrics(extended_image, recovered)
    exterior_energy = float(
        np.vdot(recovered[: support.start], recovered[: support.start]).real
        + np.vdot(recovered[support.stop :], recovered[support.stop :]).real
    )
    total_energy = float(np.vdot(recovered, recovered).real)
    full_sampling_metrics["recovered_exterior_energy_fraction"] = (
        exterior_energy / total_energy if total_energy > 0 else float("nan")
    )
    return {
        "psf_one_no_wave_identity": unity_metrics,
        "full_sampling_wave_inverse": full_sampling_metrics,
        "readout_embedding_half_open": [int(support.start), int(support.stop)],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Dedicated output directory; defaults below the dataset output root at "
            "evaluation/full_sampling_wave_operator_validation."
        ),
    )
    parser.add_argument("--fft-workers", type=int, default=4)
    parser.add_argument("--relative-l2-tolerance", type=float, default=5e-6)
    parser.add_argument("--relative-max-tolerance", type=float, default=2e-5)
    parser.add_argument(
        "--psf-magnitude-tolerance",
        type=float,
        default=3e-7,
        help="Maximum absolute |PSF|-1 deviation; 3e-7 is approximately three float32 ULPs.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _reusable(
    path: Path,
    *,
    dataset_sha256: str,
    source_report_sha256: str,
    synthesis_sha256: str,
    source_identity: dict[str, Any],
    wave_identity: dict[str, Any],
    psf_identity: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_json(path)
        return (
            payload.get("status") == "passed"
            and payload.get("dataset_manifest", {}).get("sha256") == dataset_sha256
            and payload.get("source_reconstruction_report", {}).get("sha256")
            == source_report_sha256
            and payload.get("synthesis_manifest", {}).get("sha256") == synthesis_sha256
            and payload.get("inputs", {}).get("source_no_wave_kspace") == source_identity
            and payload.get("inputs", {}).get("full_wave_kspace") == wave_identity
            and {
                key: payload.get("inputs", {})
                .get("theoretical_psf", {})
                .get(key)
                for key in psf_identity
            }
            == psf_identity
            and payload.get("settings") == settings
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset_manifest(args.dataset_manifest)
    output_dir = (
        dataset.output_root / "evaluation" / "full_sampling_wave_operator_validation"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_path = output_dir / "operator_validation_manifest.json"
    if args.fft_workers < 1:
        raise ValueError("FFT workers must be positive.")
    for value, name in (
        (args.relative_l2_tolerance, "relative L2 tolerance"),
        (args.relative_max_tolerance, "relative maximum tolerance"),
        (args.psf_magnitude_tolerance, "PSF magnitude tolerance"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    reconstruction = dataset.payload["reconstruction"]
    matrix = tuple(int(value) for value in dataset.payload["geometry"]["matrix"])
    coils = int(reconstruction["virtual_coils"])
    prefix = dataset.output_path("source_reconstruction_prefix")
    source_path = prefix.with_name(prefix.name + f"_full_ncc{coils}.npy")
    source_report_path = prefix.with_name(prefix.name + "_report.json")
    synthesis_manifest_path = dataset.output_path("wave_synthesis_dir") / "manifest.json"
    for path in (source_path, source_report_path, synthesis_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_report = _load_json(source_report_path)
    synthesis = _load_json(synthesis_manifest_path)
    if source_report.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Source report does not match the current dataset manifest.")
    if synthesis.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Synthesis manifest does not match the current dataset manifest.")
    if synthesis.get("status") != "awaiting_visual_review_before_mask_and_bart":
        raise ValueError("Full-Wave synthesis manifest has an unexpected status.")

    full_wave_path = Path(synthesis["full_wave_kspace"]["path"]).resolve()
    psf_path = Path(synthesis["psf"]["npy"]).resolve()
    for path in (full_wave_path, psf_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_identity = _file_identity(source_path)
    wave_identity = _file_identity(full_wave_path)
    psf_identity = _file_identity(psf_path)
    source_report_sha256 = sha256_file(source_report_path)
    synthesis_sha256 = sha256_file(synthesis_manifest_path)
    settings = {
        "fft_workers": int(args.fft_workers),
        "relative_l2_tolerance": float(args.relative_l2_tolerance),
        "relative_max_tolerance": float(args.relative_max_tolerance),
        "psf_magnitude_tolerance": float(args.psf_magnitude_tolerance),
        "all_virtual_coils_required": True,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume and _reusable(
            output_path,
            dataset_sha256=dataset.sha256,
            source_report_sha256=source_report_sha256,
            synthesis_sha256=synthesis_sha256,
            source_identity=source_identity,
            wave_identity=wave_identity,
            psf_identity=psf_identity,
            settings=settings,
        ):
            print(f"Reusing passed operator validation: {output_path}")
            return _load_json(output_path)
        raise FileExistsError(f"Operator-validation output is not safely reusable: {output_dir}")

    source = np.load(source_path, mmap_mode="r")
    full_wave = np.load(full_wave_path, mmap_mode="r")
    psf = np.load(psf_path, mmap_mode="r")
    expected_source = (*matrix, coils)
    expected_wave = (psf.shape[0], matrix[1], matrix[2], coils)
    if source.shape != expected_source or source.dtype != np.complex64:
        raise ValueError(f"Unexpected no-Wave source array: {source.shape}, {source.dtype}.")
    if full_wave.shape != expected_wave or full_wave.dtype != np.complex64:
        raise ValueError(f"Unexpected full-Wave array: {full_wave.shape}, {full_wave.dtype}.")
    if psf.shape != expected_wave[:3] or psf.dtype != np.complex64:
        raise ValueError(f"Unexpected theoretical PSF: {psf.shape}, {psf.dtype}.")
    if logical_array_sha256(psf) != synthesis["psf"]["logical_sha256"]:
        raise ValueError("Theoretical PSF logical hash changed.")
    maximum_psf_magnitude_deviation = float(np.max(np.abs(np.abs(psf) - 1.0)))
    if maximum_psf_magnitude_deviation > args.psf_magnitude_tolerance:
        raise ValueError("Full-sampling inversion requires a unit-magnitude PSF.")

    coil_records = []
    for coil_index in range(coils):
        metrics = validate_coil_operator(
            source[..., coil_index],
            full_wave[..., coil_index],
            psf,
            workers=args.fft_workers,
        )
        metrics["coil"] = coil_index + 1
        coil_records.append(metrics)
        print(f"Validated Wave operator coil {coil_index + 1:02d}/{coils:02d}", flush=True)

    gate_names = ("psf_one_no_wave_identity", "full_sampling_wave_inverse")
    aggregate = {}
    passed = True
    for gate_name in gate_names:
        gate = {
            "maximum_relative_complex_l2": max(
                record[gate_name]["relative_complex_l2"] for record in coil_records
            ),
            "maximum_relative_maximum_complex_error": max(
                record[gate_name]["relative_maximum_complex_error"]
                for record in coil_records
            ),
        }
        gate["passed"] = (
            gate["maximum_relative_complex_l2"] <= args.relative_l2_tolerance
            and gate["maximum_relative_maximum_complex_error"]
            <= args.relative_max_tolerance
        )
        aggregate[gate_name] = gate
        passed &= bool(gate["passed"])
    aggregate["maximum_recovered_exterior_energy_fraction"] = max(
        record["full_sampling_wave_inverse"]["recovered_exterior_energy_fraction"]
        for record in coil_records
    )
    if not passed:
        raise RuntimeError(f"Full-sampling Wave operator validation failed: {aggregate}")

    payload = {
        "format_version": 1,
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "real-data PSF=1 identity and full-sampling Wave inverse gates",
        "dataset_manifest": dataset.provenance(),
        "source_reconstruction_report": {
            "path": str(source_report_path),
            "sha256": source_report_sha256,
        },
        "synthesis_manifest": {
            "path": str(synthesis_manifest_path),
            "sha256": synthesis_sha256,
        },
        "inputs": {
            "source_no_wave_kspace": source_identity,
            "full_wave_kspace": wave_identity,
            "theoretical_psf": {
                **psf_identity,
                "logical_sha256": synthesis["psf"]["logical_sha256"],
            },
        },
        "settings": settings,
        "scientific_scope": {
            "presentation_processing_used": False,
            "bart_reconstruction_used": False,
            "psf_one_role": "source-grid no-Wave forward identity",
            "full_sampling_role": (
                "adjoint is the exact inverse for the unit-magnitude theoretical PSF"
            ),
        },
        "aggregate": aggregate,
        "maximum_psf_magnitude_deviation": maximum_psf_magnitude_deviation,
        "coils": coil_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    print(f"Operator validation manifest: {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
