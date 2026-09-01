#!/usr/bin/env python3
"""Apply a validated post-Wave sampling mask and export BART inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from checkpoint_io import write_json_atomic
from dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
)
from sampling_mask import (
    historical_cartesian_with_full_pe1_acs_mask,
    load_product_mask,
    write_masked_bart_kspace,
)
from wave_synthesis import logical_array_sha256, logical_bart_cfl_sha256, sha256_file


def _build_parser() -> argparse.ArgumentParser:
    """Build the BART Wave input finalization command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--synthesis-dir", type=Path)
    parser.add_argument("--sampling-report", type=Path)
    parser.add_argument(
        "--visual-review-approved",
        action="store_true",
        help="Required acknowledgement that the full-Wave diagnostics were approved.",
    )
    parser.add_argument("--pe2-chunk", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file to avoid partial manifests."""
    write_json_atomic(path, payload)


def link_bart_pair(source_base: Path, output_base: Path, *, replace: bool) -> None:
    """Link a BART header/CFL pair without duplicating the theoretical PSF."""
    for suffix in (".hdr", ".cfl"):
        source = source_base.with_suffix(suffix)
        output = output_base.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(source)
        if output.exists() or output.is_symlink():
            if not replace:
                raise FileExistsError(output)
            output.unlink()
        output.symlink_to(source)


def _manifest_mask_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact post-Wave target-mask contract used for restart matching."""
    sampling = contract["sampling"]
    acceleration = sampling["synthetic_wave_acceleration_pe1_pe2"]
    residues = sampling["synthetic_wave_residue_pe1_pe2"]
    return {
        "mask_kind": sampling["synthetic_wave_mask_kind"],
        "pe1_acceleration": int(acceleration[0]),
        "pe2_acceleration": int(acceleration[1]),
        "pe1_residue": int(residues[0]),
        "pe2_residue": int(residues[1]),
        "acs_pe1_start": int(sampling["synthetic_wave_acs_pe1_start"]),
        "acs_pe1_stop_exclusive": int(
            sampling["synthetic_wave_acs_pe1_stop_exclusive"]
        ),
    }


def _completed_manifest_export_reusable(
    manifest_path: Path,
    *,
    dataset_sha256: str,
    source_manifest_sha256: str,
    config: Mapping[str, Any],
) -> bool:
    """Accept reuse only when provenance, configuration, and payload hashes match."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "manifest_bart_inputs_ready":
            return False
        if manifest.get("dataset_manifest", {}).get("sha256") != dataset_sha256:
            return False
        if manifest.get("source_synthesis_manifest", {}).get(
            "sha256"
        ) != source_manifest_sha256:
            return False
        if manifest.get("config") != dict(config):
            return False
        mask_path = Path(manifest["sampling_mask"]["path"])
        kspace_path = Path(manifest["masked_wave_kspace"]["cfl"])
        psf_path = Path(manifest["psf"]["cfl"])
        return (
            mask_path.is_file()
            and kspace_path.is_file()
            and psf_path.is_file()
            and logical_array_sha256(np.load(mask_path, mmap_mode="r"))
            == manifest["sampling_mask"]["logical_sha256"]
            and sha256_file(kspace_path)
            == manifest["masked_wave_kspace"]["cfl_sha256"]
            and sha256_file(psf_path) == manifest["psf"]["cfl_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Export a manifest-defined retrospective mask after full Wave encoding."""
    if args.synthesis_dir is not None or args.sampling_report is not None or args.overwrite:
        raise ValueError(
            "--dataset-manifest cannot be combined with --synthesis-dir, "
            "--sampling-report, or --overwrite"
        )
    if not args.visual_review_approved:
        raise ValueError("Refusing export without full-Wave visual-review approval.")

    dataset = load_dataset_manifest(args.dataset_manifest)
    load_passed_inspection(dataset)
    source_dir = dataset.output_path("wave_synthesis_dir")
    output_dir = dataset.output_path("bart_export_dir")
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "awaiting_visual_review_before_mask_and_bart":
        raise ValueError("Full-Wave synthesis has not reached the visual-review gate.")
    if source_manifest.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Full-Wave synthesis uses a stale dataset manifest.")

    full_wave_info = source_manifest.get("full_wave_kspace", {})
    if full_wave_info.get("sampling_mask_applied") is not False:
        raise ValueError("The source must be unmasked full Wave k-space.")
    contract = dataset.payload
    matrix = tuple(int(value) for value in contract["geometry"]["matrix"])
    expected_shape = (
        int(contract["wave_synthesis"]["extended_readout_samples"]),
        matrix[1],
        matrix[2],
        int(contract["reconstruction"]["virtual_coils"]),
    )
    source_shape = tuple(int(value) for value in full_wave_info.get("shape", ()))
    source_full_wave = Path(full_wave_info.get("path", "")).resolve()
    if source_shape != expected_shape:
        raise ValueError(
            f"Expected full Wave shape {expected_shape}, got {source_shape}."
        )
    if not source_full_wave.is_file():
        raise FileNotFoundError(source_full_wave)
    source_array = np.load(source_full_wave, mmap_mode="r")
    if source_array.shape != expected_shape or source_array.dtype != np.complex64:
        raise ValueError("Full Wave file shape or dtype differs from its manifest contract.")

    source_psf = source_manifest.get("psf", {})
    source_psf_base = Path(source_psf.get("bart_base", "")).resolve()
    psf_shape = tuple(int(value) for value in source_psf.get("bart_shape", ()))
    expected_psf_shape = (*expected_shape[:3], 1, 1)
    if psf_shape != expected_psf_shape:
        raise ValueError(
            f"Expected BART PSF shape {expected_psf_shape}, got {psf_shape}."
        )
    source_psf_hash = logical_bart_cfl_sha256(source_psf_base, psf_shape)
    if source_psf_hash != source_psf.get("logical_sha256"):
        raise ValueError("BART PSF differs from the validated synthesis PSF.")

    config = _manifest_mask_config(contract)
    manifest_path = output_dir / "manifest.json"
    if args.resume and _completed_manifest_export_reusable(
        manifest_path,
        dataset_sha256=dataset.sha256,
        source_manifest_sha256=source_manifest_sha256,
        config=config,
    ):
        print(f"Reusing validated manifest BART inputs: {output_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    recover_incomplete = False
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            recover_incomplete = (
                prior.get("status") == "exporting_manifest_bart_inputs"
                and prior.get("dataset_manifest", {}).get("sha256") == dataset.sha256
                and prior.get("source_synthesis_manifest", {}).get("sha256")
                == source_manifest_sha256
                and prior.get("config") == config
            )
        if not recover_incomplete:
            raise FileExistsError(f"Output directory is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = {
        "format_version": 1,
        "status": "exporting_manifest_bart_inputs",
        "dataset_manifest": dataset.provenance(),
        "source_synthesis_manifest": {
            "path": str(source_manifest_path),
            "sha256": source_manifest_sha256,
        },
        "config": config,
        "mask_application_order": "full source -> Wave encoding -> target sampling mask",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, started)

    mask, mask_info = historical_cartesian_with_full_pe1_acs_mask(
        expected_shape[1:3],
        accelerations=(config["pe1_acceleration"], config["pe2_acceleration"]),
        residues=(config["pe1_residue"], config["pe2_residue"]),
        fully_sampled_pe1_lines=np.arange(
            config["acs_pe1_start"], config["acs_pe1_stop_exclusive"]
        ),
    )
    mask_path = output_dir / "sampling_mask.npy"
    if mask_path.exists() and not recover_incomplete:
        raise FileExistsError(mask_path)
    np.save(mask_path, mask)
    mask_info.update(
        {
            "path": str(mask_path),
            "dtype": str(mask.dtype),
            "logical_sha256": logical_array_sha256(mask),
            "applied_after_wave_encoding": True,
        }
    )

    bart_dir = output_dir / "bart_inputs"
    bart_dir.mkdir(exist_ok=True)
    kspace_info = write_masked_bart_kspace(
        source_full_wave,
        mask,
        bart_dir / "wave_kspace",
        pe2_chunk=args.pe2_chunk,
        overwrite=recover_incomplete,
    )
    kspace_info["cfl_sha256"] = sha256_file(Path(kspace_info["cfl"]))

    output_psf_base = bart_dir / "psf"
    link_bart_pair(source_psf_base, output_psf_base, replace=recover_incomplete)
    output_psf_hash = logical_bart_cfl_sha256(output_psf_base, psf_shape)
    if output_psf_hash != source_psf_hash:
        raise ValueError("Exported BART PSF differs from the full-Wave synthesis PSF.")
    psf_info = {
        "basename": "psf",
        "base": str(output_psf_base),
        "header": str(output_psf_base.with_suffix(".hdr")),
        "cfl": str(output_psf_base.with_suffix(".cfl")),
        "shape": list(psf_shape),
        "logical_sha256": output_psf_hash,
        "cfl_sha256": sha256_file(output_psf_base.with_suffix(".cfl")),
        "symlink_source_base": str(source_psf_base),
        "identical_to_source_synthesis_psf": True,
    }

    if sha256_file(source_manifest_path) != source_manifest_sha256:
        raise ValueError("Full-Wave synthesis manifest changed during BART export.")
    completed_at = datetime.now(timezone.utc).isoformat()
    bart_manifest = {
        "format_version": 1,
        "format": "BART CFL",
        "status": "masked_wave_inputs_ready_for_map_estimation_and_reconstruction",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "dataset_manifest": dataset.provenance(),
        "source_synthesis_manifest": started["source_synthesis_manifest"],
        "sampling_mask": mask_info,
        "full_wave_kspace": full_wave_info,
        "masked_wave_kspace": kspace_info,
        "psf": psf_info,
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": kspace_info["shape"],
                "wave_kspace_norm": kspace_info["norm"],
                "psf": "psf",
                "psf_shape": list(psf_shape),
            }
        ],
        "finalized_at_utc": completed_at,
    }
    bart_manifest_path = bart_dir / "manifest.json"
    _write_json(bart_manifest_path, bart_manifest)

    completed = {
        **started,
        "status": "manifest_bart_inputs_ready",
        "sampling_mask": mask_info,
        "masked_wave_kspace": kspace_info,
        "psf": psf_info,
        "bart_input_manifest": str(bart_manifest_path),
        "completed_at_utc": completed_at,
    }
    _write_json(manifest_path, completed)
    print(f"Manifest BART input manifest: {bart_manifest_path}")
    return completed


def _run_product(args: argparse.Namespace) -> dict[str, Any]:
    """Apply the verified product mask and finalize BART input provenance."""
    if args.synthesis_dir is None or args.sampling_report is None:
        raise ValueError(
            "Use --dataset-manifest, or provide --synthesis-dir and --sampling-report"
        )
    if args.resume:
        raise ValueError("--resume requires --dataset-manifest")
    if not args.visual_review_approved:
        raise ValueError("Refusing to apply the product mask before visual-review approval.")
    synthesis_dir = args.synthesis_dir.expanduser().resolve()
    sampling_report = args.sampling_report.expanduser().resolve()
    manifest_path = synthesis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mask, mask_info, inspection_report = load_product_mask(sampling_report)
    report_twix = Path(inspection_report["twix"]["path"]).resolve()
    synthesis_twix = Path(manifest["source_twix"]).resolve()
    if report_twix != synthesis_twix:
        raise ValueError(
            f"Sampling report TWIX {report_twix} differs from synthesis TWIX {synthesis_twix}."
        )
    if mask.shape != tuple(manifest["full_wave_kspace"]["shape"][1:3]):
        raise ValueError("Product mask dimensions do not match the full Wave PE grid.")

    mask_path = synthesis_dir / "r3x1_product_sampling_mask.npy"
    if mask_path.exists() and not args.overwrite:
        raise FileExistsError(mask_path)
    np.save(mask_path, mask)
    mask_hash = logical_array_sha256(mask)

    bart_dir = synthesis_dir / "bart_inputs"
    kspace_info = write_masked_bart_kspace(
        manifest["full_wave_kspace"]["path"],
        mask,
        bart_dir / "wave_kspace",
        pe2_chunk=args.pe2_chunk,
        overwrite=args.overwrite,
    )

    psf_info = manifest["psf"]
    psf = np.load(psf_info["npy"], mmap_mode="r")
    canonical_hash = logical_array_sha256(psf)
    bart_shape = tuple(int(value) for value in psf_info["bart_shape"])
    bart_hash = logical_bart_cfl_sha256(psf_info["bart_base"], bart_shape)
    if canonical_hash != bart_hash or canonical_hash != psf_info["logical_sha256"]:
        raise ValueError("The canonical and BART theoretical PSF payloads no longer match.")

    finalized_at = datetime.now(timezone.utc).isoformat()
    mask_provenance = {
        **mask_info,
        "path": str(mask_path),
        "dtype": str(mask.dtype),
        "logical_sha256": mask_hash,
        "sampling_report": str(sampling_report),
        "sampling_report_sha256": sha256_file(sampling_report),
        "selected_measurement_index": int(
            inspection_report["twix"]["selected_measurement_index"]
        ),
    }
    bart_manifest = {
        "format_version": 1,
        "format": "BART CFL",
        "status": "masked_wave_inputs_ready_for_map_estimation_and_reconstruction",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "visual_review": {
            "approved": True,
            "approval_source": "user confirmation after reviewing full-Wave diagnostics",
        },
        "sampling_mask": mask_provenance,
        "full_wave_kspace": manifest["full_wave_kspace"],
        "masked_wave_kspace": kspace_info,
        "psf": {
            "basename": "psf",
            "shape": list(bart_shape),
            "logical_sha256": bart_hash,
            "identical_to_synthesis_psf": True,
        },
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": kspace_info["shape"],
                "wave_kspace_norm": kspace_info["norm"],
                "psf": "psf",
                "psf_shape": list(bart_shape),
            }
        ],
        "finalized_at_utc": finalized_at,
    }
    bart_manifest_path = bart_dir / "manifest.json"
    _write_json(bart_manifest_path, bart_manifest)

    manifest["status"] = "masked_bart_inputs_ready"
    manifest["visual_review"] = {
        "approved": True,
        "approval_source": "user confirmation after reviewing full-Wave diagnostics",
    }
    manifest["sampling_mask"] = mask_provenance
    manifest["masked_wave_kspace"] = kspace_info
    manifest["bart_input_manifest"] = str(bart_manifest_path)
    manifest["finalized_at_utc"] = finalized_at
    _write_json(manifest_path, manifest)
    print(f"BART input manifest: {bart_manifest_path}")
    return bart_manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch to the portable manifest route or the compatible product route."""
    if args.dataset_manifest is not None:
        return _run_manifest(args)
    return _run_product(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Run BART input finalization from command-line arguments."""
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
        print(f"Error: {exc}")
        raise SystemExit(2) from exc
