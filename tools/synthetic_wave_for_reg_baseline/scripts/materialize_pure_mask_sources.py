#!/usr/bin/env python3
"""Rebuild cleanup-approved synthetic source intermediates in the new run tree."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from assemble_fully_sampled_no_wave_kspace import run as assemble_no_wave
from evaluate_pure_mask_sweeps import _canonical_mask
from pure_mask_rerun import (
    json_object_sha256,
    load_json,
    resolve_config_path,
    sha256_file,
    validate_bart_artifact,
    validate_csm_rss_normalization,
    validate_manifest_binding,
    validate_psf_unit_magnitude,
    write_json_atomic,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RETRO_ROOT = REPOSITORY_ROOT / "tools" / "wave_retro_lr_recon"
if str(RETRO_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRO_ROOT))

from wave_retro_lr.bart_io import open_cfl, read_shape  # noqa: E402
from wave_retro_lr.retrospective import synthesize_wave_from_no_wave_crop  # noqa: E402


CASE_SHAPES = {
    "native_r3x1": ((256, 256, 256, 12, 1), (1024, 256, 256, 1, 1)),
    "native_r3x2": ((256, 256, 256, 12, 1), (1024, 256, 256, 1, 1)),
    "lr_x_r3x2": ((256, 256, 172, 12, 1), (1024, 256, 172, 1, 1)),
    "lr_y_r3x2": ((256, 172, 256, 12, 1), (1024, 172, 256, 1, 1)),
    "lr_xy_r3x2": ((256, 204, 204, 12, 1), (1024, 204, 204, 1, 1)),
}


def _parser() -> argparse.ArgumentParser:
    """Build the source-materialization command interface.

    Returns:
        Parser requiring the ignored rerun configuration.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _artifact_path(
    specification: Mapping[str, Any], config_dir: Path, label: str
) -> tuple[Path, str]:
    """Validate one file path and expected SHA-256 digest.

    Args:
        specification: Object containing ``path`` and ``sha256``.
        config_dir: Directory resolving relative configuration paths.
        label: Human-readable error label.

    Returns:
        Resolved file path and validated digest.
    """
    path = resolve_config_path(specification.get("path"), config_dir, f"{label}.path")
    digest = str(specification.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} is absent or differs from its accepted hash: {path}")
    return path, digest


def _validate_reused_case_inputs(config: Mapping[str, Any], config_dir: Path) -> None:
    """Validate every accepted CSM and theoretical PSF before materialization.

    Args:
        config: Parsed ignored rerun configuration.
        config_dir: Directory resolving its relative binding-manifest paths.

    Returns:
        None. Invalid hashes, provenance, shapes, or values raise ``ValueError``.
    """
    case_configs = config.get("cases")
    if not isinstance(case_configs, Mapping) or set(case_configs) != set(CASE_SHAPES):
        raise ValueError("Pure-mask source validation requires exactly five cases.")
    csm_cache: dict[str, dict[str, Any]] = {}
    psf_cache: dict[str, dict[str, Any]] = {}
    validation = config.get("validation", {})
    for case_id, (csm_shape, psf_shape) in CASE_SHAPES.items():
        case = case_configs[case_id]
        csm_key = str(case["csm"]["base"])
        if csm_key not in csm_cache:
            csm_base, _record = validate_bart_artifact(
                case["csm"],
                config_dir,
                expected_shape=csm_shape,
                required_assertion_labels={
                    "dataset", "fov", "dimensions", "coil_order", "calibration_source"
                },
                label=f"{case_id} accepted CSM",
            )
            csm = np.asarray(open_cfl(csm_base)).reshape(csm_shape, order="F")[..., 0]
            csm_cache[csm_key] = validate_csm_rss_normalization(
                csm,
                support_threshold=float(validation.get("csm_support_threshold", 1e-6)),
                tolerance=float(validation.get("csm_rss_tolerance", 5e-3)),
            )
        psf_key = str(case["psf"]["base"])
        if psf_key not in psf_cache:
            psf_base, _record = validate_bart_artifact(
                case["psf"],
                config_dir,
                expected_shape=psf_shape,
                required_assertion_labels={
                    "dataset", "fov", "dimensions", "trajectory", "psf_model",
                    "wave_data_origin",
                },
                label=f"{case_id} theoretical PSF",
            )
            psf = np.asarray(open_cfl(psf_base)).reshape(psf_shape, order="F")[..., 0, 0]
            psf_cache[psf_key] = validate_psf_unit_magnitude(
                psf,
                tolerance=float(validation.get("psf_unit_magnitude_tolerance", 2e-5)),
            )
    source = config.get("source", {})
    bet = source.get("approved_bet_mask", {})
    bet_path, _bet_hash = _artifact_path(bet, config_dir, "approved BET mask")
    _canonical_mask(
        bet_path,
        expected_shape_xyz=(256, 256, 256),
        expected_fov_mm_xyz=(256.0, 256.0, 256.0),
    )
    validate_manifest_binding(
        bet.get("manifest", {}),
        config_dir,
        required_assertion_labels={"approval", "geometry"},
        label="approved BET mask",
    )
    evaluation = config.get("evaluation", {})
    validate_manifest_binding(
        evaluation.get("orientation_manifest", {}),
        config_dir,
        required_assertion_labels={"orientation", "canonical_ras"},
        label="accepted logical-to-canonical orientation",
    )


def validate_materialization_config(config_path: str | Path) -> dict[str, Any]:
    """Validate immutable inputs for deterministic source reconstruction.

    Args:
        config_path: Ignored pure-mask rerun configuration.

    Returns:
        Resolved inputs and fixed source-materialization destinations.
    """
    path = Path(config_path).expanduser().resolve()
    config = load_json(path, "pure-mask rerun configuration")
    spec = config.get("source_materialization")
    if not isinstance(spec, Mapping):
        raise ValueError("source_materialization must be a JSON object.")
    config_dir = path.parent
    output_root = resolve_config_path(config.get("output_root"), config_dir, "output_root")
    dataset_manifest, dataset_hash = _artifact_path(
        spec.get("dataset_manifest", {}), config_dir, "dataset manifest"
    )
    coil_basis, coil_basis_hash = _artifact_path(
        spec.get("coil_basis", {}), config_dir, "accepted coil basis"
    )
    source_report, source_report_hash = _artifact_path(
        spec.get("source_report", {}), config_dir, "accepted source report"
    )
    inventory, inventory_hash = _artifact_path(
        spec.get("historical_inventory", {}), config_dir, "historical cleanup inventory"
    )
    dataset = load_json(dataset_manifest, "dataset manifest")
    report = load_json(source_report, "accepted no-Wave source report")
    inventory_payload = load_json(inventory, "historical cleanup inventory")
    no_wave_hash = str(spec.get("expected_no_wave_sha256", ""))
    full_wave_hash = str(spec.get("expected_full_wave_sha256", ""))
    archived = {
        str(item.get("path")): str(item.get("sha256"))
        for item in inventory_payload.get("entries", [])
        if isinstance(item, Mapping)
    }
    if archived.get("reconstructions/no_wave/source_full_ncc12.npy") != no_wave_hash:
        raise ValueError("Historical inventory does not bind the accepted no-Wave hash.")
    if archived.get("synthetic_wave/full_encoding/full_wave_kspace.npy") != full_wave_hash:
        raise ValueError("Historical inventory does not bind the accepted full-Wave hash.")
    if report.get("dataset_manifest", {}).get("sha256") != dataset_hash:
        raise ValueError("Accepted source report and dataset manifest differ.")
    if report.get("run_signature", {}).get("coil_basis", {}).get("sha256") != coil_basis_hash:
        raise ValueError("Accepted source report and coil basis differ.")
    _validate_reused_case_inputs(config, config_dir)
    twix = Path(dataset["inputs"]["twix"]).expanduser().resolve()
    sequence = Path(dataset["inputs"]["wave_sequence"]).expanduser().resolve()
    for required in (twix, sequence):
        if not required.is_file():
            raise FileNotFoundError(required)
    recorded_twix = report["run_signature"]["twix"]
    twix_stat = twix.stat()
    if (
        twix_stat.st_size != int(recorded_twix["size_bytes"])
        or twix_stat.st_mtime_ns != int(recorded_twix["mtime_ns"])
    ):
        raise ValueError("Raw TWIX identity differs from the accepted source report.")
    psf_spec = spec.get("accepted_native_theoretical_psf")
    if not isinstance(psf_spec, Mapping):
        raise ValueError("accepted_native_theoretical_psf must be an object.")
    psf_base = resolve_config_path(psf_spec.get("base"), config_dir, "accepted PSF base")
    psf_shape = read_shape(psf_base)
    expected_psf_shape = (1024, 256, 256, 1, 1)
    if psf_shape[:5] != expected_psf_shape or any(value != 1 for value in psf_shape[5:]):
        raise ValueError(f"Accepted native theoretical PSF has wrong shape: {psf_shape}")
    for suffix, key in ((".hdr", "header_sha256"), (".cfl", "payload_sha256")):
        if sha256_file(psf_base.with_suffix(suffix)) != psf_spec.get(key):
            raise ValueError("Accepted native theoretical PSF hash changed.")
    materialization_root = output_root / "source_materialization"
    no_wave_prefix = materialization_root / "no_wave" / "source"
    no_wave_path = no_wave_prefix.with_name("source_full_ncc12.npy")
    full_wave_path = materialization_root / "full_wave_kspace.npy"
    return {
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "config": config,
        "dataset_manifest": str(dataset_manifest),
        "dataset_sha256": dataset_hash,
        "coil_basis": str(coil_basis),
        "coil_basis_sha256": coil_basis_hash,
        "source_report": str(source_report),
        "source_report_sha256": source_report_hash,
        "historical_inventory": str(inventory),
        "historical_inventory_sha256": inventory_hash,
        "twix": str(twix),
        "sequence": str(sequence),
        "psf_base": str(psf_base),
        "expected_no_wave_sha256": no_wave_hash,
        "expected_full_wave_sha256": full_wave_hash,
        "output_root": str(output_root),
        "materialization_root": str(materialization_root),
        "no_wave_prefix": str(no_wave_prefix),
        "no_wave_path": str(no_wave_path),
        "full_wave_path": str(full_wave_path),
    }


def _materialize_full_wave(validated: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    """Synthesize the accepted full Wave array using the reused theoretical PSF.

    Args:
        validated: Resolved immutable source-materialization contract.
        resume: Continue a matching per-coil checkpoint when true.

    Returns:
        Hash and shape record for the reproduced full-Wave NPY.
    """
    no_wave = np.load(validated["no_wave_path"], mmap_mode="r", allow_pickle=False)
    psf = np.asarray(open_cfl(validated["psf_base"])).squeeze()
    expected_no_wave_shape = (256, 256, 256, 12)
    expected_full_wave_shape = (1024, 256, 256, 12)
    if no_wave.shape != expected_no_wave_shape or no_wave.dtype != np.complex64:
        raise ValueError("Rebuilt no-Wave source has the wrong shape or dtype.")
    if psf.shape != expected_full_wave_shape[:3] or not np.isfinite(psf).all():
        raise ValueError("Accepted theoretical PSF is non-finite or has the wrong grid.")
    output = Path(validated["full_wave_path"])
    progress_path = output.with_name("full_wave_progress.json")
    signature = json_object_sha256(
        {
            "no_wave_sha256": validated["expected_no_wave_sha256"],
            "psf_payload_sha256": validated["config"]["source_materialization"]
            ["accepted_native_theoretical_psf"]["payload_sha256"],
            "shape": list(expected_full_wave_shape),
        }
    )
    start = 0
    if output.exists():
        if not resume or not progress_path.is_file():
            raise FileExistsError("Full-Wave checkpoint exists without an approved resume.")
        progress = load_json(progress_path, "full-Wave materialization progress")
        if progress.get("signature_sha256") != signature:
            raise ValueError("Full-Wave checkpoint has stale provenance.")
        start = int(progress.get("next_coil", -1))
        full_wave = np.load(output, mmap_mode="r+", allow_pickle=False)
        if full_wave.shape != expected_full_wave_shape or full_wave.dtype != np.complex64:
            raise ValueError("Full-Wave checkpoint has the wrong shape or dtype.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        full_wave = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.complex64,
            shape=expected_full_wave_shape,
            fortran_order=True,
        )
    if not 0 <= start <= expected_full_wave_shape[3]:
        raise ValueError("Full-Wave checkpoint has an invalid next-coil index.")
    workers = int(validated["config"].get("runtime", {}).get("fft_workers", 4))
    for coil in range(start, expected_full_wave_shape[3]):
        full_wave[..., coil] = synthesize_wave_from_no_wave_crop(
            no_wave[..., coil],
            psf,
            readout_oversampled=expected_full_wave_shape[0],
            target_mask=None,
            fft_workers=workers,
        )
        full_wave.flush()
        write_json_atomic(
            progress_path,
            {
                "format_version": 1,
                "signature_sha256": signature,
                "next_coil": coil + 1,
                "complete": coil + 1 == expected_full_wave_shape[3],
            },
        )
        print(f"Materialized full synthetic Wave coil {coil + 1:02d}/12", flush=True)
    del full_wave
    digest = sha256_file(output)
    if digest != validated["expected_full_wave_sha256"]:
        raise ValueError("Rebuilt full-Wave source differs from its archived accepted hash.")
    return {
        "path": str(output),
        "sha256": digest,
        "shape": list(expected_full_wave_shape),
        "dtype": "complex64",
        "wave_data_origin": "synthetic_from_fully_sampled_no_wave",
        "psf_model": "theoretical_sequence_trajectory_without_calibrated_correction",
        "calibration_samples_merged_into_wave_kspace": False,
    }


def materialize(
    config_path: str | Path,
    *,
    validate_only: bool,
    confirmed_output_root: Path | None,
    resume: bool,
) -> dict[str, Any]:
    """Validate or reproduce the cleanup-approved source intermediates.

    Args:
        config_path: Ignored pure-mask rerun configuration.
        validate_only: Validate immutable inputs without output writes.
        confirmed_output_root: Exact user-approved run root required for writes.
        resume: Reuse matching source and full-Wave checkpoints.

    Returns:
        Validation snapshot or completed materialization manifest.
    """
    validated = validate_materialization_config(config_path)
    if validate_only:
        if confirmed_output_root is not None:
            raise ValueError("--confirm-output-root is not used with --validate-only.")
        assemble_no_wave(
            Namespace(
                dataset_manifest=Path(validated["dataset_manifest"]),
                output_prefix=Path(validated["no_wave_prefix"]),
                pe2_chunk=4,
                resume=False,
                validate_only=True,
            )
        )
        return validated
    root = Path(validated["output_root"])
    if confirmed_output_root is None or confirmed_output_root.expanduser().resolve() != root:
        raise ValueError("Source materialization requires the exact confirmed output root.")
    unexpected = [] if not root.exists() else [
        item for item in root.iterdir() if item.name != "source_materialization"
    ]
    if unexpected:
        raise FileExistsError("Output root contains artifacts outside source_materialization.")
    report = assemble_no_wave(
        Namespace(
            dataset_manifest=Path(validated["dataset_manifest"]),
            output_prefix=Path(validated["no_wave_prefix"]),
            pe2_chunk=4,
            resume=resume,
            validate_only=False,
        )
    )
    no_wave_path = Path(validated["no_wave_path"])
    no_wave_digest = sha256_file(no_wave_path)
    if no_wave_digest != validated["expected_no_wave_sha256"]:
        raise ValueError("Rebuilt no-Wave source differs from its archived accepted hash.")
    full_wave = _materialize_full_wave(validated, resume=resume)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "dataset": {"sha256": validated["dataset_sha256"]},
        "geometry": {"fov_mm_xyz": [256.0, 256.0, 256.0]},
        "coil_order": {"basis_sha256": validated["coil_basis_sha256"]},
        "trajectory": {"sequence_sha256": sha256_file(Path(validated["sequence"]))},
        "source_report": {
            "path": validated["source_report"],
            "sha256": validated["source_report_sha256"],
        },
        "historical_inventory": {
            "path": validated["historical_inventory"],
            "sha256": validated["historical_inventory_sha256"],
        },
        "no_wave": {
            "path": str(no_wave_path),
            "sha256": no_wave_digest,
            "shape": [256, 256, 256, 12],
        },
        "full_wave": full_wave,
        "accepted_theoretical_psf": validated["config"]["source_materialization"]
        ["accepted_native_theoretical_psf"],
        "assembly_report": report,
    }
    manifest_path = Path(validated["materialization_root"]) / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"Pure-mask source materialization manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and validate or materialize accepted sources.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after validation or successful source reproduction.
    """
    args = _parser().parse_args(argv)
    materialize(
        args.config,
        validate_only=args.validate_only,
        confirmed_output_root=args.confirm_output_root,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
