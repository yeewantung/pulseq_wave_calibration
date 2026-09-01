#!/usr/bin/env python3
"""Validate or prepare the five corrected pure-mask synthetic Wave cases."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pure_mask_rerun import (
    CASE_IDS,
    PureMaskCase,
    array_is_finite,
    bart_base,
    direct_fft_reference,
    link_bart_pair,
    logical_array_sha256,
    open_cfl,
    output_layout,
    sha256_file,
    synthesize_wave_from_no_wave_crop,
    validate_config,
    write_json_atomic,
    write_masked_wave_cfl,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and validate or prepare pure-mask BART inputs.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after validation or completed preparation.
    """
    args = _parser().parse_args(argv)
    result = prepare(
        args.config,
        validate_only=args.validate_only,
        confirmed_output_root=args.confirm_output_root,
        resume=args.resume,
    )
    if args.validate_only:
        print(json.dumps(result["layout"], indent=2))
        print("Validated five pure-mask input contracts; no output was written.")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the pure-mask preparation command-line interface.

    Returns:
        Parser requiring an ignored local configuration.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read and validate every immutable input without creating outputs.",
    )
    parser.add_argument(
        "--confirm-output-root",
        type=Path,
        help="Exact user-approved output root; required for production preparation.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form.

    Returns:
        Timezone-aware UTC string.
    """
    return datetime.now(timezone.utc).isoformat()


def _complete_preparation_reusable(
    manifest_path: Path, validated: dict[str, Any]
) -> bool:
    """Check whether a complete preparation remains hash-identical.

    Args:
        manifest_path: Existing root preparation manifest.
        validated: Fresh read-only validation result.

    Returns:
        ``True`` only when configuration, source records, and every case output match.
    """
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            return False
        if manifest.get("config", {}).get("immutable_contract_sha256") != validated[
            "config"
        ]["immutable_contract_sha256"]:
            return False
        if manifest.get("source") != validated["source"]:
            return False
        for case_id in CASE_IDS:
            record = manifest["cases"][case_id]
            case_manifest = Path(record["case_manifest"])
            if not case_manifest.is_file() or sha256_file(case_manifest) != record["case_manifest_sha256"]:
                return False
            case_payload = json.loads(case_manifest.read_text(encoding="utf-8"))
            if case_payload.get("status") != "pure_mask_bart_inputs_ready":
                return False
            mask = np.load(case_payload["sampling_mask"]["path"], allow_pickle=False)
            if logical_array_sha256(mask) != case_payload["sampling_mask"]["logical_sha256"]:
                return False
            for artifact in ("wave_kspace", "coil_sens", "psf"):
                payload_path = Path(case_payload["bart_inputs"][artifact]["base"]).with_suffix(".cfl")
                if sha256_file(payload_path) != case_payload["bart_inputs"][artifact]["payload_sha256"]:
                    return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _case_full_wave(
    case: PureMaskCase,
    no_wave_crop: np.ndarray,
    native_full_wave: np.ndarray,
    case_root: Path,
    *,
    extended_readout: int,
    fft_workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve accepted native or newly synthesized target-grid full Wave data.

    Args:
        case: Validated case contract.
        no_wave_crop: Direct centered no-Wave target-grid k-space.
        native_full_wave: Accepted full native synthetic Wave source.
        case_root: Destination case directory for LR full-Wave data.
        extended_readout: Fixed Wave readout dimension.
        fft_workers: Maximum FFT worker count.

    Returns:
        Full Wave array and immutable source provenance record.
    """
    if case.case_id in {"native_r3x1", "native_r3x2"}:
        return native_full_wave, {
            "role": "reused accepted native full synthetic Wave source",
            "path": str(Path(native_full_wave.filename).resolve()),
            "sha256": sha256_file(Path(native_full_wave.filename)),
            "logical_sha256": logical_array_sha256(native_full_wave),
            "generated_for_case": False,
        }

    psf = np.asarray(open_cfl(case.psf["base"])).squeeze()
    target_shape = (extended_readout, *no_wave_crop.shape[1:3], no_wave_crop.shape[3])
    full_wave_path = case_root / "full_wave_kspace.npy"
    full_wave = np.lib.format.open_memmap(
        full_wave_path, mode="w+", dtype=np.complex64, shape=target_shape
    )
    for coil in range(no_wave_crop.shape[3]):
        full_wave[..., coil] = synthesize_wave_from_no_wave_crop(
            no_wave_crop[..., coil],
            psf,
            readout_oversampled=extended_readout,
            target_mask=None,
            fft_workers=fft_workers,
        )
    full_wave.flush()
    if not array_is_finite(full_wave):
        raise ValueError(f"{case.case_id} synthesized full Wave source is non-finite.")
    return full_wave, {
        "role": "direct no-Wave PE crop followed by target-grid Wave encoding",
        "path": str(full_wave_path),
        "shape": list(full_wave.shape),
        "dtype": str(full_wave.dtype),
        "sha256": sha256_file(full_wave_path),
        "logical_sha256": logical_array_sha256(full_wave),
        "generated_for_case": True,
        "interpolation_performed": False,
    }


def _linked_artifact_record(base: Path, accepted: dict[str, Any]) -> dict[str, Any]:
    """Build a hash record for one linked accepted BART artifact.

    Args:
        base: Case-local BART basename.
        accepted: Fresh accepted-artifact validation record.

    Returns:
        Linked paths and hashes with the accepted provenance record.
    """
    header = base.with_suffix(".hdr")
    payload = base.with_suffix(".cfl")
    header_hash = sha256_file(header)
    payload_hash = sha256_file(payload)
    if header_hash != accepted["header_sha256"] or payload_hash != accepted["payload_sha256"]:
        raise ValueError(f"Linked accepted BART artifact changed: {base}")
    return {
        "base": str(base),
        "header": str(header),
        "cfl": str(payload),
        "shape": accepted["shape"],
        "header_sha256": header_hash,
        "payload_sha256": payload_hash,
        "accepted_source_base": accepted["base"],
        "accepted_provenance_manifest": accepted["provenance_manifest"],
        "recalculated": False,
    }


def _prepare_case(
    case: PureMaskCase,
    validated: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Prepare one pure-mask case and all exact validation records.

    Args:
        case: Validated immutable case contract.
        validated: Complete read-only source validation result.
        output_root: User-confirmed production root.

    Returns:
        Complete case manifest payload.
    """
    case_root = output_root / "cases" / case.case_id
    case_root.mkdir(parents=True, exist_ok=False)
    bart_inputs = case_root / "bart_inputs"
    bart_inputs.mkdir()
    np.save(case_root / "sampling_mask.npy", case.mask)
    sampling = {
        **case.mask_metadata,
        "path": str(case_root / "sampling_mask.npy"),
    }
    if logical_array_sha256(np.load(sampling["path"], allow_pickle=False)) != sampling["logical_sha256"]:
        raise ValueError(f"{case.case_id} saved mask hash differs from its exact lattice.")

    no_wave = np.load(validated["source"]["no_wave_kspace"]["path"], mmap_mode="r")
    lin = slice(*case.resolved.crop_bounds_lin)
    par = slice(*case.resolved.crop_bounds_par)
    no_wave_crop = np.asarray(no_wave[:, lin, par, :])
    expected_no_wave_shape = (*case.resolved.target_logical_matrix_ro_lin_par, validated["virtual_coils"])
    if no_wave_crop.shape != expected_no_wave_shape:
        raise ValueError(f"{case.case_id} direct no-Wave crop has shape {no_wave_crop.shape}.")
    fft_workers = int(validated["config"]["snapshot"].get("runtime", {}).get("fft_workers", 4))
    reference = direct_fft_reference(no_wave_crop, fft_workers=fft_workers)
    reference_path = case_root / "direct_fft_reference_logical.npy"
    np.save(reference_path, reference)
    reference_record = {
        "path": str(reference_path),
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "sha256": sha256_file(reference_path),
        "logical_sha256": logical_array_sha256(reference),
        "source_no_wave_crop_bounds_lin": list(case.resolved.crop_bounds_lin),
        "source_no_wave_crop_bounds_par": list(case.resolved.crop_bounds_par),
        "interpolation_performed": False,
    }

    native_full_wave = np.load(
        validated["source"]["native_full_wave_kspace"]["path"], mmap_mode="r"
    )
    full_wave, full_wave_record = _case_full_wave(
        case,
        no_wave_crop,
        native_full_wave,
        case_root,
        extended_readout=validated["extended_wave_readout"],
        fft_workers=fft_workers,
    )
    wave_record = write_masked_wave_cfl(
        full_wave, case.mask, bart_inputs / "wave_kspace"
    )
    link_bart_pair(case.csm["base"], bart_inputs / "coil_sens")
    link_bart_pair(case.psf["base"], bart_inputs / "psf")
    csm_record = _linked_artifact_record(bart_inputs / "coil_sens", case.csm)
    psf_record = _linked_artifact_record(bart_inputs / "psf", case.psf)

    bart_manifest = {
        "format_version": 1,
        "status": "pure_mask_bart_inputs_ready",
        "case": case.resolved.to_json(),
        "case_id": case.case_id,
        "dimension_order": ["READ", "LIN", "PAR", "COIL", "MAPS"],
        "sampling_mask": sampling,
        "full_wave_kspace": full_wave_record,
        "wave_kspace": wave_record,
        "coil_sens": csm_record,
        "psf": psf_record,
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": wave_record["shape"],
                "wave_kspace_norm": wave_record["wave_kspace_norm"],
                "psf": "psf",
                "psf_shape": psf_record["shape"],
            }
        ],
        "calibration_kspace_included": False,
        "ecalib_run_performed": False,
        "psf_calibration_run_performed": False,
    }
    write_json_atomic(bart_inputs / "manifest.json", bart_manifest)
    case_manifest = {
        **bart_manifest,
        "direct_fft_reference": reference_record,
        "source_no_wave_kspace": validated["source"]["no_wave_kspace"],
        "approved_bet_mask": validated["source"]["approved_bet_mask"],
        "prepared_at_utc": _utc_now(),
        "bart_inputs": {
            "manifest": str(bart_inputs / "manifest.json"),
            "wave_kspace": wave_record,
            "coil_sens": csm_record,
            "psf": psf_record,
        },
    }
    write_json_atomic(case_root / "case_manifest.json", case_manifest)
    return case_manifest


def prepare(
    config_path: str | Path,
    *,
    validate_only: bool,
    confirmed_output_root: Path | None,
    resume: bool,
) -> dict[str, Any]:
    """Validate inputs and optionally materialize all five prepared cases.

    Args:
        config_path: Ignored local configuration path.
        validate_only: Perform no writes when true.
        confirmed_output_root: Exact user-approved root required for writes.
        resume: Reuse only a complete hash-identical preparation.

    Returns:
        Validation result or completed root preparation manifest.
    """
    validated = validate_config(config_path)
    if validate_only:
        if confirmed_output_root is not None:
            raise ValueError("--confirm-output-root is not used with --validate-only.")
        return validated
    if confirmed_output_root is None:
        raise ValueError(
            "Production preparation requires --confirm-output-root after explicit user approval."
        )
    configured_root = Path(validated["layout"]["root"])
    confirmed_root = confirmed_output_root.expanduser().resolve()
    if confirmed_root != configured_root:
        raise ValueError(
            f"Confirmed output root {confirmed_root} differs from config {configured_root}."
        )
    manifest_path = configured_root / "preparation_manifest.json"
    if resume and _complete_preparation_reusable(manifest_path, validated):
        print(f"Reusing validated pure-mask preparation: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_names = (
        {item.name for item in configured_root.iterdir()} if configured_root.exists() else set()
    )
    if existing_names - {"source_materialization"}:
        raise FileExistsError(
            "Pure-mask output root contains entries outside the validated source materialization: "
            f"{configured_root}"
        )
    configured_root.mkdir(parents=True, exist_ok=True)
    running = {
        "format_version": 1,
        "status": "preparing",
        "workflow": "synthetic_wave_pure_mask_regularization_rerun",
        "config": validated["config"],
        "layout": output_layout(configured_root),
        "source": validated["source"],
        "geometry": validated["geometry"],
        "cases": {},
        "started_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, running)
    for case in validated["cases"]:
        case_manifest = _prepare_case(case, validated, configured_root)
        case_path = configured_root / "cases" / case.case_id / "case_manifest.json"
        running["cases"][case.case_id] = {
            "case_manifest": str(case_path),
            "case_manifest_sha256": sha256_file(case_path),
            "sampling_mask": case_manifest["sampling_mask"],
        }
        write_json_atomic(manifest_path, running)
    running["status"] = "complete"
    running["completed_at_utc"] = _utc_now()
    write_json_atomic(manifest_path, running)
    print(f"Pure-mask preparation manifest: {manifest_path}")
    return running


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
