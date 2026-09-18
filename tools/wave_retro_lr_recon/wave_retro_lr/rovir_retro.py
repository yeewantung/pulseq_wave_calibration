"""Prepare retrospective MPRAGE cases from one canonical normal ROVir branch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bart_io import cfl_record, read_shape, sha256_file
from .core import ResolvedCase
from .mprage import R3X3_CASE_ID, RETRO_CASES, _sampling_from_manifest
from .retrospective import link_bart_pair, resample_sensitivity_maps, write_measured_wave_crop

ROVIR_RETRO_CASES = tuple(name for name, _ in RETRO_CASES) + (R3X3_CASE_ID,)
ROVIR_REUSED_WAVELET_LAMBDAS = {
    "native_r3x2": 0.03,
    "lr_x_1p5mm_r3x2": 0.025,
    "lr_y_1p5mm_r3x2": 0.025,
    "lr_xy_1p25mm_r3x2": 0.022,
    R3X3_CASE_ID: 0.045,
}


def prepare_mprage_rovir_retro(output_root: str | Path) -> list[dict[str, Any]]:
    """Prepare every available standard retro case in the selected ROVir basis.

    Args:
        output_root: Reconstruction root containing completed normal ROVir and
            standard retrospective input manifests.

    Returns:
        One manifest for each prepared ROVir retrospective case.

    Raises:
        FileNotFoundError: If the canonical ROVir contract or standard case is absent.
        FileExistsError: If a partial incompatible destination exists.
        ValueError: If source, transform, geometry, or hashes are inconsistent.

    Side Effects:
        Writes only sibling ``retro/<case>/rovir/bart_inputs`` trees. It does
        not launch BART reconstruction or alter standard cases.
    """
    root = Path(output_root).expanduser().resolve()
    contract_path = root / "normal" / "rovir" / "manifest.json"
    contract = _read_json(contract_path)
    if contract.get("status") != "mprage_normal_rovir_complete" or not contract.get("retro_consumable"):
        raise ValueError("Canonical normal ROVir contract is incomplete or not retro-consumable.")
    for label in ("twix", "sequence"):
        _validate_source_identity(contract.get("source", {}).get(label), label)
    normal_manifest_path = root / "normal" / "bart_inputs" / "manifest.json"
    if contract.get("normal_source_manifest", {}).get("sha256") != sha256_file(normal_manifest_path):
        raise ValueError("ROVir contract no longer matches the standard normal manifest.")
    normal = _read_json(normal_manifest_path)
    normal_rovir_inputs = root / "normal" / "rovir" / "bart_inputs"
    _validate_file_record(
        contract.get("artifacts", {}).get("prepared_inputs_manifest"),
        normal_rovir_inputs / "manifest.json",
        "prepared ROVir inputs manifest",
    )
    rovir_inputs_manifest = _read_json(normal_rovir_inputs / "manifest.json")
    if rovir_inputs_manifest.get("source") != contract.get("source"):
        raise ValueError("ROVir prepared inputs and canonical source provenance disagree.")
    source_sampling = _sampling_from_manifest(normal)
    source_wave = normal_rovir_inputs / "wave_kspace"
    _validate_cfl_record(
        rovir_inputs_manifest.get("artifacts", {}).get("wave_kspace"),
        source_wave,
        "normal ROVir image k-space",
    )
    _validate_cfl_record(
        rovir_inputs_manifest.get("artifacts", {}).get("kspace_calib"),
        normal_rovir_inputs / "kspace_calib",
        "normal ROVir corrected ACS",
    )
    _validate_cfl_record(
        rovir_inputs_manifest.get("psf_calibration", {}).get("copied"),
        normal_rovir_inputs / "psf",
        "normal ROVir PSF",
    )
    transform = Path(contract["coil_processing"]["transform"]["base"])
    _validate_cfl_record(
        contract["coil_processing"]["transform"], transform, "ROVir transform"
    )
    _validate_file_record(
        contract["coil_processing"]["approved_mask_manifest"],
        Path(contract["coil_processing"]["approved_mask_manifest"]["path"]),
        "approved ROI manifest",
    )
    physical = rovir_inputs_manifest.get("physical_calibration", {})
    physical_manifest = Path(str(physical.get("manifest_path", ""))).expanduser()
    if (
        not physical_manifest.is_file()
        or physical.get("manifest_sha256") != sha256_file(physical_manifest)
    ):
        raise ValueError("Corrected physical ACS manifest differs from ROVir provenance.")
    _validate_cfl_record(
        physical.get("cfl"),
        Path(str(physical.get("cfl", {}).get("base", ""))),
        "physical corrected set-4 ACS",
    )
    basis_path = normal_rovir_inputs / str(
        rovir_inputs_manifest.get("rovir", {}).get("projection_basis_file", "")
    )
    if (
        not basis_path.is_file()
        or rovir_inputs_manifest.get("rovir", {}).get("projection_basis_sha256")
        != sha256_file(basis_path)
    ):
        raise ValueError("Selected ROVir projection basis differs from its manifest.")
    basis = np.load(basis_path, allow_pickle=False)
    rovir_record = rovir_inputs_manifest.get("rovir", {})
    expected_basis_shape = (
        int(rovir_record.get("physical_coils", -1)),
        int(rovir_record.get("virtual_coils", -1)),
    )
    if (
        basis.shape != expected_basis_shape
        or expected_basis_shape[1]
        != int(contract.get("coil_processing", {}).get("virtual_coils", -2))
        or not np.isfinite(basis).all()
    ):
        raise ValueError("ROVir basis coil ordering, dimensions, or values are invalid.")
    source_csm = root / "normal" / "rovir" / "bart_output" / "coil_sens"
    _validate_cfl_record(
        contract.get("ecalib", {}).get("coil_sens"), source_csm, "normal ROVir CSM"
    )
    ecalib_record = root / "normal" / "rovir" / "bart_output" / "ecalib_command.txt"
    _validate_file_record(
        contract.get("ecalib", {}).get("command_record"),
        ecalib_record,
        "normal ROVir ecalib command",
    )

    results: list[dict[str, Any]] = []
    for case_name in ROVIR_RETRO_CASES:
        standard_inputs = root / "retro" / case_name / "bart_inputs"
        standard_manifest_path = standard_inputs / "manifest.json"
        standard = _read_json(standard_manifest_path)
        case = _resolved_case(standard.get("case"))
        destination = root / "retro" / case_name / "rovir" / "bart_inputs"
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            existing = _read_json(manifest_path)
            _validate_reuse(existing, contract_path, standard_manifest_path, destination)
            results.append(existing)
            continue
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"ROVir retro input directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        target_mask = None
        if case_name == R3X3_CASE_ID:
            mask_path = standard_inputs / "sampling_mask.npy"
            target_mask = np.load(mask_path, allow_pickle=False)
            if target_mask.dtype != np.bool_:
                raise ValueError("Native R3x3 standard sampling mask must be boolean.")
        crop_metrics = write_measured_wave_crop(
            source_wave,
            destination / "wave_kspace",
            case,
            source_sampling.mask(),
            source_sampling.acceleration_lin_par,
            target_mask=target_mask,
        )
        link_bart_pair(standard_inputs / "psf", destination / "psf")
        target = case.target_logical_matrix_ro_lin_par
        if read_shape(source_csm)[:3] == target:
            link_bart_pair(source_csm, destination / "coil_sens")
            csm_operation = "exact link to canonical normal ROVir CSM"
        else:
            resample_sensitivity_maps(
                source_csm,
                destination / "coil_sens",
                target_lin_par=(target[1], target[2]),
            )
            csm_operation = "same-FOV centered Fourier PE resampling plus RSS normalization"
        selected = standard.get("selected_regularization")
        reused_lambda = ROVIR_REUSED_WAVELET_LAMBDAS[case_name]
        if isinstance(selected, Mapping) and float(selected.get("lambda", -1)) != reused_lambda:
            raise ValueError(
                f"Standard {case_name} selected lambda differs from the locked tool value."
            )
        manifest = {
            "format_version": 1,
            "status": "mprage_rovir_retro_bart_inputs_ready",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": contract["source"],
            "case_directory": case_name,
            "case": case.to_json(),
            "operator": "ROVir projection completed before retrospective sampling or PE crop",
            "interpolation": False,
            "forward_simulation": False,
            "coil_processing": {
                "label": "bart_rovir",
                "candidate_id": contract["coil_processing"]["candidate_id"],
                "virtual_coils": contract["coil_processing"]["virtual_coils"],
                "transform_sha256": contract["coil_processing"]["transform"]["payload_sha256"],
                "normal_source_manifest_sha256": contract["normal_source_manifest"]["sha256"],
            },
            "source_normal_rovir_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "source_standard_case_manifest": {
                "path": str(standard_manifest_path),
                "sha256": sha256_file(standard_manifest_path),
            },
            "ecalib_command_record": str(ecalib_record),
            "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
            "sampling": crop_metrics,
            "coil_sens": "coil_sens",
            "coil_sens_shape": list(read_shape(destination / "coil_sens")),
            "coil_sens_operation": csm_operation,
            "echoes": [{
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": list(read_shape(destination / "wave_kspace")),
                "wave_kspace_norm": crop_metrics["wave_kspace_norm"],
                "psf": "psf",
                "psf_shape": list(read_shape(destination / "psf")),
            }],
            "selected_regularization": {
                **(dict(selected) if isinstance(selected, Mapping) else {}),
                "method": "wavelet",
                "lambda": reused_lambda,
                "reused_from_standard_coil_experiment": True,
                "optimized_for_rovir": False,
            },
            "fista_lambda_zero_required": True,
            "artifacts": {
                name: _cfl_hash_record(destination / name)
                for name in ("wave_kspace", "psf", "coil_sens")
            },
        }
        _write_json(manifest_path, manifest)
        results.append(manifest)
    return results


def _resolved_case(payload: object) -> ResolvedCase:
    """Reconstruct a resolved-case dataclass from a strict manifest mapping.

    Args:
        payload: Standard retrospective ``case`` mapping.

    Returns:
        Validated resolved case.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Standard retrospective manifest lacks case geometry.")
    return ResolvedCase(
        requested_resolution_mm_xyz=tuple(float(v) for v in payload["requested_resolution_mm_xyz"]),
        achieved_resolution_mm_xyz=tuple(float(v) for v in payload["achieved_resolution_mm_xyz"]),
        source_logical_matrix_ro_lin_par=tuple(int(v) for v in payload["source_logical_matrix_ro_lin_par"]),
        target_logical_matrix_ro_lin_par=tuple(int(v) for v in payload["target_logical_matrix_ro_lin_par"]),
        target_physical_matrix_xyz=tuple(int(v) for v in payload["target_physical_matrix_xyz"]),
        crop_bounds_lin=tuple(int(v) for v in payload["crop_bounds_lin"]),
        crop_bounds_par=tuple(int(v) for v in payload["crop_bounds_par"]),
        acceleration_ry_rz=tuple(int(v) for v in payload["acceleration_ry_rz"]),
        case_name=str(payload["case_name"]),
        label=None if payload.get("label") is None else str(payload["label"]),
    )


def _validate_reuse(
    manifest: Mapping[str, Any],
    contract_path: Path,
    standard_manifest_path: Path,
    destination: Path,
) -> None:
    """Validate an existing ROVir retrospective input tree for exact reuse.

    Args:
        manifest: Existing branch manifest.
        contract_path: Current canonical normal ROVir contract.
        standard_manifest_path: Current standard case manifest.
        destination: Existing ROVir BART-input directory.

    Raises:
        ValueError: If provenance or artifact hashes differ.
    """
    if (
        manifest.get("status") != "mprage_rovir_retro_bart_inputs_ready"
        or manifest.get("source_normal_rovir_contract", {}).get("sha256") != sha256_file(contract_path)
        or manifest.get("source_standard_case_manifest", {}).get("sha256") != sha256_file(standard_manifest_path)
    ):
        raise ValueError(f"Existing ROVir retrospective inputs use another contract: {destination}")
    for name in ("wave_kspace", "psf", "coil_sens"):
        if manifest.get("artifacts", {}).get(name) != _cfl_hash_record(destination / name):
            raise ValueError(f"Existing ROVir retrospective {name} failed hash reuse.")


def _cfl_hash_record(base: Path) -> dict[str, Any]:
    """Record exact BART pair dimensions and hashes.

    Args:
        base: BART CFL basename.

    Returns:
        JSON-native dimensions and SHA-256 hashes.
    """
    return {
        "shape": list(read_shape(base)),
        "header_sha256": sha256_file(base.with_suffix(".hdr")),
        "payload_sha256": sha256_file(base.with_suffix(".cfl")),
    }


def _validate_cfl_record(record: object, base: Path, label: str) -> None:
    """Require a current BART pair to match its recorded geometry and hashes.

    Args:
        record: Manifest CFL record.
        base: Current BART basename.
        label: Human-readable artifact name.

    Raises:
        ValueError: If the record or current pair differs.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"Canonical contract lacks {label} provenance.")
    current = cfl_record(base)
    for key in ("shape", "header_sha256", "payload_sha256"):
        if record.get(key) != current.get(key):
            raise ValueError(f"Canonical {label} differs from its recorded artifact.")


def _validate_file_record(record: object, path: Path, label: str) -> None:
    """Require one current file to match its immutable record.

    Args:
        record: Mapping with path, size, and SHA-256.
        path: Expected current file.
        label: Human-readable artifact name.

    Raises:
        ValueError: If the file record differs.
    """
    if not isinstance(record, Mapping) or not path.is_file():
        raise ValueError(f"Canonical contract lacks {label} provenance.")
    resolved = path.resolve()
    if (
        Path(str(record.get("path", ""))).expanduser().resolve() != resolved
        or int(record.get("size_bytes", -1)) != resolved.stat().st_size
        or record.get("sha256") != sha256_file(resolved)
    ):
        raise ValueError(f"Canonical {label} differs from its recorded file.")


def _validate_source_identity(record: object, label: str) -> None:
    """Require a source file to retain its recorded path, stat, and hash.

    Args:
        record: Canonical source identity mapping.
        label: Source label used in errors.

    Raises:
        ValueError: If identity, size, time, or SHA-256 differs.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError(f"Canonical ROVir contract lacks {label} identity.")
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Canonical ROVir {label} source is unavailable: {path}")
    stat = path.stat()
    if (
        int(record.get("size_bytes", -1)) != stat.st_size
        or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        or record.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"Canonical ROVir {label} source identity changed.")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object.

    Args:
        path: JSON file path.

    Returns:
        Parsed mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON to an existing case directory.

    Args:
        path: Destination path.
        payload: JSON-native mapping.

    Side Effects:
        Writes one UTF-8 JSON file.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
