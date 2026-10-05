"""Strict resumability records for standard-PCA reconstruction branches."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import cfl_record, open_cfl, sha256_file
from .standard import (
    normalize_reconstruction_profile,
    profile_branches,
    validate_prepared_artifact_records,
    validate_standard_pca_manifest,
)


RUN_MANIFEST_STATUS = "standard_pca_reconstruction_branch_complete"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: Existing JSON path.

    Returns:
        Parsed dictionary.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping.

    Args:
        path: Destination path.
        payload: JSON-compatible mapping.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _finite_cfl(path: Path, *, axis_zero_chunk: int = 8) -> bool:
    """Check one CFL pair for finite values in bounded leading-axis blocks.

    Args:
        path: BART CFL basename.
        axis_zero_chunk: Maximum leading-axis planes per check.

    Returns:
        ``True`` only when every stored complex value is finite.
    """

    values = open_cfl(path)
    for start in range(0, values.shape[0], axis_zero_chunk):
        if not np.isfinite(np.asarray(values[start : start + axis_zero_chunk])).all():
            return False
    return True


def _cfl_records(paths: Sequence[Path], *, require_finite: bool) -> list[dict[str, Any]]:
    """Build ordered hash records for BART arrays.

    Args:
        paths: Ordered CFL basenames.
        require_finite: Whether to scan each payload for finite values.

    Returns:
        Hash-bound CFL records.
    """

    records = []
    for path in paths:
        record = dict(cfl_record(path, include_hash=True))
        if require_finite:
            if not _finite_cfl(path):
                raise ValueError(f"BART array contains non-finite values: {path}")
            record["finite"] = True
        records.append(record)
    return records


def _nifti_records(directory: Path) -> list[dict[str, Any]]:
    """Hash every branch-local NIfTI, sidecar, and conversion record.

    Args:
        directory: Branch-specific NIfTI output directory.

    Returns:
        Sorted relative-path, size, and SHA-256 records.

    Raises:
        FileNotFoundError: If no NIfTI or matching JSON sidecar is present.
    """

    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI branch directory does not exist: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    nifti = [path for path in files if path.name.endswith(".nii.gz")]
    sidecars = [path for path in files if path.suffix == ".json"]
    if not nifti or not sidecars:
        raise FileNotFoundError(
            f"NIfTI branch is incomplete (requires NIfTI and JSON files): {directory}"
        )
    return [
        {
            "path": str(path.relative_to(directory)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def _request_contract(
    *,
    prepared_manifest: Path,
    normal_manifest: Path,
    profile: str,
    case: str,
    branch: str,
    method: str,
    regularization: float,
    maps: Path,
    psfs: Sequence[Path],
    kspaces: Sequence[Path],
    images: Sequence[Path],
    command_records: Sequence[Path],
    expected_commands: Sequence[str],
    nifti_directory: Path,
) -> dict[str, Any]:
    """Build the immutable requested reconstruction contract.

    Args:
        prepared_manifest: Case-local preparation manifest.
        normal_manifest: Count-specific normal preparation manifest.
        profile: Selected reconstruction profile.
        case: Stable case identifier.
        branch: Stable output branch identifier.
        method: ``fista`` or ``wavelet``.
        regularization: Nonnegative finite lambda.
        maps: BART sensitivity-map basename.
        psfs: Ordered PSF basenames.
        kspaces: Ordered measured-k-space basenames.
        images: Ordered reconstruction output basenames.
        command_records: Ordered command-record paths.
        expected_commands: Exact shell-escaped BART commands.
        nifti_directory: Branch-specific NIfTI directory.

    Returns:
        JSON-native request contract with source and input hashes.
    """

    normalized_profile = normalize_reconstruction_profile(profile)
    value = float(regularization)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Reconstruction regularization must be finite and nonnegative.")
    if method not in {"fista", "wavelet"}:
        raise ValueError("Reconstruction method must be fista or wavelet.")
    if method == "fista" and value != 0.0:
        raise ValueError("FISTA-r0 must use exactly zero regularization.")
    selected_method = "fista_r0" if method == "fista" else "wavelet"
    if selected_method not in profile_branches(normalized_profile):
        raise ValueError(
            f"Reconstruction method {method} is not enabled by profile "
            f"{normalized_profile}."
        )
    lengths = {len(psfs), len(kspaces), len(images), len(command_records), len(expected_commands)}
    if lengths != {len(psfs)} or not psfs:
        raise ValueError("PSF, k-space, image, command-record, and command counts must match.")

    prepared = _load_json(prepared_manifest)
    normal = _load_json(normal_manifest)
    compression = validate_standard_pca_manifest(
        normal,
        int(normal.get("coil_compression", {}).get("virtual_coils", 0)),
    )
    validate_prepared_artifact_records(
        normal_manifest.parent, normal.get("output_artifacts", {})
    )
    validate_prepared_artifact_records(
        prepared_manifest.parent, prepared.get("output_artifacts", {})
    )
    variant = str(normal["standard_pca_variant"])
    if prepared.get("standard_pca_variant") != variant:
        raise ValueError("Case and normal preparation manifests use different VCC variants.")
    prepared_count = prepared.get("virtual_coils")
    if prepared_count is not None and int(prepared_count) != int(
        compression["virtual_coils"]
    ):
        raise ValueError("Case and normal preparation manifests use different Ncc values.")
    if prepared_manifest != normal_manifest:
        source_normal = prepared.get("source_normal_manifest_identity", {})
        # Early count-specific R3x3 manifests stored the same hash-bound
        # identity directly under ``source_normal_manifest``. Accept that
        # exact schema without weakening the digest comparison.
        if not isinstance(source_normal, Mapping):
            source_normal = {}
        if not source_normal:
            legacy_source_normal = prepared.get("source_normal_manifest", {})
            if isinstance(legacy_source_normal, Mapping):
                source_normal = legacy_source_normal
        if (
            not isinstance(source_normal, Mapping)
            or source_normal.get("sha256") != sha256_file(normal_manifest)
        ):
            raise ValueError(
                "Case preparation manifest is not bound to the selected normal manifest."
            )
    return {
        "standard_pca_variant": variant,
        "virtual_coils": int(compression["virtual_coils"]),
        "compression_basis": dict(compression["basis"]),
        "profile": normalized_profile,
        "case": str(case),
        "branch": str(branch),
        "method": method,
        "regularization": value,
        "prepared_manifest": {
            "path": str(prepared_manifest),
            "sha256": sha256_file(prepared_manifest),
        },
        "normal_manifest": {
            "path": str(normal_manifest),
            "sha256": sha256_file(normal_manifest),
        },
        "source": normal.get("source"),
        "geometry": prepared.get("case", prepared.get("geometry")),
        "sampling": prepared.get("sampling"),
        "inputs": {
            "maps": _cfl_records((maps,), require_finite=True)[0],
            "psf_by_echo": _cfl_records(psfs, require_finite=True),
            "kspace_by_echo": _cfl_records(kspaces, require_finite=True),
        },
        "images": [str(path) for path in images],
        "command_records": [str(path) for path in command_records],
        "expected_commands": list(expected_commands),
        "nifti_directory": str(nifti_directory),
    }


def reconstruction_status(
    run_manifest: str | Path, **request: Any
) -> str:
    """Return the safe next action for one reconstruction branch.

    Args:
        run_manifest: Branch-local immutable run-manifest path.
        **request: Keyword arguments accepted by :func:`_request_contract`.

    Returns:
        ``run``, ``convert``, ``finalize``, or ``complete``.

    Raises:
        ValueError: If existing accepted state differs from the request.
        FileNotFoundError: If recorded accepted artifacts disappeared.
    """

    path = Path(run_manifest).expanduser().resolve()
    contract = _request_contract(**request)
    command_records = [Path(value) for value in contract["command_records"]]
    images = [Path(value) for value in contract["images"]]
    expected_commands = contract["expected_commands"]

    if path.is_file():
        existing = _load_json(path)
        if existing.get("status") != RUN_MANIFEST_STATUS:
            raise ValueError(f"Unexpected reconstruction run status: {path}")
        if existing.get("request") != contract:
            raise ValueError(f"Reconstruction request differs from completed run: {path}")
        outputs = _cfl_records(images, require_finite=True)
        if existing.get("outputs") != outputs:
            raise ValueError(f"Reconstruction output changed since completion: {path}")
        niftis = _nifti_records(Path(contract["nifti_directory"]))
        if existing.get("nifti_outputs") != niftis:
            raise ValueError(f"NIfTI output changed since completion: {path}")
        _validate_command_records(command_records, expected_commands)
        return "complete"

    existing_records = [record.is_file() for record in command_records]
    if any(existing_records) and not all(existing_records):
        raise ValueError("Reconstruction branch has only some echo command records.")
    if all(existing_records):
        _validate_command_records(command_records, expected_commands)
        _cfl_records(images, require_finite=True)
        try:
            _nifti_records(Path(contract["nifti_directory"]))
        except FileNotFoundError:
            return "convert"
        return "finalize"
    if Path(contract["nifti_directory"]).exists() and any(
        Path(contract["nifti_directory"]).iterdir()
    ):
        raise ValueError("Unrecorded NIfTI outputs exist for an incomplete branch.")
    return "run"


def _validate_command_records(
    paths: Sequence[Path], expected_commands: Sequence[str]
) -> None:
    """Validate exact ordered BART command records.

    Args:
        paths: Ordered text-record paths.
        expected_commands: Exact expected command strings.

    Returns:
        None.
    """

    for path, expected in zip(paths, expected_commands, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"Reconstruction command record is missing: {path}")
        if path.read_text(encoding="utf-8").strip() != expected:
            raise ValueError(f"Reconstruction command changed: {path}")


def record_completed_reconstruction(
    run_manifest: str | Path, **request: Any
) -> dict[str, Any]:
    """Validate and record one completed standard-PCA branch.

    Args:
        run_manifest: Branch-local immutable run-manifest path.
        **request: Keyword arguments accepted by :func:`_request_contract`.

    Returns:
        The written or exactly reused manifest.

    Side Effects:
        Writes ``run_manifest`` only after commands, arrays, and NIfTI outputs
        pass exact provenance, geometry, hash, and finite-value validation.
    """

    path = Path(run_manifest).expanduser().resolve()
    contract = _request_contract(**request)
    command_records = [Path(value) for value in contract["command_records"]]
    _validate_command_records(command_records, contract["expected_commands"])
    outputs = _cfl_records(
        [Path(value) for value in contract["images"]], require_finite=True
    )
    niftis = _nifti_records(Path(contract["nifti_directory"]))
    payload = {
        "format_version": 1,
        "status": RUN_MANIFEST_STATUS,
        "request": contract,
        "outputs": outputs,
        "nifti_outputs": niftis,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.is_file():
        existing = _load_json(path)
        stable_existing = {
            key: value for key, value in existing.items() if key != "completed_at_utc"
        }
        stable_payload = {
            key: value for key, value in payload.items() if key != "completed_at_utc"
        }
        if stable_existing != stable_payload:
            raise ValueError(f"Completed reconstruction manifest differs: {path}")
        return existing
    _write_json(path, payload)
    return payload


def validate_completed_reconstruction_manifest(
    run_manifest: str | Path,
) -> dict[str, Any]:
    """Validate a completed branch directly from its immutable manifest.

    Args:
        run_manifest: Existing branch-local reconstruction manifest.

    Returns:
        Parsed, fully validated run manifest.

    Raises:
        ValueError: If provenance, commands, hashes, or finite values changed.
        FileNotFoundError: If a recorded artifact disappeared.
    """

    path = Path(run_manifest).expanduser().resolve()
    payload = _load_json(path)
    if payload.get("status") != RUN_MANIFEST_STATUS:
        raise ValueError(f"Unexpected reconstruction run status: {path}")
    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise ValueError(f"Reconstruction manifest has no request contract: {path}")
    prepared = Path(str(request.get("prepared_manifest", {}).get("path", "")))
    normal = Path(str(request.get("normal_manifest", {}).get("path", "")))
    for label, source_path, record in (
        ("prepared", prepared, request.get("prepared_manifest", {})),
        ("normal", normal, request.get("normal_manifest", {})),
    ):
        if not source_path.is_file() or sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"Recorded {label} manifest changed: {source_path}")
    prepared_payload = _load_json(prepared)
    normal_payload = _load_json(normal)
    compression = validate_standard_pca_manifest(
        normal_payload, int(request.get("virtual_coils", 0))
    )
    if dict(compression.get("basis", {})) != request.get("compression_basis"):
        raise ValueError(f"Recorded compression basis changed: {path}")
    validate_prepared_artifact_records(
        normal.parent, normal_payload.get("output_artifacts", {})
    )
    validate_prepared_artifact_records(
        prepared.parent, prepared_payload.get("output_artifacts", {})
    )
    inputs = request.get("inputs", {})
    input_groups = [inputs.get("maps")]
    input_groups.extend(inputs.get("psf_by_echo", ()))
    input_groups.extend(inputs.get("kspace_by_echo", ()))
    for record in input_groups:
        if not isinstance(record, Mapping) or "base" not in record:
            raise ValueError(f"Invalid recorded reconstruction input: {path}")
        actual = _cfl_records((Path(str(record["base"])),), require_finite=True)[0]
        if actual != record:
            raise ValueError(f"Recorded reconstruction input changed: {record['base']}")
    images = [Path(value) for value in request.get("images", ())]
    outputs = _cfl_records(images, require_finite=True)
    if outputs != payload.get("outputs"):
        raise ValueError(f"Recorded reconstruction output changed: {path}")
    command_records = [Path(value) for value in request.get("command_records", ())]
    _validate_command_records(command_records, request.get("expected_commands", ()))
    niftis = _nifti_records(Path(str(request.get("nifti_directory", ""))))
    if niftis != payload.get("nifti_outputs"):
        raise ValueError(f"Recorded reconstruction NIfTI output changed: {path}")
    return payload
