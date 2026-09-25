"""Prepare retrospective MPRAGE cases from one canonical normal ROVir branch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, read_shape, sha256_file
from .core import (
    CaseSpec,
    Geometry,
    ResolvedCase,
    evaluate_psf_phase_planes,
    extract_psf_phase_planes,
    psf_identity_metrics,
    resolve_case,
)
from .mprage import (
    R3X3_CASE_ID,
    RETRO_CASES,
    _native_r3x3_residue,
    _sampling_from_manifest,
)
from .retrospective import (
    link_bart_pair,
    resample_sensitivity_maps,
    validate_same_grid_masked_wave,
    write_measured_wave_crop,
)
from .sampling import (
    SamplingPattern,
    pure_cartesian_image_lattice_mask,
    validate_pure_cartesian_image_lattice,
)

ROVIR_RETRO_CASES = tuple(name for name, _ in RETRO_CASES) + (R3X3_CASE_ID,)
ROVIR_REUSED_WAVELET_LAMBDAS = {
    "native_r3x2": 0.03,
    "lr_x_1p5mm_r3x2": 0.025,
    "lr_y_1p5mm_r3x2": 0.025,
    "lr_xy_1p25mm_r3x2": 0.022,
    R3X3_CASE_ID: 0.045,
}


def prepare_mprage_rovir_retro(output_root: str | Path) -> list[dict[str, Any]]:
    """Prepare all ROVir retro cases directly from canonical normal ROVir data.

    Args:
        output_root: Reconstruction root containing a completed canonical
            normal ROVir branch.

    Returns:
        One manifest for each prepared ROVir retrospective case.

    Raises:
        FileNotFoundError: If a required canonical ROVir artifact is absent.
        FileExistsError: If a partial incompatible destination exists.
        ValueError: If source, transform, geometry, or hashes are inconsistent.

    Side Effects:
        Writes only ``retro/<case>/rovir/bart_inputs`` trees. It does not
        launch BART reconstruction or create or alter standard retro cases.
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
    geometry = _geometry_from_manifest(normal)
    resolved_cases = _resolve_cases(geometry)
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
    source_psf = normal_rovir_inputs / "psf"
    source_psf_shape = read_shape(source_psf)
    if source_psf_shape[:3] != (
        read_shape(source_wave)[0],
        geometry.logical_matrix_ro_lin_par[1],
        geometry.logical_matrix_ro_lin_par[2],
    ):
        raise ValueError("Normal ROVir PSF and source geometry disagree.")
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

    psf_coefficients, psf_identity = _factor_source_psf(source_psf)
    results: list[dict[str, Any]] = []
    for case_name in ROVIR_RETRO_CASES:
        case = resolved_cases[case_name]
        target_mask, sampling = _target_sampling(source_sampling, case)
        destination = root / "retro" / case_name / "rovir" / "bart_inputs"
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            existing = _read_json(manifest_path)
            _validate_reuse(
                existing,
                contract_path,
                destination,
                expected_case=case,
                expected_sampling=sampling,
            )
            results.append(existing)
            continue
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"ROVir retro input directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        mask_path = destination / "sampling_mask.npy"
        np.save(mask_path, target_mask, allow_pickle=False)
        crop_metrics = write_measured_wave_crop(
            source_wave,
            destination / "wave_kspace",
            case,
            source_sampling.mask(),
            source_sampling.acceleration_lin_par,
            target_mask=target_mask,
        )
        if crop_metrics["sampled_coordinate_count"] != sampling["acquired_coordinate_count"]:
            raise ValueError(f"{case_name} acquired count differs from its pure mask.")
        if case.target_logical_matrix_ro_lin_par[1:] == geometry.logical_matrix_ro_lin_par[1:]:
            link_bart_pair(source_psf, destination / "psf")
            psf_operation = "exact link to canonical normal ROVir PSF"
        else:
            _write_target_psf(
                destination / "psf",
                *psf_coefficients,
                target_lin=case.target_logical_matrix_ro_lin_par[1],
                target_par=case.target_logical_matrix_ro_lin_par[2],
            )
            psf_operation = (
                "resolution-matched evaluation of canonical calibrated PSF phase planes"
            )
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
        reused_lambda = ROVIR_REUSED_WAVELET_LAMBDAS[case_name]
        same_grid_validation = None
        if target == geometry.logical_matrix_ro_lin_par:
            same_grid_validation = validate_same_grid_masked_wave(
                source_wave, destination / "wave_kspace", target_mask
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
            "standard_retro_inputs_required": False,
            "ecalib_command_record": str(ecalib_record),
            "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
            "sampling": {**sampling, **crop_metrics, "path": str(mask_path)},
            "sampling_validation": same_grid_validation,
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
                "psf_operation": psf_operation,
            }],
            "selected_regularization": {
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
            "psf_identity": psf_identity,
        }
        _write_json(manifest_path, manifest)
        results.append(manifest)
    return results


def _geometry_from_manifest(manifest: Mapping[str, Any]) -> Geometry:
    """Load the native physical geometry from the bound normal manifest.

    Args:
        manifest: Canonical source normal-input manifest.

    Returns:
        Native MPRAGE geometry.

    Raises:
        ValueError: If the recorded geometry is incomplete or invalid.
    """
    payload = manifest.get("geometry")
    if not isinstance(payload, Mapping):
        raise ValueError("Normal manifest lacks native MPRAGE geometry.")
    try:
        return Geometry(
            physical_fov_mm_xyz=tuple(float(v) for v in payload["physical_fov_mm_xyz"]),
            logical_matrix_ro_lin_par=tuple(
                int(v) for v in payload["logical_matrix_ro_lin_par"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Normal manifest contains invalid MPRAGE geometry.") from exc


def _resolve_cases(geometry: Geometry) -> dict[str, ResolvedCase]:
    """Resolve the five fixed ROVir retrospective case geometries.

    Args:
        geometry: Native physical FOV and logical matrix.

    Returns:
        Mapping from stable case directory name to resolved geometry.

    Raises:
        ValueError: If requested LR cases collapse to duplicate grids.
    """
    native_resolution = geometry.physical_resolution_mm_xyz
    cases: dict[str, ResolvedCase] = {}
    for directory_name, requested_xy in RETRO_CASES:
        requested = (
            native_resolution
            if requested_xy is None
            else (requested_xy[0], requested_xy[1], native_resolution[2])
        )
        cases[directory_name] = resolve_case(
            CaseSpec(requested, (3, 2), directory_name), geometry
        )
    cases[R3X3_CASE_ID] = resolve_case(
        CaseSpec(native_resolution, (3, 3), "native R3x3"), geometry
    )
    matrices = [
        cases[name].target_logical_matrix_ro_lin_par for name, _ in RETRO_CASES
    ]
    if len(set(matrices)) != len(matrices):
        raise ValueError("ROVir LR requests collapse to duplicate PE grids.")
    return cases


def _target_sampling(
    source: SamplingPattern, case: ResolvedCase
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct an exact pure target lattice compatible with measured data.

    Args:
        source: Validated source ``SamplingPattern``.
        case: Resolved retrospective case.

    Returns:
        Boolean target mask and canonical exact-count/hash metadata.

    Raises:
        ValueError: If the source is not a regular R1 or R3x1 lattice.
    """
    source_acceleration = tuple(int(v) for v in source.acceleration_lin_par)
    if source_acceleration not in {(1, 1), (3, 1)}:
        raise ValueError("ROVir MPRAGE retro requires regular R1 or R3x1 source sampling.")
    source_lin, source_par = source.matrix_lin_par
    expected_lin = (
        tuple(range(source_lin))
        if source_acceleration[0] == 1
        else tuple(range(int(source.lin_residue), source_lin, 3))
    )
    if tuple(source.acquired_lin) != expected_lin or tuple(source.acquired_par) != tuple(
        range(source_par)
    ):
        raise ValueError("Source sampling is not a pure regular image lattice.")
    target_lin, target_par = case.target_logical_matrix_ro_lin_par[1:]
    if case.acceleration_ry_rz == (3, 3):
        residues = _native_r3x3_residue(source, target_par)
        mask, metadata = pure_cartesian_image_lattice_mask(
            (target_lin, target_par),
            acceleration_lin_par=case.acceleration_ry_rz,
            residue_lin_par=residues,
        )
        validate_pure_cartesian_image_lattice(mask, metadata)
        return mask, metadata
    residues: list[int] = []
    for axis, (source_factor, target_factor, crop_start, target_size) in enumerate(
        zip(
            source_acceleration,
            case.acceleration_ry_rz,
            (case.crop_bounds_lin[0], case.crop_bounds_par[0]),
            (target_lin, target_par),
            strict=True,
        )
    ):
        if source_factor > 1:
            if source_factor != target_factor:
                raise ValueError("Target acceleration is incompatible with measured sampling.")
            source_residue = int(source.lin_residue) if axis == 0 else 0
            residues.append((source_residue - crop_start) % target_factor)
        else:
            residues.append((target_size // 2) % target_factor)
    mask, metadata = pure_cartesian_image_lattice_mask(
        (target_lin, target_par),
        acceleration_lin_par=case.acceleration_ry_rz,
        residue_lin_par=(residues[0], residues[1]),
    )
    validate_pure_cartesian_image_lattice(mask, metadata)
    return mask, metadata


def _factor_source_psf(
    source_base: Path,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Extract and strictly validate canonical calibrated PSF phase planes.

    Args:
        source_base: Canonical normal ROVir PSF basename.

    Returns:
        Per-readout phase coefficients and source-grid identity metrics.

    Raises:
        ValueError: If the source PSF is not a finite unit-magnitude phase plane.
    """
    source = open_cfl(source_base)
    values = np.asarray(source).squeeze()
    if values.ndim != 3:
        raise ValueError("Canonical normal ROVir PSF must reduce to three dimensions.")
    coefficients = extract_psf_phase_planes(values, readout_chunk=8)
    metrics = psf_identity_metrics(values, *coefficients, readout_chunk=8)
    if (
        not all(np.isfinite(value) for value in metrics.values())
        or metrics["relative_complex_l2"] > 2e-5
        or metrics["maximum_complex_error"] > 2e-4
    ):
        raise ValueError(f"Canonical normal ROVir PSF phase-plane gate failed: {metrics}")
    return coefficients, metrics


def _write_target_psf(
    output_base: Path,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    *,
    target_lin: int,
    target_par: int,
) -> None:
    """Evaluate one calibrated PSF on a resolution-matched PE grid.

    Args:
        output_base: Destination BART basename.
        alpha: Per-readout normalized LIN phase slope.
        beta: Per-readout normalized PAR phase slope.
        gamma: Per-readout constant phase.
        target_lin: Target LIN matrix size.
        target_par: Target PAR matrix size.

    Side Effects:
        Writes one finite complex64 BART CFL pair in bounded readout chunks.
    """
    output = create_cfl(output_base, (alpha.size, target_lin, target_par, 1, 1))
    for start in range(0, alpha.size, 8):
        stop = min(start + 8, alpha.size)
        output[start:stop, :, :, 0, 0] = evaluate_psf_phase_planes(
            alpha[start:stop], beta[start:stop], gamma[start:stop], target_lin, target_par
        )
    output.flush()
    del output


def _validate_reuse(
    manifest: Mapping[str, Any],
    contract_path: Path,
    destination: Path,
    *,
    expected_case: ResolvedCase,
    expected_sampling: Mapping[str, Any],
) -> None:
    """Validate an existing ROVir retrospective input tree for exact reuse.

    Args:
        manifest: Existing branch manifest.
        contract_path: Current canonical normal ROVir contract.
        destination: Existing ROVir BART-input directory.
        expected_case: Case geometry resolved from the canonical normal source.
        expected_sampling: Exact pure-mask contract for this case.

    Raises:
        ValueError: If provenance or artifact hashes differ.
    """
    if (
        manifest.get("status") != "mprage_rovir_retro_bart_inputs_ready"
        or manifest.get("source_normal_rovir_contract", {}).get("sha256") != sha256_file(contract_path)
        or manifest.get("case") != expected_case.to_json()
        or manifest.get("sampling", {}).get("logical_sha256")
        != expected_sampling.get("logical_sha256")
    ):
        raise ValueError(f"Existing ROVir retrospective inputs use another contract: {destination}")
    mask = np.load(destination / "sampling_mask.npy", allow_pickle=False)
    validate_pure_cartesian_image_lattice(mask, dict(manifest["sampling"]))
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
