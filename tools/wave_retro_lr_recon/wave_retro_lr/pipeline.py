"""Config-driven retrospective low-resolution Wave-MPRAGE pipeline."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import zoom

from .bart_io import bart_base, cfl_record, create_cfl, open_cfl, read_shape, sha256_file
from .core import (
    PE_MATRIX_MULTIPLE,
    CaseSpec,
    Geometry,
    ResolvedCase,
    apply_wave_forward,
    build_case_mask,
    build_wave_options,
    evaluate_psf_phase_planes,
    extract_psf_phase_planes,
    psf_identity_metrics,
    resolve_case,
)


OUTPUT_FOLDER_NAME = "retrospective_low_resolution"
SOURCE_OPERATOR_RELATIVE_TOLERANCE = 2e-5
PSF_RELATIVE_TOLERANCE = 1e-5
PSF_MAXIMUM_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SourceContract:
    bart_input_dir: Path
    bart_manifest: Path
    wave_kspace: Path
    psf: Path
    no_wave_kspace: Path
    sampling_mask: Path
    existing_maps: Path
    calibration_kspace: Path
    sequence: Path
    twix: Path
    synthesis_manifest: Path | None
    companion_manifests: tuple[Path, ...]
    source_acceleration_ry_rz: tuple[int, int]
    subject: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a manifest so partial runs never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field {label} must be a path string.")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return (path if path.is_absolute() else base / path).resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _manifest_path(value: Any, manifest: Mapping[str, Any], base: Path) -> Path:
    if isinstance(value, str) and value:
        return _resolve_path(value, base, "source.sampling_mask")
    mask = manifest.get("sampling_mask")
    if isinstance(mask, Mapping) and isinstance(mask.get("path"), str):
        path = Path(str(mask["path"])).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    raise ValueError("Sampling-mask path is absent from both config and BART manifest.")


def _source_acceleration(
    source_config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[int, int]:
    configured = source_config.get("acceleration_ry_rz")
    if isinstance(configured, list) and len(configured) == 2:
        return int(configured[0]), int(configured[1])
    mask = manifest.get("sampling_mask")
    if isinstance(mask, Mapping):
        ry = mask.get("pe1_acceleration")
        rz = mask.get("pe2_acceleration")
        if ry is not None and rz is not None:
            return int(ry), int(rz)
    raise ValueError(
        "Source acceleration is absent. Set source.acceleration_ry_rz or record "
        "pe1_acceleration/pe2_acceleration in the BART sampling-mask manifest."
    )


def load_source_contract(config: Mapping[str, Any], config_dir: Path) -> tuple[SourceContract, dict[str, Any]]:
    """Resolve the explicit source/companion inputs without recursive discovery."""
    source_config = config.get("source")
    if not isinstance(source_config, Mapping):
        raise ValueError("Configuration must contain a source object.")
    bart_input_dir = _resolve_path(source_config.get("bart_input_dir"), config_dir, "source.bart_input_dir")
    if not bart_input_dir.is_dir():
        raise NotADirectoryError(f"BART input directory not found: {bart_input_dir}")
    manifest_path = _require_file(bart_input_dir / "manifest.json", "BART input manifest")
    manifest = _load_json(manifest_path)
    if manifest.get("dimension_order") != ["READ", "PHS1", "PHS2", "COIL", "MAPS"]:
        raise ValueError(f"Unsupported BART dimension order in {manifest_path}.")
    echoes = manifest.get("echoes")
    if not isinstance(echoes, list) or len(echoes) != 1 or echoes[0].get("echo") != 1:
        raise ValueError("This MPRAGE integration requires exactly one manifest echo numbered 1.")
    echo = echoes[0]
    wave_name = echo.get("wave_kspace")
    psf_name = echo.get("psf")
    if not isinstance(wave_name, str) or not isinstance(psf_name, str):
        raise ValueError("Manifest echo must name wave_kspace and psf basenames.")
    synthesis_value = source_config.get("synthesis_manifest")
    synthesis_manifest = (
        _resolve_path(synthesis_value, config_dir, "source.synthesis_manifest")
        if synthesis_value not in (None, "")
        else None
    )
    if synthesis_manifest is not None:
        _require_file(synthesis_manifest, "Synthesis manifest")
    raw_companions = source_config.get("companion_manifests", [])
    if not isinstance(raw_companions, list):
        raise ValueError("source.companion_manifests must be a list of JSON paths.")
    companion_manifests = tuple(
        _require_file(
            _resolve_path(value, config_dir, "source.companion_manifests"),
            "Companion provenance manifest",
        )
        for value in raw_companions
    )
    contract = SourceContract(
        bart_input_dir=bart_input_dir,
        bart_manifest=manifest_path,
        wave_kspace=bart_input_dir / wave_name,
        psf=bart_input_dir / psf_name,
        no_wave_kspace=_resolve_path(source_config.get("no_wave_kspace"), config_dir, "source.no_wave_kspace"),
        sampling_mask=_manifest_path(source_config.get("sampling_mask"), manifest, bart_input_dir),
        existing_maps=_resolve_path(source_config.get("existing_maps"), config_dir, "source.existing_maps"),
        calibration_kspace=_resolve_path(
            source_config.get("calibration_kspace"), config_dir, "source.calibration_kspace"
        ),
        sequence=_resolve_path(source_config.get("sequence"), config_dir, "source.sequence"),
        twix=_resolve_path(source_config.get("twix"), config_dir, "source.twix"),
        synthesis_manifest=synthesis_manifest,
        companion_manifests=companion_manifests,
        source_acceleration_ry_rz=_source_acceleration(source_config, manifest),
        subject=str(source_config.get("subject", "retro-low-resolution")),
    )
    for label, path in (
        ("no-wave k-space", contract.no_wave_kspace),
        ("sampling mask", contract.sampling_mask),
        ("sequence", contract.sequence),
        ("TWIX", contract.twix),
    ):
        _require_file(path, label)
    for label, base in (
        ("source Wave k-space", contract.wave_kspace),
        ("source PSF", contract.psf),
        ("existing maps", contract.existing_maps),
        ("calibration k-space", contract.calibration_kspace),
    ):
        read_shape(base)
    return contract, manifest


def _read_geometry(sequence: Path, logical_shape: tuple[int, int, int]) -> Geometry:
    import pypulseq as pp

    seq = pp.Sequence()
    seq.read(str(sequence), remove_duplicates=False)
    definitions = seq.definitions
    if str(definitions.get("OrientationMapping", "")).upper() != "SAG":
        raise ValueError("Current retrospective adapter supports validated sagittal MPRAGE only.")
    expected_axes = {"ReadoutAxis": "z", "InnerPEAxis": "x", "OuterPEAxis": "y"}
    for key, expected in expected_axes.items():
        if key in definitions and str(definitions[key]).lower() != expected:
            raise ValueError(f"Sequence {key}={definitions[key]!r}; expected {expected!r}.")
    fov = np.asarray(definitions.get("FOV"), dtype=float).reshape(-1)
    if fov.size != 3 or not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("Sequence FOV must contain positive physical XYZ values.")
    geometry = Geometry(
        physical_fov_mm_xyz=tuple(float(value) * 1000.0 for value in fov),
        logical_matrix_ro_lin_par=logical_shape,
    )
    os_factor = int(float(definitions.get("ReadoutOversamplingFactor", 1)))
    return geometry, os_factor


def _case_specs(config: Mapping[str, Any]) -> list[CaseSpec]:
    raw_cases = config.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Configuration must contain a non-empty cases list.")
    specs: list[CaseSpec] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Case {index} must be an object.")
        resolution = raw.get("resolution_mm_xyz", raw.get("resolution_mm"))
        acceleration = raw.get("acceleration_ry_rz", raw.get("acceleration"))
        if not isinstance(resolution, list) or len(resolution) != 3:
            raise ValueError(f"Case {index} resolution must be physical [X,Y,Z].")
        if not isinstance(acceleration, list) or len(acceleration) != 2:
            raise ValueError(f"Case {index} acceleration must be [Ry,Rz].")
        specs.append(
            CaseSpec(
                tuple(float(value) for value in resolution),
                tuple(int(value) for value in acceleration),
                None if raw.get("label") in (None, "") else str(raw["label"]),
            )
        )
    return specs


def validate_contract(
    contract: SourceContract, config: Mapping[str, Any]
) -> tuple[Geometry, list[ResolvedCase], dict[str, Any]]:
    """Validate all shapes and resolve cases without reading full CFL payloads."""
    configured_policy = config.get("pe_matrix_policy")
    expected_policy = f"nearest multiple of {PE_MATRIX_MULTIPLE}"
    if configured_policy not in (None, expected_policy):
        raise ValueError(
            f"pe_matrix_policy must be {expected_policy!r}; got {configured_policy!r}."
        )
    wave_shape = read_shape(contract.wave_kspace) + (1,) * (5 - len(read_shape(contract.wave_kspace)))
    psf_shape = read_shape(contract.psf) + (1,) * (5 - len(read_shape(contract.psf)))
    maps_shape = read_shape(contract.existing_maps) + (1,) * (5 - len(read_shape(contract.existing_maps)))
    calib_shape = read_shape(contract.calibration_kspace) + (1,) * (4 - len(read_shape(contract.calibration_kspace)))
    if any(value != 1 for value in wave_shape[4:]) or any(value != 1 for value in psf_shape[3:]):
        raise ValueError("Source Wave k-space/PSF must contain one MPRAGE echo and map set.")
    ro_os, lin, par, coils = wave_shape[:4]
    if psf_shape[:3] != (ro_os, lin, par):
        raise ValueError(f"PSF shape {psf_shape} disagrees with Wave k-space {wave_shape}.")
    ro, map_lin, map_par, map_coils, maps = maps_shape[:5]
    if (map_lin, map_par, map_coils, maps) != (lin, par, coils, 1):
        raise ValueError(f"Map shape {maps_shape} disagrees with Wave k-space {wave_shape}.")
    if calib_shape[:4] != (ro, lin, par, coils):
        raise ValueError(f"Calibration shape {calib_shape} disagrees with maps/Wave inputs.")
    no_wave = np.load(contract.no_wave_kspace, mmap_mode="r", allow_pickle=False)
    if no_wave.shape != (ro, lin, par, coils) or no_wave.dtype != np.complex64:
        raise ValueError(
            f"No-wave source must be complex64 {(ro, lin, par, coils)}; "
            f"got {no_wave.shape} {no_wave.dtype}."
        )
    mask = np.load(contract.sampling_mask, mmap_mode="r", allow_pickle=False)
    if mask.shape != (lin, par):
        raise ValueError(f"Sampling mask shape {mask.shape} disagrees with {(lin, par)}.")
    geometry, os_factor = _read_geometry(contract.sequence, (ro, lin, par))
    if ro_os != ro * os_factor:
        raise ValueError(
            f"Wave readout {ro_os} does not equal logical readout {ro} x OS {os_factor}."
        )
    cases = [resolve_case(spec, geometry) for spec in _case_specs(config)]
    keys: set[tuple[tuple[int, int, int], tuple[int, int]]] = set()
    for case in cases:
        key = (case.target_logical_matrix_ro_lin_par, case.acceleration_ry_rz)
        if key in keys:
            raise ValueError(f"Duplicate case resolves to matrix/acceleration {key}.")
        keys.add(key)
        build_case_mask(mask, case, contract.source_acceleration_ry_rz)
    return geometry, cases, {
        "wave_shape": list(wave_shape[:5]),
        "psf_shape": list(psf_shape[:5]),
        "maps_shape": list(maps_shape[:5]),
        "calibration_shape": list(calib_shape[:4]),
        "no_wave_shape": list(no_wave.shape),
        "sampling_mask_shape": list(mask.shape),
        "readout_oversampling_factor": os_factor,
    }


def _source_provenance(contract: SourceContract, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Record source manifests and authoritative hashes without duplicating large data."""
    mask_metadata = manifest.get("sampling_mask") if isinstance(manifest.get("sampling_mask"), Mapping) else {}
    return {
        "bart_input_manifest": str(contract.bart_manifest),
        "bart_input_manifest_sha256": sha256_file(contract.bart_manifest),
        "synthesis_manifest": None if contract.synthesis_manifest is None else str(contract.synthesis_manifest),
        "synthesis_manifest_sha256": (
            None if contract.synthesis_manifest is None else sha256_file(contract.synthesis_manifest)
        ),
        "companion_manifests": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in contract.companion_manifests
        ],
        "no_wave_kspace": str(contract.no_wave_kspace),
        "sampling_mask": str(contract.sampling_mask),
        "sampling_mask_logical_sha256": mask_metadata.get("logical_sha256"),
        "wave_kspace": cfl_record(contract.wave_kspace, include_hash=False),
        "psf": cfl_record(contract.psf, include_hash=False),
        "existing_maps": cfl_record(contract.existing_maps, include_hash=False),
        "calibration_kspace": cfl_record(contract.calibration_kspace, include_hash=False),
        "sequence": str(contract.sequence),
        "sequence_sha256": sha256_file(contract.sequence),
        "twix": str(contract.twix),
        "source_acceleration_ry_rz": list(contract.source_acceleration_ry_rz),
    }


def _validate_psf(source_psf: np.ndarray, runtime: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    chunk = int(runtime.get("readout_chunk", 8))
    alpha, beta, gamma = extract_psf_phase_planes(source_psf, readout_chunk=chunk)
    metrics = psf_identity_metrics(
        source_psf, alpha, beta, gamma, readout_chunk=chunk
    )
    if (
        metrics["relative_complex_l2"] > PSF_RELATIVE_TOLERANCE
        or metrics["maximum_complex_error"] > PSF_MAXIMUM_TOLERANCE
    ):
        raise ValueError(f"Source PSF phase-plane identity gate failed: {metrics}")
    return alpha, beta, gamma, metrics


def _validate_source_operator(
    contract: SourceContract,
    source_psf: np.ndarray,
    mask: np.ndarray,
    *,
    fft_workers: int,
) -> dict[str, float]:
    """Prove crop-first synthesis reproduces the supplied native Wave input."""
    no_wave = np.load(contract.no_wave_kspace, mmap_mode="r", allow_pickle=False)
    reference = open_cfl(contract.wave_kspace)
    ro_os, _, _, coils = reference.shape[:4]
    error_squared = 0.0
    reference_squared = 0.0
    maximum_error = 0.0
    for coil in range(coils):
        synthesized = apply_wave_forward(
            np.asarray(no_wave[..., coil]),
            source_psf,
            readout_oversampled=ro_os,
            fft_workers=fft_workers,
        )
        synthesized *= mask[None, :, :]
        current = np.asarray(reference[:, :, :, coil, ...]).squeeze()
        difference = synthesized - current
        error_squared += float(np.vdot(difference, difference).real)
        reference_squared += float(np.vdot(current, current).real)
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
        print(f"Validated source Wave operator coil {coil + 1:02d}/{coils:02d}", flush=True)
    relative = float(np.sqrt(error_squared / reference_squared))
    metrics = {"relative_complex_l2": relative, "maximum_complex_error": maximum_error}
    if relative > SOURCE_OPERATOR_RELATIVE_TOLERANCE:
        raise ValueError(f"Native-grid Wave operator identity gate failed: {metrics}")
    return metrics


def _prepared_validation_metrics(
    batch: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Reuse finite operator gates from a hash-matched completed preparation."""
    records: list[dict[str, float]] = []
    for key in ("psf_source_identity", "native_wave_operator_identity"):
        value = batch.get(key)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"Prepared batch is missing {key}.")
        record = {str(name): float(metric) for name, metric in value.items()}
        if not all(np.isfinite(metric) for metric in record.values()):
            raise ValueError(f"Prepared batch contains non-finite {key} metrics.")
        records.append(record)
    return records[0], records[1]


def _write_target_psf(
    output_base: Path,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    target_lin: int,
    target_par: int,
    *,
    readout_chunk: int,
) -> None:
    output = create_cfl(output_base, (alpha.size, target_lin, target_par, 1, 1))
    for start in range(0, alpha.size, readout_chunk):
        stop = min(start + readout_chunk, alpha.size)
        output[start:stop, :, :, 0, 0] = evaluate_psf_phase_planes(
            alpha[start:stop], beta[start:stop], gamma[start:stop], target_lin, target_par
        )
    output.flush()
    del output


def _fit_spatial_shape(array: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if array.shape == target:
        return array
    output = np.zeros(target, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(array.shape, target, strict=True):
        size = min(source_size, target_size)
        source_start = source_size // 2 - size // 2
        target_start = target_size // 2 - size // 2
        source_slices.append(slice(source_start, source_start + size))
        target_slices.append(slice(target_start, target_start + size))
    output[tuple(target_slices)] = array[tuple(source_slices)]
    return output


def _write_target_maps(source_base: Path, output_base: Path, target_shape: tuple[int, int, int]) -> None:
    source = open_cfl(source_base)
    ro, source_lin, source_par, coils = source.shape[:4]
    target_ro, target_lin, target_par = target_shape
    if target_ro != ro:
        raise ValueError("Sensitivity-map readout interpolation is forbidden.")
    output = create_cfl(output_base, (ro, target_lin, target_par, coils, 1))
    rss_squared = np.zeros(target_shape, dtype=np.float32)
    factors = (1.0, target_lin / source_lin, target_par / source_par)
    for coil in range(coils):
        current = np.asarray(source[:, :, :, coil, ...]).squeeze()
        interpolated = zoom(current.real, factors, order=1) + 1j * zoom(
            current.imag, factors, order=1
        )
        interpolated = _fit_spatial_shape(
            interpolated.astype(np.complex64, copy=False), target_shape
        )
        output[:, :, :, coil, 0] = interpolated
        rss_squared += np.abs(interpolated) ** 2
    rss = np.sqrt(rss_squared)
    support = rss > 1e-8
    for coil in range(coils):
        current = np.asarray(output[:, :, :, coil, 0])
        current[support] /= rss[support]
        current[~support] = 0
        output[:, :, :, coil, 0] = current
    output.flush()
    del output


def _write_target_calibration(
    source_base: Path, output_base: Path, case: ResolvedCase
) -> None:
    source = open_cfl(source_base)
    ro, _, _, coils = source.shape[:4]
    _, target_lin, target_par = case.target_logical_matrix_ro_lin_par
    output = create_cfl(output_base, (ro, target_lin, target_par, coils))
    lin = slice(*case.crop_bounds_lin)
    par = slice(*case.crop_bounds_par)
    for coil in range(coils):
        output[:, :, :, coil] = source[:, lin, par, coil]
    output.flush()
    del output


def _write_target_wave_kspace(
    contract: SourceContract,
    output_base: Path,
    target_psf_base: Path,
    case: ResolvedCase,
    case_mask: np.ndarray,
    *,
    fft_workers: int,
) -> float:
    no_wave = np.load(contract.no_wave_kspace, mmap_mode="r", allow_pickle=False)
    source_wave_shape = read_shape(contract.wave_kspace)
    ro_os, _, _, coils = source_wave_shape[:4]
    _, target_lin, target_par = case.target_logical_matrix_ro_lin_par
    output = create_cfl(output_base, (ro_os, target_lin, target_par, coils, 1))
    target_psf = open_cfl(target_psf_base)[:, :, :, 0, 0]
    lin = slice(*case.crop_bounds_lin)
    par = slice(*case.crop_bounds_par)
    squared_norm = 0.0
    for coil in range(coils):
        cropped_no_wave = np.asarray(no_wave[:, lin, par, coil], dtype=np.complex64)
        encoded = apply_wave_forward(
            cropped_no_wave,
            target_psf,
            readout_oversampled=ro_os,
            fft_workers=fft_workers,
        )
        encoded *= case_mask[None, :, :]
        output[:, :, :, coil, 0] = encoded
        squared_norm += float(np.vdot(encoded, encoded).real)
        print(f"Prepared {case.case_name} coil {coil + 1:02d}/{coils:02d}", flush=True)
    output.flush()
    del output
    return float(np.sqrt(squared_norm))


def _target_manifest(
    case: ResolvedCase,
    bart_input_dir: Path,
    kspace_norm: float,
    source: SourceContract,
) -> dict[str, Any]:
    wave_shape = read_shape(bart_input_dir / "wave_kspace")
    psf_shape = read_shape(bart_input_dir / "psf")
    return {
        "format_version": 1,
        "format": "BART CFL",
        "status": "retrospective_low_resolution_inputs_ready",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "source_bart_input_manifest": str(source.bart_manifest),
        "retrospective_case": case.to_json(),
        "coil_sens": "coil_sens",
        "coil_sens_shape": list(read_shape(bart_input_dir / "coil_sens")),
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(read_shape(bart_input_dir / "kspace_calib")),
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": list(wave_shape),
                "wave_kspace_norm": kspace_norm,
                "psf": "psf",
                "psf_shape": list(psf_shape),
            }
        ],
    }


def _prepare_case(
    case: ResolvedCase,
    contract: SourceContract,
    source_mask: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    case_dir: Path,
    runtime: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    config_sha256: str,
    *,
    recover: bool = False,
) -> dict[str, Any]:
    bart_inputs = case_dir / "bart_inputs"
    if bart_inputs.exists():
        if not recover:
            raise FileExistsError(f"BART inputs already exist: {bart_inputs}")
        # --resume explicitly authorizes replacement of an incomplete case's
        # generated inputs. Source inputs and completed cases are never touched.
        shutil.rmtree(bart_inputs)
    bart_inputs.mkdir(parents=True, exist_ok=False)
    target_ro, target_lin, target_par = case.target_logical_matrix_ro_lin_par
    case_mask = build_case_mask(source_mask, case, contract.source_acceleration_ry_rz)
    _write_target_psf(
        bart_inputs / "psf",
        alpha,
        beta,
        gamma,
        target_lin,
        target_par,
        readout_chunk=int(runtime.get("readout_chunk", 8)),
    )
    _write_target_maps(
        contract.existing_maps,
        bart_inputs / "coil_sens",
        (target_ro, target_lin, target_par),
    )
    _write_target_calibration(contract.calibration_kspace, bart_inputs / "kspace_calib", case)
    kspace_norm = _write_target_wave_kspace(
        contract,
        bart_inputs / "wave_kspace",
        bart_inputs / "psf",
        case,
        case_mask,
        fft_workers=int(runtime.get("fft_workers", 4)),
    )
    manifest = _target_manifest(case, bart_inputs, kspace_norm, contract)
    _write_json(bart_inputs / "manifest.json", manifest)
    payload = {
        "format_version": 1,
        "status": "prepared",
        "case": case.to_json(),
        "config_sha256": config_sha256,
        "source": source_provenance,
        "operator_order": [
            "center-crop no-wave k-space in LIN/PAR",
            "inverse FFT on target grid",
            "embed unchanged readout into oversampled FOV",
            "apply target-grid Wave PSF",
            "FFT LIN/PAR",
            "apply cropped retrospective sampling mask",
        ],
        "readout_cropped": False,
        "sampling_mask_saved": False,
        "sampled_coordinate_count": int(np.count_nonzero(case_mask)),
        "sampling_fraction": float(np.mean(case_mask)),
        "bart_inputs": {
            "manifest": str(bart_inputs / "manifest.json"),
            "wave_kspace": cfl_record(bart_inputs / "wave_kspace"),
            "psf": cfl_record(bart_inputs / "psf"),
            "coil_sens": cfl_record(bart_inputs / "coil_sens"),
            "kspace_calib": cfl_record(bart_inputs / "kspace_calib"),
            "wave_kspace_norm": kspace_norm,
        },
        "prepared_at_utc": _utc_now(),
    }
    _write_json(case_dir / "case_manifest.json", payload)
    return payload


def _reuse_prepared_case(
    case: ResolvedCase,
    source_case_dir: Path,
    case_dir: Path,
    source_provenance: Mapping[str, Any],
    config_sha256: str,
    *,
    recover: bool = False,
) -> dict[str, Any]:
    """Link a hash-validated prior case's BART inputs into a new solver run."""
    source_manifest_path = source_case_dir / "case_manifest.json"
    source_payload = _load_json(
        _require_file(source_manifest_path, "Prepared case manifest")
    )
    if source_payload.get("status") != "complete":
        raise ValueError(f"Prepared source case is not complete: {source_manifest_path}")
    if source_payload.get("case") != case.to_json():
        raise ValueError(f"Prepared source case geometry differs: {source_manifest_path}")
    if source_payload.get("source") != source_provenance:
        raise ValueError(f"Prepared source provenance differs: {source_manifest_path}")

    source_inputs = source_case_dir / "bart_inputs"
    _require_file(source_inputs / "manifest.json", "Prepared BART input manifest")
    for name in ("wave_kspace", "psf", "coil_sens", "kspace_calib"):
        expected = source_payload["bart_inputs"][name]
        if cfl_record(source_inputs / name) != expected:
            raise ValueError(f"Prepared BART input changed: {source_inputs / name}")

    linked_inputs = case_dir / "bart_inputs"
    if linked_inputs.exists() or linked_inputs.is_symlink():
        if not recover or not linked_inputs.is_symlink():
            raise FileExistsError(f"Reused BART input link already exists: {linked_inputs}")
        linked_inputs.unlink()
    linked_inputs.symlink_to(source_inputs, target_is_directory=True)

    payload = {
        key: source_payload[key]
        for key in (
            "format_version",
            "case",
            "operator_order",
            "readout_cropped",
            "sampling_mask_saved",
            "sampled_coordinate_count",
            "sampling_fraction",
            "bart_inputs",
        )
    }
    payload.update(
        {
            "status": "prepared",
            "config_sha256": config_sha256,
            "source": source_provenance,
            "prepared_inputs_reused_from": {
                "case_manifest": str(source_manifest_path),
                "case_manifest_sha256": sha256_file(source_manifest_path),
                "bart_inputs": str(source_inputs),
            },
            "prepared_at_utc": _utc_now(),
        }
    )
    _write_json(case_dir / "case_manifest.json", payload)
    return payload


def _stream_command(command: Sequence[str], log_path: Path) -> float:
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    lines: list[str] = []
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    if returncode:
        raise RuntimeError(f"BART reconstruction failed with status {returncode}.")
    return time.perf_counter() - started


def _load_bart_image(image_base: Path, regularizer: str) -> tuple[np.ndarray, dict[str, Any]]:
    image = open_cfl(image_base)
    shape = image.shape + (1,) * max(0, 9 - image.ndim)
    if regularizer == "llr":
        if shape[8] != 2 or any(value != 1 for value in shape[3:8] + shape[9:]):
            raise ValueError(f"Unexpected split-complex LLR output shape: {image.shape}")
        real_component = np.take(np.asarray(image), 0, axis=8)
        imaginary_component = np.take(np.asarray(image), 1, axis=8)
        residual = max(
            float(np.max(np.abs(real_component.imag))),
            float(np.max(np.abs(imaginary_component.real))),
        )
        if residual > 1e-6:
            raise ValueError(f"Split-complex off-axis residual is {residual}.")
        combined = real_component.real + 1j * imaginary_component.imag
        return np.asarray(combined).squeeze().astype(np.complex64), {
            "split_complex": True,
            "rule": "real(ITER[0]) + 1j * imag(ITER[1])",
            "maximum_off_axis_residual": residual,
        }
    if any(value != 1 for value in shape[3:]):
        raise ValueError(f"Expected one 3D BART image; got {image.shape}.")
    return np.asarray(image).squeeze().astype(np.complex64), {"split_complex": False}


def _load_upstream_exporter(repo_root: Path):
    script = repo_root / "external" / "wave-mprage" / "recon" / "recon_wave_mprage_from_twix_integrated_nifti.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pinned Wave-MPRAGE exporter not found: {script}")
    recon_dir = script.parent
    if str(recon_dir) not in sys.path:
        sys.path.insert(0, str(recon_dir))
    spec = importlib.util.spec_from_file_location("wave_mprage_retro_export", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import NIfTI exporter: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reconstruction_settings(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, list[str]]:
    """Return the declared regularizer and its validated BART Wave options."""
    reconstruction = config.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError("Configuration must contain a reconstruction object.")
    regularizer = str(reconstruction.get("regularizer", "none"))
    lambda_value = reconstruction.get("lambda")
    options = build_wave_options(
        regularizer,
        None if lambda_value is None else float(lambda_value),
        block_size=int(reconstruction.get("block_size", 8)),
        iterations=int(reconstruction.get("iterations", 100)),
        tolerance=float(reconstruction.get("tolerance", 1e-6)),
        maximum_eigenvalue=(
            None
            if reconstruction.get("maximum_eigenvalue") in (None, "")
            else float(reconstruction["maximum_eigenvalue"])
        ),
    )
    if "-g" not in options:
        raise AssertionError("Every BART reconstruction must use GPU option -g.")
    return reconstruction, regularizer, options


def _run_reconstruction(
    case: ResolvedCase,
    case_dir: Path,
    contract: SourceContract,
    config: Mapping[str, Any],
    repo_root: Path,
    *,
    recover: bool = False,
) -> dict[str, Any]:
    reconstruction, regularizer, options = _reconstruction_settings(config)
    bart_value = str(reconstruction.get("bart", os.environ.get("BART_BIN", "bart")))
    bart_executable = shutil.which(bart_value)
    if bart_executable is None:
        raise FileNotFoundError(
            f"BART executable not found: {bart_value}. Source bart_startup.sh before running."
        )
    bart_inputs = case_dir / "bart_inputs"
    bart_output = case_dir / "bart_output"
    nifti_root = case_dir / "nifti"
    if bart_output.exists() or nifti_root.exists():
        if not recover:
            raise FileExistsError(
                f"Reconstruction output already exists beneath {case_dir}. Use --resume "
                "only for an incomplete recorded case."
            )
        for generated in (bart_output, nifti_root):
            if generated.exists():
                shutil.rmtree(generated)
    bart_output.mkdir(exist_ok=False)
    image_base = bart_output / "image_wave"
    command = [
        bart_executable,
        "wave",
        *options,
        str(bart_inputs / "coil_sens"),
        str(bart_inputs / "psf"),
        str(bart_inputs / "wave_kspace"),
        str(image_base),
    ]
    elapsed = _stream_command(command, case_dir / "bart_wave.log")
    image, split_metadata = _load_bart_image(image_base, regularizer)
    expected_shape = case.target_logical_matrix_ro_lin_par
    if image.shape != expected_shape or not np.isfinite(image).all():
        raise ValueError(f"BART image shape/finite gate failed: {image.shape}, expected {expected_shape}.")
    input_manifest = _load_json(bart_inputs / "manifest.json")
    kspace_norm = float(input_manifest["echoes"][0]["wave_kspace_norm"])
    restored = image * kspace_norm
    native = _load_upstream_exporter(repo_root)
    achieved_x, achieved_y, achieved_z = case.achieved_resolution_mm_xyz
    metadata = {
        "Reconstruction": "BART Wave retrospective low resolution",
        "ReconstructionSoftware": "BART wave",
        "RetrospectiveLowResolution": True,
        "RetrospectiveCase": case.to_json(),
        "RetrospectiveReadoutCropped": False,
        "BARTCommand": command,
        "BARTGPURequired": True,
        "BARTWaveKspaceNormRestored": kspace_norm,
        "BARTInternalNormalizationRestored": True,
        "BARTOutputAlreadyReadoutDeoversampled": True,
        "BARTSplitComplex": split_metadata,
    }
    native.save_mprage_output_to_nifti(
        image=restored,
        twix_file=str(contract.twix),
        out_folder=str(nifti_root),
        nifti_sub=contract.subject,
        suffix="BARTWaveRetrospectiveLowResolution",
        tag_wave="wave",
        voxel_size_mm=(achieved_z, achieved_y, achieved_x),
        crop_readout_os=1,
        save_phase=True,
        twix_array_axis_roles=("phase", "readout", "slice"),
        twix_array_axis_flips=(True, False, False),
        metadata=metadata,
    )
    outputs = sorted(str(path) for path in nifti_root.rglob("*.nii.gz"))
    if len(outputs) != 2:
        raise ValueError(f"Expected magnitude and phase NIfTIs, found {outputs}.")
    return {
        "command": command,
        "gpu": True,
        "elapsed_seconds": elapsed,
        "bart_output": cfl_record(image_base),
        "split_complex": split_metadata,
        "kspace_norm_restored": kspace_norm,
        "nifti_outputs": outputs,
    }


def run_config(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    validate_only: bool = False,
    prepare_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate, prepare, and optionally reconstruct all configured cases."""
    config_path = Path(config_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    config = _load_json(_require_file(config_path, "Configuration"))
    contract, source_manifest = load_source_contract(config, config_path.parent)
    geometry, cases, shape_validation = validate_contract(contract, config)
    _, regularizer, wave_options = _reconstruction_settings(config)
    output_root = _resolve_path(config.get("output_root"), config_path.parent, "output_root")
    summary = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "pe_matrix_policy": f"nearest multiple of {PE_MATRIX_MULTIPLE}",
        "output_root": str(output_root),
        "geometry": asdict(geometry),
        "shape_validation": shape_validation,
        "source": _source_provenance(contract, source_manifest),
        "cases": [case.to_json() for case in cases],
        "reconstruction": {
            "backend": "BART wave",
            "regularizer": regularizer,
            "wave_options": wave_options,
        },
    }
    prepared_cases_value = config.get("prepared_cases_root")
    prepared_cases_root = (
        None
        if prepared_cases_value in (None, "")
        else _resolve_path(
            prepared_cases_value, config_path.parent, "prepared_cases_root"
        )
    )
    prepared_batch: dict[str, Any] | None = None
    if prepared_cases_root is not None:
        source_batch_path = _require_file(
            prepared_cases_root / "batch_manifest.json", "Prepared batch manifest"
        )
        prepared_batch = _load_json(source_batch_path)
        if prepared_batch.get("status") != "complete":
            raise ValueError("The prepared source batch is not complete.")
        if prepared_batch.get("source") != summary["source"]:
            raise ValueError("The prepared source batch has different source provenance.")
        for case in cases:
            source_case_path = (
                prepared_cases_root / case.case_name / "case_manifest.json"
            )
            source_case = _load_json(
                _require_file(source_case_path, "Prepared case manifest")
            )
            if source_case.get("status") != "complete" or source_case.get(
                "case"
            ) != case.to_json():
                raise ValueError(
                    "Prepared case is incomplete or has different geometry: "
                    f"{source_case_path}"
                )
        summary["prepared_cases_reused_from"] = {
            "root": str(prepared_cases_root),
            "batch_manifest_sha256": sha256_file(source_batch_path),
        }
    print(f"Validated retrospective source: {contract.bart_input_dir}")
    for case in cases:
        print(
            f"  {case.case_name}: requested={case.requested_resolution_mm_xyz} mm, "
            f"achieved={case.achieved_resolution_mm_xyz} mm, "
            f"logical matrix={case.target_logical_matrix_ro_lin_par}"
        )
    if validate_only:
        print("Structural validation complete; no output was written.")
        return {**summary, "status": "validated"}
    workflow_root = output_root / OUTPUT_FOLDER_NAME
    if prepared_cases_root == workflow_root:
        raise ValueError("prepared_cases_root must differ from the new workflow root.")
    if workflow_root.exists() and any(workflow_root.iterdir()) and not resume:
        raise FileExistsError(
            f"Output tree is not empty: {workflow_root}. Use a new output_root or --resume."
        )
    workflow_root.mkdir(parents=True, exist_ok=True)
    batch_manifest = workflow_root / "batch_manifest.json"
    runtime = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}
    alpha = beta = gamma = source_mask = None
    if prepared_batch is None:
        source_psf_raw = open_cfl(contract.psf)
        source_psf = source_psf_raw[:, :, :, 0, 0]
        alpha, beta, gamma, psf_metrics = _validate_psf(source_psf, runtime)
        source_mask = np.asarray(
            np.load(contract.sampling_mask, mmap_mode="r", allow_pickle=False),
            dtype=bool,
        )
        operator_metrics = _validate_source_operator(
            contract,
            source_psf,
            source_mask,
            fft_workers=int(runtime.get("fft_workers", 4)),
        )
    else:
        psf_metrics, operator_metrics = _prepared_validation_metrics(prepared_batch)
    batch = {
        **summary,
        "status": "running",
        "psf_source_identity": psf_metrics,
        "native_wave_operator_identity": operator_metrics,
        "started_at_utc": _utc_now(),
        "case_manifests": [],
    }
    _write_json(batch_manifest, batch)
    for case in cases:
        case_dir = workflow_root / case.case_name
        case_manifest_path = case_dir / "case_manifest.json"
        recover_preparation = False
        recover_reconstruction = False
        if case_manifest_path.is_file() and resume:
            case_payload = _load_json(case_manifest_path)
            if case_payload.get("case") != case.to_json():
                raise ValueError(f"Existing case configuration differs: {case_manifest_path}")
            if case_payload.get("config_sha256") != summary["config_sha256"]:
                raise ValueError(f"Existing run configuration differs: {case_manifest_path}")
            if case_payload.get("source") != summary["source"]:
                raise ValueError(f"Existing source provenance differs: {case_manifest_path}")
            status = case_payload.get("status")
            if status == "complete":
                batch["case_manifests"].append(str(case_manifest_path))
                _write_json(batch_manifest, batch)
                continue
            if status == "preparing":
                recover_preparation = True
            elif status in {"prepared", "reconstructing"}:
                recover_reconstruction = status == "reconstructing"
            else:
                raise ValueError(
                    f"Unsupported recorded case status {status!r}: {case_manifest_path}"
                )
        else:
            if case_dir.exists():
                raise FileExistsError(f"Incomplete or unapproved existing case directory: {case_dir}")
            case_dir.mkdir(parents=True)
            case_payload = {
                "format_version": 1,
                "status": "preparing",
                "case": case.to_json(),
                "config_sha256": summary["config_sha256"],
                "source": summary["source"],
                "started_at_utc": _utc_now(),
            }
            _write_json(case_manifest_path, case_payload)
        if case_payload.get("status") == "preparing":
            if prepared_cases_root is None:
                assert source_mask is not None
                assert alpha is not None and beta is not None and gamma is not None
                case_payload = _prepare_case(
                    case,
                    contract,
                    source_mask,
                    alpha,
                    beta,
                    gamma,
                    case_dir,
                    runtime,
                    summary["source"],
                    summary["config_sha256"],
                    recover=recover_preparation,
                )
            else:
                case_payload = _reuse_prepared_case(
                    case,
                    prepared_cases_root / case.case_name,
                    case_dir,
                    summary["source"],
                    summary["config_sha256"],
                    recover=recover_preparation,
                )
        if not prepare_only and case_payload.get("status") != "complete":
            case_payload = {
                **case_payload,
                "status": "reconstructing",
                "reconstruction_started_at_utc": _utc_now(),
            }
            _write_json(case_manifest_path, case_payload)
            reconstruction = _run_reconstruction(
                case,
                case_dir,
                contract,
                config,
                repo_root,
                recover=recover_reconstruction,
            )
            case_payload = {
                **case_payload,
                "status": "complete",
                "reconstruction": reconstruction,
                "completed_at_utc": _utc_now(),
            }
            _write_json(case_manifest_path, case_payload)
        batch["case_manifests"].append(str(case_manifest_path))
        _write_json(batch_manifest, batch)
    batch["status"] = "prepared" if prepare_only else "complete"
    batch["completed_at_utc"] = _utc_now()
    _write_json(batch_manifest, batch)
    print(f"Batch manifest: {batch_manifest}")
    return batch
