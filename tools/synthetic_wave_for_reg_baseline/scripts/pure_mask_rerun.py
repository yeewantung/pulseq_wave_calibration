"""Shared contracts for the corrected pure-mask synthetic Wave rerun.

This module owns validation and deterministic layout only. Reconstruction is
performed by BART through the dedicated runner, while crop-first synthetic
Wave encoding, geometry, FFT, mask, and CFL operations come from
``wave_retro_lr_recon``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = SCRIPT_ROOT.parent
REPOSITORY_ROOT = TOOL_ROOT.parents[1]
RETRO_TOOL_ROOT = REPOSITORY_ROOT / "tools" / "wave_retro_lr_recon"

import sys

if str(RETRO_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRO_TOOL_ROOT))

from wave_retro_lr.bart_io import (  # noqa: E402
    bart_base,
    cfl_record,
    create_cfl,
    logical_array_sha256,
    open_cfl,
    read_shape,
    sha256_file,
)
from wave_retro_lr.core import (  # noqa: E402
    CaseSpec,
    Geometry,
    ResolvedCase,
    centered_fftn,
    resolve_case,
)
from wave_retro_lr.retrospective import synthesize_wave_from_no_wave_crop  # noqa: E402
from wave_retro_lr.sampling import (  # noqa: E402
    PURE_CARTESIAN_IMAGE_LATTICE,
    pure_cartesian_image_lattice_mask,
    validate_pure_cartesian_image_lattice,
)


WORKFLOW_NAME = "synthetic_wave_pure_mask_regularization_rerun"
THEORETICAL_PSF_MODEL = "theoretical_sequence_trajectory_without_calibrated_correction"
SYNTHETIC_WAVE_ORIGIN = "synthetic_from_fully_sampled_no_wave"
CASE_IDS = (
    "native_r3x1",
    "native_r3x2",
    "lr_x_r3x2",
    "lr_y_r3x2",
    "lr_xy_r3x2",
)
COARSE_WAVELET_LAMBDAS = (0.002, 0.005, 0.01, 0.015, 0.022, 0.03, 0.05)
COARSE_LLR_LAMBDAS = (0.002, 0.005, 0.01, 0.02, 0.04)
LLR_BLOCK_SIZES = (4, 8, 16)
FINE_LAMBDA_POOL = (
    0.003,
    0.004,
    0.006,
    0.0075,
    0.0085,
    0.01,
    0.0115,
    0.0125,
    0.015,
    0.0175,
    0.02,
    0.025,
    0.0275,
    0.03,
    0.0325,
    0.035,
    0.04,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PureMaskCase:
    """Resolved geometry, acceleration, residue, and accepted inputs for one case."""

    case_id: str
    resolved: ResolvedCase
    acceleration_lin_par: tuple[int, int]
    residue_lin_par: tuple[int, int]
    mask: np.ndarray
    mask_metadata: dict[str, Any]
    csm: dict[str, Any]
    psf: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Convert the complete immutable case contract to JSON-native values."""
        return {
            "case_id": self.case_id,
            "geometry": self.resolved.to_json(),
            "acceleration_lin_par": list(self.acceleration_lin_par),
            "residue_lin_par": list(self.residue_lin_par),
            "sampling_mask": self.mask_metadata,
            "csm": self.csm,
            "psf": self.psf,
        }


def load_json(path: str | Path, label: str = "JSON file") -> dict[str, Any]:
    """Load one JSON object from a required file.

    Args:
        path: JSON path to read without modification.
        label: Human-readable input name used in validation errors.

    Returns:
        Parsed JSON object.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON object through a sibling temporary file.

    Args:
        path: Destination manifest path.
        payload: JSON-serializable mapping to persist.

    Returns:
        None. The destination is replaced only after a complete write.
    """
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_object_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON object with deterministic key and separator encoding.

    Args:
        payload: JSON-native mapping whose semantic values define a contract.

    Returns:
        Lowercase SHA-256 digest of canonical compact JSON bytes.
    """
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_config_path(value: Any, config_dir: Path, label: str) -> Path:
    """Resolve a nonempty path relative to a local configuration file.

    Args:
        value: String or path-like configuration value.
        config_dir: Directory containing the local configuration.
        label: Field name used in validation errors.

    Returns:
        Expanded absolute path without requiring it to exist.
    """
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{label} must be a nonempty path.")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def output_layout(output_root: str | Path) -> dict[str, Any]:
    """Return the exact production tree used by preparation and sweep stages.

    Args:
        output_root: User-confirmed root directory and run name.

    Returns:
        JSON-native directory and manifest layout without creating any path.
    """
    root = Path(output_root).expanduser().resolve()
    return {
        "root": str(root),
        "source_materialization": {
            "root": str(root / "source_materialization"),
            "no_wave_kspace": str(
                root / "source_materialization" / "no_wave" / "source_full_ncc12.npy"
            ),
            "no_wave_report": str(
                root / "source_materialization" / "no_wave" / "source_report.json"
            ),
            "no_wave_progress": str(
                root / "source_materialization" / "no_wave" / "source_recon_progress.json"
            ),
            "full_wave_kspace": str(root / "source_materialization" / "full_wave_kspace.npy"),
            "full_wave_progress": str(
                root / "source_materialization" / "full_wave_progress.json"
            ),
            "manifest": str(root / "source_materialization" / "manifest.json"),
        },
        "preparation_manifest": str(root / "preparation_manifest.json"),
        "cases": {
            case_id: {
                "root": str(root / "cases" / case_id),
                "case_manifest": str(root / "cases" / case_id / "case_manifest.json"),
                "sampling_mask": str(root / "cases" / case_id / "sampling_mask.npy"),
                "direct_fft_reference": str(
                    root / "cases" / case_id / "direct_fft_reference_logical.npy"
                ),
                "bart_inputs": str(root / "cases" / case_id / "bart_inputs"),
                "full_wave_kspace": str(
                    root / "source_materialization" / "full_wave_kspace.npy"
                    if case_id in {"native_r3x1", "native_r3x2"}
                    else root / "cases" / case_id / "full_wave_kspace.npy"
                ),
            }
            for case_id in CASE_IDS
        },
        "sweeps": {
            "coarse": str(root / "sweeps" / "coarse"),
            "fine": str(root / "sweeps" / "fine"),
        },
        "evaluation": {
            "coarse": str(root / "evaluation" / "coarse"),
            "fine": str(root / "evaluation" / "fine"),
            "review": str(root / "evaluation" / "review"),
        },
    }


def _require_sha256(value: Any, label: str) -> str:
    """Validate one lowercase SHA-256 string and return it.

    Args:
        value: Candidate digest.
        label: Field name used in errors.

    Returns:
        Validated lowercase digest.
    """
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters.")
    return digest


def _json_path(payload: Mapping[str, Any], path: Sequence[Any]) -> Any:
    """Resolve one explicit object/list path in provenance JSON.

    Args:
        payload: Parsed provenance manifest.
        path: String object keys or integer list indices.

    Returns:
        Value at the declared path.
    """
    current: Any = payload
    for component in path:
        if isinstance(current, Mapping) and isinstance(component, str):
            if component not in current:
                raise KeyError(component)
            current = current[component]
        elif isinstance(current, list) and isinstance(component, int):
            current = current[component]
        else:
            raise KeyError(component)
    return current


def validate_manifest_binding(
    specification: Mapping[str, Any],
    config_dir: Path,
    *,
    required_assertion_labels: set[str],
    label: str,
) -> dict[str, Any]:
    """Validate a hash-bound provenance manifest and named exact assertions.

    Args:
        specification: Manifest path, digest, and assertion list.
        config_dir: Base directory for relative paths.
        required_assertion_labels: Scientific provenance claims that must be present.
        label: Human-readable artifact name.

    Returns:
        Manifest record with the validated assertion snapshot.
    """
    path = resolve_config_path(specification.get("path"), config_dir, f"{label}.manifest.path")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = _require_sha256(
        specification.get("sha256"), f"{label}.manifest.sha256"
    )
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} provenance manifest hash changed: {path}")
    payload = load_json(path, f"{label} provenance manifest")
    upstream_records = []
    for upstream in payload.get("upstream_manifests", []):
        if not isinstance(upstream, Mapping):
            raise ValueError(f"{label} upstream-manifest records must be objects.")
        upstream_path = resolve_config_path(
            upstream.get("path"), path.parent, f"{label}.upstream_manifest.path"
        )
        upstream_hash = _require_sha256(
            upstream.get("sha256"), f"{label}.upstream_manifest.sha256"
        )
        if not upstream_path.is_file() or sha256_file(upstream_path) != upstream_hash:
            raise ValueError(f"{label} upstream provenance manifest changed: {upstream_path}")
        upstream_records.append({"path": str(upstream_path), "sha256": upstream_hash})
    raw_assertions = specification.get("assertions")
    if not isinstance(raw_assertions, list):
        raise ValueError(f"{label}.manifest.assertions must be a list.")
    labels: set[str] = set()
    snapshots = []
    for assertion in raw_assertions:
        if not isinstance(assertion, Mapping):
            raise ValueError(f"{label} provenance assertions must be objects.")
        assertion_label = str(assertion.get("label", "")).strip()
        path_components = assertion.get("json_path")
        if not assertion_label or not isinstance(path_components, list):
            raise ValueError(f"{label} assertion requires label and json_path.")
        if assertion_label in labels:
            raise ValueError(f"{label} repeats assertion label {assertion_label!r}.")
        labels.add(assertion_label)
        try:
            found = _json_path(payload, path_components)
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{label} manifest lacks assertion path {path_components!r}."
            ) from exc
        if found != assertion.get("equals"):
            raise ValueError(
                f"{label} provenance assertion {assertion_label!r} changed."
            )
        snapshots.append(
            {"label": assertion_label, "json_path": path_components, "value": found}
        )
    missing = sorted(required_assertion_labels - labels)
    if missing:
        raise ValueError(f"{label} lacks required provenance assertions: {missing}.")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "upstream_manifests": upstream_records,
        "validated_assertions": snapshots,
    }


def provenance_assertions(record: Mapping[str, Any]) -> dict[str, Any]:
    """Index a validated provenance record by its unique assertion labels.

    Args:
        record: A manifest-binding result, directly or under an artifact's
            ``provenance_manifest`` field.

    Returns:
        Assertion values keyed by their required scientific labels.
    """
    manifest = record.get("provenance_manifest", record)
    snapshots = manifest.get("validated_assertions")
    if not isinstance(snapshots, list):
        raise ValueError("Validated provenance record lacks assertion snapshots.")
    result = {str(item["label"]): item["value"] for item in snapshots}
    if len(result) != len(snapshots):
        raise ValueError("Validated provenance record repeats an assertion label.")
    return result


def array_is_finite(values: np.ndarray, *, axis_zero_chunk: int = 8) -> bool:
    """Check a large array for finite values with bounded temporary memory.

    Args:
        values: At least one-dimensional numeric array or memory map.
        axis_zero_chunk: Maximum leading-axis planes checked together.

    Returns:
        ``True`` only when every real and imaginary component is finite.
    """
    array = np.asarray(values)
    if array.ndim < 1 or axis_zero_chunk < 1:
        raise ValueError("Finite-value validation requires an array and positive chunk size.")
    return all(
        bool(np.isfinite(array[start : start + axis_zero_chunk]).all())
        for start in range(0, array.shape[0], axis_zero_chunk)
    )


def validate_bart_artifact(
    specification: Mapping[str, Any],
    config_dir: Path,
    *,
    expected_shape: tuple[int, ...],
    required_assertion_labels: set[str],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Validate one accepted BART pair by shape, hashes, finiteness, and provenance.

    Args:
        specification: BART base, stored hashes, and manifest binding.
        config_dir: Base directory for relative paths.
        expected_shape: Required leading dimensions; trailing dimensions must be one.
        required_assertion_labels: Required manifest assertion names.
        label: Human-readable artifact name.

    Returns:
        Normalized BART base and complete immutable validation record.
    """
    base = bart_base(resolve_config_path(specification.get("base"), config_dir, f"{label}.base"))
    shape = read_shape(base)
    padded = shape + (1,) * max(0, len(expected_shape) - len(shape))
    if padded[: len(expected_shape)] != expected_shape or any(
        value != 1 for value in padded[len(expected_shape) :]
    ):
        raise ValueError(f"{label} shape {shape} differs from required {expected_shape}.")
    expected_header = _require_sha256(
        specification.get("header_sha256"), f"{label}.header_sha256"
    )
    expected_payload = _require_sha256(
        specification.get("payload_sha256"), f"{label}.payload_sha256"
    )
    header_hash = sha256_file(base.with_suffix(".hdr"))
    payload_hash = sha256_file(base.with_suffix(".cfl"))
    if header_hash != expected_header or payload_hash != expected_payload:
        raise ValueError(f"{label} BART pair differs from its accepted hashes.")
    values = open_cfl(base)
    if not array_is_finite(values):
        raise ValueError(f"{label} contains non-finite complex values.")
    provenance = validate_manifest_binding(
        specification.get("manifest", {}),
        config_dir,
        required_assertion_labels=required_assertion_labels,
        label=label,
    )
    return base, {
        **cfl_record(base),
        "header_sha256": header_hash,
        "payload_sha256": payload_hash,
        "provenance_manifest": provenance,
    }


def validate_npy_artifact(
    specification: Mapping[str, Any],
    config_dir: Path,
    *,
    expected_shape: tuple[int, ...],
    required_assertion_labels: set[str],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Validate one immutable complex64 NPY source and its provenance manifest.

    Args:
        specification: NPY path, file digest, and manifest binding.
        config_dir: Base directory for relative paths.
        expected_shape: Exact logical array shape.
        required_assertion_labels: Required manifest assertion names.
        label: Human-readable artifact name.

    Returns:
        Resolved NPY path and immutable validation record.
    """
    path = resolve_config_path(specification.get("path"), config_dir, f"{label}.path")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} NPY hash changed: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape or values.dtype != np.complex64:
        raise ValueError(
            f"{label} must be complex64 {expected_shape}; got {values.shape} {values.dtype}."
        )
    if not array_is_finite(values):
        raise ValueError(f"{label} contains non-finite complex values.")
    provenance = validate_manifest_binding(
        specification.get("manifest", {}),
        config_dir,
        required_assertion_labels=required_assertion_labels,
        label=label,
    )
    return path, {
        "path": str(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": actual_hash,
        "logical_sha256": logical_array_sha256(values),
        "provenance_manifest": provenance,
    }


def validate_csm_rss_normalization(
    csm: np.ndarray,
    *,
    support_threshold: float,
    tolerance: float,
    axis_zero_chunk: int = 8,
) -> dict[str, Any]:
    """Validate finite coil-RSS normalization on supported anatomy.

    Args:
        csm: Complex sensitivity maps in RO/LIN/PAR/COIL order.
        support_threshold: Minimum RSS magnitude defining supported voxels.
        tolerance: Maximum absolute supported deviation from unit RSS.
        axis_zero_chunk: Maximum readout planes checked together.

    Returns:
        Support count and maximum unit-RSS error.
    """
    values = np.asarray(csm)
    if values.ndim != 4 or values.shape[3] < 1 or not array_is_finite(
        values, axis_zero_chunk=axis_zero_chunk
    ):
        raise ValueError("CSM values must be a finite RO/LIN/PAR/COIL array.")
    if support_threshold <= 0 or tolerance <= 0 or axis_zero_chunk < 1:
        raise ValueError("CSM support threshold and RSS tolerance must be positive.")
    support_count = 0
    maximum_error = 0.0
    for start in range(0, values.shape[0], axis_zero_chunk):
        block = values[start : start + axis_zero_chunk]
        rss = np.sqrt(np.sum(np.abs(block) ** 2, axis=3))
        support = rss > support_threshold
        support_count += int(support.sum())
        if np.any(support):
            maximum_error = max(
                maximum_error, float(np.max(np.abs(rss[support] - 1.0)))
            )
    if support_count == 0:
        raise ValueError("CSM has no supported voxel above the declared threshold.")
    if maximum_error > tolerance:
        raise ValueError("Accepted CSM fails coil-RSS normalization.")
    return {
        "support_threshold": support_threshold,
        "tolerance": tolerance,
        "support_voxel_count": support_count,
        "maximum_absolute_error_from_one": maximum_error,
    }


def validate_psf_unit_magnitude(
    psf: np.ndarray, *, tolerance: float, axis_zero_chunk: int = 8
) -> dict[str, Any]:
    """Validate finite unit-magnitude behavior of an accepted Wave PSF.

    Args:
        psf: Complex accepted Wave PSF on one case grid.
        tolerance: Maximum absolute magnitude deviation from one.
        axis_zero_chunk: Maximum readout planes checked together.

    Returns:
        Tolerance and measured maximum magnitude error.
    """
    values = np.asarray(psf)
    if values.ndim != 3 or not np.iscomplexobj(values) or not array_is_finite(
        values, axis_zero_chunk=axis_zero_chunk
    ):
        raise ValueError("Calibrated PSF must be one finite complex RO/LIN/PAR array.")
    if tolerance <= 0 or axis_zero_chunk < 1:
        raise ValueError("PSF unit-magnitude tolerance must be positive.")
    maximum_error = max(
        float(np.max(np.abs(np.abs(values[start : start + axis_zero_chunk]) - 1.0)))
        for start in range(0, values.shape[0], axis_zero_chunk)
    )
    if maximum_error > tolerance:
        raise ValueError("Accepted Wave PSF is not unit magnitude.")
    return {
        "tolerance": tolerance,
        "maximum_absolute_error_from_one": maximum_error,
    }


def _case_specifications(geometry: Geometry) -> tuple[tuple[str, CaseSpec], ...]:
    """Build the fixed five-case geometry and acceleration requests.

    Args:
        geometry: Accepted native physical FOV and logical matrix.

    Returns:
        Ordered case identifiers and resolution/acceleration specifications.
    """
    native_resolution = geometry.physical_resolution_mm_xyz
    return (
        ("native_r3x1", CaseSpec(native_resolution, (3, 1), "native R3x1")),
        ("native_r3x2", CaseSpec(native_resolution, (3, 2), "native R3x2")),
        ("lr_x_r3x2", CaseSpec((1.5, 1.0, native_resolution[2]), (3, 2), "R3x2 LR-X")),
        ("lr_y_r3x2", CaseSpec((1.0, 1.5, native_resolution[2]), (3, 2), "R3x2 LR-Y")),
        (
            "lr_xy_r3x2",
            CaseSpec((1.25, 1.25, native_resolution[2]), (3, 2), "R3x2 LR-XY"),
        ),
    )


def _remapped_residue(
    native_residue_lin_par: tuple[int, int],
    case: ResolvedCase,
) -> tuple[int, int]:
    """Map native lattice residues through exact centered PE crop bounds.

    Args:
        native_residue_lin_par: Accepted residues on the native logical grid.
        case: Resolved target crop and acceleration.

    Returns:
        Target-grid LIN/PAR residues preserving the native coordinates.
    """
    lin_start = case.crop_bounds_lin[0]
    par_start = case.crop_bounds_par[0]
    acceleration_lin, acceleration_par = case.acceleration_ry_rz
    return (
        (native_residue_lin_par[0] - lin_start) % acceleration_lin,
        (native_residue_lin_par[1] - par_start) % acceleration_par,
    )


def validate_config(config_path: str | Path) -> dict[str, Any]:
    """Validate all five immutable input contracts without writing outputs.

    Args:
        config_path: Ignored machine-local JSON configuration.

    Returns:
        Resolved output layout, source records, and five validated cases.
    """
    path = Path(config_path).expanduser().resolve()
    config = load_json(path, "pure-mask rerun configuration")
    if config.get("format_version") != 1 or config.get("workflow") != WORKFLOW_NAME:
        raise ValueError("Unsupported pure-mask rerun configuration schema.")
    config_dir = path.parent
    output_root = resolve_config_path(config.get("output_root"), config_dir, "output_root")
    geometry_config = config.get("geometry")
    if not isinstance(geometry_config, Mapping):
        raise ValueError("geometry must be a JSON object.")
    fov = tuple(float(value) for value in geometry_config["physical_fov_mm_xyz"])
    native_matrix = tuple(
        int(value) for value in geometry_config["native_logical_matrix_ro_lin_par"]
    )
    if (
        len(fov) != 3
        or len(native_matrix) != 3
        or any(not math.isfinite(value) or value <= 0 for value in fov)
        or any(value <= 0 for value in native_matrix)
    ):
        raise ValueError("Native FOV and logical matrix must contain three positive values.")
    geometry = Geometry(fov, native_matrix)
    extended_readout = int(geometry_config["extended_wave_readout"])
    coils = int(geometry_config["virtual_coils"])
    if extended_readout < native_matrix[0] or coils < 1:
        raise ValueError("Extended Wave readout and virtual-coil count are invalid.")

    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a JSON object.")
    if sampling.get("mask_kind") != PURE_CARTESIAN_IMAGE_LATTICE:
        raise ValueError(
            "Historical ACS-union masks are forbidden; sampling.mask_kind must be "
            f"{PURE_CARTESIAN_IMAGE_LATTICE!r}."
        )
    native_residue = tuple(int(value) for value in sampling["native_residue_lin_par"])
    if len(native_residue) != 2 or not 0 <= native_residue[0] < 3 or not 0 <= native_residue[1] < 2:
        raise ValueError("Native residues must contain a valid R3 LIN and R2 PAR residue.")

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source must be a JSON object.")
    no_wave_path, no_wave_record = validate_npy_artifact(
        source.get("no_wave_kspace", {}),
        config_dir,
        expected_shape=(*native_matrix, coils),
        required_assertion_labels={"dataset", "fov", "dimensions", "coil_order"},
        label="fully sampled no-Wave source",
    )
    full_wave_path, full_wave_record = validate_npy_artifact(
        source.get("native_full_wave_kspace", {}),
        config_dir,
        expected_shape=(extended_readout, native_matrix[1], native_matrix[2], coils),
        required_assertion_labels={
            "dataset",
            "fov",
            "dimensions",
            "coil_order",
            "trajectory",
            "psf_model",
            "wave_data_origin",
            "calibration_samples_merged",
        },
        label="accepted native full synthetic Wave source",
    )
    no_wave_claims = provenance_assertions(no_wave_record)
    full_wave_claims = provenance_assertions(full_wave_record)
    if no_wave_claims["fov"] != list(fov) or full_wave_claims["fov"] != list(fov):
        raise ValueError("Accepted no-Wave/Wave source provenance has the wrong FOV.")
    if no_wave_claims["dimensions"] != [*native_matrix, coils]:
        raise ValueError("Accepted no-Wave source provenance has the wrong dimensions.")
    if full_wave_claims["dimensions"] != [extended_readout, *native_matrix[1:], coils]:
        raise ValueError("Accepted full-Wave source provenance has the wrong dimensions.")
    if no_wave_claims["dataset"] != full_wave_claims["dataset"]:
        raise ValueError("Accepted no-Wave and full-Wave sources do not share one dataset.")
    if no_wave_claims["coil_order"] != full_wave_claims["coil_order"]:
        raise ValueError("Accepted no-Wave and full-Wave sources do not share coil order.")
    if full_wave_claims["psf_model"] != THEORETICAL_PSF_MODEL:
        raise ValueError("Accepted full-Wave source must use the approved theoretical PSF.")
    if full_wave_claims["wave_data_origin"] != SYNTHETIC_WAVE_ORIGIN:
        raise ValueError("Accepted full-Wave source must be synthesized from no-Wave data.")
    if full_wave_claims["calibration_samples_merged"] is not False:
        raise ValueError("Calibration samples must remain separate from full synthetic Wave data.")
    bet_spec = source.get("approved_bet_mask")
    if not isinstance(bet_spec, Mapping):
        raise ValueError("source.approved_bet_mask must be an artifact object.")
    bet_path = resolve_config_path(bet_spec.get("path"), config_dir, "approved BET mask")
    if not bet_path.is_file():
        raise FileNotFoundError(bet_path)
    bet_hash = _require_sha256(bet_spec.get("sha256"), "approved_bet_mask.sha256")
    if sha256_file(bet_path) != bet_hash:
        raise ValueError("The approved BET brain-mask hash changed.")
    bet_manifest = validate_manifest_binding(
        bet_spec.get("manifest", {}),
        config_dir,
        required_assertion_labels={"approval", "geometry"},
        label="approved BET mask",
    )
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a JSON object.")
    orientation_manifest = validate_manifest_binding(
        evaluation.get("orientation_manifest", {}),
        config_dir,
        required_assertion_labels={"orientation", "canonical_ras"},
        label="accepted logical-to-canonical orientation",
    )

    case_configs = config.get("cases")
    if not isinstance(case_configs, Mapping) or set(case_configs) != set(CASE_IDS):
        raise ValueError(f"cases must contain exactly {list(CASE_IDS)}.")
    cases: list[PureMaskCase] = []
    for case_id, specification in _case_specifications(geometry):
        case_config = case_configs[case_id]
        if not isinstance(case_config, Mapping):
            raise ValueError(f"cases.{case_id} must be a JSON object.")
        resolved = resolve_case(specification, geometry)
        target_ro, target_lin, target_par = resolved.target_logical_matrix_ro_lin_par
        residue = _remapped_residue(native_residue, resolved)
        mask, mask_metadata = pure_cartesian_image_lattice_mask(
            (target_lin, target_par),
            acceleration_lin_par=resolved.acceleration_ry_rz,
            residue_lin_par=residue,
        )
        expected_mask_hash = _require_sha256(
            case_config.get("expected_mask_logical_sha256"),
            f"cases.{case_id}.expected_mask_logical_sha256",
        )
        if mask_metadata["logical_sha256"] != expected_mask_hash:
            raise ValueError(f"{case_id} pure-mask hash differs from the approved exact hash.")
        validate_pure_cartesian_image_lattice(mask, mask_metadata)
        csm_base, csm_record = validate_bart_artifact(
            case_config.get("csm", {}),
            config_dir,
            expected_shape=(target_ro, target_lin, target_par, coils, 1),
            required_assertion_labels={
                "dataset", "fov", "dimensions", "coil_order", "calibration_source"
            },
            label=f"{case_id} accepted CSM",
        )
        csm = np.asarray(open_cfl(csm_base)).reshape(
            (target_ro, target_lin, target_par, coils, -1), order="F"
        )[..., 0]
        csm_claims = provenance_assertions(csm_record)
        if csm_claims["dataset"] != no_wave_claims["dataset"]:
            raise ValueError(f"{case_id} accepted CSM belongs to a different dataset.")
        if csm_claims["fov"] != list(fov):
            raise ValueError(f"{case_id} accepted CSM provenance has the wrong FOV.")
        if csm_claims["dimensions"] != [target_ro, target_lin, target_par, coils, 1]:
            raise ValueError(f"{case_id} accepted CSM provenance has the wrong dimensions.")
        if csm_claims["coil_order"] != no_wave_claims["coil_order"]:
            raise ValueError(f"{case_id} accepted CSM has the wrong coil order.")
        if csm_claims["calibration_source"] != "fully_sampled_image_kspace":
            raise ValueError(f"{case_id} accepted CSM has the wrong calibration source.")
        threshold = float(config.get("validation", {}).get("csm_support_threshold", 1e-6))
        tolerance = float(config.get("validation", {}).get("csm_rss_tolerance", 5e-3))
        try:
            csm_record["rss_validation"] = validate_csm_rss_normalization(
                csm, support_threshold=threshold, tolerance=tolerance
            )
        except ValueError as exc:
            raise ValueError(f"{case_id} accepted CSM validation failed: {exc}") from exc
        psf_base, psf_record = validate_bart_artifact(
            case_config.get("psf", {}),
            config_dir,
            expected_shape=(extended_readout, target_lin, target_par, 1, 1),
            required_assertion_labels={
                "dataset",
                "fov",
                "dimensions",
                "trajectory",
                "psf_model",
                "wave_data_origin",
            },
            label=f"{case_id} theoretical PSF",
        )
        psf = np.asarray(open_cfl(psf_base)).reshape(
            (extended_readout, target_lin, target_par, -1), order="F"
        )[..., 0]
        psf_claims = provenance_assertions(psf_record)
        if psf_claims["dataset"] != no_wave_claims["dataset"]:
            raise ValueError(f"{case_id} theoretical PSF belongs to a different dataset.")
        if psf_claims["fov"] != list(fov):
            raise ValueError(f"{case_id} theoretical PSF provenance has the wrong FOV.")
        if psf_claims["dimensions"] != [extended_readout, target_lin, target_par, 1, 1]:
            raise ValueError(f"{case_id} theoretical PSF provenance has the wrong dimensions.")
        if psf_claims["trajectory"] != full_wave_claims["trajectory"]:
            raise ValueError(f"{case_id} theoretical PSF has the wrong trajectory provenance.")
        if psf_claims["psf_model"] != THEORETICAL_PSF_MODEL:
            raise ValueError(f"{case_id} PSF is not the approved theoretical model.")
        if psf_claims["wave_data_origin"] != SYNTHETIC_WAVE_ORIGIN:
            raise ValueError(f"{case_id} PSF provenance is not synthetic no-Wave to Wave.")
        psf_tolerance = float(
            config.get("validation", {}).get("psf_unit_magnitude_tolerance", 2e-5)
        )
        try:
            psf_record["unit_magnitude_validation"] = validate_psf_unit_magnitude(
                psf, tolerance=psf_tolerance
            )
        except ValueError as exc:
            raise ValueError(f"{case_id} theoretical PSF validation failed: {exc}") from exc
        cases.append(
            PureMaskCase(
                case_id=case_id,
                resolved=resolved,
                acceleration_lin_par=resolved.acceleration_ry_rz,
                residue_lin_par=residue,
                mask=mask,
                mask_metadata=mask_metadata,
                csm={**csm_record, "base": str(csm_base)},
                psf={**psf_record, "base": str(psf_base)},
            )
        )
    immutable_contract = {
        "format_version": config["format_version"],
        "workflow": config["workflow"],
        "output_root": config["output_root"],
        "geometry": config["geometry"],
        "sampling": config["sampling"],
        "source": config["source"],
        "cases": config["cases"],
        "validation": config.get("validation", {}),
        "runtime_preparation": {
            "fft_workers": config.get("runtime", {}).get("fft_workers", 4)
        },
        "evaluation_geometry": {
            "logical_to_canonical_axis_order": evaluation.get(
                "logical_to_canonical_axis_order"
            ),
            "logical_to_canonical_axis_flips": evaluation.get(
                "logical_to_canonical_axis_flips"
            ),
            "orientation_manifest": evaluation.get("orientation_manifest"),
        },
    }
    return {
        "status": "validated",
        "config": {
            "path": str(path),
            "sha256": sha256_file(path),
            "immutable_contract_sha256": json_object_sha256(immutable_contract),
            "snapshot": config,
        },
        "layout": output_layout(output_root),
        "geometry": asdict(geometry),
        "extended_wave_readout": extended_readout,
        "virtual_coils": coils,
        "native_residue_lin_par": list(native_residue),
        "source": {
            "no_wave_kspace": {**no_wave_record, "path": str(no_wave_path)},
            "native_full_wave_kspace": {**full_wave_record, "path": str(full_wave_path)},
            "approved_bet_mask": {
                "path": str(bet_path),
                "sha256": bet_hash,
                "provenance_manifest": bet_manifest,
                "bet_rerun_performed": False,
            },
            "accepted_orientation": orientation_manifest,
        },
        "cases": cases,
    }


def link_bart_pair(source_base: str | Path, destination_base: str | Path) -> None:
    """Symlink one accepted BART pair without changing its payload.

    Args:
        source_base: Existing accepted BART basename.
        destination_base: New case-local basename.

    Returns:
        None. Two relative symbolic links are created.
    """
    source = bart_base(source_base)
    destination = bart_base(destination_base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".hdr", ".cfl"):
        source_path = source.with_suffix(suffix)
        destination_path = destination.with_suffix(suffix)
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError(destination_path)
        destination_path.symlink_to(os.path.relpath(source_path, destination_path.parent))


def write_masked_wave_cfl(
    full_wave: np.ndarray,
    mask: np.ndarray,
    output_base: str | Path,
    *,
    readout_chunk: int = 8,
) -> dict[str, Any]:
    """Write and fully validate masked Wave k-space in BART layout.

    Args:
        full_wave: Complex64 ``(RO_os, LIN, PAR, COIL)`` full Wave source.
        mask: Boolean pure image-lattice mask in LIN/PAR order.
        output_base: Destination BART basename.
        readout_chunk: Maximum oversampled-readout planes copied together.

    Returns:
        BART provenance plus exact acquired/zero validation counts.
    """
    source = np.asarray(full_wave)
    logical_mask = np.asarray(mask)
    if source.ndim != 4 or source.dtype != np.complex64 or readout_chunk < 1:
        raise ValueError("Full Wave source must be a complex64 four-dimensional array.")
    if logical_mask.dtype != np.bool_ or logical_mask.shape != source.shape[1:3]:
        raise ValueError("Pure mask shape or dtype differs from full Wave k-space.")
    if not array_is_finite(source, axis_zero_chunk=readout_chunk):
        raise ValueError("Full Wave source contains non-finite values.")
    output = create_cfl(output_base, (*source.shape, 1))
    acquired_mismatch = 0
    unacquired_nonzero = 0
    nonfinite = 0
    squared_norm = 0.0
    for start in range(0, source.shape[0], readout_chunk):
        stop = min(start + readout_chunk, source.shape[0])
        source_block = np.asarray(source[start:stop])
        output_block = output[start:stop, ..., 0]
        output_block[...] = 0
        output_block[:, logical_mask, :] = source_block[:, logical_mask, :]
        acquired_mismatch += int(
            np.count_nonzero(
                output_block[:, logical_mask, :] != source_block[:, logical_mask, :]
            )
        )
        unacquired_nonzero += int(np.count_nonzero(output_block[:, ~logical_mask, :]))
        nonfinite += int(np.count_nonzero(~np.isfinite(output_block)))
        squared_norm += float(np.vdot(output_block, output_block).real)
    output.flush()
    norm = math.sqrt(squared_norm)
    del output
    if any((acquired_mismatch, unacquired_nonzero, nonfinite)) or norm <= 0:
        raise ValueError("Masked Wave CFL failed acquired-sample, zero, finite, or norm gates.")
    return {
        **cfl_record(output_base),
        "wave_kspace_norm": norm,
        "acquired_mismatch_count": acquired_mismatch,
        "unacquired_nonzero_count": unacquired_nonzero,
        "nonfinite_count": nonfinite,
        "acquired_samples_equal_full_wave_bitwise": True,
        "unacquired_samples_are_exact_zero": True,
    }


def direct_fft_reference(no_wave_kspace: np.ndarray, *, fft_workers: int) -> np.ndarray:
    """Build a resolution-matched direct-FFT RSS reference without interpolation.

    Args:
        no_wave_kspace: Complex64 target-grid no-Wave k-space in RO/LIN/PAR/COIL order.
        fft_workers: Maximum SciPy FFT worker count.

    Returns:
        Finite float32 root-sum-of-squares magnitude on the target logical grid.
    """
    values = np.asarray(no_wave_kspace)
    if values.ndim != 4 or values.dtype != np.complex64 or not np.isfinite(values).all():
        raise ValueError("Direct-FFT source must be finite complex64 RO/LIN/PAR/COIL data.")
    rss_squared = np.zeros(values.shape[:3], dtype=np.float64)
    for coil in range(values.shape[3]):
        image = centered_fftn(values[..., coil], axes=(0, 1, 2), inverse=True, workers=fft_workers)
        rss_squared += np.abs(image).astype(np.float64) ** 2
    reference = np.sqrt(rss_squared).astype(np.float32)
    if not np.isfinite(reference).all() or float(np.linalg.norm(reference)) <= 0:
        raise ValueError("Resolution-matched direct-FFT reference is invalid.")
    return reference


def build_wave_command(
    bart: str | Path,
    *,
    method: str,
    lambda_value: float,
    block_size: int | None,
    csm_base: str | Path,
    psf_base: str | Path,
    wave_kspace_base: str | Path,
    output_base: str | Path,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> list[str]:
    """Build one exact GPU BART Wave command for a rerun candidate.

    Args:
        bart: BART executable.
        method: ``fista_lambda0``, ``wavelet``, or ``llr``.
        lambda_value: Nonnegative regularization weight.
        block_size: Required LLR block size, otherwise ``None``.
        csm_base: Accepted case-matched CSM basename.
        psf_base: Accepted case-matched theoretical PSF basename.
        wave_kspace_base: Prepared pure-mask Wave k-space basename.
        output_base: Destination BART image basename.
        iterations: Fixed FISTA iteration count.
        tolerance: Fixed positive stopping tolerance.

    Returns:
        Executable argument vector with mandatory GPU option ``-g``.
    """
    if iterations != 100 or not math.isclose(tolerance, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("The approved rerun contract requires 100 iterations and tolerance 1e-6.")
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("Lambda must be finite and nonnegative.")
    if method == "fista_lambda0":
        if lambda_value != 0 or block_size is not None:
            raise ValueError("FISTA control must have lambda zero and no block size.")
        options = ["-g", "-w", "-f", "-r", "0"]
    elif method == "wavelet":
        if lambda_value <= 0 or block_size is not None:
            raise ValueError("Wavelet candidates require positive lambda and no block size.")
        options = ["-g", "-w", "-f", "-r", f"{lambda_value:.12g}"]
    elif method == "llr":
        if lambda_value <= 0 or block_size not in LLR_BLOCK_SIZES:
            raise ValueError("LLR requires positive lambda and block size 4, 8, or 16.")
        options = [
            "-g",
            "-l",
            "-v",
            "-b",
            str(block_size),
            "-f",
            "-r",
            f"{lambda_value:.12g}",
        ]
    else:
        raise ValueError(f"Unsupported rerun method: {method!r}.")
    options.extend(["-i", "100", "-t", "1e-6"])
    return [
        str(bart),
        "wave",
        *options,
        str(bart_base(csm_base)),
        str(bart_base(psf_base)),
        str(bart_base(wave_kspace_base)),
        str(bart_base(output_base)),
    ]


def coarse_candidate_settings() -> list[dict[str, Any]]:
    """Return the fixed coarse-stage candidate settings for one case.

    Returns:
        One FISTA control, seven Wavelet candidates, and fifteen corrected-LLR
        candidates ordered deterministically.
    """
    settings: list[dict[str, Any]] = [
        {"method": "fista_lambda0", "lambda": 0.0, "block_size": None}
    ]
    settings.extend(
        {"method": "wavelet", "lambda": value, "block_size": None}
        for value in COARSE_WAVELET_LAMBDAS
    )
    settings.extend(
        {"method": "llr", "lambda": value, "block_size": block_size}
        for block_size in LLR_BLOCK_SIZES
        for value in COARSE_LLR_LAMBDAS
    )
    return settings
