"""Load and validate the shared dataset contract for synthetic-Wave studies.

The contract deliberately separates the acquired no-Wave sampling from the
retrospective synthetic-Wave target.  This matters for an R1 reference scan
that is later masked to R3x2 after Wave encoding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REFERENCE_KINDS = {"grappa", "dicom", "nifti"}


class DatasetManifestError(ValueError):
    """Report one or more structural errors in a dataset manifest."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _positive_integer(value: Any, name: str, errors: list[str]) -> None:
    if not _is_integer(value) or value < 1:
        errors.append(f"{name} must be a positive integer")


def _positive_number(value: Any, name: str, errors: list[str]) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{name} must be a positive number")


def _vector(
    value: Any,
    name: str,
    length: int,
    errors: list[str],
    *,
    integer: bool,
) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        errors.append(f"{name} must contain exactly {length} values")
        return
    for index, item in enumerate(value):
        if integer:
            _positive_integer(item, f"{name}[{index}]", errors)
        else:
            _positive_number(item, f"{name}[{index}]", errors)


def _nonempty_string(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")


def validate_dataset_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the portable acquisition, reconstruction, and evaluation contract."""
    errors: list[str] = []
    if payload.get("format_version") != 1:
        errors.append("format_version must be 1")

    for field in ("dataset_id", "subject"):
        value = payload.get(field)
        _nonempty_string(value, field, errors)
        if isinstance(value, str) and value and not _IDENTIFIER_PATTERN.fullmatch(value):
            errors.append(f"{field} may contain only letters, digits, '.', '_', and '-'")

    inputs = _mapping(payload.get("inputs"), "inputs", errors)
    for field in ("twix", "wave_sequence"):
        _nonempty_string(inputs.get(field), f"inputs.{field}", errors)
    dicom = _mapping(inputs.get("dicom"), "inputs.dicom", errors)
    _nonempty_string(dicom.get("directory"), "inputs.dicom.directory", errors)
    for field in ("required_image_type_tokens", "excluded_image_type_tokens"):
        tokens = dicom.get(field)
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) and token for token in tokens
        ):
            errors.append(f"inputs.dicom.{field} must be a list of non-empty strings")

    outputs = _mapping(payload.get("outputs"), "outputs", errors)
    relative_output_fields = (
        "inspection_report",
        "coil_compression_prefix",
        "source_reconstruction_prefix",
    )
    for field in ("root", *relative_output_fields):
        _nonempty_string(outputs.get(field), f"outputs.{field}", errors)
    for field in relative_output_fields:
        value = outputs.get(field)
        if isinstance(value, str) and value:
            output_path = Path(value)
            if not output_path.is_absolute() and ".." not in output_path.parts:
                continue
            errors.append(
                f"outputs.{field} must be a relative path contained by outputs.root"
            )

    geometry = _mapping(payload.get("geometry"), "geometry", errors)
    axes = geometry.get("logical_axes")
    if axes != ["readout", "phase_encode_1", "phase_encode_2"]:
        errors.append(
            "geometry.logical_axes must be "
            "['readout', 'phase_encode_1', 'phase_encode_2']"
        )
    _vector(geometry.get("matrix"), "geometry.matrix", 3, errors, integer=True)
    _vector(geometry.get("fov_mm"), "geometry.fov_mm", 3, errors, integer=False)

    sampling = _mapping(payload.get("sampling"), "sampling", errors)
    _vector(
        sampling.get("source_acceleration_pe1_pe2"),
        "sampling.source_acceleration_pe1_pe2",
        2,
        errors,
        integer=True,
    )
    _vector(
        sampling.get("synthetic_wave_acceleration_pe1_pe2"),
        "sampling.synthetic_wave_acceleration_pe1_pe2",
        2,
        errors,
        integer=True,
    )
    _positive_number(
        sampling.get("readout_oversampling_factor"),
        "sampling.readout_oversampling_factor",
        errors,
    )
    if not isinstance(sampling.get("require_complete_source_grid"), bool):
        errors.append("sampling.require_complete_source_grid must be boolean")
    if "expected_acs_pe1_pe2" in sampling:
        _vector(
            sampling.get("expected_acs_pe1_pe2"),
            "sampling.expected_acs_pe1_pe2",
            2,
            errors,
            integer=True,
        )

    reconstruction = _mapping(payload.get("reconstruction"), "reconstruction", errors)
    _positive_integer(
        reconstruction.get("physical_coils"), "reconstruction.physical_coils", errors
    )
    _positive_integer(
        reconstruction.get("virtual_coils"), "reconstruction.virtual_coils", errors
    )
    if reconstruction.get("coil_compression_source") not in {"image", "refscan"}:
        errors.append(
            "reconstruction.coil_compression_source must be 'image' or 'refscan'"
        )
    physical_coils = reconstruction.get("physical_coils")
    virtual_coils = reconstruction.get("virtual_coils")
    if (
        _is_integer(physical_coils)
        and _is_integer(virtual_coils)
        and virtual_coils > physical_coils
    ):
        errors.append("reconstruction.virtual_coils cannot exceed physical_coils")

    grappa = _mapping(reconstruction.get("grappa"), "reconstruction.grappa", errors)
    _vector(grappa.get("kernel"), "reconstruction.grappa.kernel", 3, errors, integer=True)
    if isinstance(grappa.get("kernel"), list) and any(
        _is_integer(value) and value % 2 == 0 for value in grappa["kernel"]
    ):
        errors.append("reconstruction.grappa.kernel values must be odd")
    _positive_number(
        grappa.get("regularization"), "reconstruction.grappa.regularization", errors
    )

    bart = _mapping(reconstruction.get("bart"), "reconstruction.bart", errors)
    if bart.get("use_gpu") is not True:
        errors.append("reconstruction.bart.use_gpu must be true")
    maximum_eigenvalue = bart.get("maximum_eigenvalue")
    if maximum_eigenvalue is not None:
        _positive_number(
            maximum_eigenvalue, "reconstruction.bart.maximum_eigenvalue", errors
        )

    evaluation = _mapping(payload.get("evaluation"), "evaluation", errors)
    reference = _mapping(
        evaluation.get("ranking_reference"), "evaluation.ranking_reference", errors
    )
    reference_kind = reference.get("kind")
    if reference_kind not in _REFERENCE_KINDS:
        errors.append(
            "evaluation.ranking_reference.kind must be one of "
            + ", ".join(sorted(_REFERENCE_KINDS))
        )
    if reference_kind in {"grappa", "nifti"}:
        _nonempty_string(reference.get("path"), "evaluation.ranking_reference.path", errors)
    dicom_ranking = evaluation.get("dicom_intensity_ranking_enabled")
    if not isinstance(dicom_ranking, bool):
        errors.append("evaluation.dicom_intensity_ranking_enabled must be boolean")
    elif dicom_ranking != (reference_kind == "dicom"):
        errors.append(
            "evaluation.dicom_intensity_ranking_enabled must be true exactly when "
            "the ranking reference kind is 'dicom'"
        )
    brain_mask = _mapping(evaluation.get("brain_mask"), "evaluation.brain_mask", errors)
    if brain_mask.get("usage") != "metrics_only":
        errors.append("evaluation.brain_mask.usage must be 'metrics_only'")
    mask_path = brain_mask.get("path")
    if mask_path is not None:
        _nonempty_string(mask_path, "evaluation.brain_mask.path", errors)

    if errors:
        raise DatasetManifestError(
            "Invalid dataset manifest:\n- " + "\n- ".join(errors)
        )


def sha256_file(path: Path) -> str:
    """Hash a manifest-sized file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    """Validated dataset contract with consistent path-resolution rules."""

    path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def dataset_id(self) -> str:
        return str(self.payload["dataset_id"])

    @property
    def subject(self) -> str:
        return str(self.payload["subject"])

    def _manifest_relative_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()

    def input_path(self, field: str) -> Path:
        """Resolve one input path relative to the manifest location."""
        return self._manifest_relative_path(str(self.payload["inputs"][field]))

    @property
    def dicom_directory(self) -> Path:
        return self._manifest_relative_path(
            str(self.payload["inputs"]["dicom"]["directory"])
        )

    @property
    def output_root(self) -> Path:
        return self._manifest_relative_path(str(self.payload["outputs"]["root"]))

    @property
    def inspection_report(self) -> Path:
        return self.output_path("inspection_report")

    def output_path(self, field: str) -> Path:
        """Resolve one validated artifact path or prefix below the output root."""
        path = Path(str(self.payload["outputs"][field]))
        return (self.output_root / path).resolve()

    def resolved_contract(self) -> dict[str, Any]:
        """Return a self-contained contract snapshot for generated manifests."""
        resolved = copy.deepcopy(dict(self.payload))
        resolved["inputs"]["twix"] = str(self.input_path("twix"))
        resolved["inputs"]["wave_sequence"] = str(self.input_path("wave_sequence"))
        resolved["inputs"]["dicom"]["directory"] = str(self.dicom_directory)
        resolved["outputs"]["root"] = str(self.output_root)
        for field in (
            "inspection_report",
            "coil_compression_prefix",
            "source_reconstruction_prefix",
        ):
            resolved["outputs"][field] = str(self.output_path(field))
        for container in (
            resolved["evaluation"].get("ranking_reference", {}),
            resolved["evaluation"].get("brain_mask", {}),
        ):
            value = container.get("path")
            if value:
                container["path"] = str(self._manifest_relative_path(str(value)))
        return resolved

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "dataset_id": self.dataset_id,
            "subject": self.subject,
        }


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load one JSON contract and resolve no paths until they are requested."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"Invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DatasetManifestError("Dataset manifest root must be a JSON object")
    validate_dataset_manifest(payload)
    return DatasetManifest(manifest_path, payload, sha256_file(manifest_path))


def load_passed_inspection(manifest: DatasetManifest) -> Mapping[str, Any]:
    """Require an inspection report produced from this exact manifest revision."""
    report_path = manifest.inspection_report
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Passed dataset inspection is required before reconstruction: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"Invalid inspection report JSON: {report_path}") from exc
    if not isinstance(report, Mapping):
        raise DatasetManifestError("Dataset inspection report root must be a JSON object")
    provenance = report.get("dataset_manifest")
    if not isinstance(provenance, Mapping) or provenance.get("sha256") != manifest.sha256:
        raise DatasetManifestError(
            "Dataset inspection was not produced from the current manifest SHA-256"
        )
    checks = report.get("contract_checks")
    if not isinstance(checks, Mapping) or checks.get("all_passed") is not True:
        raise DatasetManifestError("Dataset inspection contract checks have not passed")
    return report
