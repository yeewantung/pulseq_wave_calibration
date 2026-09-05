"""Collect validated GRE magnitude and phase NIfTIs without masking."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from .bart_io import sha256_file
from .gre import resolve_gre_wavelet_lambda


COLLECTION_BUILDER = "wave_retro_lr.gre_nifti_collection"
RECONSTRUCTION_BRANCHES = ("fista_r0", "selected_wavelet")
CASE_LOCATIONS = (
    ("native_r3x1", Path("normal")),
    ("native_r3x2", Path("retro") / "native_r3x2"),
    (
        "lin_low_resolution_r3x2",
        Path("retro") / "lin_low_resolution_r3x2",
    ),
)


def build_gre_nifti_collection(
    output_root: str | Path,
    *,
    require_retro: bool = False,
) -> dict[str, Any]:
    """Copy complete GRE NIfTI/JSON sets into one unmasked collection.

    Args:
        output_root: Reconstruction root containing normal and optional
            retrospective branch-specific NIfTI directories.
        require_retro: Require both retrospective geometries when true.

    Returns:
        JSON-native collection manifest installed with the copied files.

    Raises:
        FileExistsError: If an existing collection is not owned by this builder
            or its manifested files have changed.
        FileNotFoundError: If required reconstruction outputs are absent.
        ValueError: If source manifests, NIfTIs, sidecars, or provenance are
            incomplete or inconsistent.

    Side Effects:
        Creates or safely refreshes ``OUTPUT_ROOT/nifti_collection`` atomically.
        Source reconstruction outputs are read-only and no mask or masked
        derivative is generated.
    """

    source_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"GRE reconstruction root does not exist: {source_root}")
    destination = source_root / "nifti_collection"
    _validate_existing_collection(destination)

    sources = _discover_sources(source_root, require_retro=require_retro)
    staging = Path(
        tempfile.mkdtemp(prefix=".nifti_collection-", dir=source_root)
    )
    try:
        manifest = _materialize_collection(staging, source_root, destination, sources)
        _write_json(staging / "manifest.json", manifest)
        _replace_owned_collection(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _discover_sources(
    source_root: Path, *, require_retro: bool
) -> list[dict[str, Any]]:
    """Discover and validate complete branch outputs for available geometries.

    Args:
        source_root: Existing GRE reconstruction root.
        require_retro: Require both retrospective geometries when true.

    Returns:
        Ordered case/branch records containing validated source artifacts.

    Raises:
        FileNotFoundError: If normal or required retrospective outputs are absent.
        ValueError: If a geometry is only partially available across branches.
    """

    records: list[dict[str, Any]] = []
    for geometry_id, case_location in CASE_LOCATIONS:
        branch_directories = {
            branch: source_root / case_location / "nifti" / branch
            for branch in RECONSTRUCTION_BRANCHES
        }
        availability = {
            branch: directory.is_dir()
            for branch, directory in branch_directories.items()
        }
        required = geometry_id == "native_r3x1" or require_retro
        if not any(availability.values()):
            if required:
                raise FileNotFoundError(
                    f"No GRE NIfTI outputs found for required geometry {geometry_id}."
                )
            continue
        if not all(availability.values()):
            raise ValueError(
                f"GRE geometry {geometry_id} is incomplete across reconstruction branches."
            )
        for branch, directory in branch_directories.items():
            records.append(_validate_branch(directory, geometry_id, branch, case_location))
    reference_echo_count = records[0]["echo_count"]
    reference_echo_times = records[0]["echo_times_s"]
    if any(
        record["echo_count"] != reference_echo_count
        or record["echo_times_s"] != reference_echo_times
        for record in records[1:]
    ):
        raise ValueError("GRE collection sources disagree on echo count or echo times.")
    for geometry_id, _ in CASE_LOCATIONS:
        geometry_records = [
            record for record in records if record["geometry_id"] == geometry_id
        ]
        if geometry_records and any(
            record["reference_shape_xyz"] != geometry_records[0]["reference_shape_xyz"]
            or not np.allclose(
                record["reference_affine"],
                geometry_records[0]["reference_affine"],
                rtol=0.0,
                atol=1e-6,
            )
            for record in geometry_records[1:]
        ):
            raise ValueError(f"GRE branches disagree on {geometry_id} NIfTI geometry.")
    return records


def _validate_branch(
    directory: Path,
    geometry_id: str,
    branch: str,
    case_location: Path,
) -> dict[str, Any]:
    """Validate one branch's conversion manifest and echo/part outputs.

    Args:
        directory: Branch-specific canonical NIfTI directory.
        geometry_id: Expected GRE geometry identifier.
        branch: Expected reconstruction branch.
        case_location: Stable normal or retrospective relative case location.

    Returns:
        Validated source record ready for byte-for-byte materialization.

    Raises:
        FileNotFoundError: If the conversion manifest or an output is absent.
        ValueError: If provenance, command, echo, image-part, or geometry checks fail.
    """

    conversion_path = directory / "conversion_manifest.json"
    if not conversion_path.is_file():
        raise FileNotFoundError(f"Missing GRE conversion manifest: {conversion_path}")
    conversion = _load_json(conversion_path)
    if conversion.get("status") != "multi_echo_gre_nifti_export_complete":
        raise ValueError(f"GRE conversion is not complete: {conversion_path}")
    if conversion.get("case_id") != geometry_id:
        raise ValueError(f"GRE conversion geometry mismatch: {conversion_path}")
    if conversion.get("presentation_mask_applied") is not False:
        raise ValueError(f"GRE collection accepts only unmasked source NIfTIs: {directory}")

    echo_count = int(conversion.get("echo_count", 0))
    echo_times = tuple(float(value) for value in conversion.get("echo_times_s", ()))
    if echo_count <= 0 or len(echo_times) != echo_count:
        raise ValueError(f"GRE conversion echo metadata is incomplete: {conversion_path}")
    selection = conversion.get("wavelet_selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"GRE conversion has no Wavelet selection provenance: {conversion_path}")
    selected_lambda = resolve_gre_wavelet_lambda(
        geometry_id,
        method=str(selection.get("method", "")),
        echo_ids=selection.get("echo_ids"),
        shared_lambda=selection.get("shared_lambda"),
        selection_manifest_basename=str(selection.get("selection_manifest_basename", "")),
        selection_manifest_sha256=str(selection.get("selection_manifest_sha256", "")),
    )
    _validate_wave_commands(conversion, branch, echo_count)

    declared = _declared_outputs(conversion, directory, echo_count)
    discovered = set(directory.rglob("*.nii.gz"))
    if discovered != set(declared):
        raise ValueError(
            f"GRE NIfTI files differ from the conversion manifest in {directory}."
        )

    files: list[dict[str, Any]] = []
    echo_geometry: dict[int, tuple[tuple[int, ...], np.ndarray]] = {}
    for nifti_path in sorted(declared):
        sidecar_path = nifti_path.with_name(nifti_path.name.removesuffix(".nii.gz") + ".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"GRE NIfTI has no JSON sidecar: {nifti_path}")
        metadata = _load_json(sidecar_path)
        declared_echo = declared[nifti_path]
        sidecar_echo = metadata.get("EchoNumber")
        echo_number = declared_echo if sidecar_echo is None else int(sidecar_echo)
        image_part = str(metadata.get("ImagePart", ""))
        if (
            echo_number != declared_echo
            or not 1 <= echo_number <= echo_count
            or image_part not in {"mag", "phase"}
        ):
            raise ValueError(f"Invalid GRE echo/image-part sidecar: {sidecar_path}")
        if metadata.get("CaseID") != geometry_id:
            raise ValueError(f"GRE sidecar geometry mismatch: {sidecar_path}")
        if metadata.get("PresentationMaskApplied") is not False:
            raise ValueError(f"GRE sidecar is not an unmasked source: {sidecar_path}")
        if metadata.get("GRESharedWaveletSelection") != dict(selection):
            raise ValueError(f"GRE sidecar selection provenance mismatch: {sidecar_path}")
        if float(metadata.get("GRESelectedWaveletLambda", np.nan)) != selected_lambda:
            raise ValueError(f"GRE sidecar selected lambda mismatch: {sidecar_path}")
        sidecar_echo_time = metadata.get("EchoTime")
        if sidecar_echo_time is not None and not np.isclose(
            float(sidecar_echo_time), echo_times[echo_number - 1]
        ):
            raise ValueError(f"GRE sidecar echo time mismatch: {sidecar_path}")
        _validate_orientation(metadata, sidecar_path)
        shape, affine = _validate_nifti(nifti_path)
        prior = echo_geometry.get(echo_number)
        if prior is None:
            echo_geometry[echo_number] = (shape, affine)
        elif prior[0] != shape or not np.allclose(prior[1], affine, rtol=0.0, atol=1e-6):
            raise ValueError(f"GRE magnitude/phase geometry mismatch for echo {echo_number}.")
        files.append(
            {
                "echo": echo_number,
                "te_s": echo_times[echo_number - 1],
                "image_part": image_part,
                "source_nifti": nifti_path,
                "source_sidecar": sidecar_path,
                "shape_xyz": list(shape),
            }
        )

    expected_pairs = {(echo, part) for echo in range(1, echo_count + 1) for part in ("mag", "phase")}
    actual_pairs = {(item["echo"], item["image_part"]) for item in files}
    if actual_pairs != expected_pairs or len(files) != len(expected_pairs):
        raise ValueError(f"GRE branch does not contain one magnitude/phase pair per echo: {directory}")
    reference_shape, reference_affine = echo_geometry[1]
    for echo_number, (shape, affine) in echo_geometry.items():
        if shape != reference_shape or not np.allclose(
            affine, reference_affine, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"GRE echo {echo_number} geometry differs within {directory}.")

    return {
        "geometry_id": geometry_id,
        "case_location": case_location,
        "branch": branch,
        "source_directory": directory,
        "conversion_manifest": conversion_path,
        "echo_count": echo_count,
        "echo_times_s": list(echo_times),
        "reference_shape_xyz": reference_shape,
        "reference_affine": reference_affine,
        "files": files,
    }


def _declared_outputs(
    conversion: Mapping[str, Any], directory: Path, echo_count: int
) -> dict[Path, int]:
    """Resolve the exact NIfTI set declared by a conversion manifest.

    Args:
        conversion: Parsed branch conversion manifest.
        directory: Branch-specific source directory.
        echo_count: Expected number of consecutive echoes.

    Returns:
        Resolved NIfTI paths confined to ``directory``, mapped to their
        authoritative one-based echo numbers.

    Raises:
        FileNotFoundError: If a declared file is absent.
        ValueError: If echo records or paths are malformed or escape the branch.
    """

    echo_records = conversion.get("nifti")
    if not isinstance(echo_records, list) or len(echo_records) != echo_count:
        raise ValueError("GRE conversion NIfTI echo records are incomplete.")
    paths: dict[Path, int] = {}
    for expected_echo, record in enumerate(echo_records, start=1):
        if not isinstance(record, Mapping) or int(record.get("echo", 0)) != expected_echo:
            raise ValueError("GRE conversion NIfTI echoes are not consecutive and ordered.")
        outputs = record.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise ValueError("GRE conversion must declare magnitude and phase for every echo.")
        for output in outputs:
            if not isinstance(output, Mapping) or "nifti" not in output or "json" not in output:
                raise ValueError("GRE conversion contains a malformed NIfTI output record.")
            path = Path(str(output["nifti"])).expanduser().resolve()
            if directory not in path.parents:
                raise ValueError(f"GRE conversion output escapes its branch directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Declared GRE NIfTI does not exist: {path}")
            sidecar = Path(str(output["json"])).expanduser().resolve()
            expected_sidecar = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
            if sidecar != expected_sidecar or directory not in sidecar.parents:
                raise ValueError(f"GRE conversion sidecar does not match its NIfTI: {path}")
            if not sidecar.is_file():
                raise FileNotFoundError(f"Declared GRE sidecar does not exist: {sidecar}")
            if path in paths:
                raise ValueError("GRE conversion declares a NIfTI more than once.")
            paths[path] = expected_echo
    return paths


def _validate_wave_commands(
    conversion: Mapping[str, Any], branch: str, echo_count: int
) -> None:
    """Validate branch lambda and echo-specific command provenance.

    Args:
        conversion: Parsed branch conversion manifest.
        branch: ``fista_r0`` or ``selected_wavelet``.
        echo_count: Expected number of Wave commands.

    Returns:
        None.

    Raises:
        ValueError: If commands are missing, use the wrong lambda, or reuse
            echo-specific PSF, k-space, or output paths.
    """

    command_section = conversion.get("bart_commands")
    commands = command_section.get("wave_by_echo") if isinstance(command_section, Mapping) else None
    if not isinstance(commands, list) or len(commands) != echo_count:
        raise ValueError("GRE conversion must record one BART Wave command per echo.")
    expected_lambda = 0.0 if branch == "fista_r0" else 0.015
    echo_inputs: list[tuple[str, str, str]] = []
    for command in commands:
        import shlex

        arguments = shlex.split(str(command))
        if arguments[:2] != ["bart", "wave"] or "-w" not in arguments or "-f" not in arguments:
            raise ValueError("GRE collection requires explicit BART Wave/FISTA commands.")
        try:
            value = float(arguments[arguments.index("-r") + 1])
        except (ValueError, IndexError) as exc:
            raise ValueError("GRE BART command has no valid -r value.") from exc
        if value != expected_lambda:
            raise ValueError(f"GRE {branch} command uses unexpected lambda {value}.")
        echo_inputs.append(tuple(arguments[-3:]))
    if any(len({values[index] for values in echo_inputs}) != echo_count for index in range(3)):
        raise ValueError("GRE echoes must retain distinct PSF, k-space, and output command paths.")


def _validate_orientation(metadata: Mapping[str, Any], sidecar_path: Path) -> None:
    """Require canonical-RAS, no-interpolation GRE sidecar provenance.

    Args:
        metadata: Parsed NIfTI JSON sidecar.
        sidecar_path: Sidecar path used in validation errors.

    Returns:
        None.

    Raises:
        ValueError: If canonical orientation or no-interpolation provenance is absent.
    """

    policy = metadata.get("OrientationPolicy")
    canonical = metadata.get("CanonicalRASReorientation")
    if not isinstance(policy, Mapping) or not isinstance(canonical, Mapping):
        raise ValueError(f"GRE sidecar has incomplete orientation provenance: {sidecar_path}")
    if policy.get("canonical_coordinate_system") != "RAS" or policy.get("interpolation") is not False:
        raise ValueError(f"GRE sidecar does not record no-interpolation RAS output: {sidecar_path}")
    if canonical.get("StoredAxisCodes") != ["R", "A", "S"] or canonical.get("Interpolation") is not False:
        raise ValueError(f"GRE sidecar does not store canonical RAS: {sidecar_path}")


def _validate_nifti(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    """Validate one finite real three-dimensional canonical-RAS NIfTI.

    Args:
        path: Source NIfTI path.

    Returns:
        Image shape and affine matrix.

    Raises:
        ValueError: If dimensionality, samples, affine, or axis codes are invalid.
    """

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError(f"GRE collection requires finite real 3D NIfTI data: {path}")
    if not np.isfinite(image.affine).all() or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError(f"GRE collection requires canonical-RAS NIfTI geometry: {path}")
    return tuple(int(value) for value in data.shape), np.asarray(image.affine)


def _materialize_collection(
    staging: Path,
    source_root: Path,
    destination: Path,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy all validated sources and construct their hash-bound manifest.

    Args:
        staging: Empty temporary directory beside the final destination.
        source_root: Reconstruction root used for relative source provenance.
        destination: User-selected final collection directory.
        sources: Validated case/branch source records.

    Returns:
        JSON-native collection manifest.

    Side Effects:
        Copies NIfTIs, JSON sidecars, and conversion manifests into ``staging``.
    """

    case_records: list[dict[str, Any]] = []
    owned_files: dict[str, str] = {}
    for source in sources:
        relative_leaf = Path(str(source["branch"])) / Path(source["case_location"])
        target_directory = staging / "original_nifti" / relative_leaf
        target_directory.mkdir(parents=True)
        conversion_source = Path(source["conversion_manifest"])
        conversion_target = target_directory / "conversion_manifest.json"
        shutil.copy2(conversion_source, conversion_target)
        conversion_hash = sha256_file(conversion_target)
        if conversion_hash != sha256_file(conversion_source):
            raise RuntimeError(f"GRE conversion-manifest copy hash mismatch: {conversion_source}")
        owned_files[str(conversion_target.relative_to(staging))] = conversion_hash

        file_records: list[dict[str, Any]] = []
        for source_file in source["files"]:
            nifti_source = Path(source_file["source_nifti"])
            sidecar_source = Path(source_file["source_sidecar"])
            nifti_target = target_directory / nifti_source.name
            sidecar_target = target_directory / sidecar_source.name
            shutil.copy2(nifti_source, nifti_target)
            shutil.copy2(sidecar_source, sidecar_target)
            nifti_hash = sha256_file(nifti_target)
            sidecar_hash = sha256_file(sidecar_target)
            if nifti_hash != sha256_file(nifti_source) or sidecar_hash != sha256_file(sidecar_source):
                raise RuntimeError(f"GRE NIfTI/sidecar copy hash mismatch: {nifti_source}")
            nifti_relative = str(nifti_target.relative_to(staging))
            sidecar_relative = str(sidecar_target.relative_to(staging))
            owned_files[nifti_relative] = nifti_hash
            owned_files[sidecar_relative] = sidecar_hash
            file_records.append(
                {
                    "echo": source_file["echo"],
                    "te_s": source_file["te_s"],
                    "image_part": source_file["image_part"],
                    "shape_xyz": source_file["shape_xyz"],
                    "source_nifti": str(nifti_source.relative_to(source_root)),
                    "source_nifti_sha256": nifti_hash,
                    "source_sidecar": str(sidecar_source.relative_to(source_root)),
                    "source_sidecar_sha256": sidecar_hash,
                    "collection_nifti": nifti_relative,
                    "collection_sidecar": sidecar_relative,
                }
            )
        case_records.append(
            {
                "geometry_id": source["geometry_id"],
                "branch": source["branch"],
                "echo_count": source["echo_count"],
                "echo_times_s": source["echo_times_s"],
                "source_conversion_manifest": str(conversion_source.relative_to(source_root)),
                "source_conversion_manifest_sha256": conversion_hash,
                "collection_conversion_manifest": str(conversion_target.relative_to(staging)),
                "files": file_records,
            }
        )

    return {
        "format_version": 1,
        "builder": COLLECTION_BUILDER,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_output_root": str(source_root),
        "collection_directory": str(destination),
        "scientific_scope": {
            "canonical_reconstruction_outputs_modified": False,
            "nifti_and_sidecars_copied_byte_for_byte": True,
            "magnitude_and_wrapped_phase_included": True,
            "all_echoes_retained": True,
            "masking_applied": False,
            "masked_derivatives_generated": False,
            "quantitative_complex_arrays_copied": False,
        },
        "case_branch_count": len(case_records),
        "nifti_count": sum(len(record["files"]) for record in case_records),
        "cases": case_records,
        "owned_files": owned_files,
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON object.

    Args:
        path: Existing JSON path.

    Returns:
        Parsed top-level mapping.

    Raises:
        ValueError: If the top-level JSON value is not an object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic JSON object.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping.

    Returns:
        None.

    Side Effects:
        Creates or replaces ``path`` inside the new staging directory.
    """

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_existing_collection(collection: Path) -> None:
    """Allow replacement only for an intact collection owned by this builder.

    Args:
        collection: Intended ``OUTPUT_ROOT/nifti_collection`` directory.

    Returns:
        None.

    Raises:
        FileExistsError: If the path is not a directory, is unmanifested, was
            built by another utility, or differs from its owned-file hashes.
    """

    if not collection.exists():
        return
    if collection.is_symlink() or not collection.is_dir():
        raise FileExistsError(f"Refusing to replace non-directory collection: {collection}")
    manifest_path = collection / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Existing collection is not GRE-tool-owned: {collection}")
    manifest = _load_json(manifest_path)
    if manifest.get("builder") != COLLECTION_BUILDER:
        raise FileExistsError(f"Existing collection has a different builder: {collection}")
    owned = manifest.get("owned_files")
    if not isinstance(owned, Mapping) or not owned:
        raise FileExistsError("Existing GRE collection manifest has no owned-file hashes.")
    expected_paths = {"manifest.json", *(str(path) for path in owned)}
    actual_paths = {
        str(path.relative_to(collection))
        for path in collection.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != expected_paths:
        raise FileExistsError(
            f"Existing GRE collection has added or missing files: {collection}"
        )
    for relative_path, expected_hash in owned.items():
        path = collection / str(relative_path)
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"Existing GRE collection file is invalid: {path}")
        if sha256_file(path) != str(expected_hash):
            raise FileExistsError(
                f"Existing GRE collection file changed since its manifest: {path}"
            )


def _replace_owned_collection(staging: Path, collection: Path) -> None:
    """Install a staged collection atomically while retaining rollback safety.

    Args:
        staging: Complete new collection beside ``collection``.
        collection: Final ``OUTPUT_ROOT/nifti_collection`` path.

    Returns:
        None.

    Side Effects:
        Replaces only a collection already validated as builder-owned. The old
        tree remains as a temporary sibling backup until installation succeeds.
    """

    if not collection.exists():
        os.replace(staging, collection)
        return
    backup = collection.with_name(f".{collection.name}-backup-{uuid.uuid4().hex}")
    os.replace(collection, backup)
    try:
        os.replace(staging, collection)
    except Exception:
        os.replace(backup, collection)
        raise
    shutil.rmtree(backup)
