"""Archive copied source NIfTIs after strict collection verification."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bart_io import sha256_file
from .gre_nifti_collection import COLLECTION_BUILDER as GRE_COLLECTION_BUILDER
from .gre_nifti_collection import _validate_existing_collection as validate_gre_collection
from .nifti_collection import COLLECTION_BUILDER as MPRAGE_COLLECTION_BUILDER
from .nifti_collection import (
    _manifest_owned_files as mprage_manifest_owned_files,
)
from .nifti_collection import (
    _validate_existing_collection as validate_mprage_collection,
)


ARCHIVAL_MANIFEST_NAME = "source_nifti_archival.json"


def archive_collection_source_niftis(
    output_root: str | Path,
    *,
    dry_run: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Remove source NIfTI pairs only after validating collection copies.

    Args:
        output_root: Reconstruction root containing a completed NIfTI collection.
        dry_run: Validate and report the exact archival set without deleting files.
        verify_hashes: Recompute every source and copy SHA-256 digest. If false,
            require matching recorded digests and file presence without rereading
            image payloads; intended for a batch after representative full checks.

    Returns:
        JSON-native archival summary, including released allocated-byte estimates.

    Raises:
        FileNotFoundError: If the reconstruction root, collection, source, or copy
            is absent.
        FileExistsError: If the collection differs from its owned-file manifest.
        ValueError: If source/copy provenance is inconsistent or unsafe.

    Side Effects:
        Unless ``dry_run`` is true, deletes only collection-recorded source NIfTI
        and JSON pairs, writes ``source_nifti_archival.json``, and updates the
        top-level collection manifest. Quantitative arrays, conversion manifests,
        reconstruction manifests, and collection copies are preserved.
    """

    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reconstruction output root does not exist: {root}")
    collection = root / "nifti_collection"
    manifest_path = collection / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NIfTI collection manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    builder = str(manifest.get("builder", ""))
    if builder == MPRAGE_COLLECTION_BUILDER:
        if verify_hashes:
            manifest = validate_mprage_collection(collection) or {}
        else:
            _validate_collection_presence(collection, manifest, builder)
    elif builder == GRE_COLLECTION_BUILDER:
        if verify_hashes:
            manifest = validate_gre_collection(collection) or {}
        else:
            _validate_collection_presence(collection, manifest, builder)
    else:
        raise ValueError(f"Unsupported NIfTI collection builder: {builder}")

    existing = manifest.get("source_archival")
    if isinstance(existing, Mapping) and existing.get("status") == "complete":
        record_path = collection / str(existing.get("manifest", ""))
        expected_hash = str(existing.get("manifest_sha256", ""))
        if not record_path.is_file() or sha256_file(record_path) != expected_hash:
            raise ValueError("Completed source archival metadata is missing or changed.")
        result = _load_json(record_path)
        result["already_complete"] = True
        return result

    entries = _collection_source_entries(root, collection, manifest, builder)
    if not entries:
        raise ValueError("Collection manifest contains no source NIfTI pairs to archive.")
    _validate_archive_entries(
        root, collection, entries, verify_hashes=verify_hashes
    )
    allocated_bytes = sum(int(entry["source_allocated_bytes"]) for entry in entries)
    apparent_bytes = sum(int(entry["source_size_bytes"]) for entry in entries)
    nifti_count = sum(1 for entry in entries if entry["kind"] == "nifti")
    sidecar_count = sum(1 for entry in entries if entry["kind"] == "json_sidecar")
    summary: dict[str, Any] = {
        "format_version": 1,
        "status": "dry_run" if dry_run else "verified_pending_removal",
        "builder": "wave_retro_lr.collection_archive",
        "collection_builder": builder,
        "output_root": ".",
        "collection": "nifti_collection",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_mode": (
            "recomputed_sha256" if verify_hashes else "recorded_sha256_and_copy_presence"
        ),
        "source_nifti_count": nifti_count,
        "source_sidecar_count": sidecar_count,
        "source_file_count": len(entries),
        "source_apparent_bytes": apparent_bytes,
        "source_allocated_bytes": allocated_bytes,
        "preserved_collection_copies": True,
        "quantitative_complex_arrays_removed": False,
        "conversion_manifests_removed": False,
        "reconstruction_resumability": "intentionally_disabled_for_archived_nifti_outputs",
        "files": entries,
    }
    if dry_run:
        return summary

    archival_path = collection / ARCHIVAL_MANIFEST_NAME
    if archival_path.exists() or archival_path.is_symlink():
        raise FileExistsError(f"Unexpected archival metadata already exists: {archival_path}")
    _write_json_atomic(archival_path, summary)
    removed: list[Path] = []
    try:
        for entry in entries:
            source = root / str(entry["source"])
            source.unlink()
            removed.append(source)
    except Exception:
        summary["status"] = "incomplete_removal"
        summary["removed_source_files"] = [str(path.relative_to(root)) for path in removed]
        _write_json_atomic(archival_path, summary)
        raise

    summary["status"] = "complete"
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["removed_source_file_count"] = len(removed)
    _write_json_atomic(archival_path, summary)
    archival_hash = sha256_file(archival_path)

    owned = manifest.get("owned_files")
    if not isinstance(owned, dict):
        raise ValueError("Collection manifest has no explicit owned-file hash map.")
    owned[ARCHIVAL_MANIFEST_NAME] = archival_hash
    manifest["source_archival"] = {
        "status": "complete",
        "manifest": ARCHIVAL_MANIFEST_NAME,
        "manifest_sha256": archival_hash,
        "source_nifti_count": nifti_count,
        "source_sidecar_count": sidecar_count,
        "source_allocated_bytes": allocated_bytes,
        "completed_at_utc": summary["completed_at_utc"],
    }
    storage = manifest.setdefault("storage_policy", {})
    storage["source_nifti_outputs"] = {
        "status": "removed_after_verified_byte_identical_collection_copy",
        "collection_copies_preserved": True,
        "reconstruction_resumability": "intentionally_disabled",
    }
    scope = manifest.setdefault("scientific_scope", {})
    scope["canonical_reconstruction_outputs_modified"] = True
    scope["source_nifti_outputs_removed_after_verified_copy"] = True
    scope["quantitative_complex_arrays_removed"] = False
    _write_json_atomic(manifest_path, manifest)
    return summary


def prune_mprage_head_masks(
    output_root: str | Path, *, validate_hashes: bool = False
) -> dict[str, Any]:
    """Remove MPRAGE mask products and rewrite manifests as original-only.

    Args:
        output_root: Reconstruction root containing an MPRAGE collection.
        validate_hashes: Recompute all existing collection hashes first. The
            default performs a metadata/path-presence migration for batch use.

    Returns:
        JSON-native summary of removed directories, files, and allocated bytes.

    Raises:
        FileNotFoundError: If the collection manifest is absent.
        ValueError: If the collection builder or manifest structure is invalid.

    Side Effects:
        Deletes only collection-local ``head_masked_nifti`` and ``masks``
        directories, rewrites variant manifests, and atomically updates the
        top-level collection manifest with the removal record.
    """

    root = Path(output_root).expanduser().resolve()
    collection = root / "nifti_collection"
    manifest_path = collection / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MPRAGE collection manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("builder") != MPRAGE_COLLECTION_BUILDER:
        raise ValueError(f"Not an MPRAGE NIfTI collection: {collection}")
    policy = manifest.get("storage_policy", {})
    if (
        isinstance(policy, Mapping)
        and policy.get("collection_payload") == "original_nifti_only"
    ):
        return {
            "status": "already_original_only",
            "removed_directory_count": 0,
            "removed_file_count": 0,
            "removed_allocated_bytes": 0,
        }
    if validate_hashes:
        manifest = validate_mprage_collection(collection) or {}

    owned = mprage_manifest_owned_files(manifest)
    mask_paths = {
        path
        for path in owned
        if "head_masked_nifti" in Path(path).parts or "masks" in Path(path).parts
    }
    if not validate_hashes:
        for relative, digest in owned.items():
            path = collection / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Recorded collection file is absent: {path}")
            if len(str(digest)) != 64:
                raise ValueError(f"Recorded collection digest is invalid: {path}")

    mask_directories = sorted(
        {
            path
            for path in collection.rglob("*")
            if path.is_dir() and path.name in {"head_masked_nifti", "masks"}
        },
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removed_file_count = 0
    removed_allocated_bytes = 0
    for directory in mask_directories:
        for path in directory.rglob("*"):
            if path.is_file() or path.is_symlink():
                removed_file_count += 1
                if path.is_file() and not path.is_symlink():
                    removed_allocated_bytes += path.stat().st_blocks * 512

    completed_at = datetime.now(timezone.utc).isoformat()
    variants = manifest.get("variants")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("MPRAGE collection variant record is invalid.")
            subtree = variant.get("subtree_manifest")
            if isinstance(subtree, Mapping):
                subtree_path = collection / str(subtree.get("path", ""))
                subtree_manifest = _load_json(subtree_path)
                _mark_original_only(subtree_manifest, completed_at)
                subtree_owned = subtree_manifest.get("owned_files")
                if isinstance(subtree_owned, dict):
                    subtree_manifest["owned_files"] = {
                        path: digest
                        for path, digest in subtree_owned.items()
                        if "head_masked_nifti" not in Path(path).parts
                        and "masks" not in Path(path).parts
                    }
                _write_json_atomic(subtree_path, subtree_manifest)
                subtree["sha256"] = sha256_file(subtree_path)
                owned[str(subtree_path.relative_to(collection))] = subtree["sha256"]
            variant["head_mask"] = {
                "status": "removed_by_original_only_storage_policy",
                "generated": False,
            }

    _mark_original_only(manifest, completed_at)
    manifest["owned_files"] = {
        path: digest for path, digest in owned.items() if path not in mask_paths
    }
    manifest["storage_policy"]["head_mask_outputs"].update(
        {
            "removed_during_this_sync": True,
            "historical_outputs_removed": True,
            "removed_at_utc": completed_at,
            "removed_file_count": removed_file_count,
            "removed_allocated_bytes": removed_allocated_bytes,
            "migration_validation": (
                "recomputed_sha256"
                if validate_hashes
                else "recorded_manifest_and_path_presence"
            ),
        }
    )
    synchronization = manifest.setdefault("synchronization", {})
    synchronization["head_mask_storage_migration"] = {
        "status": "complete",
        "removed_at_utc": completed_at,
        "removed_file_count": removed_file_count,
        "removed_allocated_bytes": removed_allocated_bytes,
    }

    for directory in mask_directories:
        if directory.exists():
            shutil.rmtree(directory)
    _write_json_atomic(manifest_path, manifest)
    return {
        "status": "complete",
        "removed_directory_count": len(mask_directories),
        "removed_file_count": removed_file_count,
        "removed_allocated_bytes": removed_allocated_bytes,
    }


def _mark_original_only(manifest: dict[str, Any], completed_at: str) -> None:
    """Remove masked-file fields and record an original-only storage policy.

    Args:
        manifest: Top-level or variant MPRAGE collection manifest.
        completed_at: UTC timestamp shared by the migration records.

    Returns:
        None. The supplied mapping is updated in place.
    """

    masked_fields = {
        "masked_nifti",
        "masked_nifti_sha256",
        "masked_json",
        "masked_json_sha256",
        "mask_mapping",
        "masked_nonzero_voxels",
    }
    for case in manifest.get("cases", []):
        if not isinstance(case, dict):
            continue
        for record in case.get("files", []):
            if isinstance(record, dict):
                for field in masked_fields:
                    record.pop(field, None)
    manifest["format_version"] = max(int(manifest.get("format_version", 0)), 6)
    manifest["head_mask"] = {
        "status": "removed_by_original_only_storage_policy",
        "generated": False,
    }
    manifest["storage_policy"] = {
        **(
            manifest.get("storage_policy", {})
            if isinstance(manifest.get("storage_policy"), Mapping)
            else {}
        ),
        "collection_payload": "original_nifti_only",
        "head_mask_outputs": {
            "default": "omitted",
            "requested": False,
            "status": "omitted_by_policy",
            "historical_outputs_removed": True,
            "removed_at_utc": completed_at,
        },
    }
    scope = manifest.setdefault("scientific_scope", {})
    scope["head_mask_outputs_generated"] = False
    scope["same_whole_head_mask_applied_to_all_branches"] = False
    scope["masked_outputs_for_presentation_only"] = False
    scope["masked_outputs_excluded_from_regularization_evaluation"] = False


def _collection_source_entries(
    root: Path,
    collection: Path,
    manifest: Mapping[str, Any],
    builder: str,
) -> list[dict[str, Any]]:
    """Extract unique source/copy records from all collection variants.

    Args:
        root: Reconstruction output root.
        collection: Top-level NIfTI collection directory.
        manifest: Validated top-level collection manifest.
        builder: Collection builder identifier.

    Returns:
        Sorted file-level archival records for NIfTIs and JSON sidecars.
    """

    records: dict[str, dict[str, Any]] = {}
    variants = manifest.get("variants")
    legacy_top_level = not isinstance(variants, list)
    if legacy_top_level:
        variants = [
            {
                "collection_id": "legacy_vcc12",
                "collection_relative": ".",
            }
        ]
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("Collection variant record is invalid.")
        relative = Path(str(variant.get("collection_relative", ".")))
        variant_root = collection if relative == Path(".") else collection / relative
        if relative == Path("."):
            if legacy_top_level:
                cases = manifest.get("cases", [])
            else:
                collection_id = str(variant.get("collection_id", ""))
                cases = [
                    case
                    for case in manifest.get("cases", [])
                    if isinstance(case, Mapping)
                    and str(case.get("collection_id", "")) == collection_id
                ]
        else:
            subtree = variant.get("subtree_manifest")
            if not isinstance(subtree, Mapping):
                raise ValueError("Count-specific collection has no subtree manifest.")
            subtree_path = collection / str(subtree.get("path", ""))
            cases = _load_json(subtree_path).get("cases", [])
        if not isinstance(cases, list):
            raise ValueError("Collection variant cases are invalid.")
        for case in cases:
            if not isinstance(case, Mapping) or not isinstance(case.get("files"), list):
                raise ValueError("Collection case record is invalid.")
            for file_record in case["files"]:
                if not isinstance(file_record, Mapping):
                    raise ValueError("Collection file record is invalid.")
                if builder == MPRAGE_COLLECTION_BUILDER:
                    pairs = (
                        (
                            "nifti",
                            "source_nifti",
                            "source_nifti_sha256",
                            "original_nifti",
                            "original_nifti_sha256",
                        ),
                        (
                            "json_sidecar",
                            "source_json",
                            "source_json_sha256",
                            "original_json",
                            "original_json_sha256",
                        ),
                    )
                else:
                    pairs = (
                        (
                            "nifti",
                            "source_nifti",
                            "source_nifti_sha256",
                            "collection_nifti",
                            "source_nifti_sha256",
                        ),
                        (
                            "json_sidecar",
                            "source_sidecar",
                            "source_sidecar_sha256",
                            "collection_sidecar",
                            "source_sidecar_sha256",
                        ),
                    )
                for kind, source_key, source_hash_key, copy_key, copy_hash_key in pairs:
                    source_relative = str(file_record.get(source_key, ""))
                    copy_relative = str(file_record.get(copy_key, ""))
                    if not source_relative or not copy_relative:
                        raise ValueError("Collection file record lacks archival paths.")
                    source = root / source_relative
                    copied = variant_root / copy_relative
                    entry = {
                        "kind": kind,
                        "source": source_relative,
                        "source_sha256": str(file_record.get(source_hash_key, "")),
                        "collection_copy": str(copied.relative_to(collection)),
                        "collection_copy_sha256": str(file_record.get(copy_hash_key, "")),
                        "source_size_bytes": source.stat().st_size if source.is_file() else 0,
                        "source_allocated_bytes": (
                            source.stat().st_blocks * 512 if source.is_file() else 0
                        ),
                    }
                    prior = records.get(source_relative)
                    if prior is not None and prior != entry:
                        raise ValueError(f"Source has conflicting collection copies: {source}")
                    records[source_relative] = entry
    return [records[path] for path in sorted(records)]


def _validate_archive_entries(
    root: Path,
    collection: Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    verify_hashes: bool,
) -> None:
    """Validate source/copy hashes and path confinement before deletion.

    Args:
        root: Reconstruction output root.
        collection: Top-level collection directory.
        entries: File-level archival records.
        verify_hashes: Whether to recompute file content digests.

    Returns:
        None.

    Raises:
        FileNotFoundError: If a source or collection copy is absent.
        ValueError: If a path escapes its root or any hash differs.
    """

    for entry in entries:
        source = (root / str(entry["source"])).resolve()
        copied = (collection / str(entry["collection_copy"])).resolve()
        if not source.is_relative_to(root) or not copied.is_relative_to(collection):
            raise ValueError("Archival path escapes its declared root.")
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Archival source is missing or invalid: {source}")
        if copied.is_symlink() or not copied.is_file():
            raise FileNotFoundError(f"Collection copy is missing or invalid: {copied}")
        recorded_source = str(entry["source_sha256"])
        recorded_copy = str(entry["collection_copy_sha256"])
        if len(recorded_source) != 64 or recorded_source != recorded_copy:
            raise ValueError(f"Recorded source/copy hashes differ: {source}")
        if verify_hashes:
            source_hash = sha256_file(source)
            copied_hash = sha256_file(copied)
            if source_hash != recorded_source:
                raise ValueError(f"Archival source hash mismatch: {source}")
            if copied_hash != recorded_copy:
                raise ValueError(f"Collection copy hash mismatch: {copied}")


def _validate_collection_presence(
    collection: Path, manifest: Mapping[str, Any], builder: str
) -> None:
    """Validate collection ownership and file presence without payload hashing.

    Args:
        collection: Existing top-level collection directory.
        manifest: Parsed top-level collection manifest.
        builder: Expected MPRAGE or GRE collection builder identifier.

    Returns:
        None.

    Raises:
        FileExistsError: If the collection has missing, added, symlinked, or
            invalidly hashed files relative to its ownership manifest.
    """

    if builder == MPRAGE_COLLECTION_BUILDER:
        owned = mprage_manifest_owned_files(dict(manifest))
    elif builder == GRE_COLLECTION_BUILDER:
        raw_owned = manifest.get("owned_files")
        if not isinstance(raw_owned, Mapping) or not raw_owned:
            raise FileExistsError("Existing GRE collection has no owned-file hashes.")
        owned = {str(path): str(digest) for path, digest in raw_owned.items()}
    else:
        raise ValueError(f"Unsupported NIfTI collection builder: {builder}")
    if any(not path or len(digest) != 64 for path, digest in owned.items()):
        raise FileExistsError("Collection manifest contains an invalid file hash.")
    expected = {"manifest.json", *owned}
    actual = {
        str(path.relative_to(collection))
        for path in collection.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise FileExistsError(
            f"Existing collection has added or missing files: {collection}"
        )
    for relative in owned:
        path = collection / relative
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"Existing collection file is invalid: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk.

    Args:
        path: UTF-8 JSON file.

    Returns:
        Parsed top-level mapping.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write stable JSON beside its final path.

    Args:
        path: Destination JSON path.
        payload: JSON-native mapping to serialize.

    Returns:
        None.

    Side Effects:
        Creates a temporary sibling and atomically replaces ``path``.
    """

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
