#!/usr/bin/env python3
"""Build a manifested presentation collection of magnitude NIfTIs.

Available entries are copied byte-for-byte from accepted outputs. Pending
reconstructions receive JSON placeholder records, never fake or empty NIfTIs.
The collection manifest carries display order so filenames remain stable and
scientifically descriptive rather than phase-numbered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np


KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_nifti(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(int(value) < 1 for value in image.shape):
        raise ValueError(f"Magnitude NIfTI must be a nonempty 3D image: {path}")
    axis_codes = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    if axis_codes != ("R", "A", "S"):
        raise ValueError(f"Presentation NIfTI is not canonical RAS: {path} {axis_codes}")
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"Presentation NIfTI contains non-finite samples: {path}")
    if np.iscomplexobj(data):
        raise ValueError(f"Presentation NIfTI must contain magnitude data: {path}")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    return {
        "shape": [int(value) for value in image.shape],
        "voxel_size_mm": list(zooms),
        "axis_codes": list(axis_codes),
        "dtype": str(image.get_data_dtype()),
        "all_samples_finite": True,
    }


def _validate_entries(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("Collection configuration requires a nonempty entries list.")
    normalized: list[dict[str, Any]] = []
    keys: set[str] = set()
    orders: set[int] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Every collection entry must be a JSON object.")
        entry = dict(raw)
        key = str(entry.get("key", ""))
        if not KEY_PATTERN.fullmatch(key) or key in keys:
            raise ValueError(f"Invalid or duplicate collection key: {key!r}")
        order = int(entry.get("display_order", 0))
        if order < 1 or order in orders:
            raise ValueError(f"Invalid or duplicate display order: {order}")
        status = str(entry.get("status", ""))
        if status not in {"available", "placeholder"}:
            raise ValueError(f"Invalid status for {key}: {status!r}")
        if not str(entry.get("label", "")).strip():
            raise ValueError(f"Entry has no presentation label: {key}")
        if status == "available" and not str(entry.get("source_nifti", "")):
            raise ValueError(f"Available entry has no source NIfTI: {key}")
        if status == "placeholder" and not str(entry.get("reason", "")).strip():
            raise ValueError(f"Placeholder has no reason: {key}")
        keys.add(key)
        orders.add(order)
        normalized.append(entry)
    return sorted(normalized, key=lambda item: int(item["display_order"]))


def _copy_available(
    entry: dict[str, Any], output_dir: Path, *, refresh: bool
) -> dict[str, Any]:
    source = Path(str(entry["source_nifti"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    validation = _validate_nifti(source)
    source_sha256 = sha256_file(source)
    destination = output_dir / f"{entry['key']}.nii.gz"
    placeholder = output_dir / f"{entry['key']}.placeholder.json"
    if placeholder.exists():
        if not refresh:
            raise FileExistsError(
                f"Use --refresh to replace the owned placeholder for {entry['key']}."
            )
        placeholder.unlink()
    if destination.exists():
        if sha256_file(destination) != source_sha256:
            raise FileExistsError(f"Refusing to overwrite a changed collection file: {destination}")
    else:
        temporary = Path(str(destination) + ".tmp")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Copied NIfTI hash differs from its source: {source}")
        os.replace(temporary, destination)
    record: dict[str, Any] = {
        "display_order": int(entry["display_order"]),
        "key": entry["key"],
        "label": entry["label"],
        "status": "available",
        "collection_file": destination.name,
        "source_nifti": str(source),
        "source_sha256": source_sha256,
        "collection_sha256": sha256_file(destination),
        "nifti": validation,
    }
    if entry.get("source_manifest"):
        manifest = Path(str(entry["source_manifest"])).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        record["source_manifest"] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
    if entry.get("notes"):
        record["notes"] = str(entry["notes"])
    return record


def _write_placeholder(
    entry: dict[str, Any], output_dir: Path, *, refresh: bool
) -> dict[str, Any]:
    destination = output_dir / f"{entry['key']}.placeholder.json"
    nifti_path = output_dir / f"{entry['key']}.nii.gz"
    if nifti_path.exists():
        raise FileExistsError(
            f"Refusing to replace an available NIfTI with a placeholder: {nifti_path}"
        )
    payload = {
        "format_version": 1,
        "status": "placeholder_pending_reconstruction",
        "display_order": int(entry["display_order"]),
        "key": entry["key"],
        "label": entry["label"],
        "reason": entry["reason"],
        "expected_collection_file": nifti_path.name,
    }
    if entry.get("planned_source"):
        payload["planned_source"] = str(entry["planned_source"])
    if destination.exists() and not refresh:
        existing = _load_json(destination)
        if existing != payload:
            raise FileExistsError(f"Placeholder differs; use --refresh: {destination}")
    else:
        _write_json_atomic(destination, payload)
    return {
        "display_order": int(entry["display_order"]),
        "key": entry["key"],
        "label": entry["label"],
        "status": "placeholder",
        "collection_file": destination.name,
        "reason": entry["reason"],
    }


def run(config_path: Path, *, refresh: bool) -> dict[str, Any]:
    """Build or safely refresh one configured presentation collection."""
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if int(config.get("format_version", 0)) != 1:
        raise ValueError("Unsupported collection configuration format version.")
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    entries = _validate_entries(config.get("entries"))
    manifest_path = output_dir / "collection_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned collection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in entries:
        if entry["status"] == "available":
            records.append(_copy_available(entry, output_dir, refresh=refresh))
        else:
            records.append(_write_placeholder(entry, output_dir, refresh=refresh))
    payload = {
        "format_version": 1,
        "status": (
            "complete" if all(item["status"] == "available" for item in records)
            else "complete_with_placeholders"
        ),
        "collection_name": str(config.get("collection_name", "presentation magnitude NIfTIs")),
        "scientific_scope": {
            "magnitude_niftis_only": True,
            "available_sources_copied_byte_for_byte": True,
            "spatial_resampling_performed": False,
            "cross_volume_intensity_normalization_performed": False,
            "placeholders_are_never_nifti_files": True,
        },
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "output_dir": str(output_dir),
        "entry_count": len(records),
        "available_count": sum(item["status"] == "available" for item in records),
        "placeholder_count": sum(item["status"] == "placeholder" for item in records),
        "entries": records,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    readme = output_dir / "README.md"
    lines = [
        f"# {payload['collection_name']}",
        "",
        "Files are ordered by `display_order` in `collection_manifest.json`.",
        "Available NIfTIs are byte-for-byte copies of canonical outputs.",
        "JSON placeholder files mark reconstructions that are not complete yet.",
        "No spatial resampling or cross-volume intensity normalization is performed.",
        "`presentation_metrics.csv` lists metric status and values in display order.",
        "`orientation_slices_index-128/` contains manifested presentation TIFFs.",
        "Phase NIfTIs remain in their canonical reconstruction output trees.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"Presentation collection: {output_dir}")
    print(
        f"Available: {payload['available_count']}; placeholders: {payload['placeholder_count']}"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace owned placeholders with newly available files; never overwrite changed NIfTIs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run(args.config, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
