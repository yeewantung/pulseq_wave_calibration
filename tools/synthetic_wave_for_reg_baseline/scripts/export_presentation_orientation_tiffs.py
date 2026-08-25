#!/usr/bin/env python3
"""Export fixed-index orthogonal presentation slices as 16-bit TIFF images."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from PIL import Image

from bart_cfl import sha256_file


ORIENTATIONS = ("sagittal", "coronal", "axial")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def orientation_slices(volume: np.ndarray, index: int) -> dict[str, np.ndarray]:
    """Return neurological RAS display slices with superior/anterior at top."""
    if volume.ndim != 3 or any(index < 0 or index >= size for size in volume.shape):
        raise ValueError(f"Slice index {index} is outside volume shape {volume.shape}")
    return {
        "sagittal": np.flip(volume[index, :, :].T, axis=0),
        "coronal": np.flip(volume[:, index, :].T, axis=0),
        "axial": np.flip(volume[:, :, index].T, axis=0),
    }


def _to_uint16(slice_data: np.ndarray, display_max: float) -> np.ndarray:
    scaled = np.clip(slice_data, 0.0, display_max) / display_max
    return np.rint(scaled * 65535.0).astype(np.uint16)


def _save_tiff_atomic(path: Path, pixels: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(pixels).save(temporary, format="TIFF", compression="tiff_lzw")
    with Image.open(temporary) as saved:
        if saved.size != (pixels.shape[1], pixels.shape[0]):
            raise RuntimeError(f"TIFF shape validation failed: {path}")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    collection_path = args.collection_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not collection_path.is_file():
        raise FileNotFoundError(collection_path)
    if not 0.0 < args.display_percentile <= 100.0:
        raise ValueError("Display percentile must be in (0, 100].")
    collection = _load_json(collection_path)
    if collection.get("status") not in {"complete", "complete_with_placeholders"}:
        raise ValueError("Presentation collection manifest is not complete")
    manifest_path = output_dir / "orientation_slices_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"Nonempty TIFF output is not owned: {output_dir}")
        if not args.refresh:
            raise FileExistsError("TIFF outputs exist; use --refresh.")
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_dir = collection_path.parent
    records = []
    for entry in sorted(collection["entries"], key=lambda item: item["display_order"]):
        if entry["status"] != "available":
            continue
        source = collection_dir / entry["collection_file"]
        if not source.is_file() or sha256_file(source) != entry["collection_sha256"]:
            raise ValueError(f"Collection NIfTI is missing or changed: {source}")
        image = nib.load(str(source))
        if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
            raise ValueError(f"Collection NIfTI is not canonical RAS: {source}")
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(volume).all() or np.iscomplexobj(volume):
            raise ValueError(f"Collection NIfTI is not finite real magnitude: {source}")
        positive = volume[volume > 0]
        if not positive.size:
            raise ValueError(f"Collection NIfTI has no positive magnitude: {source}")
        display_max = float(np.percentile(positive, args.display_percentile))
        if not np.isfinite(display_max) or display_max <= 0:
            raise ValueError(f"Invalid display maximum for {source}")

        output_files = {}
        for orientation, slice_data in orientation_slices(volume, args.index).items():
            destination = output_dir / (
                f"{entry['key']}_{orientation}_index-{args.index}.tiff"
            )
            _save_tiff_atomic(destination, _to_uint16(slice_data, display_max))
            output_files[orientation] = {
                "file": destination.name,
                "sha256": sha256_file(destination),
                "pixel_shape": [int(value) for value in slice_data.shape],
            }
        records.append(
            {
                "display_order": int(entry["display_order"]),
                "key": entry["key"],
                "label": entry["label"],
                "source_nifti": str(source),
                "source_sha256": entry["collection_sha256"],
                "source_shape": [int(value) for value in image.shape],
                "slice_index_each_array_axis": int(args.index),
                "display_scaling": {
                    "method": "per-volume positive-finite percentile to uint16",
                    "percentile": float(args.display_percentile),
                    "input_percentile_value": display_max,
                    "output_range": [0, 65535],
                    "clipped_for_display": True,
                },
                "orientation_convention": (
                    "canonical RAS neurological: right/anterior increase rightward; "
                    "superior or anterior displayed upward as applicable"
                ),
                "outputs": output_files,
            }
        )
        print(f"Exported TIFF slices: {entry['key']}", flush=True)

    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "fixed-index orthogonal TIFF slices for presentation",
        "collection_manifest": {
            "path": str(collection_path),
            "sha256": sha256_file(collection_path),
        },
        "slice_index_each_array_axis": int(args.index),
        "display_percentile": float(args.display_percentile),
        "entry_count": len(records),
        "tiff_count": len(records) * len(ORIENTATIONS),
        "entries": records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = Path(str(manifest_path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Orientation TIFF manifest: {manifest_path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index", type=int, default=128)
    parser.add_argument("--display-percentile", type=float, default=99.5)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
