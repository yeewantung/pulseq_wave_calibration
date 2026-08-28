#!/usr/bin/env python3
"""Build one presentation-ordered CSV from approved reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import sha256_file


STANDARD_METRICS = (
    "intensity_scale_lsq",
    "ncc_brain",
    "ncc_full_fov",
    "nrmse_brain",
    "mae_brain",
    "nmae_brain",
    "psnr_p99_db",
    "ssim_3d_brain_bbox",
    "ssim_axial_brain_mean",
    "ssim_axial_brain_slice_count",
    "gradient_ncc_brain_edge",
    "edge_preservation_ratio",
    "background_std_normalized_p99",
    "background_mean_abs_normalized_p99",
    "background_p95_abs_normalized_p99",
    "anatomy_missed_brain_fraction",
)

NATIVE_METRICS = (
    "voxel_volume_mm3",
    "smooth_region_signal_to_residual_proxy",
    "edge_gradient_mean_per_mm",
    "edge_gradient_mean_per_mm_ratio_to_full",
    "edge_abs_gradient_x_mean_per_mm_ratio_to_full",
    "edge_abs_gradient_y_mean_per_mm_ratio_to_full",
    "edge_abs_gradient_z_mean_per_mm_ratio_to_full",
)

OUTPUT_FIELDS = (
    "display_order",
    "key",
    "label",
    "collection_file",
    "availability",
    "metric_status",
    "reference",
    "comparison_grid",
    "metric_source",
    *STANDARD_METRICS,
    *NATIVE_METRICS,
    "notes",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _merge_regularization_metrics(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Merge metric tables while rejecting conflicting hash-bound records."""
    rows_by_hash: dict[str, dict[str, str]] = {}
    rows_without_hash: list[dict[str, str]] = []
    for path in paths:
        for row in _load_csv(path):
            source_hash = row.get("source_nifti_sha256", "")
            if not source_hash:
                rows_without_hash.append(row)
                continue
            existing = rows_by_hash.get(source_hash)
            if existing is not None and existing != row:
                raise ValueError(
                    "Conflicting regularization metrics for source NIfTI hash "
                    f"{source_hash}"
                )
            rows_by_hash[source_hash] = row
    return [*rows_by_hash.values(), *rows_without_hash]


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def _load_retrospective_supplement(
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load exact collection-key mappings for an additional retro metric package."""
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError(f"Unsupported retrospective supplement: {config_path}")
    matched_path = _resolve_config_path(config_path, str(config["matched_metrics"]))
    native_path = _resolve_config_path(config_path, str(config["native_metrics"]))
    matched_rows = {
        row["candidate"]: row
        for row in _load_csv(matched_path)
        if row.get("reference") == "direct_fft_rss"
    }
    native_rows = {row["case"]: row for row in _load_csv(native_path)}
    mapped: dict[str, dict[str, Any]] = {}
    for entry in config.get("entries", []):
        key = str(entry["key"])
        case = str(entry["case"])
        if key in mapped or case not in matched_rows or case not in native_rows:
            raise ValueError(f"Invalid retrospective supplement mapping: {key} -> {case}")
        mapped[key] = {
            "matched": matched_rows[case],
            "native": native_rows[case],
            "metric_source": str(config_path),
        }
    if not mapped:
        raise ValueError(f"Retrospective supplement contains no entries: {config_path}")
    provenance = {
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "matched_metrics": {
            "path": str(matched_path),
            "sha256": sha256_file(matched_path),
        },
        "native_metrics": {
            "path": str(native_path),
            "sha256": sha256_file(native_path),
        },
    }
    return mapped, provenance


def _empty_row(entry: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "display_order": int(entry["display_order"]),
            "key": entry["key"],
            "label": entry["label"],
            "collection_file": entry["collection_file"],
            "availability": entry["status"],
            "notes": entry.get("notes", ""),
        }
    )
    return row


def _copy_standard_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    for field in STANDARD_METRICS:
        if field in metrics:
            row[field] = metrics[field]


def _copy_native_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    for field in NATIVE_METRICS:
        if field in metrics:
            row[field] = metrics[field]


def _copy_retrospective_metrics(
    row: dict[str, Any], matched: dict[str, str], native: dict[str, str]
) -> None:
    aliases = {
        "intensity_scale_lsq": "intensity_scale_lsq",
        "ncc_brain": "ncc_brain",
        "nrmse_brain": "nrmse_brain",
        "nmae_brain": "nmae_brain",
        "ssim_axial_brain_mean": "ssim_axial_brain_bbox_mean",
        "ssim_axial_brain_slice_count": "ssim_axial_slice_count",
        "gradient_ncc_brain_edge": "gradient_ncc_fixed_edge",
        "edge_preservation_ratio": "edge_gradient_preservation_ratio",
    }
    for destination, source in aliases.items():
        if matched.get(source, "") != "":
            row[destination] = matched[source]
    _copy_native_metrics(row, native)


def _source_manifest(entry: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    record = entry.get("source_manifest")
    if not isinstance(record, dict) or not record.get("path"):
        return None, {}
    path = Path(record["path"]).resolve()
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Collection source-manifest hash changed: {path}")
    return path, _load_json(path)


def build_rows(
    collection: dict[str, Any],
    regularization_rows: list[dict[str, str]],
    retrospective_matched_rows: list[dict[str, str]],
    retrospective_native_rows: list[dict[str, str]],
    supplemental_retrospective: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return presentation-ordered rows with metric scope kept explicit."""
    regularization_by_hash = {
        item["source_nifti_sha256"]: item
        for item in regularization_rows
        if item.get("source_nifti_sha256")
    }
    matched_by_candidate = {
        item["candidate"]: item
        for item in retrospective_matched_rows
        if item.get("reference") == "direct_fft_rss"
    }
    native_by_case = {
        item["case"]: item for item in retrospective_native_rows if item.get("case")
    }
    supplemental_retrospective = supplemental_retrospective or {}

    output = []
    for entry in sorted(collection["entries"], key=lambda item: item["display_order"]):
        row = _empty_row(entry)
        if entry["status"] == "placeholder":
            row["metric_status"] = "pending_reconstruction"
            row["notes"] = entry.get("reason", row["notes"])
            output.append(row)
            continue
        if entry["key"].startswith("dicom_"):
            row["metric_status"] = "not_evaluated_qualitative_only"
            row["notes"] = "DICOM is presentation context, not an intensity-ranking reference."
            output.append(row)
            continue

        manifest_path, source_manifest = _source_manifest(entry)
        direct_metrics = source_manifest.get("direct_fft_metrics", {})
        if direct_metrics.get("status") == "complete":
            row.update(
                {
                    "metric_status": "complete",
                    "reference": "direct_fft_rss",
                    "comparison_grid": "exact_native_1mm_grid",
                    "metric_source": str(manifest_path),
                }
            )
            _copy_standard_metrics(row, direct_metrics["metrics"])
            output.append(row)
            continue

        regularization = regularization_by_hash.get(entry.get("source_sha256", ""))
        if regularization is not None:
            row.update(
                {
                    "metric_status": "complete",
                    "reference": "direct_fft_rss",
                    "comparison_grid": "exact_native_1mm_grid",
                    "metric_source": "regularization_metrics_csv",
                }
            )
            _copy_standard_metrics(row, regularization)
            output.append(row)
            continue

        supplemental = supplemental_retrospective.get(entry["key"])
        if supplemental is not None:
            row.update(
                {
                    "metric_status": "complete_descriptive_retrospective",
                    "reference": "direct_fft_rss",
                    "comparison_grid": "matched_full_resolution_grid",
                    "metric_source": supplemental["metric_source"],
                }
            )
            _copy_retrospective_metrics(
                row, supplemental["matched"], supplemental["native"]
            )
            output.append(row)
            continue

        case_name = source_manifest.get("case", {}).get("case_name")
        if case_name and case_name in matched_by_candidate:
            row.update(
                {
                    "metric_status": "complete_descriptive_retrospective",
                    "reference": "direct_fft_rss",
                    "comparison_grid": "matched_full_resolution_grid",
                    "metric_source": "retrospective_matched_and_native_metrics_csv",
                }
            )
            _copy_retrospective_metrics(
                row,
                matched_by_candidate[case_name],
                native_by_case.get(case_name, {}),
            )
            output.append(row)
            continue

        if entry["key"] == "direct_fft_rss":
            identity = matched_by_candidate.get("direct_fft_rss")
            if identity is None:
                raise ValueError("Direct-FFT identity metrics are missing")
            row.update(
                {
                    "metric_status": "reference_identity",
                    "reference": "direct_fft_rss",
                    "comparison_grid": "exact_native_1mm_grid",
                    "metric_source": "retrospective_matched_metrics_csv",
                }
            )
            _copy_retrospective_metrics(
                row, identity, native_by_case.get("direct_fft_rss", {})
            )
            output.append(row)
            continue
        raise ValueError(f"No approved metric source found for {entry['key']}")
    return output


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    collection_path = args.collection_manifest.expanduser().resolve()
    regularization_paths = [
        path.expanduser().resolve() for path in args.regularization_metrics
    ]
    matched_path = args.retrospective_matched_metrics.expanduser().resolve()
    native_path = args.retrospective_native_metrics.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = output_path.with_name("presentation_metrics_manifest.json")
    for path in (collection_path, *regularization_paths, matched_path, native_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (output_path.exists() or manifest_path.exists()) and not args.refresh:
        raise FileExistsError("Presentation metrics output exists; use --refresh.")

    collection = _load_json(collection_path)
    if collection.get("status") not in {"complete", "complete_with_placeholders"}:
        raise ValueError("Presentation collection manifest is not complete")
    supplemental: dict[str, dict[str, Any]] = {}
    supplemental_provenance = []
    for path in args.retrospective_supplement:
        mapped, provenance = _load_retrospective_supplement(path)
        overlap = supplemental.keys() & mapped.keys()
        if overlap:
            raise ValueError(
                "Duplicate supplemental retrospective keys: "
                + ", ".join(sorted(overlap))
            )
        supplemental.update(mapped)
        supplemental_provenance.append(provenance)
    rows = build_rows(
        collection,
        _merge_regularization_metrics(regularization_paths),
        _load_csv(matched_path),
        _load_csv(native_path),
        supplemental,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_path, rows)
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "presentation-ordered approved metric list",
        "output_csv": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "row_count": len(rows),
        },
        "inputs": {
            "collection_manifest": {
                "path": str(collection_path),
                "sha256": sha256_file(collection_path),
            },
            "regularization_metrics": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in regularization_paths
            ],
            "retrospective_matched_metrics": {
                "path": str(matched_path),
                "sha256": sha256_file(matched_path),
            },
            "retrospective_native_metrics": {
                "path": str(native_path),
                "sha256": sha256_file(native_path),
            },
            "retrospective_supplements": supplemental_provenance,
        },
        "scope_notes": [
            "DICOM rows are qualitative only and intentionally have no metrics.",
            "Standard reconstructions use the approved direct-FFT reference on the exact 1 mm grid.",
            "Retrospective-resolution fidelity metrics use the documented matched full-resolution grid.",
            "Native retrospective sharpness and signal/local-residual proxy fields remain descriptive and are not true SNR.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = Path(str(manifest_path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Presentation metrics CSV: {output_path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument(
        "--regularization-metrics",
        required=True,
        action="append",
        type=Path,
        help="Repeat for each hash-bound regularization metric table.",
    )
    parser.add_argument("--retrospective-matched-metrics", required=True, type=Path)
    parser.add_argument("--retrospective-native-metrics", required=True, type=Path)
    parser.add_argument(
        "--retrospective-supplement", action="append", default=[], type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
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
