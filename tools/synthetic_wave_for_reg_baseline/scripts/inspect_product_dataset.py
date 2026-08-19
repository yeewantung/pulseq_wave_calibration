#!/usr/bin/env python3
"""Inspect Siemens product TWIX sampling metadata and matching DICOM series.

The TWIX path is metadata-only by default: mapVBVD indexes MDH headers without
requesting image or refscan arrays. ``--probe-samples`` optionally reads one
small RO-by-coil block from each relevant stream. The resulting JSON captures
the information needed to design later loaders without allocating full k-space.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


COUNTER_NAMES = (
    "Lin",
    "Par",
    "Sli",
    "Ave",
    "Phs",
    "Eco",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
)

DICOM_TAGS = {
    "0008,0008": "image_type",
    "0008,103e": "series_description",
    "0018,0050": "slice_thickness_mm",
    "0020,000e": "series_instance_uid",
    "0020,0013": "instance_number",
    "0028,0010": "rows",
    "0028,0011": "columns",
    "0028,0030": "pixel_spacing_mm",
    "0028,0100": "bits_allocated",
    "0028,0101": "bits_stored",
    "0028,0102": "high_bit",
    "0028,0103": "pixel_representation",
}

INTEGER_DICOM_FIELDS = {
    "instance_number",
    "rows",
    "columns",
    "bits_allocated",
    "bits_stored",
    "high_bit",
    "pixel_representation",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the metadata-inspection command interface."""
    parser = argparse.ArgumentParser(
        description="Create a metadata-only report for TWIX and DICOM inputs."
    )
    parser.add_argument("--twix", required=True, type=Path, help="Siemens TWIX .dat file.")
    parser.add_argument(
        "--dicom-dir", required=True, type=Path, help="Directory containing reference DICOMs."
    )
    parser.add_argument("--output", required=True, type=Path, help="JSON report path.")
    parser.add_argument(
        "--probe-samples",
        action="store_true",
        help="Read one image and one refscan RO-by-coil block to verify payload access.",
    )
    return parser


def _integer_array(values: Any, name: str) -> list[int]:
    """Convert integral-valued mapVBVD counters to ordinary Python integers."""
    if values is None:
        return []
    result: list[int] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} contains a non-integral counter value: {value!r}")
        result.append(int(numeric))
    return result


def _unique_summary(values: Sequence[int], include_values: bool = True) -> dict[str, Any]:
    """Summarize the count and support of an integer counter sequence."""
    unique = sorted(set(values))
    result: dict[str, Any] = {
        "count": len(values),
        "unique_count": len(unique),
        "min": unique[0] if unique else None,
        "max": unique[-1] if unique else None,
    }
    if include_values:
        result["unique_values"] = unique
    return result


def _compress_partition_patterns(
    lines: Sequence[int], partitions: Sequence[int]
) -> list[dict[str, Any]]:
    """Group PE2 partitions that have exactly the same acquired PE1 lines."""
    if len(lines) != len(partitions):
        raise ValueError("PE1 and PE2 counter arrays must have equal lengths.")

    by_partition: dict[int, set[int]] = defaultdict(set)
    for line, partition in zip(lines, partitions):
        by_partition[int(partition)].add(int(line))

    by_pattern: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for partition, acquired_lines in sorted(by_partition.items()):
        by_pattern[tuple(sorted(acquired_lines))].append(partition)

    return [
        {"partitions": grouped_partitions, "pe1_lines": list(pattern)}
        for pattern, grouped_partitions in sorted(
            by_pattern.items(), key=lambda item: (item[1][0], item[0])
        )
    ]


def _infer_regular_stride(lines: Sequence[int]) -> int | None:
    """Infer the regular PE1 stride from unique acquired line differences."""
    unique = sorted(set(lines))
    differences = [right - left for left, right in zip(unique, unique[1:]) if right > left]
    if not differences:
        return None
    return math.gcd(*differences)


def summarize_sampling(
    image_lines: Sequence[int],
    image_partitions: Sequence[int],
    ref_lines: Sequence[int],
    ref_partitions: Sequence[int],
    *,
    matrix_pe1: int | None,
    matrix_pe2: int | None,
) -> dict[str, Any]:
    """Summarize exact MDH-coordinate masks without reading k-space samples."""
    image_pairs = set(zip(image_lines, image_partitions))
    ref_pairs = set(zip(ref_lines, ref_partitions))
    union_pairs = image_pairs | ref_pairs

    ref_unique_lines = sorted(set(ref_lines))
    ref_unique_partitions = sorted(set(ref_partitions))
    ref_is_rectangle = len(ref_pairs) == len(ref_unique_lines) * len(ref_unique_partitions)
    ref_covers_full_pe2 = (
        matrix_pe2 is not None
        and ref_unique_partitions == list(range(matrix_pe2))
    )

    stride = _infer_regular_stride(image_lines)
    residues = sorted({line % stride for line in image_lines}) if stride and stride > 1 else []

    union_lines = [pair[0] for pair in sorted(union_pairs)]
    union_partitions = [pair[1] for pair in sorted(union_pairs)]
    result: dict[str, Any] = {
        "matrix_pe1": matrix_pe1,
        "matrix_pe2": matrix_pe2,
        "image_unique_coordinate_count": len(image_pairs),
        "refscan_unique_coordinate_count": len(ref_pairs),
        "union_unique_coordinate_count": len(union_pairs),
        "image_duplicate_coordinate_count": len(image_lines) - len(image_pairs),
        "refscan_duplicate_coordinate_count": len(ref_lines) - len(ref_pairs),
        "image_inferred_pe1_stride": stride,
        "image_pe1_residues_for_inferred_stride": residues,
        "image_patterns_by_pe2": _compress_partition_patterns(image_lines, image_partitions),
        "refscan_patterns_by_pe2": _compress_partition_patterns(ref_lines, ref_partitions),
        "merged_patterns_by_pe2": _compress_partition_patterns(union_lines, union_partitions),
        "refscan_unique_pe1_lines": ref_unique_lines,
        "refscan_unique_pe2_partitions": ref_unique_partitions,
        "refscan_is_cartesian_rectangle": ref_is_rectangle,
        "refscan_covers_full_pe2": ref_covers_full_pe2,
    }

    if matrix_pe1 is not None and matrix_pe2 is not None:
        result["full_grid_coordinate_count"] = matrix_pe1 * matrix_pe2
        result["merged_sampling_fraction"] = len(union_pairs) / (matrix_pe1 * matrix_pe2)
        out_of_range = sorted(
            [
                [line, partition]
                for line, partition in union_pairs
                if not (0 <= line < matrix_pe1 and 0 <= partition < matrix_pe2)
            ]
        )
        result["out_of_range_coordinates"] = out_of_range

    return result


def _header_get(mapping: Any, key: tuple[str, ...], default: Any = None) -> Any:
    """Read a mapVBVD header key while tolerating its mapping variants."""
    try:
        value = mapping.get(key, default)
    except Exception:
        try:
            value = mapping[key]
        except Exception:
            return default
    return default if value is None else value


def _header_string(value: Any) -> str:
    """Remove Siemens ASCCONV quote delimiters from scalar header strings."""
    return str(value).strip().strip('"')


def _header_summary(scan: Any) -> dict[str, Any]:
    """Extract the compact protocol and matrix fields used by this workflow."""
    hdr = scan["hdr"] if isinstance(scan, Mapping) else scan.hdr
    yaps = hdr["MeasYaps"]
    fields = {
        "protocol_name": (("tProtocolName",), _header_string),
        "sequence_file_name": (("tSequenceFileName",), _header_string),
        "base_resolution": (("sKSpace", "lBaseResolution"), int),
        "phase_encoding_lines": (("sKSpace", "lPhaseEncodingLines"), int),
        "partitions": (("sKSpace", "lPartitions"), int),
        "acceleration_pe1": (("sPat", "lAccelFactPE"), int),
        "acceleration_pe2": (("sPat", "lAccelFact3D"), int),
        "reference_lines_pe1": (("sPat", "lRefLinesPE"), int),
        "reference_lines_pe2": (("sPat", "lRefLines3D"), int),
        "pat_mode": (("sPat", "ucPATMode"), int),
        "reference_scan_mode": (("sPat", "ucRefScanMode"), int),
    }
    result: dict[str, Any] = {}
    for label, (key, converter) in fields.items():
        value = _header_get(yaps, key)
        if value is None:
            result[label] = None
            continue
        try:
            result[label] = converter(value)
        except (TypeError, ValueError):
            result[label] = str(value)
    return result


def _stream_summary(stream: Any) -> dict[str, Any]:
    """Summarize mapVBVD stream dimensions and MDH counters without payload I/O."""
    stream.flagRemoveOS = True
    counters = {
        name: _unique_summary(_integer_array(getattr(stream, name, None), name))
        for name in COUNTER_NAMES
    }
    center_lines = _integer_array(getattr(stream, "centerLin", None), "centerLin")
    center_partitions = _integer_array(getattr(stream, "centerPar", None), "centerPar")
    reflected_raw = getattr(stream, "IsReflected", None)
    reflected = [] if reflected_raw is None else list(reflected_raw)

    acquired_columns = int(stream.NCol)
    output_columns = int(stream.dataSize[0])
    return {
        "type": str(stream.dataType),
        "acquisition_count": int(stream.NAcq),
        "data_dimensions": list(stream.dataDims),
        "data_size_remove_os": [int(value) for value in stream.dataSize],
        "squeezed_dimensions": list(stream.sqzDims),
        "squeezed_size_remove_os": [int(value) for value in stream.sqzSize],
        "acquired_readout_samples": acquired_columns,
        "output_readout_samples_after_remove_os": output_columns,
        "readout_oversampling_factor": acquired_columns / output_columns,
        "coil_count": int(stream.NCha),
        "counters": counters,
        "center_line": _unique_summary(center_lines),
        "center_partition": _unique_summary(center_partitions),
        "reflected_acquisition_count": sum(bool(value) for value in reflected),
    }


def _probe_stream_sample(stream: Any) -> dict[str, Any]:
    """Read only the first acquired PE coordinate and report its array layout."""
    import numpy as np

    raw_line = min(_integer_array(stream.Lin, "stream.Lin"))
    raw_partition = min(_integer_array(stream.Par, "stream.Par"))
    local_line = raw_line - int(stream.skipLin)
    local_partition = raw_partition - int(stream.skipPar)
    stream.flagRemoveOS = True
    stream.squeeze = True
    block = np.asarray(stream[:, :, local_line, local_partition])
    return {
        "raw_pe1_line": raw_line,
        "raw_pe2_partition": raw_partition,
        "mapvbvd_local_pe1_index": local_line,
        "mapvbvd_local_pe2_index": local_partition,
        "shape": list(block.shape),
        "dtype": str(block.dtype),
        "all_finite": bool(np.isfinite(block).all()),
        "nonzero_sample_count": int(np.count_nonzero(block)),
        "l2_norm": float(np.linalg.norm(block)),
    }


def inspect_twix(path: Path, *, probe_samples: bool = False) -> dict[str, Any]:
    """Index all TWIX measurements and analyze the largest image measurement."""
    try:
        import mapvbvd
    except ImportError as exc:
        raise RuntimeError(
            "TWIX inspection requires pymapvbvd>=0.6.1. Install the "
            "tool's requirements into a Python 3.11+ environment."
        ) from exc

    twix_root = mapvbvd.mapVBVD(str(path), quiet=True)
    measurements = list(twix_root) if isinstance(twix_root, (list, tuple)) else [twix_root]

    reports: list[dict[str, Any]] = []
    for index, scan in enumerate(measurements):
        stream_names = scan.MDH_flags() if hasattr(scan, "MDH_flags") else []
        stream_reports = {
            name: _stream_summary(scan[name])
            for name in stream_names
            if name != "hdr" and getattr(scan[name], "NAcq", 0)
        }
        reports.append(
            {
                "measurement_index": index,
                "header": _header_summary(scan),
                "streams": stream_reports,
            }
        )

    candidates = [
        (report["streams"]["image"]["acquisition_count"], report["measurement_index"])
        for report in reports
        if "image" in report["streams"]
    ]
    if not candidates:
        raise RuntimeError("No TWIX measurement contains an image stream.")
    selected_index = max(candidates)[1]
    selected_scan = measurements[selected_index]
    selected_report = reports[selected_index]
    header = selected_report["header"]

    image = selected_scan["image"]
    image_lines = _integer_array(image.Lin, "image.Lin")
    image_partitions = _integer_array(image.Par, "image.Par")
    if "refscan" in selected_report["streams"]:
        refscan = selected_scan["refscan"]
        ref_lines = _integer_array(refscan.Lin, "refscan.Lin")
        ref_partitions = _integer_array(refscan.Par, "refscan.Par")
    else:
        ref_lines = []
        ref_partitions = []

    sampling = summarize_sampling(
        image_lines,
        image_partitions,
        ref_lines,
        ref_partitions,
        matrix_pe1=header.get("phase_encoding_lines"),
        matrix_pe2=header.get("partitions"),
    )

    probes: dict[str, Any] = {}
    if probe_samples:
        probes["image"] = _probe_stream_sample(image)
        if "refscan" in selected_report["streams"]:
            probes["refscan"] = _probe_stream_sample(selected_scan["refscan"])

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "pymapvbvd_version": metadata.version("pymapvbvd"),
        "measurement_count": len(measurements),
        "selected_measurement_index": selected_index,
        "measurement_selection_rule": "largest image-stream acquisition count",
        "measurements": reports,
        "selected_measurement_sampling": sampling,
        "sample_probes": probes,
    }


def parse_dcmdump_records(text: str) -> list[dict[str, Any]]:
    """Parse the deliberately small dcmdump output requested by this script."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    filename_pattern = re.compile(r"^# dcmdump \(\d+/\d+\): (.+)$")
    tag_pattern = re.compile(r"^\(([0-9a-fA-F]{4},[0-9a-fA-F]{4})\)\s+\w+\s+(.*?)\s+#")

    for line in text.splitlines():
        filename_match = filename_pattern.match(line)
        if filename_match:
            current = {"path": filename_match.group(1), "filename": Path(filename_match.group(1)).name}
            records.append(current)
            continue
        if current is None:
            continue
        tag_match = tag_pattern.match(line)
        if not tag_match:
            continue
        tag, raw_value = tag_match.groups()
        value = raw_value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        elif value == "(no value available)":
            value = ""
        field = DICOM_TAGS.get(tag.lower())
        if field:
            current[field] = int(value) if field in INTEGER_DICOM_FIELDS and value else value
    return records


def _constant_or_values(records: Sequence[Mapping[str, Any]], field: str) -> Any:
    """Return one shared DICOM value or the sorted set of differing values."""
    values = sorted({str(record.get(field, "")) for record in records})
    if len(values) == 1:
        value: Any = records[0].get(field, "")
        return value
    return values


def inspect_dicom_directory(path: Path) -> dict[str, Any]:
    """Group DICOM metadata by series and identify the unfiltered ND baseline."""
    executable = shutil.which("dcmdump")
    if not executable:
        raise RuntimeError("DICOM inspection requires the DCMTK 'dcmdump' executable.")

    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".dcm")
    if not files:
        raise RuntimeError(f"No .dcm files found in {path}.")

    command = [executable, "+F"]
    for tag in DICOM_TAGS:
        command.extend(["+P", tag])
    command.extend(str(item) for item in files)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    records = parse_dcmdump_records(completed.stdout)
    if len(records) != len(files):
        raise RuntimeError(f"Parsed {len(records)} DICOM records but found {len(files)} files.")

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_uid[str(record.get("series_instance_uid", ""))].append(record)

    series_reports: list[dict[str, Any]] = []
    for uid, series_records in by_uid.items():
        instance_numbers = sorted(
            int(record["instance_number"])
            for record in series_records
            if "instance_number" in record
        )
        image_type = str(_constant_or_values(series_records, "image_type"))
        components = image_type.split("\\")
        unfiltered = "ND" in components and "DIS2D" not in components and "DIS3D" not in components
        series_reports.append(
            {
                "series_instance_uid": uid,
                "file_count": len(series_records),
                "first_filename": min(record["filename"] for record in series_records),
                "last_filename": max(record["filename"] for record in series_records),
                "instance_number_min": min(instance_numbers) if instance_numbers else None,
                "instance_number_max": max(instance_numbers) if instance_numbers else None,
                "instance_numbers_are_contiguous": instance_numbers
                == list(range(min(instance_numbers), max(instance_numbers) + 1))
                if instance_numbers
                else False,
                "series_description": _constant_or_values(series_records, "series_description"),
                "image_type": _constant_or_values(series_records, "image_type"),
                "rows": _constant_or_values(series_records, "rows"),
                "columns": _constant_or_values(series_records, "columns"),
                "pixel_spacing_mm": _constant_or_values(series_records, "pixel_spacing_mm"),
                "slice_thickness_mm": _constant_or_values(series_records, "slice_thickness_mm"),
                "bits_allocated": _constant_or_values(series_records, "bits_allocated"),
                "bits_stored": _constant_or_values(series_records, "bits_stored"),
                "high_bit": _constant_or_values(series_records, "high_bit"),
                "pixel_representation": _constant_or_values(series_records, "pixel_representation"),
                "is_unfiltered_nd_baseline": unfiltered,
            }
        )

    series_reports.sort(key=lambda item: item["first_filename"])
    baseline_indices = [
        index for index, report in enumerate(series_reports) if report["is_unfiltered_nd_baseline"]
    ]
    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "series_count": len(series_reports),
        "series": series_reports,
        "unfiltered_baseline_series_indices": baseline_indices,
        "dcmdump_stderr": completed.stderr.strip(),
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    """Print the key inspection findings and report location."""
    twix = report["twix"]
    selected = twix["measurements"][twix["selected_measurement_index"]]
    sampling = twix["selected_measurement_sampling"]
    image = selected["streams"]["image"]
    refscan = selected["streams"].get("refscan")
    print(f"TWIX measurements: {twix['measurement_count']} (selected {twix['selected_measurement_index']})")
    print(f"Matrix from header: {selected['header']['base_resolution']} x {sampling['matrix_pe1']} x {sampling['matrix_pe2']}")
    print(f"Image stream: {image['acquisition_count']} acquisitions, {image['coil_count']} coils")
    if refscan:
        print(f"Refscan stream: {refscan['acquisition_count']} acquisitions")
        print(
            "Refscan support: "
            f"{len(sampling['refscan_unique_pe1_lines'])} PE1 lines x "
            f"{len(sampling['refscan_unique_pe2_partitions'])} PE2 partitions"
        )
    print(
        "Image PE1 stride/residue: "
        f"{sampling['image_inferred_pe1_stride']} / "
        f"{sampling['image_pe1_residues_for_inferred_stride']}"
    )
    print(f"DICOM series: {report['dicom']['series_count']}; unfiltered baseline indices: {report['dicom']['unfiltered_baseline_series_indices']}")
    print(f"Report: {report['report_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the requested inputs and write the reproducible JSON report."""
    args = _build_parser().parse_args(argv)
    twix_path = args.twix.expanduser().resolve()
    dicom_path = args.dicom_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not twix_path.is_file():
        raise FileNotFoundError(f"TWIX file not found: {twix_path}")
    if not dicom_path.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {dicom_path}")

    report = {
        "format_version": 1,
        "pipeline_step": "data and mask verification",
        "twix": inspect_twix(twix_path, probe_samples=args.probe_samples),
        "dicom": inspect_dicom_directory(dicom_path),
        "report_path": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
