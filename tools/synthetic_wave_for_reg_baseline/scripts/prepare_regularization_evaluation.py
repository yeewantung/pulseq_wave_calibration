#!/usr/bin/env python3
"""Consolidate a completed regularization sweep and convert one exact DICOM series."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nifti_sidecar(path: Path) -> Path:
    """Return the JSON sidecar path for a compressed or uncompressed NIfTI."""
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7] + ".json")
    if name.endswith(".nii"):
        return path.with_suffix(".json")
    raise ValueError(f"Not a NIfTI path: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_niftis(manifest_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize lambda-zero and regularized NIfTI records."""
    if "nifti_outputs" in manifest:
        raw_records = manifest["nifti_outputs"]
        normalized = []
        for record in raw_records:
            normalized.append(
                {
                    "part": record["part"],
                    "path": Path(record["nifti"]),
                    "expected_sha256": record.get("nifti_sha256"),
                    "json": Path(record["json"]),
                    "expected_json_sha256": record.get("json_sha256"),
                }
            )
        return normalized

    normalized = []
    for record in manifest.get("outputs", []):
        path = Path(record.get("path", ""))
        if path.name.endswith((".nii", ".nii.gz")):
            part = "phase" if "part-phase" in path.name else "mag"
            normalized.append(
                {
                    "part": part,
                    "path": path,
                    "expected_sha256": record.get("sha256"),
                    "json": nifti_sidecar(path),
                    "expected_json_sha256": None,
                }
            )
    if not normalized:
        raise ValueError(f"No NIfTI outputs found in {manifest_path}")
    return normalized


def discover_cases(recon_root: Path) -> list[dict[str, Any]]:
    """Discover lambda-zero and all complete regularization manifests."""
    manifests = [recon_root / "bart_lambda0_existing_csm_c050" / "manifest.json"]
    manifests.extend(sorted((recon_root / "regularization").glob("*/manifest.json")))
    cases: list[dict[str, Any]] = []
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"Manifest is not complete: {manifest_path}")
        is_lambda_zero = manifest_path.parent.name == "bart_lambda0_existing_csm_c050"
        config = manifest.get("config", {})
        case = {
            "case": "lambda0" if is_lambda_zero else manifest_path.parent.name,
            "kind": "lambda0" if is_lambda_zero else config.get("regularizer"),
            "lambda": 0.0 if is_lambda_zero else config.get("lambda"),
            "lambda_label": "0" if is_lambda_zero else config.get("lambda_label"),
            "block_size": None if is_lambda_zero else config.get("block_size"),
            "backend": manifest.get("backend") if is_lambda_zero else config.get("backend"),
            "manifest": manifest_path,
            "manifest_payload": manifest,
            "niftis": _manifest_niftis(manifest_path, manifest),
        }
        parts = sorted(record["part"] for record in case["niftis"])
        if parts != ["mag", "phase"]:
            raise ValueError(f"Expected mag/phase pair in {manifest_path}, found {parts}")
        cases.append(case)
    names = [case["case"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate case names discovered: {names}")
    return cases


def _nifti_metadata(path: Path) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    return {
        "shape": [int(value) for value in image.shape],
        "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "storage_dtype": str(image.get_data_dtype()),
    }


def _copy_verified(source: Path, destination: Path, expected_hash: str | None) -> dict[str, Any]:
    """Verify a source, copy it idempotently, and prove byte identity."""
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256_file(source)
    if expected_hash and source_hash != expected_hash:
        raise ValueError(
            f"Source hash disagrees with reconstruction manifest for {source}: "
            f"{source_hash} != {expected_hash}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    action = "reused"
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise FileExistsError(
                f"Existing destination is not byte-identical; refusing to overwrite: {destination}"
            )
    else:
        shutil.copy2(source, destination)
        action = "copied"
    destination_hash = sha256_file(destination)
    if destination_hash != source_hash:
        raise OSError(f"Copied file hash mismatch: {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
        "action": action,
    }


def consolidate_cases(
    cases: Iterable[dict[str, Any]], nifti_root: Path
) -> list[dict[str, Any]]:
    """Copy every reconstruction NIfTI/sidecar pair into one case-organized tree."""
    inventory = []
    for case in cases:
        destination_dir = nifti_root / case["case"]
        outputs = []
        for record in case["niftis"]:
            nifti_record = _copy_verified(
                record["path"],
                destination_dir / record["path"].name,
                record["expected_sha256"],
            )
            json_record = _copy_verified(
                record["json"],
                destination_dir / record["json"].name,
                record["expected_json_sha256"],
            )
            nifti_record.update(_nifti_metadata(Path(nifti_record["destination"])))
            outputs.append(
                {"part": record["part"], "nifti": nifti_record, "json": json_record}
            )
        inventory.append(
            {
                "case": case["case"],
                "kind": case["kind"],
                "lambda": case["lambda"],
                "lambda_label": case["lambda_label"],
                "block_size": case["block_size"],
                "backend": case["backend"],
                "source_manifest": str(case["manifest"]),
                "source_manifest_sha256": sha256_file(case["manifest"]),
                "outputs": outputs,
            }
        )
    return inventory


def scan_dicom_series(dicom_dir: Path) -> dict[str, list[tuple[Path, Any]]]:
    """Read header-only DICOM metadata and group files by SeriesInstanceUID."""
    import pydicom

    groups: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    tags = [
        "SeriesInstanceUID",
        "SeriesDescription",
        "InstanceNumber",
        "SOPInstanceUID",
        "ImageType",
        "Rows",
        "Columns",
        "PixelSpacing",
        "SliceThickness",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PatientPosition",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation",
    ]
    paths = sorted(path for path in dicom_dir.iterdir() if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files found in DICOM directory: {dicom_dir}")
    for path in paths:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        except Exception as error:  # malformed/non-DICOM files are not silently accepted
            raise ValueError(f"Could not read DICOM header {path}: {error}") from error
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not uid:
            raise ValueError(f"DICOM is missing SeriesInstanceUID: {path}")
        groups[uid].append((path, dataset))
    return dict(groups)


def select_dicom_series(
    groups: dict[str, list[tuple[Path, Any]]],
    expected_uid: str,
    expected_description: str,
    expected_count: int,
) -> tuple[list[tuple[Path, Any]], list[dict[str, Any]]]:
    """Select exactly one normalized, unfiltered ND magnitude series."""
    summary = []
    for uid, records in sorted(groups.items()):
        first = records[0][1]
        summary.append(
            {
                "series_instance_uid": uid,
                "series_description": str(getattr(first, "SeriesDescription", "")),
                "image_type": [str(value) for value in getattr(first, "ImageType", [])],
                "file_count": len(records),
            }
        )
    if expected_uid not in groups:
        raise ValueError(f"Requested DICOM SeriesInstanceUID not found: {expected_uid}")
    selected = groups[expected_uid]
    if len(selected) != expected_count:
        raise ValueError(
            f"Expected {expected_count} DICOMs in selected series, found {len(selected)}"
        )
    expected_image_type = ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
    instance_numbers = []
    sop_uids = set()
    for path, dataset in selected:
        description = str(getattr(dataset, "SeriesDescription", ""))
        image_type = [str(value) for value in getattr(dataset, "ImageType", [])]
        if description != expected_description:
            raise ValueError(f"Unexpected SeriesDescription in {path}: {description}")
        if image_type != expected_image_type or any(
            component in {"DIS2D", "DIS3D"} for component in image_type
        ):
            raise ValueError(
                "Selected series is not normalized, unfiltered ND magnitude "
                f"in {path}: {image_type}"
            )
        if int(dataset.Rows) != 256 or int(dataset.Columns) != 256:
            raise ValueError(f"Unexpected DICOM matrix in {path}: {dataset.Rows}x{dataset.Columns}")
        if int(dataset.BitsAllocated) != 16 or int(dataset.BitsStored) != 12:
            raise ValueError(f"Unexpected DICOM bit depth in {path}")
        if int(dataset.PixelRepresentation) != 0:
            raise ValueError(f"Expected unsigned DICOM pixels in {path}")
        instance_numbers.append(int(dataset.InstanceNumber))
        sop_uid = str(dataset.SOPInstanceUID)
        if sop_uid in sop_uids:
            raise ValueError(f"Duplicate SOPInstanceUID in selected series: {sop_uid}")
        sop_uids.add(sop_uid)
    if sorted(instance_numbers) != list(range(1, expected_count + 1)):
        raise ValueError("Selected DICOM InstanceNumber values are not contiguous from 1")
    selected.sort(key=lambda pair: int(pair[1].InstanceNumber))
    return selected, summary


def stage_selected_dicoms(
    selected: Iterable[tuple[Path, Any]], staging_dir: Path
) -> list[dict[str, Any]]:
    """Create an exact symlink-only series directory for dcm2niix."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    expected_names = set()
    records = []
    for path, dataset in selected:
        name = f"instance-{int(dataset.InstanceNumber):04d}.dcm"
        expected_names.add(name)
        link = staging_dir / name
        if link.is_symlink():
            if link.resolve() != path.resolve():
                raise FileExistsError(f"Existing staging link has wrong target: {link}")
            action = "reused"
        elif link.exists():
            raise FileExistsError(f"Non-symlink exists in DICOM staging directory: {link}")
        else:
            link.symlink_to(path.resolve())
            action = "linked"
        records.append(
            {
                "instance_number": int(dataset.InstanceNumber),
                "sop_instance_uid": str(dataset.SOPInstanceUID),
                "source": str(path),
                "source_sha256": sha256_file(path),
                "staging_link": str(link),
                "action": action,
            }
        )
    unexpected = [path for path in staging_dir.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(f"Unexpected files in DICOM staging directory: {unexpected}")
    return records


def convert_dicom(
    dcm2niix: Path, staging_dir: Path, output_dir: Path
) -> tuple[list[str], str, Path, Path]:
    """Convert the staged exact series with orientation-preserving settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(dcm2niix),
        "-b", "y",
        "-ba", "y",
        "-d", "0",
        "-f", "dicom_normalized_unfiltered_nd",
        "-l", "o",
        "-m", "n",
        "-q", "n",
        "-w", "1",
        "-x", "i",
        "-z", "i",
        "-o", str(output_dir),
        str(staging_dir),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    combined = process.stdout + process.stderr
    if process.returncode:
        raise RuntimeError(
            f"dcm2niix failed with status {process.returncode}:\n{combined}"
        )
    niftis = sorted(output_dir.glob("dicom_normalized_unfiltered_nd*.nii.gz"))
    if len(niftis) != 1:
        raise ValueError(f"Expected exactly one converted DICOM NIfTI, found {niftis}")
    sidecar = nifti_sidecar(niftis[0])
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    return command, combined, niftis[0], sidecar


def canonicalize_to_ras(source: Path, destination: Path) -> Path:
    """Save a dtype-preserving copy on the nearest canonical RAS voxel grid."""
    import nibabel as nib
    import numpy as np

    image = nib.load(str(source))
    canonical = nib.as_closest_canonical(image)
    if tuple(nib.aff2axcodes(canonical.affine)) != ("R", "A", "S"):
        raise ValueError(f"Could not canonicalize DICOM reference to RAS: {source}")
    if destination.exists():
        raise FileExistsError(f"Canonical DICOM output already exists: {destination}")
    header = canonical.header.copy()
    header.set_data_dtype(image.get_data_dtype())
    data = np.asanyarray(canonical.dataobj)
    nib.save(nib.Nifti1Image(data, canonical.affine, header), str(destination))
    return destination


def _dicom_series_metadata(selected: list[tuple[Path, Any]]) -> dict[str, Any]:
    first = selected[0][1]
    last = selected[-1][1]
    return {
        "series_instance_uid": str(first.SeriesInstanceUID),
        "series_description": str(first.SeriesDescription),
        "image_type": [str(value) for value in first.ImageType],
        "file_count": len(selected),
        "instance_number_range": [int(first.InstanceNumber), int(last.InstanceNumber)],
        "matrix": [int(first.Rows), int(first.Columns), len(selected)],
        "pixel_spacing_mm": [float(value) for value in first.PixelSpacing],
        "slice_thickness_mm": float(first.SliceThickness),
        "image_orientation_patient": [float(value) for value in first.ImageOrientationPatient],
        "first_image_position_patient": [float(value) for value in first.ImagePositionPatient],
        "last_image_position_patient": [float(value) for value in last.ImagePositionPatient],
        "patient_position": str(first.PatientPosition),
        "bits_allocated": int(first.BitsAllocated),
        "bits_stored": int(first.BitsStored),
        "pixel_representation": int(first.PixelRepresentation),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-root", required=True, type=Path)
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--dicom-series-uid", required=True)
    parser.add_argument("--dicom-series-description", default="t1_mprage_sag_p2_ND")
    parser.add_argument("--expected-cases", type=int, default=25)
    parser.add_argument("--expected-dicom-count", type=int, default=256)
    parser.add_argument("--dcm2niix", type=Path, default=Path("/path/to/software/bin/dcm2niix"))
    parser.add_argument(
        "--nifti-root",
        type=Path,
        help="Destination; defaults to <recon-root>/nifti.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    recon_root = args.recon_root.expanduser().resolve()
    dicom_dir = args.dicom_dir.expanduser().resolve()
    nifti_root = (
        args.nifti_root.expanduser().resolve()
        if args.nifti_root
        else recon_root / "nifti"
    )
    if not recon_root.is_dir() or not dicom_dir.is_dir():
        raise FileNotFoundError("The reconstruction root or DICOM directory is missing")
    if not args.dcm2niix.is_file():
        raise FileNotFoundError(args.dcm2niix)

    cases = discover_cases(recon_root)
    if len(cases) != args.expected_cases:
        raise ValueError(f"Expected {args.expected_cases} reconstruction cases, found {len(cases)}")
    inventory = consolidate_cases(cases, nifti_root)

    groups = scan_dicom_series(dicom_dir)
    selected, series_summary = select_dicom_series(
        groups,
        args.dicom_series_uid,
        args.dicom_series_description,
        args.expected_dicom_count,
    )
    staging_dir = recon_root / "evaluation" / "dicom_normalized_unfiltered_nd_selection"
    selected_records = stage_selected_dicoms(selected, staging_dir)
    reference_dir = nifti_root / "dicom_reference"
    command, converter_output, converted_dicom_nifti, dicom_json = convert_dicom(
        args.dcm2niix.resolve(), staging_dir, reference_dir
    )
    dicom_nifti = canonicalize_to_ras(
        converted_dicom_nifti,
        reference_dir / "dicom_normalized_unfiltered_nd_ras.nii.gz",
    )
    dicom_nifti_record = {
        "path": str(dicom_nifti),
        "sha256": sha256_file(dicom_nifti),
        "size_bytes": dicom_nifti.stat().st_size,
        **_nifti_metadata(dicom_nifti),
    }
    if dicom_nifti_record["shape"] != [256, 256, 256]:
        raise ValueError(f"Unexpected converted DICOM shape: {dicom_nifti_record['shape']}")
    if any(abs(value - 1.0) > 1e-5 for value in dicom_nifti_record["voxel_size_mm"]):
        raise ValueError(
            f"Unexpected converted DICOM voxel size: {dicom_nifti_record['voxel_size_mm']}"
        )

    version_process = subprocess.run(
        [str(args.dcm2niix.resolve()), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    converter_version = (version_process.stdout + version_process.stderr).strip()
    if not converter_version:
        raise RuntimeError("dcm2niix did not report a version string")
    manifest = {
        "format_version": 1,
        "status": "orientation_review_pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reconstruction_root": str(recon_root),
        "nifti_root": str(nifti_root),
        "case_count": len(inventory),
        "reconstruction_cases": inventory,
        "dicom_source_directory": str(dicom_dir),
        "dicom_series_discovered": series_summary,
        "selected_dicom_series": _dicom_series_metadata(selected),
        "selected_dicom_files": selected_records,
        "dicom_staging_directory": str(staging_dir),
        "dcm2niix": {
            "executable": str(args.dcm2niix.resolve()),
            "version": converter_version,
            "version_command_returncode": version_process.returncode,
            "command": command,
            "combined_output": converter_output,
            "converted_nifti_before_ras_canonicalization": str(converted_dicom_nifti),
        },
        "dicom_reference_nifti": dicom_nifti_record,
        "dicom_reference_json": {
            "path": str(dicom_json),
            "sha256": sha256_file(dicom_json),
            "size_bytes": dicom_json.stat().st_size,
        },
    }
    manifest_path = recon_root / "evaluation" / "evaluation_inputs_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        reference_dir / "dicom_series_selection.json",
        {
            "selected_dicom_series": manifest["selected_dicom_series"],
            "source_directory": str(dicom_dir),
            "selection_rule": {
                "SeriesInstanceUID": args.dicom_series_uid,
                "SeriesDescription": args.dicom_series_description,
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
                "expected_file_count": args.expected_dicom_count,
            },
            "excluded_series": [
                item
                for item in series_summary
                if item["series_instance_uid"] != args.dicom_series_uid
            ],
        },
    )
    print(f"Consolidated {len(inventory)} cases below: {nifti_root}")
    print(f"Converted exact unfiltered DICOM series: {dicom_nifti}")
    print(f"Evaluation input manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
