#!/usr/bin/env python3
"""Audit relocated standard MPRAGE inputs and write an explicit immutable receipt.

Only run after the user confirms the reconstruction root and receipt destination.
This does not move arrays, edit original manifests, or launch reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shlex
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wave_retro_lr.metal_workflow import validate_cfl
from wave_retro_lr.relocation import RECEIPT, _view, _command_view, checked_hash
from wave_retro_lr.standard import validate_prepared_artifact_records, validate_standard_pca_manifest


def build_receipt(root: Path, prefixes: list[str], count: int) -> dict:
    """Audit a moved vccN input tree and return a receipt without writing files.

    Require the historical raw size/mtime and current content hash, verify all
    prepared artifact hashes and finite CFLs, and verify completed CSM lineage.
    Old raw hashes cannot be reconstructed: explicitly label that limitation.
    """
    root = root.resolve(strict=True)
    variant = root / f"vcc{count}"
    normal_path = variant / "normal/bart_inputs/manifest.json"
    normal_bytes = normal_path.read_bytes()
    original = json.loads(normal_bytes)
    receipt = {"schema": "mprage-path-relocation-v1", "root": str(root),
               "target_dataset": str(root.parent), "source_prefixes": prefixes,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "documents": {}, "raw_sources": [], "validated_artifacts": {}}
    for prefix in prefixes:
        if (not Path(prefix).is_absolute() or ".." in Path(prefix).parts
                or len(Path(prefix).parts) < 4 or prefix.rstrip("/") != prefix
                or prefix == str(root.parent)):
            raise ValueError("Supply an exact historical dataset prefix, not a broad filesystem root")
    normal = _view(original, receipt)
    raw = normal["source"]["twix"]
    if raw["path"] == original["source"]["twix"]["path"] or raw.get("sha256"):
        raise ValueError("Expected a mapped historical raw identity with size/mtime only")
    raw_path = Path(raw["path"]).resolve(strict=True)
    if raw_path.parent != root.parent:
        raise ValueError("Raw must be a file in the confirmed dataset directory")
    stat = raw_path.stat()
    if stat.st_size != raw["size_bytes"] or stat.st_mtime_ns != raw["mtime_ns"]:
        raise ValueError("Relocated raw size/mtime differs from historical source")
    print("Checking current raw SHA-256 (historical raw hash unavailable)", flush=True)
    receipt["raw_sources"].append({**raw, "sha256": checked_hash(raw_path),
        "historical_sha256_available": False,
        "historical_path": original["source"]["twix"]["path"],
        "verification": "user-confirmed-relocation-plus-size-mtime; current-sha256"})
    sequence = normal["source"]["sequence"]
    if checked_hash(Path(sequence["path"])) != sequence["sha256"]:
        raise ValueError("Sequence hash mismatch")
    validate_standard_pca_manifest(normal, count)
    # Hash every historical document whose view may be consumed later, but do
    # not require pruned reconstruction images/NIfTIs merely to reuse CSMs.
    documents = sorted(set(variant.rglob("*.json")) | set(variant.rglob("*_command.txt")))
    for path in documents:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Historical document symlink escapes reconstruction root")
        receipt["documents"][str(path.relative_to(root))] = checked_hash(path)
    prepared = [normal_path] + sorted((variant / "retro").glob("*/bart_inputs/manifest.json"))
    for path in prepared:
        manifest = _view(json.loads(path.read_text()), receipt)
        if manifest["source"] != normal["source"]:
            raise ValueError(f"Prepared source differs from selected normal inputs: {path}")
        required = {"wave_kspace", "psf"}
        if path == normal_path:
            required.update(("kspace_calib", "coil_compression_basis.npy"))
        if not required.issubset(manifest["output_artifacts"]):
            raise ValueError("Missing required hash-bound prepared artifact records")
        print(f"Checking prepared artifacts: {path.relative_to(root)}", flush=True)
        # CFL validation below already checks both hashes, shape and finite
        # values; avoid a redundant full read of network-storage payloads.
        regular_files = {name: record for name, record in manifest["output_artifacts"].items()
                         if "payload_sha256" not in record}
        if regular_files:
            validate_prepared_artifact_records(path.parent, regular_files)
        for name, record in manifest["output_artifacts"].items():
            if "payload_sha256" in record:
                if Path(record["base"]).resolve() != (path.parent/name).resolve():
                    raise ValueError("Prepared artifact path does not match its tree")
                actual = validate_cfl(record)
                receipt["validated_artifacts"][str((path.parent/name).relative_to(root))] = actual
    branches = sorted((variant / "normal/bart_output").glob("*/reconstruction_manifest.json"))
    if not branches:
        raise ValueError("No completed standard branch to attest original CSM lineage")
    branch = _view(json.loads(branches[0].read_text()), receipt)
    if (branch["status"] != "standard_pca_reconstruction_branch_complete"
            or branch["request"]["normal_manifest"]["sha256"] != checked_hash(normal_path)
            or Path(branch["request"]["normal_manifest"]["path"]) != normal_path
            or branch["request"]["compression_basis"] != normal["coil_compression"]["basis"]):
        raise ValueError("Completed CSM lineage mismatch")
    maps = validate_cfl(branch["request"]["inputs"]["maps"])
    if Path(maps["base"]) != variant / "normal/bart_output/coil_sens":
        raise ValueError("Original CSM path mismatch")
    command_path = variant / "normal/bart_output/ecalib_command.txt"
    tokens = shlex.split(_command_view(command_path.read_text().strip(), receipt))
    if (len(tokens) != 8 or Path(tokens[0]).name != "bart" or tokens[1:5] != ["ecalib", "-m", "1", "-c"]
            or not 0 <= float(tokens[5]) <= 1
            or Path(tokens[6]) != normal_path.parent / "kspace_calib"
            or Path(tokens[7]) != Path(maps["base"])):
        raise ValueError("Original ecalib command does not bind relocated ACS/CSM")
    receipt["validated_artifacts"]["normal_csm"] = maps
    for relative, digest in receipt["documents"].items():
        if checked_hash(root/relative) != digest:
            raise ValueError("Historical document changed during audit")
    return receipt


def main() -> None:
    """Parse explicitly confirmed roots/prefixes and exclusively create one receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-prefix", action="append", required=True)
    parser.add_argument("--virtual-coils", type=int, default=24)
    args = parser.parse_args()
    destination = args.root.resolve()/RECEIPT
    if destination.exists():
        raise ValueError("Relocation receipt already exists; never overwrite it")
    receipt = build_receipt(args.root, args.source_prefix, args.virtual_coils)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Registered relocation: {destination}", flush=True)


if __name__ == "__main__":
    main()
