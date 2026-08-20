#!/usr/bin/env python3
"""Export a validated retrospective Cartesian mask from reusable full Wave data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sampling_mask import retrospective_cartesian_mask, write_masked_bart_kspace
from wave_synthesis import logical_array_sha256, logical_bart_cfl_sha256, sha256_file


def _build_parser() -> argparse.ArgumentParser:
    """Build the retrospective Wave-input export command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-synthesis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pe1-acceleration", type=int, default=3)
    parser.add_argument("--pe2-acceleration", type=int, default=2)
    parser.add_argument("--pe1-residue", type=int, default=1)
    parser.add_argument("--pe2-residue", type=int, default=0)
    parser.add_argument("--acs-pe1-start", type=int, default=115)
    parser.add_argument("--acs-pe1-stop", type=int, default=139)
    parser.add_argument("--tag", default="r3x2")
    parser.add_argument("--pe2-chunk", type=int, default=8)
    parser.add_argument(
        "--full-wave-review-approved",
        action="store_true",
        help="Acknowledge that the reusable full-Wave diagnostics were approved.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file."""
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _link_pair(source_base: Path, output_base: Path, *, replace: bool) -> None:
    """Create an explicit BART CFL symlink pair without duplicating the PSF."""
    for suffix in (".hdr", ".cfl"):
        source = source_base.with_suffix(suffix)
        output = output_base.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(source)
        if output.exists() or output.is_symlink():
            if not replace:
                raise FileExistsError(output)
            output.unlink()
        output.symlink_to(source)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the exact mask configuration used for resume matching."""
    return {
        "tag": args.tag,
        "pe1_acceleration": args.pe1_acceleration,
        "pe2_acceleration": args.pe2_acceleration,
        "pe1_residue": args.pe1_residue,
        "pe2_residue": args.pe2_residue,
        "acs_pe1_start": args.acs_pe1_start,
        "acs_pe1_stop_exclusive": args.acs_pe1_stop,
    }


def _completed_reusable(
    manifest_path: Path,
    config: dict[str, Any],
    source_synthesis_dir: Path,
) -> bool:
    """Require matching provenance and intact hashes before reusing an export."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "retrospective_bart_inputs_ready":
            return False
        if manifest.get("config") != config:
            return False
        if Path(manifest["source_synthesis_dir"]).resolve() != source_synthesis_dir:
            return False
        mask = Path(manifest["sampling_mask"]["path"])
        kspace = Path(manifest["masked_wave_kspace"]["cfl"])
        psf = Path(manifest["psf"]["cfl"])
        return (
            mask.is_file()
            and kspace.is_file()
            and psf.is_file()
            and logical_array_sha256(np.load(mask, mmap_mode="r"))
            == manifest["sampling_mask"]["logical_sha256"]
            and sha256_file(kspace) == manifest["masked_wave_kspace"]["cfl_sha256"]
            and sha256_file(psf) == manifest["psf"]["cfl_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build the mask and export a separate, validated BART input tree."""
    if not args.full_wave_review_approved:
        raise ValueError("Refusing export without full-Wave visual-review approval.")
    if args.acs_pe1_stop <= args.acs_pe1_start:
        raise ValueError("--acs-pe1-stop must exceed --acs-pe1-start.")
    source_dir = args.source_synthesis_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_full_wave = Path(source_manifest["full_wave_kspace"]["path"]).resolve()
    source_psf_base = Path(source_manifest["psf"]["bart_base"]).resolve()
    source_shape = tuple(int(value) for value in source_manifest["full_wave_kspace"]["shape"])
    if source_shape != (1024, 256, 256, 12):
        raise ValueError(f"Expected reusable full Wave shape [1024,256,256,12], got {source_shape}.")
    if not source_full_wave.is_file():
        raise FileNotFoundError(source_full_wave)

    config = _config(args)
    manifest_path = output_dir / "manifest.json"
    if args.resume and _completed_reusable(manifest_path, config, source_dir):
        print(f"Reusing validated retrospective BART inputs: {output_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    recover_incomplete = False
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            recover_incomplete = (
                prior.get("status") == "exporting_retrospective_bart_inputs"
                and prior.get("config") == config
                and Path(prior.get("source_synthesis_dir", "")).resolve() == source_dir
            )
        if not recover_incomplete:
            raise FileExistsError(f"Output directory is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = {
        "format_version": 1,
        "status": "exporting_retrospective_bart_inputs",
        "config": config,
        "source_synthesis_dir": str(source_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, started)

    mask, mask_info = retrospective_cartesian_mask(
        source_shape[1:3],
        accelerations=(args.pe1_acceleration, args.pe2_acceleration),
        residues=(args.pe1_residue, args.pe2_residue),
        fully_sampled_pe1_lines=np.arange(args.acs_pe1_start, args.acs_pe1_stop),
    )
    mask_path = output_dir / f"{args.tag}_sampling_mask.npy"
    if mask_path.exists() and not recover_incomplete:
        raise FileExistsError(mask_path)
    np.save(mask_path, mask)
    mask_info.update(
        {
            "path": str(mask_path),
            "dtype": str(mask.dtype),
            "logical_sha256": logical_array_sha256(mask),
        }
    )

    bart_dir = output_dir / "bart_inputs"
    bart_dir.mkdir(exist_ok=True)
    kspace_info = write_masked_bart_kspace(
        source_full_wave,
        mask,
        bart_dir / "wave_kspace",
        pe2_chunk=args.pe2_chunk,
        overwrite=recover_incomplete,
    )
    kspace_info["cfl_sha256"] = sha256_file(Path(kspace_info["cfl"]))

    output_psf_base = bart_dir / "psf"
    _link_pair(source_psf_base, output_psf_base, replace=recover_incomplete)
    psf_shape = tuple(int(value) for value in source_manifest["psf"]["bart_shape"])
    psf_logical_hash = logical_bart_cfl_sha256(output_psf_base, psf_shape)
    expected_psf_hash = source_manifest["psf"]["logical_sha256"]
    if psf_logical_hash != expected_psf_hash:
        raise ValueError("Reused BART PSF differs from the full-Wave synthesis PSF.")
    psf_info = {
        "basename": "psf",
        "base": str(output_psf_base),
        "header": str(output_psf_base.with_suffix(".hdr")),
        "cfl": str(output_psf_base.with_suffix(".cfl")),
        "shape": list(psf_shape),
        "logical_sha256": psf_logical_hash,
        "cfl_sha256": sha256_file(output_psf_base.with_suffix(".cfl")),
        "symlink_source_base": str(source_psf_base),
        "identical_to_source_synthesis_psf": True,
    }

    completed_at = datetime.now(timezone.utc).isoformat()
    bart_manifest = {
        "format_version": 1,
        "format": "BART CFL",
        "status": "masked_wave_inputs_ready_for_reconstruction_with_existing_maps",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "sampling_mask": mask_info,
        "full_wave_kspace": source_manifest["full_wave_kspace"],
        "masked_wave_kspace": kspace_info,
        "psf": psf_info,
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": kspace_info["shape"],
                "wave_kspace_norm": kspace_info["norm"],
                "psf": "psf",
                "psf_shape": list(psf_shape),
            }
        ],
        "finalized_at_utc": completed_at,
    }
    _write_json(bart_dir / "manifest.json", bart_manifest)

    manifest = {
        **started,
        "status": "retrospective_bart_inputs_ready",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "sampling_mask": mask_info,
        "masked_wave_kspace": kspace_info,
        "psf": psf_info,
        "bart_input_manifest": str(bart_dir / "manifest.json"),
        "completed_at_utc": completed_at,
    }
    _write_json(manifest_path, manifest)
    print(f"Retrospective BART input manifest: {bart_dir / 'manifest.json'}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run retrospective BART input export from command-line arguments."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc
