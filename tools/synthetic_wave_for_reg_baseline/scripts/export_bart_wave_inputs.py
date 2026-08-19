#!/usr/bin/env python3
"""Apply the verified R3x1 product mask and finalize BART Wave inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sampling_mask import load_product_mask, write_masked_bart_kspace
from wave_synthesis import logical_array_sha256, logical_bart_cfl_sha256, sha256_file


def _build_parser() -> argparse.ArgumentParser:
    """Build the BART Wave input finalization command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-dir", required=True, type=Path)
    parser.add_argument("--sampling-report", required=True, type=Path)
    parser.add_argument(
        "--visual-review-approved",
        action="store_true",
        help="Required acknowledgement that the full-Wave diagnostics were approved.",
    )
    parser.add_argument("--pe2-chunk", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file to avoid partial manifests."""
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Apply the verified product mask and finalize BART input provenance."""
    if not args.visual_review_approved:
        raise ValueError("Refusing to apply the product mask before visual-review approval.")
    synthesis_dir = args.synthesis_dir.expanduser().resolve()
    sampling_report = args.sampling_report.expanduser().resolve()
    manifest_path = synthesis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mask, mask_info, inspection_report = load_product_mask(sampling_report)
    report_twix = Path(inspection_report["twix"]["path"]).resolve()
    synthesis_twix = Path(manifest["source_twix"]).resolve()
    if report_twix != synthesis_twix:
        raise ValueError(
            f"Sampling report TWIX {report_twix} differs from synthesis TWIX {synthesis_twix}."
        )
    if mask.shape != tuple(manifest["full_wave_kspace"]["shape"][1:3]):
        raise ValueError("Product mask dimensions do not match the full Wave PE grid.")

    mask_path = synthesis_dir / "r3x1_product_sampling_mask.npy"
    if mask_path.exists() and not args.overwrite:
        raise FileExistsError(mask_path)
    np.save(mask_path, mask)
    mask_hash = logical_array_sha256(mask)

    bart_dir = synthesis_dir / "bart_inputs"
    kspace_info = write_masked_bart_kspace(
        manifest["full_wave_kspace"]["path"],
        mask,
        bart_dir / "wave_kspace",
        pe2_chunk=args.pe2_chunk,
        overwrite=args.overwrite,
    )

    psf_info = manifest["psf"]
    psf = np.load(psf_info["npy"], mmap_mode="r")
    canonical_hash = logical_array_sha256(psf)
    bart_shape = tuple(int(value) for value in psf_info["bart_shape"])
    bart_hash = logical_bart_cfl_sha256(psf_info["bart_base"], bart_shape)
    if canonical_hash != bart_hash or canonical_hash != psf_info["logical_sha256"]:
        raise ValueError("The canonical and BART theoretical PSF payloads no longer match.")

    finalized_at = datetime.now(timezone.utc).isoformat()
    mask_provenance = {
        **mask_info,
        "path": str(mask_path),
        "dtype": str(mask.dtype),
        "logical_sha256": mask_hash,
        "sampling_report": str(sampling_report),
        "sampling_report_sha256": sha256_file(sampling_report),
        "selected_measurement_index": int(
            inspection_report["twix"]["selected_measurement_index"]
        ),
    }
    bart_manifest = {
        "format_version": 1,
        "format": "BART CFL",
        "status": "masked_wave_inputs_ready_for_map_estimation_and_reconstruction",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "visual_review": {
            "approved": True,
            "approval_source": "user confirmation after reviewing full-Wave diagnostics",
        },
        "sampling_mask": mask_provenance,
        "full_wave_kspace": manifest["full_wave_kspace"],
        "masked_wave_kspace": kspace_info,
        "psf": {
            "basename": "psf",
            "shape": list(bart_shape),
            "logical_sha256": bart_hash,
            "identical_to_synthesis_psf": True,
        },
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": kspace_info["shape"],
                "wave_kspace_norm": kspace_info["norm"],
                "psf": "psf",
                "psf_shape": list(bart_shape),
            }
        ],
        "finalized_at_utc": finalized_at,
    }
    bart_manifest_path = bart_dir / "manifest.json"
    _write_json(bart_manifest_path, bart_manifest)

    manifest["status"] = "masked_bart_inputs_ready"
    manifest["visual_review"] = {
        "approved": True,
        "approval_source": "user confirmation after reviewing full-Wave diagnostics",
    }
    manifest["sampling_mask"] = mask_provenance
    manifest["masked_wave_kspace"] = kspace_info
    manifest["bart_input_manifest"] = str(bart_manifest_path)
    manifest["finalized_at_utc"] = finalized_at
    _write_json(manifest_path, manifest)
    print(f"BART input manifest: {bart_manifest_path}")
    return bart_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run BART input finalization from command-line arguments."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc
