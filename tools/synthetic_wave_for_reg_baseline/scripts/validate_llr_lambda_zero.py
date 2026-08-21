#!/usr/bin/env python3
"""Gate split-complex LLR lambda zero against native-complex FISTA lambda zero."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic
from run_bart_regularization import _relative_bart_difference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-relative-l2", type=float, default=1e-5)
    return parser


def _load_complete(path: Path, regularizer: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete reconstruction manifest: {path}")
    if config.get("regularizer") != regularizer or float(config.get("lambda", -1)) != 0:
        raise ValueError(f"Expected {regularizer} lambda-zero manifest: {path}")
    if config.get("backend") != "gpu":
        raise ValueError(f"Lambda-zero reconstruction did not use GPU BART: {path}")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    split_path = args.split_manifest.expanduser().resolve()
    native_path = args.native_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    split = _load_complete(split_path, "llr")
    native = _load_complete(native_path, "wavelet")
    if split["config"].get("block_size") != 8:
        raise ValueError("The presentation LLR lambda-zero gate requires block size 8.")
    for field in ("dataset_manifest_sha256", "lambda_zero_manifest_sha256"):
        if split["config"].get(field) != native["config"].get(field):
            raise ValueError(f"Lambda-zero provenance mismatch for {field}.")
    if split.get("maps", {}).get("cfl_sha256") != native.get("maps", {}).get("cfl_sha256"):
        raise ValueError("Lambda-zero cases use different ESPIRiT maps.")

    difference = _relative_bart_difference(
        Path(split["bart_output"]["base"]), Path(native["bart_output"]["base"])
    )
    report = {
        "format_version": 1,
        "status": "accepted" if difference <= args.maximum_relative_l2 else "rejected",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "split-complex LLR lambda-zero equivalence gate",
        "maximum_relative_l2": args.maximum_relative_l2,
        "measured_relative_l2": difference,
        "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "native_manifest": {"path": str(native_path), "sha256": sha256_file(native_path)},
    }
    write_json_atomic(output_path, report)
    print(f"LLR lambda-zero equivalence: {difference:.12g}")
    print(f"Gate report: {output_path}")
    if report["status"] != "accepted":
        raise ValueError(
            f"LLR lambda-zero equivalence failed: {difference:.12g} > "
            f"{args.maximum_relative_l2:.12g}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
