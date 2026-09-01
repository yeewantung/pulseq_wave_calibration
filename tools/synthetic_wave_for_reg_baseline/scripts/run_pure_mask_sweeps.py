#!/usr/bin/env python3
"""Run or validate manifest-backed coarse/fine pure-mask BART Wave sweeps."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pure_mask_rerun import (
    CASE_IDS,
    FINE_LAMBDA_POOL,
    LLR_BLOCK_SIZES,
    bart_base,
    build_wave_command,
    coarse_candidate_settings,
    load_json,
    open_cfl,
    resolve_config_path,
    sha256_file,
    validate_config,
    validate_pure_cartesian_image_lattice,
    write_json_atomic,
)

RETRO_TOOL_ROOT = Path(__file__).resolve().parents[2] / "wave_retro_lr_recon"
if str(RETRO_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRO_TOOL_ROOT))

from wave_retro_lr.bart_io import recombine_split_complex_cfl  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and validate or execute one sweep stage.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after successful validation or completion.
    """
    args = _parser().parse_args(argv)
    result = run_sweep(
        args.config,
        stage=args.stage,
        validate_only=args.validate_only,
        confirmed_output_root=args.confirm_output_root,
        resume=args.resume,
    )
    if args.validate_only:
        print(
            f"Validated {result['candidate_count']} manifest-backed {args.stage} "
            "candidates; no reconstruction was launched."
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the pure-mask sweep command-line interface.

    Returns:
        Parser for one coarse or fine stage.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("coarse", "fine"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _utc_now() -> str:
    """Return the current timezone-aware UTC timestamp.

    Returns:
        ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def lambda_label(value: float) -> str:
    """Format a finite nonnegative lambda for stable path names.

    Args:
        value: Lambda value.

    Returns:
        Compact scientific label without a plus sign or decimal point.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError("Lambda must be finite and nonnegative.")
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.8e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def candidate_name(setting: Mapping[str, Any]) -> str:
    """Build one deterministic candidate directory name.

    Args:
        setting: Method, lambda, and optional block size.

    Returns:
        Stable filesystem-safe candidate name.
    """
    method = str(setting["method"])
    value = lambda_label(float(setting["lambda"]))
    if method == "fista_lambda0":
        return "fista_lambda-0"
    if method == "wavelet":
        return f"wavelet_lambda-{value}"
    if method == "llr":
        return f"llr_block-{int(setting['block_size'])}_lambda-{value}"
    raise ValueError(f"Unknown candidate method: {method!r}.")


def _validated_preparation(validated: dict[str, Any]) -> dict[str, Any]:
    """Load and revalidate the completed pure-mask preparation manifest.

    Args:
        validated: Fresh local configuration validation result.

    Returns:
        Completed preparation manifest with verified case inputs.
    """
    root = Path(validated["layout"]["root"])
    manifest_path = root / "preparation_manifest.json"
    preparation = load_json(manifest_path, "pure-mask preparation manifest")
    if preparation.get("status") != "complete":
        raise ValueError("Pure-mask preparation is not complete.")
    if preparation.get("config", {}).get("immutable_contract_sha256") != validated[
        "config"
    ]["immutable_contract_sha256"]:
        raise ValueError("Pure-mask preparation uses a different immutable input contract.")
    if set(preparation.get("cases", {})) != set(CASE_IDS):
        raise ValueError("Pure-mask preparation does not contain exactly five cases.")
    for case_id in CASE_IDS:
        record = preparation["cases"][case_id]
        case_path = Path(record["case_manifest"])
        if sha256_file(case_path) != record["case_manifest_sha256"]:
            raise ValueError(f"Prepared case manifest changed: {case_path}")
        case = load_json(case_path, f"{case_id} prepared case")
        if case.get("status") != "pure_mask_bart_inputs_ready":
            raise ValueError(f"{case_id} BART inputs are not ready.")
        mask = np.load(case["sampling_mask"]["path"], allow_pickle=False)
        validate_pure_cartesian_image_lattice(mask, case["sampling_mask"])
        if case.get("calibration_kspace_included") is not False:
            raise ValueError(f"{case_id} improperly includes calibration in Wave inputs.")
        if case.get("ecalib_run_performed") is not False:
            raise ValueError(f"{case_id} does not record accepted CSM reuse.")
        for artifact in ("wave_kspace", "coil_sens", "psf"):
            artifact_record = case["bart_inputs"][artifact]
            payload = Path(artifact_record["base"]).with_suffix(".cfl")
            if sha256_file(payload) != artifact_record["payload_sha256"]:
                raise ValueError(f"Prepared {case_id} {artifact} payload changed.")
    return preparation


def _fine_settings(
    config: Mapping[str, Any], config_path: Path, output_root: Path
) -> dict[str, list[dict[str, Any]]]:
    """Validate explicitly reviewed fine settings and their coarse manifests.

    Args:
        config: Complete local rerun configuration.
        config_path: Configuration path used for relative resolution.
        output_root: Confirmed workflow root.

    Returns:
        Ordered per-case positive-lambda settings.
    """
    fine = config.get("fine_sweep")
    if not isinstance(fine, Mapping):
        raise ValueError("fine_sweep must be configured after coarse evaluation.")
    coarse_sweep_path = resolve_config_path(
        fine.get("coarse_sweep_manifest"), config_path.parent, "fine_sweep.coarse_sweep_manifest"
    )
    coarse_evaluation_path = resolve_config_path(
        fine.get("coarse_evaluation_manifest"),
        config_path.parent,
        "fine_sweep.coarse_evaluation_manifest",
    )
    coarse_sweep = load_json(coarse_sweep_path, "coarse sweep manifest")
    coarse_evaluation = load_json(coarse_evaluation_path, "coarse evaluation manifest")
    if coarse_sweep.get("status") != "complete" or coarse_sweep.get("stage") != "coarse":
        raise ValueError("Fine sweep requires one completed coarse sweep.")
    if coarse_evaluation.get("status") != "complete":
        raise ValueError("Fine sweep requires completed coarse evaluation.")
    if coarse_evaluation.get("stage") != "coarse":
        raise ValueError("Fine sweep requires a coarse-stage evaluation manifest.")
    scope = coarse_evaluation.get("scientific_scope", {})
    if scope.get("automatic_composite_selection_performed") is not False:
        raise ValueError("Coarse evaluation must explicitly prohibit automatic composite selection.")
    if coarse_sweep_path != output_root / "sweeps" / "coarse" / "sweep_manifest.json":
        raise ValueError("Fine sweep coarse manifest is outside the confirmed workflow root.")
    if coarse_evaluation_path != output_root / "evaluation" / "coarse" / "evaluation_manifest.json":
        raise ValueError("Fine sweep evaluation manifest is outside the workflow root.")
    evaluation_sweep = coarse_evaluation.get("sweep_manifest", {})
    if (
        Path(evaluation_sweep.get("path", "")).resolve() != coarse_sweep_path
        or evaluation_sweep.get("sha256") != sha256_file(coarse_sweep_path)
    ):
        raise ValueError("Coarse evaluation is not hash-bound to the declared coarse sweep.")
    coarse_keys = {
        (
            record["case_id"],
            record["setting"]["method"],
            record["setting"]["block_size"],
            float(record["setting"]["lambda"]),
        )
        for record in coarse_sweep["candidate_manifests"]
    }
    raw_cases = fine.get("cases")
    if not isinstance(raw_cases, Mapping) or set(raw_cases) != set(CASE_IDS):
        raise ValueError("fine_sweep.cases must explicitly contain all five cases.")
    allowed = set(FINE_LAMBDA_POOL)
    result: dict[str, list[dict[str, Any]]] = {}
    for case_id in CASE_IDS:
        raw_settings = raw_cases[case_id]
        if not isinstance(raw_settings, list) or not raw_settings:
            raise ValueError(f"fine_sweep.cases.{case_id} must be a nonempty list.")
        settings = []
        keys = set()
        for item in raw_settings:
            if not isinstance(item, Mapping):
                raise ValueError(f"{case_id} fine settings must be objects.")
            method = str(item.get("method"))
            block = None if item.get("block_size") is None else int(item["block_size"])
            value = float(item["lambda"])
            if value not in allowed or method not in {"wavelet", "llr"}:
                raise ValueError(f"{case_id} fine setting is outside the approved pool.")
            if (method == "wavelet" and block is not None) or (
                method == "llr" and block not in LLR_BLOCK_SIZES
            ):
                raise ValueError(f"{case_id} fine setting has an invalid block contract.")
            key = (method, block, value)
            if key in keys:
                raise ValueError(f"{case_id} repeats fine setting {key}.")
            if (case_id, *key) in coarse_keys:
                raise ValueError(
                    f"{case_id} fine setting {key} already exists in the coarse sweep."
                )
            keys.add(key)
            settings.append({"method": method, "block_size": block, "lambda": value})
        result[case_id] = sorted(
            settings,
            key=lambda item: (
                item["method"],
                -1 if item["block_size"] is None else item["block_size"],
                item["lambda"],
            ),
        )
    return result


def _candidate_settings(
    validated: dict[str, Any], stage: str
) -> dict[str, list[dict[str, Any]]]:
    """Resolve fixed coarse or review-gated fine settings for all cases.

    Args:
        validated: Fresh local configuration validation result.
        stage: ``coarse`` or ``fine``.

    Returns:
        Ordered per-case candidate settings.
    """
    if stage == "coarse":
        return {case_id: coarse_candidate_settings() for case_id in CASE_IDS}
    return _fine_settings(
        validated["config"]["snapshot"],
        Path(validated["config"]["path"]),
        Path(validated["layout"]["root"]),
    )


def _stream_command(command: Sequence[str], log_path: Path) -> float:
    """Run one BART command while saving combined output to a log.

    Args:
        command: Exact executable argument vector.
        log_path: Destination combined stdout/stderr log.

    Returns:
        Elapsed wall-clock seconds.
    """
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            list(command), stdout=stream, stderr=subprocess.STDOUT, text=True, check=False
        )
    if process.returncode:
        raise RuntimeError(f"BART failed with status {process.returncode}: {log_path}")
    return time.perf_counter() - started


def _candidate_reusable(
    manifest_path: Path, setting: Mapping[str, Any], case_manifest_hash: str
) -> bool:
    """Check whether one completed candidate remains safe to reuse.

    Args:
        manifest_path: Candidate manifest path.
        setting: Expected method/block/lambda setting.
        case_manifest_hash: Current prepared-case manifest digest.

    Returns:
        ``True`` only for a complete hash-identical candidate.
    """
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path, "candidate manifest")
        if manifest.get("status") != "complete" or manifest.get("setting") != dict(setting):
            return False
        if manifest.get("prepared_case_manifest", {}).get("sha256") != case_manifest_hash:
            return False
        for part in ("magnitude", "phase"):
            record = manifest["outputs"][part]
            if sha256_file(record["path"]) != record["sha256"]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_candidate(
    bart: Path,
    case_id: str,
    case_manifest_path: Path,
    setting: dict[str, Any],
    run_dir: Path,
    *,
    resume: bool,
    bart_version: str,
) -> dict[str, Any]:
    """Run one BART candidate and save finite magnitude plus phase arrays.

    Args:
        bart: Resolved compatible BART executable.
        case_id: Stable five-case identifier.
        case_manifest_path: Hash-bound prepared case manifest.
        setting: Method, lambda, and optional LLR block size.
        run_dir: Candidate output directory.
        resume: Reuse a complete hash-identical candidate when true.
        bart_version: Recorded BART build/version text.

    Returns:
        Completed candidate manifest.
    """
    case_manifest_hash = sha256_file(case_manifest_path)
    manifest_path = run_dir / "manifest.json"
    if resume and _candidate_reusable(manifest_path, setting, case_manifest_hash):
        return load_json(manifest_path, "candidate manifest")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Candidate output is not safely reusable: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    case = load_json(case_manifest_path, f"{case_id} prepared case")
    inputs = case_manifest_path.parent / "bart_inputs"
    bart_output = run_dir / "bart"
    bart_output.mkdir()
    split_output = bart_output / "image_wave_split"
    native_output = bart_output / "image_wave"
    command_output = split_output if setting["method"] == "llr" else native_output
    command = build_wave_command(
        bart,
        method=setting["method"],
        lambda_value=float(setting["lambda"]),
        block_size=setting["block_size"],
        csm_base=inputs / "coil_sens",
        psf_base=inputs / "psf",
        wave_kspace_base=inputs / "wave_kspace",
        output_base=command_output,
    )
    running = {
        "format_version": 1,
        "status": "running",
        "case_id": case_id,
        "setting": setting,
        "prepared_case_manifest": {
            "path": str(case_manifest_path),
            "sha256": case_manifest_hash,
        },
        "effective_bart_command": command,
        "bart_version": bart_version,
        "started_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, running)
    elapsed = _stream_command(command, run_dir / "bart_wave.log")
    split_record = None
    if setting["method"] == "llr":
        split_record = recombine_split_complex_cfl(split_output, native_output)
    image = np.asarray(open_cfl(native_output)).squeeze()
    expected_shape = tuple(int(value) for value in case["case"]["target_logical_matrix_ro_lin_par"])
    if image.shape != expected_shape or not np.isfinite(image).all():
        raise ValueError(f"{case_id} BART output shape/finite validation failed.")
    norm = float(case["bart_inputs"]["wave_kspace"]["wave_kspace_norm"])
    restored = image.astype(np.complex64, copy=False) * norm
    magnitude = np.abs(restored).astype(np.float32)
    phase = np.angle(restored).astype(np.float32)
    magnitude_path = run_dir / "magnitude_logical.npy"
    phase_path = run_dir / "phase_logical.npy"
    np.save(magnitude_path, magnitude)
    np.save(phase_path, phase)
    if not np.isfinite(magnitude).all() or not np.isfinite(phase).all():
        raise ValueError(f"{case_id} magnitude/phase outputs contain non-finite values.")
    completed = {
        **running,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "split_complex_recombination": split_record,
        "bart_internal_kspace_norm_restored": norm,
        "outputs": {
            "complex_bart": {
                "base": str(native_output),
                "payload_sha256": sha256_file(native_output.with_suffix(".cfl")),
            },
            "magnitude": {
                "path": str(magnitude_path),
                "sha256": sha256_file(magnitude_path),
                "shape": list(magnitude.shape),
            },
            "phase": {
                "path": str(phase_path),
                "sha256": sha256_file(phase_path),
                "shape": list(phase.shape),
            },
        },
        "phase_available": True,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, completed)
    return completed


def run_sweep(
    config_path: str | Path,
    *,
    stage: str,
    validate_only: bool,
    confirmed_output_root: Path | None,
    resume: bool,
) -> dict[str, Any]:
    """Validate and optionally run one manifest-owned coarse or fine sweep.

    Args:
        config_path: Ignored local configuration.
        stage: ``coarse`` or ``fine``.
        validate_only: Perform no output writes or BART execution when true.
        confirmed_output_root: Exact user-approved root required for execution.
        resume: Reuse complete hash-identical candidates.

    Returns:
        Validation summary or completed sweep manifest.
    """
    validated = validate_config(config_path)
    preparation = _validated_preparation(validated)
    settings_by_case = _candidate_settings(validated, stage)
    new_candidate_count = sum(len(values) for values in settings_by_case.values())
    if stage == "coarse" and new_candidate_count != len(CASE_IDS) * 23:
        raise AssertionError("Coarse sweep must contain 23 candidates per case.")
    reused_records: list[dict[str, Any]] = []
    if stage == "fine":
        coarse_root = Path(validated["layout"]["root"])
        coarse_manifest_path = coarse_root / "sweeps" / "coarse" / "sweep_manifest.json"
        coarse = load_json(coarse_manifest_path, "coarse sweep manifest")
        if coarse.get("status") != "complete":
            raise ValueError("Fine sweep requires a complete coarse candidate pool.")
        reused_records = list(coarse["candidate_manifests"])
    candidate_count = new_candidate_count + len(reused_records)
    if validate_only:
        if confirmed_output_root is not None:
            raise ValueError("--confirm-output-root is not used with --validate-only.")
        return {"status": "validated", "candidate_count": candidate_count}
    if confirmed_output_root is None:
        raise ValueError("Sweep execution requires --confirm-output-root after user approval.")
    configured_root = Path(validated["layout"]["root"])
    if confirmed_output_root.expanduser().resolve() != configured_root:
        raise ValueError("Confirmed output root differs from the local rerun configuration.")
    bart_name = str(validated["config"]["snapshot"].get("runtime", {}).get("bart", "bart"))
    bart_value = shutil.which(bart_name)
    if bart_value is None:
        raise FileNotFoundError("Compatible BART is not on PATH; source bart_startup.sh first.")
    bart = Path(bart_value).resolve()
    version_result = subprocess.run(
        [str(bart), "version"], check=True, capture_output=True, text=True
    )
    bart_version = (version_result.stdout + version_result.stderr).strip()
    stage_root = configured_root / "sweeps" / stage
    manifest_path = stage_root / "sweep_manifest.json"
    if stage_root.exists() and any(stage_root.iterdir()) and not resume:
        raise FileExistsError(f"Sweep stage output is nonempty; use --resume: {stage_root}")
    if stage_root.exists() and any(stage_root.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty sweep stage lacks its owned manifest: {stage_root}")
    stage_root.mkdir(parents=True, exist_ok=True)
    sweep = {
        "format_version": 1,
        "status": "running",
        "stage": stage,
        "preparation_manifest": {
            "path": str(configured_root / "preparation_manifest.json"),
            "sha256": sha256_file(configured_root / "preparation_manifest.json"),
        },
        "settings_by_case": settings_by_case,
        "candidate_count": candidate_count,
        "new_candidate_count": new_candidate_count,
        "reused_coarse_candidate_count": len(reused_records),
        "bart_version": bart_version,
        "candidate_manifests": reused_records,
        "started_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, sweep)
    for case_id in CASE_IDS:
        case_manifest_path = Path(preparation["cases"][case_id]["case_manifest"])
        for setting in settings_by_case[case_id]:
            run_dir = stage_root / case_id / candidate_name(setting)
            candidate = _run_candidate(
                bart,
                case_id,
                case_manifest_path,
                setting,
                run_dir,
                resume=resume,
                bart_version=bart_version,
            )
            record = {
                "case_id": case_id,
                "setting": setting,
                "manifest": str(run_dir / "manifest.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "magnitude": candidate["outputs"]["magnitude"],
                "phase": candidate["outputs"]["phase"],
            }
            sweep["candidate_manifests"] = [
                prior
                for prior in sweep["candidate_manifests"]
                if not (prior["case_id"] == case_id and prior["setting"] == setting)
            ] + [record]
            write_json_atomic(manifest_path, sweep)
    sweep["status"] = "complete"
    sweep["completed_at_utc"] = _utc_now()
    write_json_atomic(manifest_path, sweep)
    print(f"Pure-mask {stage} sweep manifest: {manifest_path}")
    return sweep


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
