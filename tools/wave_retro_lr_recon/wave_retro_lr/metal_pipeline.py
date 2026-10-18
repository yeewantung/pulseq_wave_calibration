"""Opt-in normal and retrospective MPRAGE integration with isolated metal CSMs.

Canonical inputs are validated/reused through the existing preparation APIs.
Only metal derivatives are reconstructed; a reviewed normal baseline is required
to bind the reference ROI to the exact logical grid, orientation and PCA basis.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shlex
import shutil

import nibabel as nib
import numpy as np

from .bart_io import cfl_record, sha256_file
from .metal_csm import MetalCSMParameters
from .metal_runner import memory_guard, run_cpu_reconstruction, run_metal_workflow
from .metal_workflow import file_record, read_json, source_contract, validate_cfl, write_json
from .reconstruction_state import _request_contract
from .retrospective import resample_sensitivity_maps


def validate_options(args) -> None:
    """Reject incompatible launcher options before reading data or creating outputs."""
    if not args.metal_csm_extrapolation:
        raise ValueError("Supply --metal-csm-extrapolation; metal mode is disabled by default")
    if args.g:
        raise ValueError("Metal CSM mode is CPU-only; remove -g")
    if args.rovir or args.include_legacy_lr:
        raise ValueError("Metal CSM mode supports standard PCA only, not --rovir/legacy LR")
    if args.n4_bias_correction:
        raise ValueError("Metal mode does not yet combine N4; omit --n4-bias-correction")
    if args.virtual_coils < 1 or not 1 <= args.cpu_threads <= 8:
        raise ValueError("Invalid virtual-coil count or CPU thread cap")
    if not math.isfinite(args.ecalib_crop) or not 0 < args.ecalib_crop <= 1:
        raise ValueError("Metal CSM requires hard-cropped single maps with crop in (0, 1]")
    if args.r3_lambda is not None:
        if args.mode != "normal" or not math.isfinite(args.r3_lambda) or args.r3_lambda < 0:
            raise ValueError("--r3-lambda is a finite nonnegative normal-only override")
    for axis in ("kx", "y", "z"):
        low, high = getattr(args, f"psf_fit_{axis}_min"), getattr(args, f"psf_fit_{axis}_max")
        if (low is None) != (high is None):
            raise ValueError(f"Manual PSF {axis} fitting requires both bounds")


def load_configuration(root: Path, config_path: Path, count: int) -> tuple[dict, dict, Path]:
    """Validate a local configuration and its completed normal source; return isolated destination.

    Relative source/reference paths are resolved relative to the configuration.
    All source hashes, coil count and exact dataset location must match. No
    filesystem mutations occur here; the full configuration hash binds later reuse.
    """
    config_path = config_path.resolve(strict=True)
    document = read_json(config_path)
    if set(document) != {"source_manifest", "reference_nifti", "parameters"}:
        raise ValueError("Metal config requires exactly source_manifest, reference_nifti, parameters")
    parameters = MetalCSMParameters.from_dict(document["parameters"])
    paths = {}
    for field in ("source_manifest", "reference_nifti"):
        value = document[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Set an explicit reviewed {field} in the local config")
        paths[field] = (config_path.parent/Path(value).expanduser()).resolve()
    destination = root/f"vcc{count}_metal_csm"
    if destination.is_symlink():
        raise ValueError("Metal derivative roots must not be symlinks")
    reference = destination/"source_reference"/paths["reference_nifti"].name
    cached_sidecar = reference.with_name(reference.name.replace(".nii.gz", ".json"))
    existing_request = destination/"metadata/request.json"
    if existing_request.is_file():
        saved = read_json(existing_request)
        if saved.get("config") != file_record(config_path):
            raise ValueError("Metal pipeline request mismatch: configuration changed")
        if reference.exists() or cached_sidecar.exists():
            for key, path in (("reference", reference), ("reference_sidecar", cached_sidecar)):
                if file_record(path) != saved["source"][key]:
                    raise ValueError("Cached metal source reference changed")
    elif reference.exists() or cached_sidecar.exists():
        raise ValueError("Unbound metal reference cache")
    source = source_contract(paths["source_manifest"], reference if reference.exists() else paths["reference_nifti"])
    expected = (root/f"vcc{count}/normal/bart_inputs/manifest.json").resolve()
    if Path(source["normal_manifest"]["path"]) != expected or source["virtual_coils"] != count:
        raise ValueError("Metal config belongs to a different dataset or VCC variant")
    settings = {"config": file_record(config_path), "parameters": parameters.to_dict(),
                "source_manifest": str(paths["source_manifest"]),
                "reference_nifti": str(reference), "reference_origin": str(paths["reference_nifti"])}
    # Bind the immutable derivative-owned copy from the first request onward.
    # Collection replacement may later replace the original reference pathname.
    source["reference"]["path"] = str(reference)
    source["reference_sidecar"]["path"] = str(cached_sidecar)
    return settings, source, destination


def preserve_reference(settings: dict, source: dict) -> None:
    """Copy the original reference pair exclusively under the held variant lock.

    Validate copies against the source contract; reject partial or modified
    caches instead of overwriting them. Inputs are configuration and source
    records returned by load_configuration; no source file is modified.
    """
    origin = Path(settings["reference_origin"])
    for key, original in (("reference", origin),
                          ("reference_sidecar", origin.with_name(origin.name.replace(".nii.gz", ".json")))):
        record = source[key]
        target = Path(record["path"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
        if file_record(target) != record:
            raise ValueError("Metal source reference copy differs from reviewed source")


@contextmanager
def locked_directory(path: Path, request: dict, folders=("metadata", "intermediate", "logs", "reconstruction", "nifti")):
    """Initialize or validate an isolated directory and hold its exclusive advisory lock.

    Refuse partial initialization, unknown files and symlink descendants. The
    request is immutable; only the execution state can advance within this root.
    """
    if path.exists() and any(path.iterdir()) and not (path/"metadata/request.json").is_file():
        raise ValueError(f"Unbound metal output directory: {path}")
    if path.is_symlink() or (path.exists() and any(p.is_symlink() for p in path.rglob("*"))):
        raise ValueError("Symlinks in metal output trees are not supported")
    path.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path/".metal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = path/"metadata/request.json"
        if record.exists():
            if read_json(record) != request:
                raise ValueError("Metal pipeline request mismatch; preserve existing outputs")
        else:
            if set(p.name for p in path.iterdir()) != {".metal.lock"}:
                raise ValueError("Partial metal output initialization")
            for folder in folders:
                (path/folder).mkdir()
            write_json(record, request)
        yield
    finally:
        os.close(descriptor)


def prepare_case_maps(root: Path, case: str, prepared_path: Path, native: dict) -> dict:
    """Reuse native candidate maps or Fourier-resample them once for a matched LR-Y grid.

    The output record binds the exact native candidate and preparation manifest.
    No original CSM or canonical retrospective manifest is edited.
    """
    source = native["request"]["source"]
    prepared = read_json(prepared_path)
    matrix = tuple(source["geometry"]["logical_matrix_ro_lin_par"])
    target = tuple(prepared.get("case", {}).get("target_logical_matrix_ro_lin_par", matrix))
    candidate = validate_cfl(native["candidate"])
    if case in ("normal", "native_r3x2", "native_r3x3"):
        if target != matrix:
            raise ValueError("Native metal case does not match the native CSM grid")
        return candidate
    if case != "lr_y_1p5mm_r3x2" or target[0] != matrix[0] or target[2] != matrix[2] or not 0 < target[1] < matrix[1]:
        raise ValueError("Only a same-FOV LIN-reduced LR-Y metal case is supported")
    request = {"native_candidate": candidate, "prepared_manifest": file_record(prepared_path),
               "target_matrix": list(target), "method": "centered_fourier_PE_then_RSS",
               "implementation_sha256": sha256_file(Path(__file__).with_name("retrospective.py"))}
    directory = root/"retro"/case/"maps"
    with locked_directory(directory, request):
        manifest = directory/"metadata/maps.json"
        if manifest.exists():
            saved = read_json(manifest)
            if saved.get("request") != request:
                raise ValueError("Metal LR map provenance changed")
            return validate_cfl(saved["candidate"])
        if any((directory/"intermediate").iterdir()):
            raise ValueError("Partial metal LR map preparation; refusing overwrite")
        memory_guard()
        output = directory/"intermediate/coil_sens"
        resample_sensitivity_maps(candidate["base"], output, target_lin_par=target[1:])
        result = validate_cfl(cfl_record(output))
        validate_cfl(candidate)
        write_json(manifest, {"candidate": result, "request": request})
        return result


def branch_settings(mode: str, profile: str | None, count: int, override: float | None) -> list[tuple]:
    """Return the existing default case/profile/lambda policy with explicitly unvalidated metal labels."""
    profile = profile or ("wavelet-only" if mode == "normal" else "fista-only")
    cases = {"normal": [("normal", override if override is not None else (.03 if count == 24 else .035))],
             "retro": [("native_r3x2", .035), ("lr_y_1p5mm_r3x2", .025), ("native_r3x3", .045)],
             "r3x3": [("native_r3x3", .045)]}[mode]
    result = []
    for case, value in cases:
        if profile in ("fista-only", "reg-full"):
            result.append((case, "fista_r0", "fista", 0., profile))
        if profile in ("wavelet-only", "reg-full"):
            result.append((case, "wavelet_metal_unvalidated", "wavelet", value, profile))
    return result


def validate_invocation(source: dict, args) -> dict:
    """Read-only validation of source paths, prewhitening and requested PSF settings.

    Reuse the existing strict native matcher without modifying its hash-bound
    implementation, or its accepted hash-bound artifact reuse policy when no
    fitting override is requested. Return the recorded/requested PSF policy.
    Retro preparation may not migrate baseline metadata behind its digest.
    """
    from .mprage import (
        _native_manifest_matches, _native_manifest_matches_reusable_artifact,
        _normalize_psf_coefficient_settings,
        _normalize_psf_spatial_settings, _psf_processing_implementation_identity,
        PSF_COEFFICIENT_PLOT_NAME, PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME,
    )

    normal = read_json(source["normal_manifest"]["path"])
    settings = _normalize_psf_coefficient_settings(args.psf_coefficient_processing,
                                                  args.psf_fit_kx_min, args.psf_fit_kx_max)
    settings.update(_normalize_psf_spatial_settings(args.psf_fit_y_min, args.psf_fit_y_max,
                                                   args.psf_fit_z_min, args.psf_fit_z_max))
    settings["processing_implementation"] = _psf_processing_implementation_identity()
    match_args = (normal, args.twix.expanduser().resolve(), args.seq.expanduser().resolve(),
                  settings, args.virtual_coils, args.noise_prewhitening)
    exact = _native_manifest_matches(*match_args)
    if not exact and not _native_manifest_matches_reusable_artifact(*match_args):
        raise ValueError("Metal invocation differs from reviewed sources, prewhitening or PSF settings")
    if args.mode != "normal" and exact:
        for key, filename in (("visual_assessment_plot_relative_to_output_root", PSF_COEFFICIENT_PLOT_NAME),
                              ("full_range_plot_relative_to_output_root", PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME)):
            if normal.get("psf_calibration", {}).get(key) != f"vcc{args.virtual_coils}/normal/{filename}":
                raise ValueError("Retro preparation would migrate the reviewed normal manifest's plot metadata; refusing mutation")
    return {"policy": "exact_native_match" if exact else "accepted_materialized_artifacts_no_refit",
            "requested_psf_settings": settings, "recorded_psf_calibration": normal.get("psf_calibration"),
            "recorded_psf_preserved": True}


def run_branch(root: Path, case: str, branch: str, method: str, value: float, profile: str,
               prepared: Path, native: dict, maps: dict, settings: dict, args) -> None:
    """Reconstruct or resume one hash-bound CPU branch and export via the standard MPRAGE converter.

    All outputs are isolated. A completed execution may be exported on retry;
    incomplete executions or partial exports are preserved and rejected.
    """
    from scripts.convert_mprage_bart_to_nifti import convert

    directory = root/("normal" if case == "normal" else f"retro/{case}")/"bart_output"/branch
    normal = Path(native["request"]["source"]["normal_manifest"]["path"])
    image = directory/"reconstruction/image_wave"
    record = image.parent/"wave_command.txt"
    inputs = prepared.parent
    command = ["bart", "wave", "-w", "-f", "-r", format(value, ".17g"), "-i", "100", "-t", "1e-6",
               maps["base"], str(inputs/"psf"), str(inputs/"wave_kspace"), str(image)]
    standard_request = _request_contract(
        prepared_manifest=prepared, normal_manifest=normal, profile=profile, case=case,
        branch=branch, method=method, regularization=value, maps=Path(maps["base"]),
        psfs=[inputs/"psf"], kspaces=[inputs/"wave_kspace"], images=[image],
        command_records=[record], expected_commands=[shlex.join(command)], nifti_directory=directory/"nifti")
    request = {"standard_request": standard_request, "native_preparation": native,
               "config": settings["config"], "cpu_threads": args.cpu_threads,
               "software_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                   for name in ("metal_pipeline.py", "metal_runner.py")},
               "converter_sha256": sha256_file(Path(__file__).parents[1]/"scripts/convert_mprage_bart_to_nifti.py")}
    with locked_directory(directory, request):
        preparation_path = directory/"metadata/preparation.json"
        preparation = {"request": {"source": {"regularization": value,
                       "records": {"psf": standard_request["inputs"]["psf_by_echo"][0],
                                   "kspace": standard_request["inputs"]["kspace_by_echo"][0]}},
                       "pipeline": request}, "candidate": maps}
        if preparation_path.exists():
            if read_json(preparation_path) != preparation:
                raise ValueError("Metal branch preparation changed")
        else:
            write_json(preparation_path, preparation)
        if args.metal_prepare_only:
            return
        execution = run_cpu_reconstruction(directory, preparation, args.cpu_threads)
        geometry = read_json(prepared)
        target = tuple(geometry.get("case", {}).get("target_logical_matrix_ro_lin_par",
                       native["request"]["source"]["geometry"]["logical_matrix_ro_lin_par"]))
        if tuple(execution["output"]["shape"][:3]) != target or any(n != 1 for n in execution["output"]["shape"][3:]):
            raise ValueError("Metal reconstructed image geometry mismatch")
        if record.exists():
            if record.read_text().strip() != shlex.join(command):
                raise ValueError("Metal wave command record changed")
        else:
            with record.open("x") as stream:
                stream.write(shlex.join(command)+"\n")
        manifest = directory/"metadata/export.json"
        if manifest.exists():
            exported = read_json(manifest)
            if exported["execution"] != file_record(directory/"metadata/reconstruction.json"):
                raise ValueError("Metal export execution identity changed")
            for item in exported["outputs"]:
                if file_record(item["path"]) != item:
                    raise ValueError("Metal NIfTI output changed")
            return
        if any((directory/"nifti").iterdir()):
            raise ValueError("Partial metal NIfTI export; refusing overwrite")
        convert(inputs, Path(execution["output"]["base"]), args.twix, args.seq,
                directory/"nifti", f"BARTWaveMPRAGE{case}MetalCSM{method}")
        # Augment new derivatives only; original exports and their sidecars remain untouched.
        import json
        for sidecar in (directory/"nifti").glob("*.json"):
            payload = read_json(sidecar)
            payload["MetalCSM"] = {"Experimental": True, "DefaultEnabled": False,
                                  "Preparation": file_record(preparation_path),
                                  "NativeCandidate": native["candidate"], "CaseCandidate": maps,
                                  "RegularizationOptimizedForMetalCSM": False,
                                  "RegularizationLambda": value, "GPU": False}
            sidecar.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        outputs = [file_record(path) for path in sorted((directory/"nifti").iterdir())]
        if len(outputs) != 4:
            raise ValueError("Expected magnitude/phase NIfTI plus two JSON sidecars")
        for path in (directory/"nifti").glob("*.nii.gz"):
            image_nifti = nib.load(path)
            if image_nifti.shape != target[::-1] or not np.isfinite(image_nifti.affine).all() or not np.isfinite(np.asarray(image_nifti.dataobj)).all():
                raise ValueError("Metal NIfTI geometry/finite-value validation failed")
        write_json(manifest, {"status": "complete", "outputs": outputs,
                             "execution": file_record(directory/"metadata/reconstruction.json")})


def run_pipeline(args) -> Path:
    """Run the explicit standard-launcher metal mode; return its independent VCC root.

    Requires a completed reviewed normal baseline. Preparation uses existing
    source/PSF/geometry validation; no baseline BART job is implicitly launched.
    """
    validate_options(args)
    memory_guard()
    root = args.output.expanduser().resolve()
    config = args.metal_csm_config or root/"metal_csm.local.json"
    settings, source, destination = load_configuration(root, config, args.virtual_coils)
    invocation = validate_invocation(source, args)
    normal = Path(source["normal_manifest"]["path"])
    # Check ecalib settings against the actual native source, not a newly requested map.
    ecalib = normal.parent.parent/"bart_output/ecalib_command.txt"
    expected = ["bart", "ecalib", "-m", "1", "-c", str(args.ecalib_crop),
                str(normal.parent/"kspace_calib"), source["records"]["maps"]["base"]]
    actual = shlex.split(ecalib.read_text())
    if (len(actual) != len(expected) or Path(actual[0]).name != "bart"
            or actual[1:5] != expected[1:5] or float(actual[5]) != args.ecalib_crop
            or [Path(p).resolve() for p in actual[6:]] != [Path(p).resolve() for p in expected[6:]]):
        raise ValueError("Requested ecalib crop/source differs from reviewed native CSM")
    kwargs = {key: getattr(args, key) for key in ("virtual_coils", "noise_prewhitening", "psf_coefficient_processing",
              "psf_fit_kx_min", "psf_fit_kx_max", "psf_fit_y_min", "psf_fit_y_max", "psf_fit_z_min", "psf_fit_z_max")}
    root_request = {"config": settings["config"], "source": source,
                    "parameters": settings["parameters"], "implementation": file_record(Path(__file__)),
                    "ecalib_record": file_record(ecalib), "source_reuse": invocation}
    with locked_directory(destination, root_request, folders=("metadata",)):
        preserve_reference(settings, source)
        return _run_validated_pipeline(args, root, destination, settings, source, kwargs)


def _run_validated_pipeline(args, root: Path, destination: Path, settings: dict, source: dict, kwargs: dict) -> Path:
    """Prepare cases and run selected branches under the already-held variant lock."""
    from .mprage import prepare_retro_mprage, prepare_retro_mprage_r3x3

    normal = Path(source["normal_manifest"]["path"])
    memory_guard()
    if args.mode == "retro":
        prepare_retro_mprage(args.twix, root, args.seq, **kwargs)
    if args.mode in ("retro", "r3x3"):
        prepare_retro_mprage_r3x3(args.twix, root, args.seq, **kwargs)
    if file_record(normal) != source["normal_manifest"]:
        raise ValueError("Normal preparation changed from the reviewed metal source")
    native = run_metal_workflow(settings["source_manifest"], settings["reference_nifti"],
                               destination/"shared_csm", MetalCSMParameters.from_dict(settings["parameters"]),
                               enabled=True, prepare_only=True, threads=args.cpu_threads)
    for case, branch, method, value, profile in branch_settings(args.mode, args.profile, args.virtual_coils, args.r3_lambda):
        prepared = normal if case == "normal" else root/f"vcc{args.virtual_coils}/retro/{case}/bart_inputs/manifest.json"
        maps = prepare_case_maps(destination, case, prepared, native)
        run_branch(destination, case, branch, method, value, profile, prepared, native, maps, settings, args)
    if source_contract(settings["source_manifest"], settings["reference_nifti"]) != source:
        raise ValueError("Reviewed native source changed during metal reconstruction")
    if file_record(settings["config"]["path"]) != settings["config"]:
        raise ValueError("Metal configuration changed during reconstruction")
    return destination
