"""Isolated MPRAGE standard-PCA/ROVir reconstruction with an explicit native CSM."""

from pathlib import Path
import math
import os
import shlex

import numpy as np

from .bart_io import cfl_record, open_cfl, sha256_file
from .csm_cache import file_record, read_json, validate_cfl, write_json
from .custom_csm import load_custom_csm, locked_output
from .metal_runner import memory_guard, run_reconstruction
from .retrospective import resample_sensitivity_maps, write_measured_wave_crop, validate_same_grid_masked_wave
from .rovir_retro import (_geometry_from_manifest, _resolve_cases, _target_sampling,
                          _factor_source_psf, _write_target_psf, _validate_source_identity)
from .mprage import _sampling_from_manifest
from .sampling import validate_pure_cartesian_image_lattice

CASES = ("native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3")


def check_recorded_array(base: Path, record: dict) -> dict:
    """Validate a full or legacy compact CFL hash record, then all finite values.

    Both header and payload digests plus shape are mandatory; missing hashes
    cannot be supplied by trusting the current file. Returns the full record.
    """
    actual = validate_cfl(base)
    for key in ("shape", "header_sha256", "payload_sha256"):
        if not isinstance(record, dict) or record.get(key) != actual[key]:
            raise ValueError(f"Custom reconstruction source {key} mismatch: {base}")
    return actual


def validate_native(args, custom: dict) -> dict:
    """Bind CLI acquisition, whitening, basis and calibrated operator without writes.

    Requires prepared native data. ROVir additionally requires its completed
    canonical contract; it is never reinterpreted as PCA by equal channel count.
    Returns verified native records, sampling anchor, and original calibration
    command. No raw preparation, ecalib or source manifest rewrite is performed.
    """
    source = custom["source"]
    rovir = source["kind"] == "rovir"
    if bool(args.rovir) != rovir:
        raise ValueError("--rovir selection differs from custom CSM basis")
    count = source["virtual_coils"]
    if args.virtual_coils is not None and args.virtual_coils != count:
        raise ValueError("Explicit virtual-coil count differs from custom CSM basis")
    root = args.output.resolve()
    expected = root/("normal/rovir" if rovir else f"vcc{count}/normal")/"bart_inputs/manifest.json"
    if Path(source["manifest"]["path"]).resolve() != expected:
        raise ValueError("Custom CSM belongs to another output root or coil variant")
    normal = source["manifest_content"]
    for name, supplied in (("twix", args.twix), ("sequence", args.seq)):
        record = normal["source"][name]
        if Path(record["path"]).resolve() != supplied.resolve():
            raise ValueError(f"Custom CSM {name} path mismatch")
        # Full source SHA is required for this new opt-in contract.
        _validate_source_identity(record, name)
    whitening = normal.get("noise_prewhitening", normal["source"].get("noise_prewhitening", {}))
    if bool(whitening.get("request", {}).get("enabled", False)) != args.noise_prewhitening:
        raise ValueError("Requested noise prewhitening differs from prepared custom CSM source")
    inputs = expected.parent
    artifact = normal.get("artifacts" if rovir else "output_artifacts", {})
    wave = check_recorded_array(inputs/"wave_kspace", artifact.get("wave_kspace"))
    psf_record = normal.get("psf_calibration", {}).get("copied") if rovir else artifact.get("psf")
    psf = check_recorded_array(inputs/"psf", psf_record)
    anchor = normal
    extra = {}
    if rovir:
        contract_path = inputs.parent/"manifest.json"
        contract = read_json(contract_path)
        if contract.get("status") != "mprage_normal_rovir_complete" or not contract.get("retro_consumable"):
            raise ValueError("Custom ROVir requires a completed canonical normal contract")
        if contract["source"] != normal["source"]:
            raise ValueError("ROVir canonical/prepared source mismatch")
        linked = contract["artifacts"]["prepared_inputs_manifest"]
        if linked.get("sha256") != sha256_file(expected):
            raise ValueError("ROVir contract prepared-manifest hash mismatch")
        for item in (contract["normal_source_manifest"], contract["coil_processing"]["approved_mask_manifest"]):
            if file_record(Path(item["path"]))["sha256"] != item.get("sha256"):
                raise ValueError("ROVir source/ROI manifest hash mismatch")
        transform = contract["coil_processing"]["transform"]
        check_recorded_array(Path(transform["base"]), transform)
        if contract["coil_processing"]["virtual_coils"] != count:
            raise ValueError("ROVir transform channel count mismatch")
        donor_command = inputs.parent/"bart_output/ecalib_command.txt"
        if sha256_file(donor_command) != contract["ecalib"]["command_record"].get("sha256"):
            raise ValueError("ROVir donor calibration-command hash mismatch")
        physical = normal["physical_calibration"]
        if sha256_file(Path(physical["manifest_path"])) != physical["manifest_sha256"]:
            raise ValueError("ROVir physical calibration provenance changed")
        check_recorded_array(Path(physical["cfl"]["base"]), physical["cfl"])
        anchor = read_json(Path(contract["normal_source_manifest"]["path"]))
        extra = {"rovir_contract": file_record(contract_path)}
    else:
        # Retain the established PSF/source matching rules, including explicit
        # fitting overrides and hash-bound accepted-artifact reuse.
        from .metal_pipeline import validate_invocation
        args.virtual_coils = count
        extra["source_match"] = validate_invocation({"normal_manifest": source["manifest"]}, args)
    geometry = _geometry_from_manifest(anchor)
    matrix = tuple(source["geometry"]["logical_matrix_ro_lin_par"])
    if geometry.logical_matrix_ro_lin_par != matrix:
        raise ValueError("Sampling anchor and custom CSM native geometry differ")
    sampling = _sampling_from_manifest(anchor)
    mask = sampling.mask()
    if tuple(mask.shape) != matrix[1:] or wave["shape"][1:4] != [*matrix[1:], count]:
        raise ValueError("Native sampling or k-space geometry mismatch")
    if psf["shape"][:3] != wave["shape"][:3]:
        raise ValueError("Native PSF and k-space geometry mismatch")
    oversampling = source["geometry"].get("readout_oversampling_factor")
    if (oversampling is not None and wave["shape"][0] != matrix[0]*oversampling
            or any(n != 1 for n in wave["shape"][4:]) or any(n != 1 for n in psf["shape"][3:])):
        raise ValueError("Native readout oversampling or trailing dimensions mismatch")
    calibration_record = inputs.parent/"bart_output/ecalib_command.txt"
    if not calibration_record.is_file():
        raise ValueError("Original CSM calibration command is required for export lineage")
    return {"source": source, "wave": wave, "psf": psf, "anchor": anchor,
            "ecalib": file_record(calibration_record), **extra}


def prepare_case(root: Path, case: str, native: dict, custom: dict) -> dict:
    """Prepare a custom-only retrospective operator and support-preserving LR maps.

    Native cases reuse the exact supplied CSM. Case inputs are derived from the
    validated native acquisition, never an unrelated old retro CSM. Only scalar
    sampling, cropped k-space, LR PSF and LR maps are created where required;
    canonical trees remain read-only. Returns hash-bound records for execution.
    """
    normal = Path(native["source"]["manifest"]["path"])
    matrix = tuple(custom["maps"]["shape"][:3])
    if case == "normal":
        return {"manifest": file_record(normal), "maps": custom["maps"],
                "psf": native["psf"], "wave": native["wave"], "target": list(matrix)}
    geometry = _geometry_from_manifest(native["anchor"])
    resolved = _resolve_cases(geometry, [case])[case]
    sampling = _sampling_from_manifest(native["anchor"])
    target_mask, target_sampling = _target_sampling(sampling, resolved)
    target = resolved.target_logical_matrix_ro_lin_par
    request = {"native": native, "custom_maps": custom["maps"], "case": resolved.to_json(),
               "sampling": target_sampling,
               "implementation": {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ("custom_pipeline.py", "retrospective.py", "rovir_retro.py")}}
    directory = root/"retro"/case
    with locked_output(directory, request, ("metadata", "bart_inputs")):
        complete = directory/"metadata/complete.json"
        if complete.exists():
            result = read_json(complete)
            if file_record(Path(result["manifest"]["path"])) != result["manifest"]:
                raise ValueError("Custom retro manifest changed")
            for key in ("maps", "wave", "psf"):
                validate_cfl(Path(result[key]["base"]), result[key])
            for record in result["provenance_files"]:
                if file_record(Path(record["path"])) != record:
                    raise ValueError("Custom retro sampling/resampling provenance changed")
            return result
        inputs = directory/"bart_inputs"
        if any(inputs.iterdir()):
            raise ValueError("Partial custom retrospective preparation; refusing overwrite")
        np.save(inputs/"sampling_mask.npy", target_mask, allow_pickle=False)
        metrics = write_measured_wave_crop(native["wave"]["base"], inputs/"wave_kspace", resolved,
                                           sampling.mask(), sampling.acceleration_lin_par, target_mask=target_mask)
        if metrics["sampled_coordinate_count"] != target_sampling["acquired_coordinate_count"]:
            raise ValueError("Custom retrospective sampling count mismatch")
        validate_pure_cartesian_image_lattice(target_mask, target_sampling)
        same_grid = None
        if target == matrix:
            same_grid = validate_same_grid_masked_wave(native["wave"]["base"], inputs/"wave_kspace", target_mask)
            psf, maps = native["psf"], custom["maps"]
            resampling = None
        else:
            coefficients, _ = _factor_source_psf(Path(native["psf"]["base"]))
            _write_target_psf(inputs/"psf", *coefficients, target_lin=target[1], target_par=target[2])
            psf = validate_cfl(inputs/"psf")
            resampling = resample_sensitivity_maps(custom["maps"]["base"], inputs/"coil_sens", target_lin_par=target[1:])
            maps = validate_cfl(inputs/"coil_sens")
        wave = validate_cfl(inputs/"wave_kspace")
        manifest = {"status": "mprage_custom_csm_retro_inputs_ready", "case_directory": case,
                    "case": resolved.to_json(), "source": native["source"]["manifest_content"]["source"],
                    "source_native_manifest": native["source"]["manifest"], "basis": native["source"]["basis"],
                    "basis_kind": native["source"]["kind"], "sampling": target_sampling,
                    "sampling_validation": same_grid, "coil_sens_resampling": resampling,
                    "custom_native_maps": custom["maps"], "artifacts": {"wave_kspace": wave, "psf": psf, "coil_sens": maps},
                    "echoes": [{"wave_kspace_norm": metrics["wave_kspace_norm"]}]}
        write_json(inputs/"manifest.json", manifest)
        files = [inputs/"sampling_mask.npy"]
        if resampling is not None:
            files.append(inputs/"coil_sens_resampling.json")
        result = {"manifest": file_record(inputs/"manifest.json"), "maps": maps, "psf": psf,
                  "wave": wave, "target": list(target), "provenance_files": [file_record(p) for p in files]}
        write_json(complete, result)
        return result


def finish_custom_export(directory: Path, case_record: dict, metadata: dict, native: dict, args) -> None:
    """Atomically publish paired NIfTIs with custom/basis provenance after BART completion.

    Original calibration commands describe only the donor, never the selected
    custom maps. Interrupted staging is preserved and rejected. Completed exports
    are hash/geometry validated and reused, not overwritten.
    """
    from scripts.convert_mprage_bart_to_nifti import convert
    from .metal_export import export_files, validate_images
    final = directory/"nifti"
    complete = directory/"metadata/export.json"
    execution = file_record(directory/"metadata/reconstruction.json")
    if complete.exists():
        record = read_json(complete)
        files = export_files(final)
        if record != {"status": "complete", "execution": execution, "custom": metadata,
                      "files": [file_record(p) for p in files]}:
            raise ValueError("Custom NIfTI completion/provenance changed")
        validate_images(files, tuple(case_record["target"]))
        return
    staging = directory/"nifti_staging"
    if final.exists() or staging.exists():
        raise ValueError("Partial custom NIfTI export; refusing overwrite")
    staging.mkdir()
    convert(Path(case_record["manifest"]["path"]).parent, directory/"reconstruction/image_wave",
            args.twix, args.seq, staging, "BARTWaveMPRAGECustomCSM"+metadata["Method"].title(),
            ecalib_command_record=Path(native["ecalib"]["path"]))
    files = export_files(staging)
    validate_images(files, tuple(case_record["target"]))
    for path in files:
        if path.suffix == ".json":
            value = read_json(path)
            value["CSMDonorEcalibCommand"] = value.pop("BARTEcalibCommand", None)
            value.update(CustomCSM=metadata, CoilProcessing=native["source"]["kind"],
                         RegularizationMethod=metadata["Method"], RegularizationLambda=metadata["Lambda"],
                         RegularizationOptimizedForCustomCSM=False)
            # Only unpublished derivative-owned staging sidecars are modified.
            import json
            path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    staging.rename(final)
    write_json(complete, {"status": "complete", "execution": execution, "custom": metadata,
                          "files": [file_record(p) for p in export_files(final)]})


def run_custom_pipeline(args) -> Path:
    """Validate and run explicit custom CSM mode; defaults never activate this path.

    Uses isolated fingerprint roots, fixed 100-iteration Wave/FISTA settings and
    the existing RAM-aware CPU/GPU runner. Prepare-only never calls BART. No
    NIfTI collection replacement, automatic N4 or parameter selection occurs.
    """
    if args.r3_lambda is not None and (args.mode != "normal" or not math.isfinite(args.r3_lambda) or args.r3_lambda <= 0):
        raise ValueError("--r3-lambda requires a positive finite normal Wavelet value")
    custom = load_custom_csm(args.custom_csm)
    native = validate_native(args, custom)
    identity = file_record(args.custom_csm.resolve())
    token = identity["sha256"][:16]
    count = custom["source"]["virtual_coils"]
    variant = "rovir" if args.rovir else f"vcc{count}"
    root = args.output.resolve()/f"{variant}_custom_csm"/token
    profile = args.profile or ("reg-full" if args.rovir and args.mode == "normal" else
                               "wavelet-only" if args.mode == "normal" else "fista-only")
    if native["anchor"]["sampling"]["name"] == "R1" and profile != "fista-only":
        raise ValueError("R1 has no approved positive Wavelet parameter; use --fista-only")
    if args.custom_check_only:
        if args.mode != "normal":
            selected = CASES if args.mode == "retro" else ("native_r3x3",)
            geometry = _geometry_from_manifest(native["anchor"])
            sampling = _sampling_from_manifest(native["anchor"])
            for resolved in _resolve_cases(geometry, selected).values():
                _target_sampling(sampling, resolved)
        print("Read-only custom CSM validation passed; no output directories created.")
        return root
    request = {"custom_manifest": identity, "source": native,
               "implementation": {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ("custom_pipeline.py", "custom_csm.py", "retrospective.py", "metal_runner.py")},
               "converter_sha256": sha256_file(Path(__file__).parents[1]/"scripts/convert_mprage_bart_to_nifti.py")}
    memory_guard()
    with locked_output(root, request, ("metadata", "normal", "retro")):
        selected = ("normal",) if args.mode == "normal" else (("native_r3x3",) if args.mode == "r3x3" else CASES)
        for case in selected:
            prepared = prepare_case(root, case, native, custom)
            default = (.035 if args.rovir else .03 if count == 24 else .035) if case == "normal" else {
                "native_r3x2": .035, "lr_y_1p5mm_r3x2": .025, "native_r3x3": .045}[case]
            branches = []
            if profile in ("reg-full", "fista-only"):
                branches.append(("fista_r0", "fista", 0.))
            if profile in ("reg-full", "wavelet-only"):
                branches.append(("wavelet_custom_unvalidated", "wavelet", args.r3_lambda or default))
            for branch, method, value in branches:
                directory = root/("normal" if case == "normal" else f"retro/{case}")/"bart_output"/branch
                branch_request = {"custom_manifest": identity, "case": prepared, "method": method,
                                  "lambda": value, "profile": profile, "gpu": args.g,
                                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if args.g else None,
                                  "cpu_threads": args.cpu_threads, "pipeline": request}
                with locked_output(directory, branch_request, ("metadata", "logs", "reconstruction")):
                    preparation = {"candidate": prepared["maps"], "request": {"source": {
                        "regularization": value, "records": {"psf": prepared["psf"], "kspace": prepared["wave"]}},
                        "custom_pipeline": branch_request}}
                    completion = directory/"metadata/preparation.json"
                    if completion.exists():
                        if read_json(completion) != preparation:
                            raise ValueError("Custom branch preparation changed")
                    else:
                        write_json(completion, preparation)
                    if args.custom_prepare_only:
                        continue
                    execution = run_reconstruction(directory, preparation, args.cpu_threads, gpu=args.g)
                    if execution["output"]["shape"][:3] != prepared["target"] or any(n != 1 for n in execution["output"]["shape"][3:]):
                        raise ValueError("Custom reconstructed image geometry mismatch")
                    command_file = directory/"reconstruction/wave_command.txt"
                    command = shlex.join(execution["identity"]["command"])
                    if command_file.exists():
                        if command_file.read_text().strip() != command:
                            raise ValueError("Custom Wave command record changed")
                    else:
                        with command_file.open("x") as stream:
                            stream.write(command+"\n")
                    metadata = {"Manifest": identity, "NativeMaps": custom["maps"], "CaseMaps": prepared["maps"],
                                "Basis": native["source"]["basis"], "Method": method, "Lambda": value,
                                "GPU": args.g, "Experimental": True, "Case": case}
                    finish_custom_export(directory, prepared, metadata, native, args)
    return root
