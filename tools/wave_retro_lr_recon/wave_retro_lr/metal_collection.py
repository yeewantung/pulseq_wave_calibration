"""Read-only validation and same-VCC collection preference for metal derivatives."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
import shlex

import nibabel as nib
import numpy as np

from .bart_io import sha256_file
from .metal_runner import wave_command
from .metal_workflow import file_record, read_json, source_contract, validate_cfl
from .reconstruction_state import _request_contract
from .metal_export import export_files


def checked_file(record: dict, within: Path | None = None) -> Path:
    """Validate a recorded file's size/hash and optional directory confinement; return its path."""
    path = Path(record["path"]).resolve(strict=True)
    if within is not None and within.resolve() not in path.parents:
        raise ValueError(f"Metal provenance escapes its directory: {path}")
    if file_record(path) != record:
        raise ValueError(f"Metal file provenance changed: {path}")
    return path


def validate_branch(directory: Path, variant: Path, count: int, case: str,
                    root_request: dict, native: dict) -> dict:
    """Validate one advertised complete metal branch and return its paired NIfTIs/provenance.

    Verify config/source/basis/geometry, native or LR candidate lineage, exact
    preparation/execution/command binding and every exported hash. Historical
    implementation hashes are retained, not compared with today's source code.
    No BART command, preparation, or output mutation is performed.
    """
    root = variant.parent
    request_path = directory/"metadata/request.json"
    request = read_json(request_path)
    standard = request["standard_request"]
    if (standard["virtual_coils"] != count or standard["case"] != case
            or standard["branch"] != directory.name
            or request["config"] != root_request["config"]
            or request["native_preparation"] != native):
        raise ValueError("Metal branch count, case, configuration or native lineage mismatch")
    expected_normal = root/f"vcc{count}/normal/bart_inputs/manifest.json"
    expected_prepared = expected_normal if case == "normal" else root/f"vcc{count}/retro/{case}/bart_inputs/manifest.json"
    if (Path(standard["normal_manifest"]["path"]).resolve() != expected_normal.resolve()
            or Path(standard["prepared_manifest"]["path"]).resolve() != expected_prepared.resolve()
            or Path(standard["nifti_directory"]).resolve() != (directory/"nifti").resolve()):
        raise ValueError("Metal branch refers to another dataset/grid/export directory")
    fresh = _request_contract(
        prepared_manifest=expected_prepared, normal_manifest=expected_normal,
        profile=standard["profile"], case=case, branch=directory.name,
        method=standard["method"], regularization=standard["regularization"],
        maps=Path(standard["inputs"]["maps"]["base"]),
        psfs=[Path(r["base"]) for r in standard["inputs"]["psf_by_echo"]],
        kspaces=[Path(r["base"]) for r in standard["inputs"]["kspace_by_echo"]],
        images=[Path(p) for p in standard["images"]],
        command_records=[Path(p) for p in standard["command_records"]],
        expected_commands=standard["expected_commands"], nifti_directory=directory/"nifti")
    if fresh != standard:
        raise ValueError("Metal reconstruction source request changed")
    preparation_path = directory/"metadata/preparation.json"
    preparation = read_json(preparation_path)
    if preparation["request"]["pipeline"] != request:
        raise ValueError("Metal preparation is not bound to its request")
    candidate = validate_cfl(preparation["candidate"])
    if dict(candidate, finite=True) != standard["inputs"]["maps"]:
        raise ValueError("Metal reconstruction uses a different CSM")
    if case == "lr_y_1p5mm_r3x2":
        maps_path = variant/f"retro/{case}/maps/metadata/maps.json"
        maps = read_json(maps_path)
        if (maps["candidate"] != candidate or maps["request"]["native_candidate"] != native["candidate"]
                or maps["request"]["prepared_manifest"] != file_record(expected_prepared)
                or maps["request"] != read_json(maps_path.with_name("request.json"))):
            raise ValueError("Metal LR map lineage changed")
    elif candidate != native["candidate"]:
        raise ValueError("Native metal case must use the shared native CSM")
    source = preparation["request"]["source"]
    if (source["regularization"] != standard["regularization"]
            or source["records"]["psf"] != standard["inputs"]["psf_by_echo"][0]
            or source["records"]["kspace"] != standard["inputs"]["kspace_by_echo"][0]):
        raise ValueError("Metal execution inputs differ from its standard source contract")
    execution_path = directory/"metadata/reconstruction.json"
    execution = read_json(execution_path)
    identity = execution["identity"]
    gpu = request.get("gpu", False)  # Historical integrated branches were CPU-only.
    if type(gpu) is not bool:
        raise ValueError("Invalid metal GPU selection")
    command = wave_command(preparation, directory, identity["bart"]["path"], gpu=gpu)
    if (execution["status"] != "complete" or identity["gpu"] is not gpu
            or identity["preparation_sha256"] != sha256_file(preparation_path)
            or identity["command"] != command
            or len(standard["expected_commands"]) != 1
            or shlex.split(standard["expected_commands"][0])[1:] != command[1:]):
        raise ValueError("Metal execution is not a complete matched reconstruction (including CPU/GPU selection)")
    output = validate_cfl(execution["output"])
    if output["base"] != str(directory/"reconstruction/image_wave") or standard["images"] != [output["base"]]:
        raise ValueError("Metal output image path mismatch")
    prepared = read_json(expected_prepared)
    source_geometry = root_request["source"]["geometry"]
    target = tuple(prepared.get("case", {}).get("target_logical_matrix_ro_lin_par",
                   source_geometry["logical_matrix_ro_lin_par"]))
    spacing = np.asarray(source_geometry["physical_fov_mm_xyz"])/np.asarray(target[::-1])
    if (tuple(output["shape"][:3]) != target or any(n != 1 for n in output["shape"][3:])
            or tuple(candidate["shape"][:4]) != target+(count,)
            or any(n != 1 for n in candidate["shape"][4:])):
        raise ValueError("Metal CSM/image geometry mismatch")
    for record in execution["logs"]:
        checked_file(record, directory/"logs")
    for path, expected in zip(standard["command_records"], standard["expected_commands"], strict=True):
        if Path(path).resolve() != directory/"reconstruction/wave_command.txt" or Path(path).read_text().strip() != expected:
            raise ValueError("Metal recorded wave command changed")
    export_path = directory/"metadata/export.json"
    export = read_json(export_path)
    if export["status"] != "complete" or export["execution"] != file_record(execution_path):
        raise ValueError("Metal export is not bound to its completed execution")
    paths = [checked_file(record, directory/"nifti") for record in export["outputs"]]
    if len(paths) != 4 or set(paths) != {p.resolve() for p in export_files(directory/"nifti")}:
        raise ValueError("Metal export must contain exactly two NIfTI/JSON pairs")
    if "export_registration" in export:
        registration = checked_file(export["export_registration"], directory/"metadata")
        if read_json(registration)["execution"] != export["execution"]:
            raise ValueError("Metal export registration execution mismatch")
    pairs = {}
    affine = None
    for part in ("mag", "phase"):
        images = [p for p in paths if f"_part-{part}_" in p.name and p.name.endswith(".nii.gz")]
        if len(images) != 1:
            raise ValueError(f"Metal export requires exactly one {part} image")
        image = images[0]
        side = image.with_name(image.name.removesuffix(".nii.gz")+".json")
        if side not in paths:
            raise ValueError("Metal image sidecar is not hash bound")
        nifti = nib.load(image)
        if (nifti.shape != target[::-1] or not np.isfinite(nifti.affine).all()
                or not np.isfinite(np.asarray(nifti.dataobj)).all()
                or not np.allclose(nifti.header.get_zooms(), spacing, rtol=1e-4, atol=1e-5)
                or (affine is not None and not np.allclose(nifti.affine, affine, rtol=0, atol=1e-6))):
            raise ValueError("Metal NIfTI geometry/finite-value mismatch")
        affine = nifti.affine
        metadata = read_json(side).get("MetalCSM", {})
        if (metadata.get("Preparation") != file_record(preparation_path)
                or metadata.get("CaseCandidate") != candidate
                or metadata.get("NativeCandidate") != native["candidate"]
                or metadata.get("RegularizationLambda") != standard["regularization"]
                or metadata.get("RegularizationOptimizedForMetalCSM") is not False
                or metadata.get("GPU") is not gpu):
            raise ValueError("Metal NIfTI sidecar provenance mismatch")
        if root_request["source"].get("kind") == "automatic_native_inputs" and (
                metadata.get("SupportSelection") != "automatic"
                or metadata.get("SupportStatistics") != native["statistics"]):
            raise ValueError("Automatic support/no-op sidecar provenance mismatch")
        pairs[part] = (image, side)
    return {"case": case, "branch": directory.name, "method": standard["method"],
            "lambda": standard["regularization"], "pairs": pairs,
            "manifest": {"kind": "metal_csm", "path": str(export_path.relative_to(root)),
                         "path_relative_to": "output_root", "sha256": sha256_file(export_path),
                         "request": file_record(request_path), "preparation": file_record(preparation_path),
                         "execution": file_record(execution_path), "profile": standard["profile"],
                         "method": standard["method"], "lambda": standard["regularization"],
                         "prepared_manifest_sha256": standard["prepared_manifest"]["sha256"]}}


def discover_metal(root: Path) -> list[dict]:
    """Discover complete count-specific metal branches read-only, skipping unfinished branches.

    A completion manifest is an integrity claim: any mismatch raises instead of
    silently falling back. A shared advisory lock prevents inspecting a live
    pipeline writer. Empty/prepare-only/failed branches without exports are not
    selected. Returned records include exact native and branch provenance.
    """
    results = []
    for variant in sorted(root.iterdir()):
        match = re.fullmatch(r"vcc([1-9][0-9]*)_metal_csm", variant.name)
        if match is None or not variant.is_dir():
            continue
        branches = list(variant.glob("normal/bart_output/*/metadata/export.json"))
        branches += list(variant.glob("retro/*/bart_output/*/metadata/export.json"))
        if not branches:
            continue
        if variant.is_symlink() or any(p.is_symlink() for p in variant.rglob("*")):
            raise ValueError("Metal collection source must not contain symlinks")
        descriptor = os.open(variant/".metal.lock", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            count = int(match[1])
            request_path = variant/"metadata/request.json"
            request = read_json(request_path)
            automatic = request.get("mode") == "automatic"
            source = read_json(variant/"metadata/source.json") if automatic else request["source"]
            if (source["virtual_coils"] != count
                    or Path(source["normal_manifest"]["path"]).resolve() != root/f"vcc{count}/normal/bart_inputs/manifest.json"):
                raise ValueError("Metal variant has mismatched normal source, count or reference")
            if automatic:
                from .metal_auto_pipeline import automatic_source
                fresh_source = automatic_source(Path(source["normal_manifest"]["path"]), source["records"]["maps"],
                    source["ecalib_record"], source["ecalib_origin"], source["ecalib_crop"])
                if request["config"] != request["parameters"]:
                    raise ValueError("Automatic metal policy provenance mismatch")
                request = dict(request, source=source)
            else:
                checked_file(request["config"])
                fresh_source = source_contract(source["manifest"]["path"], source["reference"]["path"])
            if (source["virtual_coils"] != count
                    or Path(source["normal_manifest"]["path"]).resolve() != root/f"vcc{count}/normal/bart_inputs/manifest.json"
                    or fresh_source != source):
                raise ValueError("Metal variant has mismatched normal source, count or reference")
            # A collection reference must survive replacement of the collection itself.
            if not automatic and root/"nifti_collection" in Path(source["reference"]["path"]).parents:
                raise ValueError("Metal reference still resides in the collection; preserve a derivative-owned reference first")
            native_path = variant/"shared_csm/metadata/preparation.json"
            native = read_json(native_path)
            if (native["status"] != "metal_csm_prepared" or native["request"]["source"] != source
                    or native["request"]["parameters"] != request["parameters"]
                    or native["request"] != read_json(variant/"shared_csm/metadata/request.json")):
                raise ValueError("Metal native preparation source/configuration mismatch")
            validate_cfl(native["candidate"])
            checked_file(native["support"], variant)
            if automatic:
                checked_file(native["figure"], variant)
            completed = []
            identities = set()
            for export in sorted(branches):
                directory = export.parent.parent
                relative = directory.relative_to(variant).parts
                case = "normal" if relative[0] == "normal" else relative[1]
                if case not in ("normal", "native_r3x2", "lr_y_1p5mm_r3x2", "native_r3x3"):
                    raise ValueError("Unsupported metal retrospective case")
                group = validate_branch(directory, variant, count, case, request, native)
                identity = (case, group["method"])
                if identity in identities:
                    raise ValueError("Ambiguous completed metal candidates for the same case/method")
                identities.add(identity)
                completed.append(group)
            results.append({"count": count, "source": source, "branches": completed,
                            "provenance": {"variant_request": file_record(request_path),
                                           "native_preparation": file_record(native_path)}})
        except BlockingIOError as exc:
            raise RuntimeError("Metal pipeline is writing this variant; retry collection after it finishes") from exc
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Malformed completed metal provenance in {variant.name}: {exc}") from exc
        finally:
            os.close(descriptor)
    return results


def merge_metal(root: Path, variants: list[dict], previous: dict | None = None) -> None:
    """Overlay complete metal groups on the same VCC/method, preserving canonical names.

    Modify only in-memory collection discovery. Unique method matches retain
    their branch label and filenames; ambiguous Wavelet matches require a unique
    lambda match. Metal-only cases are added, never substituted for ROVir or
    another VCC. The optional validated previous collection preserves names
    after source-NIfTI pruning. File overrides contain expected copy hashes.
    """
    for metal in discover_metal(root):
        count = metal["count"]
        variant = next((v for v in variants if v["collection_id"] == f"vcc{count}"), None)
        if variant is None:
            source = metal["source"]
            normal = read_json(source["normal_manifest"]["path"])
            variant = {"collection_id": f"vcc{count}", "kind": "standard_pca",
                       "collection_relative": Path(f"vcc{count}"), "sampling_class": normal["sampling"]["name"],
                       "sources": {}, "run_manifests": {},
                       "source_contract": {"virtual_coils": count, "source_manifest": str(Path(source["normal_manifest"]["path"]).relative_to(root)),
                                           "source_manifest_sha256": source["normal_manifest"]["sha256"],
                                           "physical_coils": source["compression"]["physical_coils"],
                                           "retained_energy": source["compression"]["retained_energy"],
                                           "compression_basis": source["compression"]["basis"], "geometry": source["geometry"],
                                           "source": source["raw_source_provenance"]}}
            variants.append(variant)
        overrides = variant.setdefault("pair_overrides", {})
        selections = variant.setdefault("metal_case_groups", [])
        variant["source_contract"]["metal_csm_preference"] = metal["provenance"]
        for group in metal["branches"]:
            case = group["case"]
            previous_cases = [c for c in (previous or {}).get("cases", [])
                              if c.get("collection_id") == f"vcc{count}" and c["case"] == case
                              and c.get("reconstruction_manifest", {}).get("method") == group["method"]]
            candidates = [branch for branch, cases in variant["sources"].items() if case in cases
                          and variant["run_manifests"].get(f"{branch}:{case}", {}).get("method") == group["method"]]
            if len(candidates) > 1:
                candidates = [branch for branch in candidates
                              if variant["run_manifests"][f"{branch}:{case}"]["lambda"] == group["lambda"]]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous standard Wavelet targets for metal collection preference")
            branch = candidates[0] if candidates else group["branch"]
            prior = None
            if not candidates and previous_cases:
                if len(previous_cases) > 1:
                    previous_cases = [c for c in previous_cases if c["reconstruction_manifest"]["lambda"] == group["lambda"]]
                if len(previous_cases) != 1:
                    raise ValueError("Ambiguous historical collection targets for metal preference")
                prior = previous_cases[0]
                branch = prior["branch"]
            key = f"{branch}:{case}"
            replaced = variant["run_manifests"].get(key)
            if replaced is None and prior is not None:
                replaced = prior.get("reconstruction_manifest")
            pairs = variant["sources"].setdefault(branch, {}).setdefault(case, list(group["pairs"].values()))
            parts = ["mag" if "_part-mag_" in p.name else "phase" if "_part-phase_" in p.name else None for p, _ in pairs]
            if sorted(p for p in parts if p is not None) != ["mag", "phase"] or len(parts) != 2:
                raise ValueError("Metal replacement requires one canonical magnitude/phase pair")
            for (canonical, _), part in zip(pairs, parts, strict=True):
                selected, sidecar = group["pairs"][part]
                overrides[str(canonical)] = {
                    "nifti": selected, "sidecar": sidecar, "selection": "metal_csm",
                    "expected_nifti": file_record(selected), "expected_sidecar": file_record(sidecar),
                    "metal_csm_provenance": {**metal["provenance"], "selected_branch": group["manifest"],
                                             "replaced_standard_branch": replaced,
                                             "regularization_optimized_for_metal_csm": False}}
                if prior is not None:
                    names = [Path(f["original_nifti"]).name for f in prior["files"]
                             if f"_part-{part}_" in Path(f["original_nifti"]).name]
                    if len(names) != 1:
                        raise ValueError("Historical collection does not contain an unambiguous paired filename")
                    overrides[str(canonical)]["output_name"] = names[0]
            variant["run_manifests"][key] = group["manifest"]
            selections.append(key)
        if not any("normal" in cases for cases in variant["sources"].values()):
            raise FileNotFoundError("A collected VCC requires a complete standard or metal normal branch")


def reject_metal_downgrade(previous: dict | None, variants: list[dict]) -> None:
    """Reject silent replacement of a previously selected metal group with a standard fallback."""
    available = {f"{v['collection_id']}:{key}" for v in variants for key in v.get("metal_case_groups", [])}
    for case in (previous or {}).get("cases", []):
        if any(f.get("collection_source_selection") == "metal_csm" for f in case.get("files", [])):
            key = f"{case['collection_id']}:{case['branch']}:{case['case']}"
            if key not in available:
                raise FileNotFoundError(f"Previously collected metal branch is now missing/incomplete: {key}")
