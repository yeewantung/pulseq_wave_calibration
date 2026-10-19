"""Explicit automatic MPRAGE metal preparation from raw data or compatible standard inputs."""

from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np

from .bart_io import cfl_record, sha256_file
from .metal_auto_csm import AUTO_POLICY, write_auto_figure, write_automatic_candidate
from .metal_runner import memory_guard, run_reconstruction
from .metal_workflow import file_record, read_json, validate_cfl, write_json
from .standard import array_sha256, validate_prepared_artifact_records, validate_standard_pca_manifest
from .relocation import read_command, relocation_record


def preparation_options(args) -> dict:
    """Return only the standard count, whitening and PSF flags shared by preparation APIs."""
    return {key: getattr(args, key) for key in (
        "virtual_coils", "noise_prewhitening", "psf_coefficient_processing",
        "psf_fit_kx_min", "psf_fit_kx_max", "psf_fit_y_min", "psf_fit_y_max", "psf_fit_z_min", "psf_fit_z_max")}


def select_psf_donor(root: Path, args) -> Path | None:
    """Find an accepted coil-independent source only if the requested vccN input is absent.

    A present requested variant is always validated by the standard preparation
    API. Alternate standard/ROVir anchors are never relabeled as vccN maps.
    A bound but corrupt ROVir source raises instead of causing a silent refit.
    """
    from .rovir_source import discover_rovir_source_inputs, resolve_rovir_source_inputs, ROVIR_SOURCE_BINDING
    from .mprage import _normalize_psf_coefficient_settings, _normalize_psf_spatial_settings
    from .metal_psf_reuse import accepted_psf

    if (root/f"vcc{args.virtual_coils}/normal/bart_inputs/manifest.json").exists():
        return None
    candidates = discover_rovir_source_inputs(root)
    if (root/ROVIR_SOURCE_BINDING).exists():
        bound = resolve_rovir_source_inputs(root)
        candidates = (bound,) + tuple(p for p in candidates if p != bound)
    if not candidates:
        return None
    settings = _normalize_psf_coefficient_settings(args.psf_coefficient_processing, args.psf_fit_kx_min, args.psf_fit_kx_max)
    settings.update(_normalize_psf_spatial_settings(args.psf_fit_y_min, args.psf_fit_y_max, args.psf_fit_z_min, args.psf_fit_z_max))
    # A source tree is an integrity claim. Do not silently ignore a corrupt or
    # source-incompatible candidate and rebuild over the same dataset root.
    accepted_psf(candidates[0], args.twix.resolve(), args.seq.resolve(), settings)
    return candidates[0]


def automatic_source(normal_path: Path, maps: dict, ecalib: dict, origin: dict, crop: float) -> dict:
    """Validate native prepared inputs, PCA and CSM lineage without requiring a reconstructed image.

    Return hash-bound geometry and provenance consumed by reconstruction and
    collection. Reject mismatches/nonfinite inputs and verify the exact ecalib
    command against its calibration source and selected crop.
    """
    from .mprage import _uses_alias_free_coil_calibration

    normal_path = normal_path.resolve()
    normal = read_json(normal_path)
    count = int(normal["coil_compression"]["virtual_coils"])
    if (normal.get("status") != "measured_wave_mprage_bart_inputs_ready"
            or not _uses_alias_free_coil_calibration(normal)
            or normal.get("sampling", {}).get("name") != "R3x1"):
        raise ValueError("Automatic metal mode requires native R3x1 standard-PCA MPRAGE inputs")
    compression = validate_standard_pca_manifest(normal, count)
    validate_prepared_artifact_records(normal_path.parent, normal["output_artifacts"])
    basis_path = normal_path.parent/compression["basis"]["path"]
    basis = np.load(basis_path, allow_pickle=False)
    if (basis.shape != (compression["physical_coils"], count) or not np.isfinite(basis).all()
            or sha256_file(basis_path) != compression["basis"]["file_sha256"]
            or array_sha256(basis) != compression["basis"]["logical_sha256"]):
        raise ValueError("Automatic metal PCA basis mismatch")
    records = {"maps": validate_cfl(maps)}
    for key, name in (("calibration", "kspace_calib"), ("psf", "psf"), ("kspace", "wave_kspace")):
        records[key] = validate_cfl(cfl_record(normal_path.parent/name))
        if any(records[key].get(k) != v for k, v in normal["output_artifacts"][name].items()):
            raise ValueError("Automatic metal prepared artifact mismatch")
    matrix = tuple(normal["geometry"]["logical_matrix_ro_lin_par"])
    os_ro = matrix[0]*normal["geometry"]["readout_oversampling_factor"]
    for key in ("maps", "calibration"):
        if tuple(records[key]["shape"][:4]) != matrix+(count,) or any(n != 1 for n in records[key]["shape"][4:]):
            raise ValueError("Automatic CSM/ACS grid or coil count mismatch")
    if (tuple(records["kspace"]["shape"][:4]) != (os_ro, *matrix[1:], count)
            or tuple(records["psf"]["shape"][:3]) != (os_ro, *matrix[1:])
            or any(n != 1 for n in records["kspace"]["shape"][4:])
            or any(n != 1 for n in records["psf"]["shape"][3:])):
        raise ValueError("Automatic metal PSF/k-space grid mismatch")
    if file_record(ecalib["path"]) != ecalib or file_record(origin["record"]["path"]) != origin["record"]:
        raise ValueError("Automatic CSM calibration provenance changed")
    tokens = shlex.split(read_command(ecalib["path"]))
    if (len(tokens) != 8 or Path(tokens[0]).name != "bart" or tokens[1:5] != ["ecalib", "-m", "1", "-c"]
            or float(tokens[5]) != crop or Path(tokens[6]).resolve() != Path(records["calibration"]["base"])
            or Path(tokens[7]).resolve() != Path(maps["base"])):
        raise ValueError("Automatic metal ecalib source/crop/command mismatch")
    bound = read_json(origin["record"]["path"])
    if origin["kind"] == "completed_standard_branch":
        request = bound["request"]
        if (bound.get("status") != "standard_pca_reconstruction_branch_complete" or request["case"] != "normal"
                or request["normal_manifest"]["sha256"] != sha256_file(normal_path)
                or Path(request["normal_manifest"]["path"]).resolve() != normal_path
                or request["virtual_coils"] != count or request["compression_basis"] != compression["basis"]
                or validate_cfl(request["inputs"]["maps"]) != maps):
            raise ValueError("Original CSM is not bound to matching standard inputs")
    elif origin["kind"] == "isolated_ecalib":
        preparation_path = Path(origin["record"]["path"]).with_name("preparation.json")
        preparation = read_json(preparation_path)
        if (bound.get("status") != "complete" or bound["identity"]["gpu"] is not False
                or bound["output"] != maps or bound["identity"]["command"] != tokens
                or bound["identity"]["preparation_sha256"] != sha256_file(preparation_path)
                or preparation["request"]["source"]["records"]["calibration"] != records["calibration"]
                or preparation["request"]["normal_manifest"] != file_record(normal_path)):
            raise ValueError("Isolated CSM calibration binding mismatch")
        for record in bound["logs"]:
            if file_record(record["path"]) != record:
                raise ValueError("Isolated CSM calibration log changed")
    else:
        raise ValueError("Unknown CSM origin")
    return {"kind": "automatic_native_inputs", "normal_manifest": file_record(normal_path),
            "basis": file_record(basis_path), "records": records, "ecalib_record": ecalib,
            "ecalib_origin": origin, "ecalib_crop": crop, "geometry": normal["geometry"],
            "virtual_coils": count, "compression": compression, "raw_source_provenance": normal["source"],
            "path_relocation": relocation_record(normal_path)}


def resolve_csm(normal: Path, destination: Path, args) -> tuple[dict, dict, dict]:
    """Reuse hash-bound standard CSM or run isolated CPU ecalib; never reconstruct a baseline image."""
    from .metal_pipeline import locked_directory

    original = normal.parent.parent/"bart_output"
    maps = original/"coil_sens"
    command_record = original/"ecalib_command.txt"
    if any(p.exists() for p in (maps.with_suffix(".cfl"), maps.with_suffix(".hdr"), command_record)):
        if not all(p.is_file() for p in (maps.with_suffix(".cfl"), maps.with_suffix(".hdr"), command_record)):
            raise ValueError("Partial original CSM; refusing overwrite or implicit recalibration")
        completed = sorted(original.glob("*/reconstruction_manifest.json"))
        if not completed:
            raise ValueError("Original CSM has no completed hash-bound branch; cannot safely adopt it")
        origin = {"kind": "completed_standard_branch", "record": file_record(completed[0])}
        return cfl_record(maps), file_record(command_record), origin
    directory = destination/"source_calibration"
    calibration = validate_cfl(cfl_record(normal.parent/"kspace_calib"))
    request = {"source": {"records": {"calibration": calibration}}, "ecalib_crop": args.ecalib_crop,
               "normal_manifest": file_record(normal), "threads": args.cpu_threads,
               "implementation": file_record(Path(__file__))}
    if args.metal_prepare_only and not (directory/"metadata/reconstruction.json").exists():
        raise ValueError("Automatic prepare-only needs existing hash-bound CSM; omit --metal-prepare-only to permit CPU ecalib")
    with locked_directory(directory, request):
        preparation = {"request": request}
        path = directory/"metadata/preparation.json"
        if path.exists():
            if read_json(path) != preparation:
                raise ValueError("CSM preparation mismatch")
        else:
            write_json(path, preparation)
        execution = run_reconstruction(directory, preparation, args.cpu_threads, operation="ecalib")
        record = directory/"reconstruction/ecalib_command.txt"
        command = shlex.join(execution["identity"]["command"])
        if record.exists():
            if record.read_text().strip() != command:
                raise ValueError("Isolated ecalib command changed")
        else:
            with record.open("x") as stream:
                stream.write(command+"\n")
        return execution["output"], file_record(record), {
            "kind": "isolated_ecalib", "record": file_record(directory/"metadata/reconstruction.json")}


def prepare_auto_maps(destination: Path, source: dict) -> dict:
    """Create/resume hash-bound automatic maps, no-op support and QC without any Wave reconstruction."""
    from .metal_pipeline import locked_directory

    directory = destination/"shared_csm"
    request = {"source": source, "parameters": AUTO_POLICY,
               "implementation": file_record(Path(__file__).with_name("metal_auto_csm.py"))}
    with locked_directory(directory, request, folders=("metadata", "intermediate", "figures")):
        path = directory/"metadata/preparation.json"
        if path.exists():
            saved = read_json(path)
            if saved["request"] != request:
                raise ValueError("Automatic CSM preparation mismatch")
            validate_cfl(saved["candidate"])
            for record in (saved["support"], saved["figure"]):
                if file_record(record["path"]) != record:
                    raise ValueError("Automatic CSM QC provenance changed")
            return saved
        if any((directory/"intermediate").iterdir()) or any((directory/"figures").iterdir()):
            raise ValueError("Partial automatic CSM preparation; refusing overwrite")
        geometry = source["geometry"]
        spacing = np.asarray(geometry["physical_fov_mm_xyz"])[::-1]/np.asarray(geometry["logical_matrix_ro_lin_par"])
        stats, arrays = write_automatic_candidate(Path(source["records"]["maps"]["base"]),
            Path(source["records"]["calibration"]["base"]), directory, tuple(spacing), memory_guard)
        figure = directory/"figures/automatic_support.png"
        write_auto_figure(figure, arrays)
        saved = {"status": "metal_csm_prepared", "request": request, "statistics": stats,
                 "candidate": validate_cfl(cfl_record(directory/"intermediate/coil_sens")),
                 "support": file_record(directory/"intermediate/support.npz"), "figure": file_record(figure)}
        write_json(path, saved)
        print(f"Automatic metal CSM: {stats['status']}; added={stats['added_voxels']}; reason={stats['reason']}", flush=True)
        return saved


def run_automatic_pipeline(args) -> Path:
    """Prepare/reuse native inputs, extend eligible support, and run only requested metal branches.

    Explicit metal opt-in is validated before I/O. A stable root request locks
    source paths/settings/algorithm; existing output or preparation mismatch is
    never cleared. Outputs use the user-selected vccN_metal_csm layout.
    """
    from .metal_pipeline import locked_directory, validate_options, branch_settings, prepare_case_maps, run_branch
    from .mprage import _file_identity, prepare_normal_mprage, prepare_retro_mprage, prepare_retro_mprage_r3x3

    validate_options(args)
    root = args.output.expanduser().resolve()
    destination = root/f"vcc{args.virtual_coils}_metal_csm"
    options = preparation_options(args)
    request = {"mode": "automatic", "config": AUTO_POLICY, "parameters": AUTO_POLICY,
               "raw": _file_identity(args.twix.resolve()), "sequence": _file_identity(args.seq.resolve(), include_hash=True),
               "options": options, "ecalib_crop": args.ecalib_crop, "cpu_threads": args.cpu_threads,
               "path_relocation": relocation_record(root / f"vcc{args.virtual_coils}/normal/bart_inputs/manifest.json"),
               "implementation": {name: file_record(Path(__file__).with_name(name)) for name in (
                   "metal_auto_pipeline.py", "metal_auto_csm.py", "metal_pipeline.py", "metal_runner.py", "metal_psf_reuse.py",
                   "relocation.py")}}
    memory_guard()
    with locked_directory(destination, request, folders=("metadata",)):
        donor = select_psf_donor(root, args)
        prepare_normal_mprage(args.twix, root, args.seq, **options,
                              allow_legacy_artifact_reuse=True, accepted_psf_inputs=donor)
        if args.mode == "retro":
            prepare_retro_mprage(args.twix, root, args.seq, **options)
        if args.mode in ("retro", "r3x3"):
            prepare_retro_mprage_r3x3(args.twix, root, args.seq, **options)
        normal = root/f"vcc{args.virtual_coils}/normal/bart_inputs/manifest.json"
        maps, ecalib, origin = resolve_csm(normal, destination, args)
        source = automatic_source(normal, maps, ecalib, origin, args.ecalib_crop)
        if source["virtual_coils"] != args.virtual_coils:
            raise ValueError("Requested metal coil count differs from prepared inputs")
        source_path = destination/"metadata/source.json"
        if source_path.exists():
            if read_json(source_path) != source:
                raise ValueError("Automatic source changed; preserve existing outputs")
        else:
            write_json(source_path, source)
        native = prepare_auto_maps(destination, source)
        settings = {"config": AUTO_POLICY}
        for case, branch, method, value, profile in branch_settings(args.mode, args.profile, args.virtual_coils, args.r3_lambda):
            prepared = normal if case == "normal" else root/f"vcc{args.virtual_coils}/retro/{case}/bart_inputs/manifest.json"
            candidate = prepare_case_maps(destination, case, prepared, native)
            run_branch(destination, case, branch, method, value, profile, prepared, native, candidate, settings, args)
        if automatic_source(normal, maps, ecalib, origin, args.ecalib_crop) != source:
            raise ValueError("Automatic source changed during reconstruction")
    return destination
