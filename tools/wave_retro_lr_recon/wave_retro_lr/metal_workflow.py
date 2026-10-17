"""Hash-bound preparation and NIfTI export for isolated MPRAGE metal derivatives.

These functions never launch BART or select a production output directory.
"""

from __future__ import annotations

import itertools
import json
import shlex
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .bart_io import cfl_record, open_cfl, sha256_file
from .metal_csm import METHOD_VERSION, MetalCSMParameters, orient, write_candidate
from .standard import array_sha256, validate_prepared_artifact_records, validate_standard_pca_manifest


def read_json(path: str | Path) -> dict:
    """Read a JSON object from path; reject non-object documents."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    """Write a new JSON artifact exclusively; never overwrite an existing file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_record(path: str | Path) -> dict:
    """Return resolved file path, size and SHA-256 for a non-mutating provenance check."""
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_cfl(record: dict) -> dict:
    """Verify a CFL's recorded geometry, size, hashes and all finite complex values.

    Args:
        record: Existing CFL provenance, including base and both hashes.
    Returns:
        Current complete CFL record; raises ValueError on any mismatch.
    """
    actual = cfl_record(record["base"])
    if any(record.get(key) != value for key, value in actual.items()):
        raise ValueError(f"CFL provenance mismatch: {record['base']}")
    values = open_cfl(record["base"]).ravel(order="F")
    for start in range(0, values.size, 1024**2):
        if not np.isfinite(values[start:start+1024**2]).all():
            raise ValueError(f"Nonfinite CFL: {record['base']}")
    return actual


def source_contract(manifest_path: str | Path, reference_path: str | Path) -> dict:
    """Validate a completed standard-PCA native R3x1 MPRAGE branch and original magnitude export.

    Source NIfTI trees may have been pruned: an exact collection copy is accepted
    by hashes, not by filename alone. ROVir, GRE, retro, N4 and multi-map sources
    are intentionally unsupported. No raw-data or PSF preparation is repeated.

    Args:
        manifest_path: Completed normal branch reconstruction manifest.
        reference_path: Original magnitude NIfTI, with its matching JSON sidecar.
    Returns:
        Exact source records, verified orientation, geometry and shared magnitude scale.
    Raises:
        ValueError: Source, count, geometry, finite-value or provenance mismatch.
    """
    manifest_path, reference_path = Path(manifest_path).resolve(), Path(reference_path).resolve()
    run = read_json(manifest_path)
    request = run.get("request", {})
    if run.get("status") != "standard_pca_reconstruction_branch_complete" or request.get("case") != "normal":
        raise ValueError("Metal workflow requires a completed standard-PCA normal branch")
    normal_path = Path(request["normal_manifest"]["path"]).resolve()
    normal = read_json(normal_path)
    if normal.get("status") != "measured_wave_mprage_bart_inputs_ready":
        raise ValueError("Only measured MPRAGE preparation is supported; not GRE/ROVir")
    from .mprage import _uses_alias_free_coil_calibration

    if not _uses_alias_free_coil_calibration(normal):
        raise ValueError("Source lacks the accepted alias-free coil calibration contract")
    for identity in (request["normal_manifest"], request["prepared_manifest"]):
        if Path(identity["path"]).resolve() != normal_path or sha256_file(normal_path) != identity["sha256"]:
            raise ValueError("Normal preparation manifest mismatch")
    for field in ("geometry", "source", "sampling", "standard_pca_variant"):
        if normal.get(field) != request.get(field):
            raise ValueError(f"Preparation/reconstruction {field} mismatch")
    if request["sampling"].get("acceleration_lin_par") != [3, 1]:
        raise ValueError("Only native R3x1 is supported")
    compression = validate_standard_pca_manifest(normal, request["virtual_coils"])
    if request["compression_basis"] != compression["basis"]:
        raise ValueError("PCA basis identity mismatch")
    validate_prepared_artifact_records(normal_path.parent, normal["output_artifacts"])
    basis_path = normal_path.parent/compression["basis"]["path"]
    basis = np.load(basis_path, allow_pickle=False)
    if (basis.shape != (compression["physical_coils"], request["virtual_coils"])
            or not np.isfinite(basis).all() or sha256_file(basis_path) != compression["basis"]["file_sha256"]
            or array_sha256(basis) != compression["basis"]["logical_sha256"]):
        raise ValueError("Invalid PCA basis shape, values or hash")
    if len(run["outputs"]) != 1 or len(request["inputs"]["psf_by_echo"]) != 1 or len(request["inputs"]["kspace_by_echo"]) != 1:
        raise ValueError("Only single-echo native MPRAGE is supported")
    records = {"maps": request["inputs"]["maps"], "image": run["outputs"][0],
               "psf": request["inputs"]["psf_by_echo"][0], "kspace": request["inputs"]["kspace_by_echo"][0]}
    for key, value in records.items():
        records[key] = validate_cfl(value)
    for key, artifact in (("psf", "psf"), ("kspace", "wave_kspace")):
        expected = normal["output_artifacts"][artifact]
        if any(records[key].get(k) != v for k, v in expected.items()):
            raise ValueError("Prepared PSF/k-space differs from completed reconstruction")
    matrix = tuple(normal["geometry"]["logical_matrix_ro_lin_par"])
    count = request["virtual_coils"]
    if (tuple(records["maps"]["shape"][:4]) != matrix+(count,)
            or any(n != 1 for n in records["maps"]["shape"][4:])
            or tuple(records["image"]["shape"][:3]) != matrix
            or any(n != 1 for n in records["image"]["shape"][3:])):
        raise ValueError("Source CSM/image geometry or coil count mismatch")
    os_ro = matrix[0]*normal["geometry"]["readout_oversampling_factor"]
    if (tuple(records["kspace"]["shape"][:4]) != (os_ro, matrix[1], matrix[2], count)
            or tuple(records["psf"]["shape"][:3]) != (os_ro, matrix[1], matrix[2])
            or any(n != 1 for n in records["kspace"]["shape"][4:])
            or any(n != 1 for n in records["psf"]["shape"][3:])):
        raise ValueError("PSF/k-space geometry mismatch")
    sidecar_path = reference_path.with_name(reference_path.name.replace(".nii.gz", ".json"))
    if not reference_path.name.endswith(".nii.gz") or "part-mag" not in reference_path.name:
        raise ValueError("Reference must be an original magnitude .nii.gz export")
    for path in (reference_path, sidecar_path):
        matches = [r for r in run["nifti_outputs"] if Path(r["path"]).name == path.name]
        if len(matches) != 1 or matches[0]["sha256"] != sha256_file(path) or matches[0]["size_bytes"] != path.stat().st_size:
            raise ValueError("Reference is not an exact original export/collection copy (N4 is unsupported)")
    reference = nib.load(reference_path)
    values = reference.get_fdata(dtype=np.float32)
    if reference.shape != matrix[::-1] or not np.isfinite(values).all() or not np.isfinite(reference.affine).all():
        raise ValueError("Reference geometry/finite-value mismatch")
    if abs(np.linalg.det(reference.affine[:3, :3])) < 1e-12:
        raise ValueError("Singular reference affine")
    spacing = np.asarray(reference.header.get_zooms(), dtype=float)
    expected_spacing = np.asarray(normal["geometry"]["physical_fov_mm_xyz"])/np.asarray(reference.shape)
    if not np.allclose(spacing, expected_spacing, rtol=1e-4, atol=1e-5):
        raise ValueError("Reference voxel spacing differs from prepared physical FOV")
    columns = reference.affine[:3, :3]/spacing
    if not np.allclose(columns.T@columns, np.eye(3), atol=1e-4):
        raise ValueError("Sheared reference grids are unsupported")
    sidecar = read_json(sidecar_path)
    scale = normal["echoes"][0]["wave_kspace_norm"]/sidecar["MagnitudeNormalization"]["InputPercentileValue"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid shared reference scale")
    image = abs(open_cfl(records["image"]["base"]).reshape(matrix, order="F"))
    target = values[::4, ::4, ::4]
    errors = [(float(np.linalg.norm(orient(image, flips)[::4, ::4, ::4]*scale-target)/max(np.linalg.norm(target), 1e-20)), flips)
              for flips in itertools.product((False, True), repeat=3)]
    error, flips = min(errors)
    full_error = float(np.linalg.norm(orient(image, flips)*scale-values)/max(np.linalg.norm(values), 1e-20))
    if not np.isfinite(error) or not np.isfinite(full_error) or not np.any(values > 0) or error > 1e-5 or full_error > 1e-5:
        raise ValueError("Reference orientation or normalization does not match complex baseline")
    regularization = float(request["regularization"])
    if not np.isfinite(regularization) or regularization < 0 or request["method"] not in ("wavelet", "fista"):
        raise ValueError("Invalid baseline reconstruction settings")
    if request["method"] == "fista" and regularization != 0:
        raise ValueError("FISTA-r0 source must use zero regularization")
    commands = request.get("expected_commands", [])
    if len(commands) != 1:
        raise ValueError("A single recorded baseline wave command is required")
    tokens = shlex.split(commands[0])
    tokens = [token for token in tokens if token != "-g"]
    if (len(tokens) != 14 or Path(tokens[0]).name != "bart"
            or tokens[1:5] != ["wave", "-w", "-f", "-r"]
            or float(tokens[5]) != regularization or tokens[6:9] != ["-i", "100", "-t"]
            or float(tokens[9]) != 1e-6):
        raise ValueError("Baseline must use wave -w -f, 100 iterations and tolerance 1e-6")
    if [Path(p).resolve() for p in tokens[10:]] != [Path(records[k]["base"]).resolve() for k in ("maps", "psf", "kspace", "image")]:
        raise ValueError("Baseline command/source array paths mismatch")
    command_records = request.get("command_records", [])
    if len(command_records) != 1 or Path(command_records[0]).read_text().strip() != commands[0]:
        raise ValueError("Baseline command record changed or is missing")
    return {"manifest": file_record(manifest_path), "normal_manifest": file_record(normal_path),
            "basis": file_record(basis_path), "command_record": file_record(command_records[0]),
            "reference": file_record(reference_path),
            "reference_sidecar": file_record(sidecar_path), "records": records,
            "geometry": normal["geometry"], "virtual_coils": count,
            "compression": compression, "raw_source_provenance": normal["source"],
            "regularization": regularization, "method": request["method"],
            "orientation_flips": list(flips), "orientation_error": full_error,
            "spacing_mm": spacing.tolist(), "shared_magnitude_scale": scale}


def validate_destination(output: Path, source: dict) -> None:
    """Reject output locations inside frozen source/collection trees or their ancestors.

    Args:
        output: Explicit user-supplied output directory.
        source: Validated source contract.
    Returns:
        None; raises ValueError for overlapping or canonical collection paths.
    """
    output = output.resolve()
    protected = [Path(source["normal_manifest"]["path"]).parent.parent,
                 Path(source["reference"]["path"]).parent]
    if "nifti_collection" in output.parts or any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Metal output must be independent of normal/retro/collection source trees")
    if any(name in output.parts for name in ("bart_inputs", "bart_output", "retro")):
        raise ValueError("Use an independent diagnostic/derivative output directory")


def prepare_candidate(output: Path, request: dict, checkpoint: Callable[[], None] | None = None) -> dict:
    """Prepare or strictly reuse candidate maps in an initialized experiment directory.

    Args:
        output: Explicit derivative root, already locked by the caller.
        request: Immutable source, parameters and method-version contract.
        checkpoint: Optional resource guard passed to map preparation.
    Returns:
        Completed preparation record; partial/mismatched outputs raise instead of overwriting.
    """
    manifest_path = output/"metadata/preparation.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["request"] != request or manifest.get("status") != "metal_csm_prepared":
            raise ValueError("Metal CSM request mismatch; use a separate output directory")
        validate_cfl(manifest["candidate"])
        if file_record(manifest["support"]["path"]) != manifest["support"]:
            raise ValueError("Support archive mismatch")
        return manifest
    if any((output/"intermediate").iterdir()):
        raise ValueError("Partial candidate preparation; preserve it and use a fresh output directory")
    source = request["source"]
    stats, native, added = write_candidate(
        source["records"]["maps"]["base"], output/"intermediate/coil_sens",
        tuple(source["orientation_flips"]), tuple(source["spacing_mm"]),
        MetalCSMParameters.from_dict(request["parameters"]), checkpoint=checkpoint,
    )
    support_path = output/"intermediate/support.npz"
    np.savez_compressed(support_path, native=native, added=added)
    manifest = {"status": "metal_csm_prepared", "request": request, "checks": stats,
                "candidate": cfl_record(output/"intermediate/coil_sens"), "support": file_record(support_path)}
    write_json(manifest_path, manifest)
    return manifest


def export_result(output: Path, preparation: dict, execution: dict) -> dict:
    """Export shared-scale magnitude/phase/support NIfTIs and quantitative comparisons.

    Inputs are completed, hash-bound preparation and execution records. Output
    geometry, qform/sform and magnitude scale match the baseline exactly. No N4,
    tissue mask, automatic winner selection or canonical collection mutation occurs.
    Partial exports are rejected; complete exports are validated and reused.
    """
    manifest_path = output/"metadata/comparison.json"
    if execution.get("status") != "complete":
        raise ValueError("Reconstruction has not completed")
    validate_cfl(execution["output"])
    if manifest_path.exists():
        result = read_json(manifest_path)
        if result["preparation_sha256"] != sha256_file(output/"metadata/preparation.json") or result["execution_sha256"] != sha256_file(output/"metadata/reconstruction.json"):
            raise ValueError("Export provenance mismatch")
        for record in result["outputs"]:
            if file_record(record["path"]) != record:
                raise ValueError("Export output hash mismatch")
        return result
    if any((output/"nifti").iterdir()) or any((output/"figures").iterdir()):
        raise ValueError("Partial export exists; preserve it and use a fresh output directory")
    source = preparation["request"]["source"]
    if file_record(source["reference"]["path"]) != source["reference"]:
        raise ValueError("Reference NIfTI changed")
    reference = nib.load(source["reference"]["path"])
    baseline = reference.get_fdata(dtype=np.float32)
    matrix = tuple(source["geometry"]["logical_matrix_ro_lin_par"])
    image = open_cfl(execution["output"]["base"])
    if tuple(image.shape[:3]) != matrix or any(n != 1 for n in image.shape[3:]):
        raise ValueError("Reconstruction output geometry mismatch")
    image = image.reshape(matrix, order="F")
    flips = tuple(source["orientation_flips"])
    magnitude = orient(abs(image), flips)*source["shared_magnitude_scale"]
    with np.load(preparation["support"]["path"], allow_pickle=False) as support:
        native = orient(support["native"], flips)
        added = orient(support["added"], flips)
    outputs = []
    provenance = {"Experimental": True, "MethodVersion": METHOD_VERSION,
                  "MetalOnlyOptIn": True, "DefaultEnabled": False, "GPU": False,
                  "N4BiasCorrection": False, "VirtualCoils": source["virtual_coils"],
                  "Lambda": source["regularization"], "SourceReference": source["reference"],
                  "SharedMagnitudeScale": source["shared_magnitude_scale"],
                  "IndependentlyNormalized": False,
                  "PreparationSHA256": sha256_file(output/"metadata/preparation.json"),
                  "ExecutionSHA256": sha256_file(output/"metadata/reconstruction.json")}
    for name, values in (("magnitude", magnitude), ("phase", orient(np.angle(image), flips)), ("added_support", added.astype(np.uint8))):
        path = output/f"nifti/metal_csm_{name}.nii.gz"
        header = reference.header.copy()
        header.set_data_dtype(values.dtype)
        header.set_slope_inter(1, 0)
        result = nib.Nifti1Image(values, reference.affine, header)
        for form in ("qform", "sform"):
            affine, code = getattr(reference, f"get_{form}")(coded=True)
            if affine is not None:
                getattr(result, f"set_{form}")(affine, int(code))
        nib.save(result, path)
        check = nib.load(path)
        if check.shape != reference.shape or not np.array_equal(check.affine, reference.affine) or not np.isfinite(np.asarray(check.dataobj)).all():
            raise ValueError("Export finite-value/geometry validation failed")
        sidecar = path.with_name(path.name.replace(".nii.gz", ".json"))
        write_json(sidecar, dict(provenance, Part=name))
        outputs.extend((file_record(path), file_record(sidecar)))
    metrics = {}
    for name, region in {"added_support": added,
                         "native_interior_5mm": ndi.distance_transform_edt(native, sampling=source["spacing_mm"]) > 5,
                         "remote_background_15mm": ndi.distance_transform_edt(~(native | added), sampling=source["spacing_mm"]) > 15}.items():
        a, b = baseline[region].astype(float), magnitude[region].astype(float)
        metrics[name] = {"voxel_count": len(a), "baseline_rms": float(np.sqrt(np.mean(a*a))) if len(a) else None,
                         "candidate_rms": float(np.sqrt(np.mean(b*b))) if len(a) else None,
                         "mae_shared_baseline_units": float(np.mean(abs(b-a))) if len(a) else None}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = preparation["request"]["parameters"]
    axis = parameters["envelope_axis"]
    low, high = parameters["roi_voxels"][axis]
    indices = sorted(set((low+(high-low)//4, (low+high)//2, high-1-(high-low)//4)))
    figure, axes = plt.subplots(len(indices), 3, squeeze=False, figsize=(12, 4*len(indices)))
    vmax = max(float(np.percentile(baseline[native], 99)), 1e-12)
    for row, index in enumerate(indices):
        for col, (values, title) in enumerate(((baseline, "Baseline"), (magnitude, "Metal CSM candidate"), (magnitude-baseline, "Difference"))):
            ax = axes[row, col]
            ax.imshow(np.take(values, index, axis=axis).T, origin="lower", cmap="coolwarm" if col == 2 else "gray",
                      vmin=-vmax*.25 if col == 2 else 0, vmax=vmax*.25 if col == 2 else vmax)
            if col < 2:
                ax.contour(np.take(native, index, axis=axis).T, levels=[.5], colors="cyan", linewidths=.5)
            ax.set_title(f"{title}; reference axis {axis}, index {index}")
            ax.axis("off")
    figure.suptitle("Shared baseline scale; visible recovery is not anatomical ground truth")
    figure.tight_layout()
    figure_path = output/"figures/comparison.png"
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    outputs.append(file_record(figure_path))
    report = {"status": "complete", "preparation_sha256": provenance["PreparationSHA256"],
              "execution_sha256": provenance["ExecutionSHA256"], "outputs": outputs, "metrics": metrics,
              "limitations": ["Historical baseline may use GPU; backend differences are not separately controlled.",
                              "No independent anatomy ground truth; signal recovery may include artifacts/noise.",
                              "Not ESPIRiT soft crop or multi-map soft-SENSE; no default pipeline change."]}
    write_json(manifest_path, report)
    return report
