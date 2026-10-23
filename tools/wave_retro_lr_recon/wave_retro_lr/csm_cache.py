"""Hash-bound, CPU-only uncropped calibration cache for opt-in diagnostics.

This module never changes production inputs, reconstructions, or collections.
Only the prepared ACS and its coil-basis contract are used; this is not a new
validation of raw TWIX, PSF, trajectory, or full reconstruction readiness.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, sha256_file
from .csm_crop import METHOD_VERSION, crop_weights, validate_map_shapes
from .standard import array_sha256, validate_standard_pca_manifest


def read_json(path: Path) -> dict:
    """Read one JSON object from path; reject non-object documents."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    """Exclusively create a JSON completion record; never replace existing metadata."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_record(path: Path) -> dict:
    """Return resolved path, length and SHA-256 for an existing file."""
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def validate_cfl(base: Path, expected: dict | None = None) -> dict:
    """Check CFL shape, byte count, hashes and finite values with bounded block memory.

    If expected is provided, all scientific artifact fields must match exactly;
    recorded paths are provenance only and are not used to redirect file reads.
    """
    actual = cfl_record(base)
    if expected is not None:
        for key in ("shape", "dtype", "payload_bytes", "header_sha256", "payload_sha256"):
            if actual[key] != expected.get(key):
                raise ValueError(f"CSM cache input/output mismatch ({key}): {base}")
    flat = open_cfl(base).reshape(-1, order="F")
    for start in range(0, flat.size, 1024*1024):
        if not np.isfinite(flat[start:start+1024*1024]).all():
            raise ValueError(f"Non-finite CFL values: {base}")
    return actual


def prepared_source(manifest_path: Path) -> dict:
    """Validate one prepared standard-PCA or ROVir ACS/basis and return its binding.

    Requires recorded ACS hashes, native geometry, physical/virtual coil bounds,
    and finite hash-matched projection basis. No physical-space basis is inferred
    from a VCC count. The original manifest and its source provenance are retained
    verbatim in the returned record. Legacy unbound inputs fail closed.
    """
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    json.dumps(manifest, allow_nan=False)
    if not manifest.get("source"):
        raise ValueError("Prepared manifest lacks source provenance")
    geometry = manifest.get("geometry", {})
    matrix = geometry.get("logical_matrix_ro_lin_par", geometry.get("matrix_ro_lin_par"))
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(type(n) is not int or n < 1 for n in matrix)):
        raise ValueError("Prepared manifest lacks a valid native geometry")
    fov = geometry.get("physical_fov_mm_xyz", geometry.get("fov_mm_ro_lin_par"))
    if (not isinstance(fov, list) or len(fov) != 3
            or not np.isfinite(fov).all() or any(v <= 0 for v in fov)):
        raise ValueError("Prepared manifest lacks a finite positive physical FOV")
    rovir = manifest.get("rovir")
    if rovir:
        kind = "rovir"
        if manifest.get("status") != "measured_wave_mprage_rovir_control_ready":
            raise ValueError("ROVir source is not a completed native control preparation")
        compression = rovir
        if manifest.get("standard_pca_variant"):
            raise ValueError("Ambiguous ROVir/standard-PCA basis")
        if not rovir.get("basis_applied_identically_to_image_and_acs"):
            raise ValueError("ROVir basis must be shared by image and ACS")
        basis_path = manifest_path.parent / rovir["projection_basis_file"]
        basis_hash = rovir["projection_basis_sha256"]
        artifact = manifest.get("artifacts", {}).get("kspace_calib")
    else:
        kind = "standard_pca"
        if manifest.get("status") not in {"measured_wave_mprage_bart_inputs_ready",
                                          "measured_multi_echo_wave_gre_bart_inputs_ready"}:
            raise ValueError("Source is not a completed native MPRAGE/GRE preparation")
        compression = manifest.get("coil_compression", {})
        validate_standard_pca_manifest(manifest, compression.get("virtual_coils", 0))
        basis_path = manifest_path.parent / compression["basis"]["path"]
        basis_hash = compression["basis"]["file_sha256"]
        artifact = manifest.get("output_artifacts", {}).get("kspace_calib")
    physical, count = compression.get("physical_coils"), compression.get("virtual_coils")
    if type(physical) is not int or type(count) is not int or not 1 <= count <= physical:
        raise ValueError("Invalid physical/virtual coil counts")
    if sha256_file(basis_path) != basis_hash:
        raise ValueError("Projection basis file hash mismatch")
    basis = np.load(basis_path, allow_pickle=False)
    if basis.shape != (physical, count) or not np.isfinite(basis).all():
        raise ValueError("Projection basis shape/finite-value mismatch")
    if kind == "standard_pca" and array_sha256(basis) != compression["basis"]["logical_sha256"]:
        raise ValueError("PCA logical basis hash mismatch")
    if not isinstance(artifact, dict):
        raise ValueError("Prepared ACS has no hash-bound artifact record")
    acs = validate_cfl(manifest_path.parent/"kspace_calib", artifact)
    if acs["shape"][:4] != [*matrix, count] or any(n != 1 for n in acs["shape"][4:]):
        raise ValueError("Prepared ACS/native geometry/coil count mismatch")
    return {"manifest": file_record(manifest_path), "manifest_content": manifest,
            "kind": kind, "geometry": geometry, "basis": file_record(basis_path),
            "physical_coils": physical, "virtual_coils": count, "acs": acs}


def available_gib() -> float:
    """Read Linux MemAvailable in GiB, failing closed on unsupported systems."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024**2
    raise RuntimeError("Cannot read Linux MemAvailable")


def idle_child() -> None:
    """Limit only our child to idle CPU scheduling, 96-GiB address space and no cores."""
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = 96*1024**3 if hard == resource.RLIM_INFINITY else min(hard, 96*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_ecalib(command: list[str], log: Path, threads: int) -> None:
    """Run one CPU ecalib child at idle priority, aborting our job under RAM pressure.

    Requires 112 GiB available at admission, preserves 64 GiB host headroom and
    caps child RSS at 64 GiB. Partial output/logs remain on failure. No GPU or
    other user's process is touched. The caller owns the output-directory lock.
    """
    if available_gib() < 112:
        raise MemoryError("Yielding to other workloads: ecalib requires 112 GiB available")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, preexec_fn=idle_child)
        try:
            while child.poll() is None:
                try:
                    status = Path(f"/proc/{child.pid}/status").read_text()
                except FileNotFoundError:
                    continue
                rss = next((int(line.split()[1])/1024**2 for line in status.splitlines()
                            if line.startswith("VmRSS:")), 0.)
                if available_gib() < 64 or rss > 64:
                    raise MemoryError("Stopped our ecalib to preserve RAM for other workloads")
                time.sleep(1)
            if child.returncode != 0:
                raise RuntimeError(f"ecalib failed ({child.returncode}); inspect {log}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def calibration_cache(root: Path, source: dict, bart: Path, threads: int) -> dict:
    """Create or strictly reuse uncropped maps/eigenvalues beneath a locked root.

    Only -m 1 -c 0 is allowed; all other ecalib settings remain BART defaults.
    Tool, source, code and thread identity must match on resume. Any uncommitted
    files from an interrupted calibration require a separate fresh directory.
    """
    maps, eigen = root/"calibration/coil_sens_uncropped", root/"calibration/eigenvalues"
    command = [str(bart), "ecalib", "-m", "1", "-c", "0", source["acs"]["base"],
               str(maps), str(eigen)]
    identity = {"source": source, "bart": file_record(bart), "command": command,
                "threads": threads, "method": METHOD_VERSION,
                "implementation_sha256": sha256_file(Path(__file__)),
                "gpu": False, "maps": 1, "calibration_crop": 0}
    completion = root/"metadata/calibration.json"
    if completion.exists():
        cached = read_json(completion)
        if cached.get("identity") != identity:
            raise ValueError("Uncropped calibration cache identity mismatch; use a fresh directory")
        validate_cfl(maps, cached["maps"])
        validate_cfl(eigen, cached["eigenvalues"])
        validate_map_shapes(tuple(cached["maps"]["shape"]), tuple(cached["eigenvalues"]["shape"]))
        return cached
    if any((root/"calibration").iterdir()) or (root/"logs/ecalib.log").exists():
        raise ValueError("Incomplete calibration cache; preserve it and use a fresh directory")
    run_ecalib(command, root/"logs/ecalib.log", threads)
    maps_record, eigen_record = validate_cfl(maps), validate_cfl(eigen)
    shape = validate_map_shapes(tuple(maps_record["shape"]), tuple(eigen_record["shape"]))
    if list(shape) != source["acs"]["shape"][:4]:
        raise ValueError("Uncropped ecalib output changed native geometry or coil count")
    # Recheck the actual inputs after calibration before committing provenance.
    if prepared_source(Path(source["manifest"]["path"])) != source or file_record(bart) != identity["bart"]:
        raise ValueError("Calibration inputs/tool changed during ecalib")
    cached = {"identity": identity, "maps": maps_record, "eigenvalues": eigen_record}
    write_json(completion, cached)
    return cached


def derive_variant(root: Path, cached: dict, crop: float, mode: str, materialize: bool) -> dict:
    """Save scalar weights and native-grid statistics, optionally complete weighted CSM.

    Maps retain complex phase and relative coil scaling. Neither normalization,
    interpolation, nor background extrapolation is performed. Reuse validates
    hashes; partial variants are never overwritten. Materialization can be added
    later to a complete weights-only variant without re-running ecalib.
    """
    crop_weights(np.zeros(1, np.complex64), crop, mode)
    tag = variant_name(mode, crop)
    directory = root/"csm_variants"/tag
    identity = {"calibration_sha256": sha256_file(root/"metadata/calibration.json"),
                "crop": float(crop), "mode": mode, "method": METHOD_VERSION,
                "numpy_version": np.__version__,
                "crop_implementation_sha256": sha256_file(Path(__file__).with_name("csm_crop.py"))}
    record_path = directory/"variant.json"
    maps = open_cfl(cached["maps"]["base"])
    eigen = open_cfl(cached["eigenvalues"]["base"])
    shape = validate_map_shapes(maps.shape, eigen.shape)
    spatial = shape[:3]
    if record_path.exists():
        record = read_json(record_path)
        if record.get("identity") != identity:
            raise ValueError(f"CSM variant identity mismatch: {directory}")
        validate_cfl(directory/"weights", record["weights"])
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Partial CSM variant; refusing overwrite: {directory}")
        directory.mkdir(exist_ok=True)
        weights = create_cfl(directory/"weights", spatial)
        eig3 = eigen.reshape(spatial, order="F")
        count, over_one, maximum = 0, 0, 0.
        for k in range(spatial[2]):
            block = crop_weights(eig3[:, :, k], crop, mode)
            weights[:, :, k] = block
            count += int(np.count_nonzero(block))
            over_one += int(np.count_nonzero(block > 1))
            maximum = max(maximum, float(block.max()))
        weights.flush()
        record = {"identity": identity, "weights": validate_cfl(directory/"weights"),
                  "nonzero_weight_voxels": count, "weight_above_one_voxels": over_one,
                  "maximum_weight": maximum, "spatial_voxels": int(np.prod(spatial)),
                  "warning": "Diagnostic only; nonzero weights do not prove recoverable tissue signal."}
        write_json(record_path, record)
    if materialize:
        completion = directory/"materialized.json"
        if completion.exists():
            done = read_json(completion)
            if done.get("variant_sha256") != sha256_file(record_path):
                raise ValueError("Materialized CSM provenance mismatch")
            validate_cfl(directory/"coil_sens", done["maps"])
        else:
            if any(directory.glob("coil_sens.*")):
                raise ValueError("Partial materialized CSM; refusing overwrite")
            output = create_cfl(directory/"coil_sens", maps.shape)
            maps4 = maps.reshape(shape, order="F")
            output4 = output.reshape(shape, order="F")
            weights = open_cfl(directory/"weights")
            for k in range(spatial[2]):
                output4[:, :, k, :] = maps4[:, :, k, :] * weights[:, :, k, None]
            output.flush()
            write_json(completion, {"variant_sha256": sha256_file(record_path),
                                    "maps": validate_cfl(directory/"coil_sens")})
    return {"directory": str(directory), **record}


def variant_name(mode: str, crop: float) -> str:
    """Return a readable crop directory label; exact numeric identity is checked separately."""
    return f"{mode}_{format(float(crop), '.9g').replace('.', 'p')}"


def required_disk_bytes(root: Path, source: dict, variants: list[tuple[str, float]], materialize: bool) -> int:
    """Estimate additional payload bytes, excluding complete existing files and small metadata.

    Full CSM is one ACS-sized complex64 payload; each eigenvalue/weight map has
    one complex64 value per spatial voxel. A separate 1-GiB reserve is enforced
    by the caller. BART transient scratch usage is not included in this estimate.
    """
    maps_bytes = source["acs"]["payload_bytes"]
    scalar_bytes = int(np.prod(source["acs"]["shape"][:3]))*8
    files = [(root/"calibration/coil_sens_uncropped.cfl", maps_bytes),
             (root/"calibration/eigenvalues.cfl", scalar_bytes)]
    for mode, crop in variants:
        directory = root/"csm_variants"/variant_name(mode, crop)
        files.append((directory/"weights.cfl", scalar_bytes))
        if materialize:
            files.append((directory/"coil_sens.cfl", maps_bytes))
    return sum(max(0, size - (path.stat().st_size if path.is_file() else 0)) for path, size in files)


def comparison_figure(root: Path, cached: dict, records: list[dict], slices: list[int] | None) -> str:
    """Save three logical-grid slice comparisons with exact source and figure hashes.

    Rows show eigenvalue magnitude, nominal hard-0.6 RSS and each candidate's
    weighted coil RSS. Slice inputs are zero-based RO/LIN/PAR, or grid centers.
    These are support diagnostics, not reconstructed tissue images or NIfTI
    anatomical views. Distinct slices/variant lists receive distinct filenames.
    """
    import hashlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = open_cfl(cached["maps"]["base"])
    eigen = open_cfl(cached["eigenvalues"]["base"])
    shape = validate_map_shapes(maps.shape, eigen.shape)
    indices = slices if slices is not None else [n//2 for n in shape[:3]]
    if len(indices) != 3 or any(type(i) is not int or not 0 <= i < n for i, n in zip(indices, shape)):
        raise ValueError("Slices must be valid zero-based RO LIN PAR indices")
    identity = {"slices_ro_lin_par": indices,
                "calibration_sha256": sha256_file(root/"metadata/calibration.json"),
                "variant_records": [file_record(Path(r["directory"])/"variant.json") for r in records],
                "implementation_sha256": sha256_file(Path(__file__))}
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    png = root/"figures"/f"support_{token}.png"
    metadata = root/"metadata"/f"support_{token}.json"
    if metadata.exists():
        stored = read_json(metadata)
        if stored.get("identity") != identity or stored.get("figure") != file_record(png):
            raise ValueError("Support figure provenance/hash mismatch")
        return str(png)
    if png.exists():
        raise ValueError("Uncommitted support figure exists; refusing overwrite")
    maps4, eig3 = maps.reshape(shape, order="F"), eigen.reshape(shape[:3], order="F")
    rows = 2 + len(records)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3*rows), squeeze=False)
    try:
        for axis, index in enumerate(indices):
            selector = [slice(None)]*3
            selector[axis] = index
            eig_slice = eig3[tuple(selector)]
            rss = np.sqrt(np.sum(np.abs(maps4[tuple(selector)])**2, axis=-1))
            panels = [("|eigenvalue|", np.abs(eig_slice)),
                      ("nominal hard 0.6: coil RSS", rss*crop_weights(eig_slice, .6))]
            for record in records:
                request = record["identity"]
                panels.append((f"{request['mode']} {request['crop']:g}: coil RSS",
                               rss*crop_weights(eig_slice, request["crop"], request["mode"])))
            for row, (label, values) in enumerate(panels):
                panel = axes[row, axis]
                picture = panel.imshow(values.T, origin="lower", vmin=0, vmax=1, cmap="gray")
                panel.set_title(f"{label}\n{'RO LIN PAR'.split()[axis]}={index}")
                panel.set_xlabel("Logical index (not anatomical orientation)")
                fig.colorbar(picture, ax=panel, fraction=.035)
        fig.suptitle("CSM support only; display clipped to [0,1]; inspect metadata for weights >1")
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(png, dpi=120)
    finally:
        plt.close(fig)
    write_json(metadata, {"identity": identity, "figure": file_record(png),
                          "display_limits": [0, 1], "anatomical_orientation": False})
    return str(png)


def run_diagnostics(manifest: Path, root: Path, variants: list[tuple[str, float]], *,
                    threads: int = 1, materialize: bool = False, check_only: bool = False,
                    slices: list[int] | None = None) -> dict:
    """Validate or execute the isolated cache/derivation workflow with an exclusive lock.

    Args:
        manifest: Existing hash-bound normal or ROVir prepared manifest.
        root: User-confirmed diagnostic destination (never an existing production tree).
        variants: Ordered hard or soft_bart_1_0 crop requests; no winner is selected.
        threads: CPU ecalib thread cap between one and eight.
        materialize: Also export full CSM arrays for each crop.
        check_only: Validate sources and report the request without writes or BART execution.
        slices: Optional zero-based RO/LIN/PAR indices for the support comparison.
    Returns:
        JSON-ready source summary or completed variant records.
    Raises:
        ValueError: Invalid request, unsafe destination, incomplete or mismatched reuse.
        BlockingIOError: Another invocation holds the output lock.
    """
    if type(threads) is not int or not 1 <= threads <= 8 or not variants:
        raise ValueError("Require 1..8 threads and at least one crop variant")
    for mode, crop in variants:
        crop_weights(np.zeros(1, np.complex64), crop, mode)
    if len({variant_name(mode, crop) for mode, crop in variants}) != len(variants):
        raise ValueError("Duplicate crop variants")
    root = root.resolve()
    manifest = manifest.resolve()
    if (root == manifest.parent or root in manifest.parents or manifest.parent in root.parents
            or "diagnostics" not in root.parts or root.name == "diagnostics"):
        raise ValueError("Use a dedicated named subdirectory beneath diagnostics, outside bart_inputs")
    source = prepared_source(manifest)
    if slices is not None and (len(slices) != 3 or any(type(i) is not int or not 0 <= i < n
                            for i, n in zip(slices, source["acs"]["shape"][:3]))):
        raise ValueError("Slices must be valid zero-based RO LIN PAR indices")
    executable = shutil.which("bart")
    if not executable:
        raise RuntimeError("Source the host bart_startup.sh before running diagnostics")
    bart = Path(executable).resolve()
    disk_bytes = required_disk_bytes(root, source, variants, materialize)
    if check_only:
        return {"check_only": True, "output": str(root), "source": source,
                "bart": file_record(bart), "variants": variants, "materialize": materialize,
                "estimated_additional_payload_bytes": disk_bytes}
    allowed = {"calibration", "csm_variants", "figures", "metadata", "logs", "scripts", ".lock"}
    if root.exists() and any(p.name not in allowed for p in root.iterdir()):
        raise ValueError("Diagnostic destination contains unrelated entries")
    existing_parent = root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < disk_bytes + 1024**3:
        raise OSError("Insufficient free disk for diagnostic payloads plus a 1-GiB reserve")
    root.mkdir(parents=True, exist_ok=True)
    if (root/".lock").is_symlink():
        raise ValueError("Diagnostic lock must not be a symlink")
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in allowed - {".lock"}:
            path = root/name
            if path.is_symlink():
                raise ValueError("Diagnostic subdirectories must not be symlinks")
            path.mkdir(exist_ok=True)
        cached = calibration_cache(root, source, bart, threads)
        records = [derive_variant(root, cached, crop, mode, materialize) for mode, crop in variants]
        figure = comparison_figure(root, cached, records, slices)
        return {"output": str(root), "calibration": str(root/"metadata/calibration.json"),
                "variants": records, "figure": figure, "production_outputs_changed": False}
