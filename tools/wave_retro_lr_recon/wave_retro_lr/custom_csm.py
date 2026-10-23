"""Hash-bound, explicitly selected native single-map CSM derivatives."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil

import numpy as np

from .bart_io import cfl_record, open_cfl, sha256_file
from .csm_cache import file_record, prepared_source, read_json, validate_cfl, write_json
from .metal_csm import MetalCSMParameters, write_candidate
from .metal_runner import memory_guard

SCHEMA = "mprage-custom-csm-v1"


@contextmanager
def locked_output(path: Path, request: dict, folders: tuple[str, ...]):
    """Lock an isolated directory, binding its immutable request and rejecting partial state.

    Args:
        path: User-approved derivative location, never a canonical input tree.
        request: JSON-native exact identity for this derivative.
        folders: Named subdirectories created on first initialization.
    Yields:
        None, with an exclusive nonblocking lock held until context exit.
    Raises:
        ValueError: Symlinks, unbound files, or a mismatched existing request.
    """
    if path.is_symlink() or (path.exists() and any(p.is_symlink() for p in path.rglob("*"))):
        raise ValueError("Custom output trees must not contain symlinks")
    path.mkdir(parents=True, exist_ok=True)
    with (path/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = path/"metadata/request.json"
        if record.exists():
            if read_json(record) != request:
                raise ValueError("Custom CSM request mismatch; preserve existing outputs")
        else:
            if set(p.name for p in path.iterdir()) != {".lock"}:
                raise ValueError("Unbound/partial custom output directory")
            for folder in folders:
                (path/folder).mkdir()
            write_json(record, request)
        yield


def load_custom_csm(path: Path, expected_manifest: Path | None = None) -> dict:
    """Validate a completed native custom CSM and its exact prepared source/basis.

    Args:
        path: Derivative completion JSON, not a bare CFL or unbound configuration.
        expected_manifest: Optional selected native preparation to enforce exact
            acquisition, geometry, whitening and physical/PCA/ROVir basis identity.
    Returns:
        Validated completion record. Never recalibrates, rewrites or relabels maps.
    Raises:
        ValueError: Changed provenance, nonfinite maps, wrong basis/grid/count or
            a multi-map input. GRE manifests are explicitly unsupported.
    """
    record = read_json(path)
    if record.get("schema") != SCHEMA or record.get("status") != "complete":
        raise ValueError("Require a completed hash-bound MPRAGE custom CSM manifest")
    source = record["source"]
    manifest = Path(source["manifest"]["path"])
    if expected_manifest is not None and manifest.resolve() != expected_manifest.resolve():
        raise ValueError("Custom CSM belongs to another prepared source/basis")
    if prepared_source(manifest) != source:
        raise ValueError("Custom CSM source/basis changed")
    if "gre" in source["manifest_content"]["status"]:
        raise ValueError("Custom CSM integration currently supports MPRAGE only")
    maps = validate_cfl(Path(record["maps"]["base"]), record["maps"])
    if maps["shape"][:4] != source["acs"]["shape"][:4] or any(n != 1 for n in maps["shape"][4:]):
        raise ValueError("Custom CSM native geometry/coil count/map sets mismatch")
    values = open_cfl(record["maps"]["base"])
    if not any(np.any(values[i] != 0) for i in range(values.shape[0])):
        raise ValueError("Custom CSM must not be identically zero")
    for item in record.get("provenance_files", []):
        if file_record(Path(item["path"])) != item:
            raise ValueError("Custom CSM provenance file changed")
    return record


def prepare_local_csm(config_path: Path, output: Path) -> Path:
    """Generate one mouth/local-ROI candidate from an existing uncropped cache.

    Configuration contains diagnostic_root, baseline_csm and explicit parameters
    in logical RO/LIN/PAR order. Outside the ROI and on native support all vectors
    remain exact. Writes csm/, masks/, figures/ and metadata/ under the approved
    output; no ecalib or reconstruction is launched. Partial results reject reuse.
    Returns the completion JSON usable by --custom-csm.
    """
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    if set(config) != {"diagnostic_root", "baseline_csm", "parameters"}:
        raise ValueError("Expected diagnostic_root, baseline_csm and parameters")
    cache = (config_path.parent/Path(config["diagnostic_root"])).resolve()
    baseline = (config_path.parent/Path(config["baseline_csm"])).resolve()
    calibration_path = cache/"metadata/calibration.json"
    calibration = read_json(calibration_path)
    source = prepared_source(Path(calibration["identity"]["source"]["manifest"]["path"]))
    if source != calibration["identity"]["source"] or "gre" in source["manifest_content"]["status"]:
        raise ValueError("Local CSM requires unchanged prepared MPRAGE ACS/basis")
    if calibration["identity"].get("maps") != 1 or calibration["identity"].get("calibration_crop") != 0:
        raise ValueError("Require single-map crop-zero calibration cache")
    original = validate_cfl(baseline)
    for key in ("maps", "eigenvalues"):
        validate_cfl(Path(calibration[key]["base"]), calibration[key])
    native = open_cfl(baseline)
    shape = tuple(source["acs"]["shape"][:4])
    if native.shape[:4] != shape or any(n != 1 for n in native.shape[4:]):
        raise ValueError("Baseline CSM differs from native ACS geometry/basis count")
    native = native.reshape(shape, order="F")
    uncropped = open_cfl(calibration["maps"]["base"]).reshape(shape, order="F")
    eigen = open_cfl(calibration["eigenvalues"]["base"]).reshape(shape[:3], order="F")
    # Establish the baseline's basis/phase by direct comparison, not channel count.
    support = np.abs(eigen) >= np.float32(.6)
    for coil in range(shape[3]):
        memory_guard()
        if not np.array_equal(native[..., coil], uncropped[..., coil]*support):
            raise ValueError("Baseline is not exactly the cache's hard-0.6 CSM")
    parameters = MetalCSMParameters.from_dict(config["parameters"])
    if parameters.global_base_extension:
        raise ValueError("Local custom CSM forbids global extension")
    # The shared numerical routine uses reversed voxel order; no anatomical flips
    # are inferred. Both the explicit ROI and physical spacing follow that mapping.
    values = parameters.to_dict()
    values["roi_voxels"] = values["roi_voxels"][::-1]
    values["envelope_axis"] = 2-values["envelope_axis"]
    internal = MetalCSMParameters.from_dict(values)
    spacing_xyz = source["geometry"].get("physical_resolution_mm_xyz")
    if spacing_xyz is None:
        fov = source["geometry"]["physical_fov_mm_xyz"]
        spacing_xyz = [f/n for f, n in zip(fov, shape[:3][::-1])]
    request = {"config": file_record(config_path), "source": source,
               "calibration": file_record(calibration_path), "baseline": original,
               "parameters_logical": parameters.to_dict(), "spacing_xyz": spacing_xyz,
               "implementation": {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ("custom_csm.py", "metal_csm.py", "csm_overlay.py")}}
    output = output.resolve()
    completion = output/"metadata/custom_csm.json"
    with locked_output(output, request, ("csm", "masks", "figures", "metadata")):
        if completion.exists():
            load_custom_csm(completion)
            return completion
        if any((output/"csm").iterdir()) or any((output/"masks").iterdir()):
            raise ValueError("Partial local CSM generation; refusing overwrite")
        if shutil.disk_usage(output).free < original["payload_bytes"]+1024**3:
            raise ValueError("Insufficient disk capacity for candidate CSM plus 1 GiB reserve")
        stats, native_mask, added = write_candidate(baseline, output/"csm/coil_sens",
                                                   (False, False, False), tuple(spacing_xyz),
                                                   internal, checkpoint=memory_guard)
        roi = np.zeros(shape[:3], bool)
        roi[tuple(slice(a, b) for a, b in parameters.roi_voxels)] = True
        if np.any(added & ~roi) or stats["unchanged_vectors_max_error"] != 0:
            raise ValueError("Local extension altered voxels outside its contract")
        np.savez_compressed(output/"masks/support.npz", native=native_mask, added=added, roi=roi)
        plot_local_csm(output/"figures/mouth_support.png", source, native_mask, added, roi,
                       [(a+b)//2 for a, b in parameters.roi_voxels])
        if validate_cfl(baseline) != original or prepared_source(Path(source["manifest"]["path"])) != source:
            raise ValueError("Source changed during local CSM generation")
        record = {"schema": SCHEMA, "status": "complete", "source": source,
                  "maps": validate_cfl(output/"csm/coil_sens"), "statistics": stats,
                  "method": "manual-local-bounded-complex-continuation",
                  "parameters_logical": parameters.to_dict(), "experimental": True,
                  "regularization_optimized": False,
                  "provenance_files": [file_record(p) for p in (output/"metadata/request.json",
                      output/"masks/support.npz", output/"figures/mouth_support.png", calibration_path,
                      config_path)]}
        write_json(completion, record)
    return completion


def plot_local_csm(path: Path, source: dict, native: np.ndarray, added: np.ndarray,
                   roi: np.ndarray, indices: list[int]) -> None:
    """Save three matched ACS/native/candidate views at logical ROI-center indices.

    Green marks additions, yellow the original support, cyan the ROI. The same
    unmasked coil-RSS background/window is used throughout; no image is reconstructed.
    """
    from .csm_overlay import acs_rss_magnitude, display_slice, draw_contour
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    magnitude = acs_rss_magnitude(Path(source["acs"]["base"]))
    window = float(np.percentile(magnitude, 99.5))
    if window <= 0:
        raise ValueError("Invalid ACS display window")
    fig, panels = plt.subplots(3, 3, figsize=(13, 14), layout="constrained")
    names = ("RO", "LIN", "PAR")
    try:
        for axis, index in enumerate(indices):
            bg = np.sqrt(np.clip(display_slice(magnitude, axis, index)/window, 0, 1))
            old = display_slice(native, axis, index)
            new = display_slice(added, axis, index)
            box = display_slice(roi, axis, index)
            remaining = [name for i, name in enumerate(names) if i != axis]
            for column, title in enumerate(("ACS RSS only", "Original hard 0.6", "Mouth-only candidate")):
                panel = panels[axis, column]
                panel.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
                if column:
                    draw_contour(panel, old.astype(float), .5, "yellow", "solid")
                    draw_contour(panel, box.astype(float), .5, "cyan", "dashed")
                if column == 2:
                    rgba = np.zeros((*new.shape, 4))
                    rgba[..., 1] = 1
                    rgba[..., 3] = new*.5
                    panel.imshow(rgba, origin="lower")
                panel.set_title(f"{title}: {names[axis]}={index}")
                panel.set_xlabel(remaining[1]+" voxel")
                panel.set_ylabel(remaining[0]+" voxel")
        fig.suptitle("Same-basis ACS RSS (no CSM mask) | yellow: original support | green: added | cyan: ROI\n"
                     f"Shared window 0..{window:.6g}, gamma=0.5; logical axes, not RAS; experimental, no reconstruction")
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
