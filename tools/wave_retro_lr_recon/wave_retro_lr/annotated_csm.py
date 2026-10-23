"""Local CSM continuation bounded by explicitly mapped, open annotation traces.

Three open traces provide local one-sided constraints, not a closed segmentation
or a patient-independent anatomical model. No annotation endpoint is connected
to another endpoint and no curve is extrapolated beyond its sampled domain.
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .bart_io import create_cfl, open_cfl, sha256_file
from .csm_cache import file_record, prepared_source, read_json, validate_cfl, write_json
from .custom_csm import SCHEMA, load_custom_csm, locked_output
from .metal_csm import smoothstep
from .metal_runner import memory_guard


def open_trace_profiles(image: Path, panels: list[dict]) -> list[dict]:
    """Digitize red pixels from three explicitly calibrated figure panels.

    Inputs are an annotation PNG and per-panel pixel bounds (left, top, width,
    height), logical array shape, and side policy. Pixel centers are mapped into
    imshow's [-.5,N-.5] extents. Returns open trace coordinates and conservative
    single-valued boundary samples, never polygons. Invalid/empty traces fail.
    """
    rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32)
    red = (rgb[..., 0] > 100) & (rgb[..., 0] > 1.8*rgb[..., 1]) & (rgb[..., 0] > 1.8*rgb[..., 2])
    profiles = []
    if len(panels) != 3:
        raise ValueError("Require exactly three explicitly mapped annotation views")
    for axis, panel in enumerate(panels):
        x, y, width, height = panel["pixel_bounds"]
        ny, nx = panel["shape"]
        if (panel["axis"] != axis or min(nx, ny, width, height) <= 0
                or x < 0 or y < 0 or x+width > rgb.shape[1] or y+height > rgb.shape[0]):
            raise ValueError("Invalid annotation panel geometry")
        yy, xx = np.where(red)
        keep = (xx >= x) & (xx < x+width) & (yy >= y) & (yy < y+height)
        px = (xx[keep]+.5-x)/width*nx-.5
        py = (y+height-yy[keep]-.5)/height*ny-.5
        if len(px) < 10:
            raise ValueError("Annotation panel has no usable open red trace")
        mode = panel["policy"]
        if mode not in ("y_upper", "y_lower", "x_upper"):
            raise ValueError("Unsupported open-boundary side policy")
        independent, dependent = (py, px) if mode == "x_upper" else (px, py)
        # Subvoxel bins remove raster thickness, without closing the trace.
        bins = np.rint(independent*2).astype(int)
        distinct = np.unique(bins)
        if len(distinct) < 2 or np.max(np.diff(distinct)) > 6:
            raise ValueError("Open trace contains a gap larger than three logical voxels")
        reducer = np.min if mode == "y_lower" else np.max
        values = [float(reducer(dependent[bins == k])) for k in distinct]
        profiles.append({"panel": panel, "x": px.tolist(), "y": py.tolist(),
                         "independent": (distinct/2).tolist(), "boundary": values,
                         "pixels": int(len(px)), "endpoint_connection": False})
    return profiles


def trace_halfspace(profile: dict, spacing: tuple[float, float], margin_mm: float) -> np.ndarray:
    """Return a local one-sided 2D constraint with a physical-distance outer margin.

    Only the trace's sampled independent-coordinate range is valid, including
    after dilation. NaN outside that range prevents endpoint extrapolation.
    This bounded halfspace is not interpreted as a closed tissue segmentation.
    """
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("Annotation margin must be finite and nonnegative")
    panel = profile["panel"]
    ny, nx = panel["shape"]
    y, x = np.indices((ny, nx))
    axis = y if panel["policy"] == "x_upper" else x
    boundary = np.interp(axis, profile["independent"], profile["boundary"], left=np.nan, right=np.nan)
    domain = np.isfinite(boundary)
    if panel["policy"] == "y_lower":
        inside = domain & (y >= boundary)
    elif panel["policy"] == "y_upper":
        inside = domain & (y <= boundary)
    else:
        inside = domain & (x <= boundary)
    if not inside.any():
        raise ValueError("Empty open-trace constraint")
    return domain & (ndi.distance_transform_edt(~inside, sampling=spacing) <= margin_mm)


def annotation_domain(shape: tuple, spacing: tuple, profiles: list[dict], roi_bounds: list,
                      margin_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Intersect three local one-sided constraints inside an explicit logical ROI.

    Returns the bounded candidate domain and rectangular review ROI. This is an
    approximate extension between marked planes, not an inferred whole-head mask.
    No unmarked independent-coordinate span is filled by curve extrapolation.
    """
    roi = np.zeros(shape, bool)
    if len(roi_bounds) != 3 or any(len(p) != 2 or any(type(v) is not int for v in p)
            or not 0 <= p[0] < p[1] <= n for p, n in zip(roi_bounds, shape)):
        raise ValueError("ROI must contain three bounded half-open logical integer ranges")
    roi[tuple(slice(a, b) for a, b in roi_bounds)] = True
    domain = roi.copy()
    if len(profiles) != 3:
        raise ValueError("Require three open-trace constraints")
    for axis, profile in enumerate(profiles):
        expected = tuple(n for i, n in enumerate(shape) if i != axis)
        if (tuple(profile["panel"]["shape"]) != expected
                or profile["panel"]["axis"] != axis
                or not 0 <= profile["panel"]["index"] < shape[axis]):
            raise ValueError("Annotation panel and native grid differ")
        plane = trace_halfspace(profile, tuple(v for i, v in enumerate(spacing) if i != axis), margin_mm)
        domain &= np.expand_dims(plane, axis)
    return domain, roi


def annotated_vectors(maps: np.ndarray, domain: np.ndarray, spacing: tuple,
                      sigmas_mm: list[float], max_distance_mm: float,
                      minimum_weight: float = .01, minimum_coherence: float = .2) -> tuple:
    """Predict complex vectors only in the annotated domain outside native support.

    Args:
        maps: Native logical single-map unit-RSS CSM, unchanged by this function.
        domain: Explicit local 3D candidate domain from open trace constraints.
        spacing: Logical RO/LIN/PAR voxel sizes in mm.
        sigmas_mm: Increasing positive Gaussian scales, smallest used where stable.
        max_distance_mm: Explicit donor-distance safety cap, not a global dilation.
        minimum_weight: Minimum smoothed native-support mass in [0,1].
        minimum_coherence: Minimum complex prediction norm in (0,1].
    Returns:
        Native and accepted masks, sparse unit vectors, and confidence/distance
        statistics. Scale transitions blend smoothly where donor weight permits.
        Weak/cancelled predictions remain excluded; supported vectors are untouched.
    """
    settings = [*spacing, *sigmas_mm, max_distance_mm, minimum_weight, minimum_coherence]
    if (len(spacing) != 3 or not sigmas_mm or not np.isfinite(settings).all()
            or min(settings) <= 0 or list(sigmas_mm) != sorted(set(sigmas_mm))
            or minimum_weight > .5 or minimum_coherence > 1 or maps.ndim != 4
            or maps.shape[:3] != domain.shape):
        raise ValueError("Invalid annotated-continuation geometry/confidence parameters")
    rss2 = np.zeros(domain.shape, np.float32)
    for coil in range(maps.shape[3]):
        memory_guard()
        field = np.asarray(maps[..., coil])
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite native CSM")
        rss2 += abs(field)**2
    rss = np.sqrt(rss2)
    if not np.all((rss < 1e-8) | (abs(rss-1) < 1e-5)):
        raise ValueError("Annotated continuation requires binary unit-RSS native maps")
    native = rss > .5
    if not native.any():
        raise ValueError("Empty native CSM support")
    distance = ndi.distance_transform_edt(~native, sampling=spacing)
    proposal = domain & ~native & (distance <= max_distance_mm)
    coordinates = np.where(proposal)
    if not proposal.any():
        raise ValueError("No annotated support additions within donor-distance cap")
    # Crop the filtering work to the proposal plus the largest kernel halo.
    pad = np.ceil(3*max(sigmas_mm)/np.asarray(spacing)).astype(int)+1
    bounds = tuple(slice(max(0, int(v.min())-int(p)), min(n, int(v.max())+int(p)+1))
                   for v, p, n in zip(coordinates, pad, domain.shape))
    local_proposal = proposal[bounds]
    local_native = native[bounds].astype(np.float32)
    predictions, weights, coherences = [], [], []
    for sigma in sigmas_mm:
        print(f"Predicting local complex coil vectors: sigma={sigma:g} mm", flush=True)
        scale = sigma/np.asarray(spacing)
        weight = ndi.gaussian_filter(local_native, scale, mode="constant", truncate=3)[local_proposal]
        prediction = np.empty((int(proposal.sum()), maps.shape[3]), np.complex64)
        for coil in range(maps.shape[3]):
            memory_guard()
            field = np.asarray(maps[bounds+(coil,)])
            prediction[:, coil] = ndi.gaussian_filter(field, scale, mode="constant", truncate=3)[local_proposal]
        prediction /= np.maximum(weight[:, None], 1e-12)
        coherence = np.linalg.norm(prediction, axis=1)
        prediction /= np.maximum(coherence[:, None], 1e-12)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite annotated coil-vector prediction")
        predictions.append(prediction)
        weights.append(weight)
        coherences.append(coherence)
    # Fill from coarse to fine, retaining fine predictions only where reliable.
    result = np.zeros_like(predictions[0])
    valid = np.zeros(len(result), bool)
    selected = np.full(len(result), -1, np.int16)
    for index in reversed(range(len(sigmas_mm))):
        good = (weights[index] >= minimum_weight) & (coherences[index] >= minimum_coherence)
        blend = smoothstep((weights[index]-minimum_weight)/minimum_weight)
        blend[~valid] = 1
        result[good] = ((1-blend[good, None])*result[good] + blend[good, None]*predictions[index][good])
        valid |= good
        selected[good] = index
    norm = np.linalg.norm(result, axis=1)
    accepted = valid & (norm >= minimum_coherence)
    vectors = result[accepted]/norm[accepted, None]
    added = np.zeros_like(native)
    added[proposal] = accepted
    stats = {"proposed_voxels": int(proposal.sum()), "added_voxels": int(added.sum()),
             "rejected_confidence_voxels": int((~accepted).sum()), "native_support_voxels": int(native.sum()),
             "added_outside_annotation_domain": int(np.count_nonzero(added & ~domain)),
             "maximum_added_distance_mm": float(distance[added].max()) if added.any() else None,
             "added_beyond_16mm_voxels": int(np.count_nonzero(added & (distance > 16))),
             "smallest_reliable_scale_voxels": {str(s): int(np.count_nonzero(accepted & (selected == i)))
                                                for i, s in enumerate(sigmas_mm)}}
    return native, added, vectors, stats


def prepare_annotated_csm(config_path: Path, output: Path) -> Path:
    """Create an independent annotation-bounded CSM/QC derivative from validated native maps.

    Configuration binds PNG panel calibration, open-side policies, local ROI,
    physical margin and prediction parameters. Writes only the confirmed output
    directory; never repairs old metadata, runs BART or changes a launcher.
    Returns a completed manifest compatible with --custom-csm. Partial outputs
    and any changed source, annotation or completion hash are rejected.
    """
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    if set(config) != {"diagnostic_root", "baseline_csm", "annotation", "panels", "roi_voxels",
                       "margin_mm", "sigmas_mm", "max_distance_mm"}:
        raise ValueError("Unexpected annotation configuration fields")
    annotation = (config_path.parent/Path(config["annotation"])).resolve()
    baseline = (config_path.parent/Path(config["baseline_csm"])).resolve()
    cache = (config_path.parent/Path(config["diagnostic_root"])).resolve()/"metadata/calibration.json"
    cached = read_json(cache)
    source = prepared_source(Path(cached["identity"]["source"]["manifest"]["path"]))
    if (source != cached["identity"]["source"] or "mprage" not in source["manifest_content"]["status"]
            or cached["identity"].get("maps") != 1 or cached["identity"].get("calibration_crop") != 0):
        raise ValueError("Require unchanged single-map uncropped MPRAGE calibration")
    original = validate_cfl(baseline)
    for key in ("maps", "eigenvalues"):
        validate_cfl(Path(cached[key]["base"]), cached[key])
    shape = tuple(source["acs"]["shape"][:4])
    if tuple(original["shape"][:4]) != shape or any(n != 1 for n in original["shape"][4:]):
        raise ValueError("Baseline CSM grid/count mismatch")
    maps = open_cfl(baseline).reshape(shape, order="F")
    raw = open_cfl(cached["maps"]["base"]).reshape(shape, order="F")
    eigen = open_cfl(cached["eigenvalues"]["base"]).reshape(shape[:3], order="F")
    hard = abs(eigen) >= np.float32(.6)
    for coil in range(shape[3]):
        memory_guard()
        if not np.array_equal(maps[..., coil], raw[..., coil]*hard):
            raise ValueError("Baseline does not exactly match cache hard-0.6 maps/basis/phase")
    spacing = tuple(f/n for f, n in zip(source["geometry"]["physical_fov_mm_xyz"][::-1], shape[:3]))
    profiles = open_trace_profiles(annotation, config["panels"])
    domain, roi = annotation_domain(shape[:3], spacing, profiles, config["roi_voxels"], config["margin_mm"])
    request = {"config": file_record(config_path), "annotation": file_record(annotation),
               "calibration": file_record(cache), "source": source, "baseline": original,
               "settings": config, "spacing_ro_lin_par_mm": spacing,
               "interpretation": "Local open one-sided trace constraints; not a closed 3D segmentation",
               "implementation": {name: sha256_file(Path(__file__).with_name(name))
                   for name in ("annotated_csm.py", "custom_csm.py", "csm_overlay.py", "metal_csm.py")}}
    # Normalize tuples before exact JSON resume comparisons.
    request["spacing_ro_lin_par_mm"] = list(spacing)
    output = output.resolve()
    completion = output/"metadata/custom_csm.json"
    with locked_output(output, request, ("csm", "masks", "figures", "metadata")):
        if completion.exists():
            load_custom_csm(completion)
            return completion
        if any((output/"csm").iterdir()) or any((output/"masks").iterdir()):
            raise ValueError("Partial annotated CSM output; refusing overwrite")
        if shutil.disk_usage(output).free < original["payload_bytes"]+1024**3:
            raise ValueError("Need candidate capacity plus 1 GiB reserve")
        native, added, vectors, stats = annotated_vectors(maps, domain, spacing, config["sigmas_mm"], config["max_distance_mm"])
        if not added.any():
            raise ValueError("No stable annotated continuation vectors")
        candidate = create_cfl(output/"csm/coil_sens", shape)
        max_change, rss2 = 0., np.zeros(shape[:3], np.float32)
        for coil in range(shape[3]):
            memory_guard()
            field = np.asarray(maps[..., coil]).copy()
            field[added] = vectors[:, coil]
            max_change = max(max_change, float(abs(field[~added]-maps[..., coil][~added]).max()))
            rss2 += abs(field)**2
            candidate[..., coil] = field
        candidate.flush()
        error = float(abs(np.sqrt(rss2)-(native | added)).max())
        if max_change != 0 or error > 1e-5 or np.any(added & ~roi):
            raise ValueError("Preservation/RSS/ROI invariant failed")
        stats.update(unchanged_vectors_max_error=max_change, candidate_rss_max_error=error,
                     added_outside_roi=int(np.count_nonzero(added & ~roi)))
        np.savez_compressed(output/"masks/support.npz", native=native, added=added, roi=roi, annotation_domain=domain)
        write_json(output/"metadata/open_traces.json", {"annotation": file_record(annotation), "profiles": profiles})
        plot_annotated_csm(output/"figures/mouth_support.png", source, native, added, roi, domain, profiles,
                           config["margin_mm"])
        if validate_cfl(baseline) != original or file_record(annotation) != request["annotation"]:
            raise ValueError("Native CSM or annotation changed during preparation")
        if prepared_source(Path(source["manifest"]["path"])) != source:
            raise ValueError("Prepared acquisition changed during preparation")
        write_json(completion, {"schema": SCHEMA, "status": "complete", "source": source,
            "maps": validate_cfl(output/"csm/coil_sens"), "statistics": stats,
            "method": "local-open-trace-bounded-multiscale-complex-continuation",
            "experimental": True, "regularization_optimized": False,
            "provenance_files": [file_record(p) for p in (output/"metadata/request.json",
                output/"metadata/open_traces.json", output/"masks/support.npz", output/"figures/mouth_support.png",
                config_path, cache, annotation)]})
    return completion


def plot_annotated_csm(path: Path, source: dict, native: np.ndarray, added: np.ndarray,
                       roi: np.ndarray, domain: np.ndarray, profiles: list[dict], margin_mm: float) -> None:
    """Save matched ACS/native/candidate QC at the annotated slices with open red traces.

    Grey is same-basis unmasked ACS RSS; green is actual accepted addition. White
    dotted contours show the local annotation domain (not a tissue segmentation).
    Red pixels are scattered, never joined into a closed contour.
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
        for axis, profile in enumerate(profiles):
            index = profile["panel"]["index"]
            bg = np.sqrt(np.clip(display_slice(magnitude, axis, index)/window, 0, 1))
            old, new, box, allowed = [display_slice(a, axis, index) for a in (native, added, roi, domain)]
            remaining = [name for i, name in enumerate(names) if i != axis]
            for column, title in enumerate(("ACS + open annotation", "Original hard 0.6", "Local annotated candidate v2")):
                panel = panels[axis, column]
                panel.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
                if column:
                    draw_contour(panel, old.astype(float), .5, "yellow", "solid")
                    draw_contour(panel, box.astype(float), .5, "cyan", "dashed")
                if column == 2:
                    rgba = np.zeros((*new.shape, 4))
                    rgba[..., 1], rgba[..., 3] = 1, new*.45
                    panel.imshow(rgba, origin="lower")
                    draw_contour(panel, allowed.astype(float), .5, "white", "dotted")
                panel.scatter(profile["x"], profile["y"], s=.5, c="red", linewidths=0)
                panel.set_title(f"{title}: {names[axis]}={index}")
                panel.set_xlabel(remaining[1]+" voxel")
                panel.set_ylabel(remaining[0]+" voxel")
        fig.suptitle("Same-basis ACS RSS | red: open annotation | yellow: original support | green: added\n"
                     f"Cyan: local ROI | white dotted: local constraint domain (+{margin_mm:g} mm margin) | not a full segmentation\n"
                     f"Common window 0..{window:.6g}, gamma=0.5 | logical axes | experimental, no reconstruction")
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
