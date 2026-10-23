"""Add reviewed slice-local CSM patches without intersecting unrelated annotations."""

from pathlib import Path
import shutil

import numpy as np
from scipy import ndimage as ndi

from .annotated_csm import annotated_vectors, trace_halfspace
from .bart_io import create_cfl, open_cfl, sha256_file
from .csm_cache import file_record, read_json, validate_cfl, write_json
from .custom_csm import SCHEMA, load_custom_csm, locked_output
from .metal_csm import smoothstep
from .metal_runner import memory_guard


def local_patch_domains(support: np.ndarray, spacing: tuple, profiles: list[dict],
                        patches: list[dict], margin_mm: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build independent bounded patches from open traces near their actual slices.

    Args:
        support: Accepted parent support on the native logical RO/LIN/PAR grid.
        spacing: Three positive voxel sizes in mm.
        profiles: Mapped open annotation profiles in logical axis order.
        patches: Named trace selectors with half-open ROI bounds, slab core/outer
            half-widths, ROI edge taper and maximum distance from parent support.
        margin_mm: Physical outward trace margin, without endpoint extrapolation.
    Returns:
        Union of patch domains and individual domains. A smoothstep slab/ROI
        taper reduces geometric extension toward patch edges. It does not soften
        CSM amplitudes, form a segmentation or change parent-supported vectors.
    Raises:
        ValueError: Invalid settings, empty support or mismatched annotation grid.
    """
    shape = support.shape
    if (support.ndim != 3 or support.dtype != bool or not support.any() or len(spacing) != 3 or len(profiles) != 3
            or not np.isfinite(spacing).all() or min(spacing) <= 0 or not patches):
        raise ValueError("Invalid slice-local grid/support")
    distance = ndi.distance_transform_edt(~support, sampling=spacing)
    domains, names = [], set()
    for patch in patches:
        if set(patch) != {"name", "trace_axis", "roi_voxels", "core_half_width_mm",
                          "outer_half_width_mm", "edge_taper_mm", "extension_cap_mm"}:
            raise ValueError("Unexpected slice-local patch fields")
        name, axis = patch["name"], patch["trace_axis"]
        if not isinstance(name, str) or not name or name in names or type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("Require unique patch names and a logical trace axis")
        names.add(name)
        profile = profiles[axis]
        if (profile["panel"]["axis"] != axis or tuple(profile["panel"]["shape"]) !=
                tuple(n for i, n in enumerate(shape) if i != axis)):
            raise ValueError("Slice-local annotation grid mismatch")
        center = profile["panel"]["index"]
        if type(center) is not int or not 0 <= center < shape[axis]:
            raise ValueError("Invalid annotated slice index")
        bounds = patch["roi_voxels"]
        if len(bounds) != 3 or any(len(p) != 2 or any(type(v) is not int for v in p)
                or not 0 <= p[0] < p[1] <= n for p, n in zip(bounds, shape)):
            raise ValueError("Patch ROI must contain bounded half-open integer ranges")
        core, outer, edge, cap = (patch[k] for k in ("core_half_width_mm", "outer_half_width_mm",
                                                    "edge_taper_mm", "extension_cap_mm"))
        if not np.isfinite([core, outer, edge, cap]).all() or not 0 <= core < outer or min(edge, cap) <= 0:
            raise ValueError("Invalid slice-local widths/taper/cap")
        plane = trace_halfspace(profile, tuple(s for i, s in enumerate(spacing) if i != axis), margin_mm)
        allowed = np.broadcast_to(np.expand_dims(plane, axis), shape)
        roi = np.zeros(shape, bool)
        roi[tuple(slice(a, b) for a, b in bounds)] = True
        # Local geometric extension vanishes outside the reviewed slab and ROI.
        depth = abs(np.arange(shape[axis])-center)*spacing[axis]
        slab = smoothstep((outer-depth)/(outer-core))
        dims = [1, 1, 1]
        dims[axis] = shape[axis]
        allowance = np.broadcast_to(slab.reshape(dims), shape).astype(np.float32).copy()
        for dim, ((lo, hi), step) in enumerate(zip(bounds, spacing)):
            coordinate = np.arange(shape[dim])
            taper = smoothstep(np.minimum(coordinate-lo+.5, hi-.5-coordinate)*step/edge)
            dims = [1, 1, 1]
            dims[dim] = shape[dim]
            allowance *= taper.reshape(dims)
        domains.append(roi & allowed & (allowance > 0) & (distance <= cap*allowance))
    return np.logical_or.reduce(domains), domains


def prepare_slice_local_csm(config_path: Path, output: Path) -> Path:
    """Generate an additive local-patch CSM/QC derivative, never reconstruction.

    Input JSON selects a completed annotation-bounded parent, patch settings and
    physical margin. Parent vectors remain bitwise exact. Predictions use only
    the original calibrated maps, not earlier extrapolated vectors as donors.
    Writes csm/masks/figures/metadata under the confirmed immutable output root.
    Source, parent, configuration, maps, annotations and outputs are hash-bound;
    partial/mismatched reuse or any preservation failure raises ValueError.
    """
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    if set(config) != {"parent_manifest", "patches", "margin_mm"}:
        raise ValueError("Expected parent_manifest, patches and margin_mm")
    parent_path = (config_path.parent/Path(config["parent_manifest"])).resolve(strict=True)
    parent_file = file_record(parent_path)
    print("Validating accepted parent CSM, source and annotation provenance.", flush=True)
    parent = load_custom_csm(parent_path)
    if parent.get("method") != "local-open-trace-bounded-multiscale-complex-continuation":
        raise ValueError("Require an annotation-bounded parent, not recursive patch continuation")
    parent_root = parent_path.parent.parent
    request_path, trace_path, mask_path = (parent_root/p for p in
        ("metadata/request.json", "metadata/open_traces.json", "masks/support.npz"))
    # Do not trust neighboring files unless the validated parent binds them.
    for path in (request_path, trace_path, mask_path):
        if file_record(path) not in parent["provenance_files"]:
            raise ValueError("Parent does not bind its maps request, traces and masks")
    previous = read_json(request_path)
    source = parent["source"]
    if previous["source"] != source:
        raise ValueError("Parent request/source mismatch")
    baseline = previous["baseline"]
    validate_cfl(Path(baseline["base"]), baseline)
    shape = tuple(source["acs"]["shape"][:4])
    if baseline["shape"][:4] != list(shape) or any(n != 1 for n in baseline["shape"][4:]):
        raise ValueError("Original calibration grid/count differs from parent")
    original = open_cfl(baseline["base"]).reshape(shape, order="F")
    accepted = open_cfl(parent["maps"]["base"]).reshape(shape, order="F")
    with np.load(mask_path) as masks:
        native, prior = masks["native"], masks["added"]
    if native.shape != shape[:3] or prior.shape != shape[:3] or native.dtype != bool or prior.dtype != bool or (native & prior).any():
        raise ValueError("Invalid parent support masks")
    original_rss2, parent_rss2 = np.zeros(shape[:3], np.float32), np.zeros(shape[:3], np.float32)
    for coil in range(shape[3]):
        memory_guard()
        original_rss2 += abs(original[..., coil])**2
        parent_rss2 += abs(accepted[..., coil])**2
        if not np.array_equal(original[..., coil][~prior], accepted[..., coil][~prior]):
            raise ValueError("Parent changed non-added calibrated vectors")
    if (not np.array_equal(original_rss2 > 1e-10, native)
            or np.max(abs(np.sqrt(parent_rss2)-(native | prior))) > 1e-5):
        raise ValueError("Parent masks do not match map support/unit RSS")
    spacing = tuple(f/n for f, n in zip(source["geometry"]["physical_fov_mm_xyz"][::-1], shape[:3]))
    profiles = read_json(trace_path)["profiles"]
    domain, domains = local_patch_domains(native | prior, spacing, profiles, config["patches"], config["margin_mm"])
    donor_settings = previous["settings"]
    request = {"config": file_record(config_path), "settings": config, "source": source,
               "parent_manifest": parent_file, "parent_maps": parent["maps"], "donor_maps": baseline,
               "spacing_ro_lin_par_mm": list(spacing), "donor_settings": donor_settings,
               "interpretation": "Independent slice-local patches; not a closed tissue segmentation",
               "implementation": {name: sha256_file(Path(__file__).with_name(name)) for name in
                    ("slice_local_csm.py", "annotated_csm.py", "custom_csm.py", "metal_csm.py", "csm_overlay.py")}}
    # Flatten parent provenance so later custom-map consumers verify all ancestors.
    inherited = [parent_file, *parent["provenance_files"],
                 file_record(Path(baseline["base"]+".hdr")), file_record(Path(baseline["base"]+".cfl")),
                 file_record(Path(parent["maps"]["base"]+".hdr")), file_record(Path(parent["maps"]["base"]+".cfl"))]
    print(f"Slice-local proposal: {int((domain & ~(native | prior)).sum())} new voxels; parent remains unchanged.", flush=True)
    output = output.resolve()
    completion = output/"metadata/custom_csm.json"
    with locked_output(output, request, ("csm", "masks", "figures", "metadata")):
        if completion.exists():
            load_custom_csm(completion)
            return completion
        if any((output/"csm").iterdir()) or any((output/"masks").iterdir()):
            raise ValueError("Partial slice-local output; refusing overwrite")
        if shutil.disk_usage(output).free < baseline["payload_bytes"]+1024**3:
            raise ValueError("Need candidate capacity plus 1 GiB reserve")
        _, added, vectors, stats = annotated_vectors(original, domain & ~(native | prior), spacing,
            donor_settings["sigmas_mm"], donor_settings["max_distance_mm"])
        if not added.any():
            raise ValueError("No stable new slice-local vectors")
        candidate = create_cfl(output/"csm/coil_sens", shape)
        rss2, max_change = np.zeros(shape[:3], np.float32), 0.
        for coil in range(shape[3]):
            memory_guard()
            field = np.asarray(accepted[..., coil]).copy()
            field[added] = vectors[:, coil]
            max_change = max(max_change, float(abs(field[~added]-accepted[..., coil][~added]).max()))
            rss2 += abs(field)**2
            candidate[..., coil] = field
        candidate.flush()
        error = float(abs(np.sqrt(rss2)-(native | prior | added)).max())
        if max_change != 0 or error > 1e-5 or (added & (native | prior | ~domain)).any():
            raise ValueError("Parent preservation/unit RSS/locality invariant failed")
        stats.update(parent_vectors_max_error=max_change, candidate_rss_max_error=error,
                     parent_added_voxels=int(prior.sum()), total_added_voxels=int((prior | added).sum()),
                     newly_added_voxels=int(added.sum()), added_outside_patch_union=int((added & ~domain).sum()),
                     patch_counts={p["name"]: int((added & d).sum()) for p, d in zip(config["patches"], domains)})
        np.savez_compressed(output/"masks/support.npz", native=native, added=prior | added,
                            parent_added=prior, new_added=added, patch_domain=domain,
                            **{f"patch_{i}": d for i, d in enumerate(domains)})
        print("Saving ACS overlays and neighboring-slice QC.", flush=True)
        plot_slice_local_csm(output/"figures", source, native, prior, added, domain, profiles, config["patches"])
        print("Revalidating ancestor hashes before publishing completion.", flush=True)
        for item in inherited:
            if file_record(Path(item["path"])) != item:
                raise ValueError("Parent/source provenance changed during generation")
        # Recheck the prepared source and parent completion before publishing ours.
        load_custom_csm(parent_path)
        artifacts = [output/"metadata/request.json", output/"masks/support.npz",
                     output/"figures/mouth_support.png", output/"figures/adjacent_slices.png", config_path]
        write_json(completion, {"schema": SCHEMA, "status": "complete", "source": source,
            "maps": validate_cfl(output/"csm/coil_sens"), "statistics": stats,
            "method": "additive-slice-local-open-trace-complex-continuation",
            "experimental": True, "regularization_optimized": False,
            "provenance_files": [*inherited, *(file_record(p) for p in artifacts)]})
    return completion


def plot_slice_local_csm(folder: Path, source: dict, native: np.ndarray, prior: np.ndarray,
                         added: np.ndarray, domain: np.ndarray, profiles: list[dict], patches: list[dict]) -> None:
    """Save ACS/v2/v3 comparisons and adjacent-slice QC without synthetic anatomy.

    Original support is yellow, retained parent additions green and new patches
    cyan. Actual red annotation pixels appear only at their annotated slice.
    Neighbor views expose the approximate out-of-plane support taper. Writes two
    PNG files to the confirmed figures directory; no reconstruction is performed.
    """
    from .csm_overlay import acs_rss_magnitude, display_slice, draw_contour
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    magnitude = acs_rss_magnitude(Path(source["acs"]["base"]))
    window = float(np.percentile(magnitude, 99.5))
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Invalid ACS display window")
    names = ("RO", "LIN", "PAR")

    def draw(panel, axis, index, mode):
        """Render one logical slice using a shared ACS window and exact support masks."""
        bg = np.sqrt(np.clip(display_slice(magnitude, axis, index)/window, 0, 1))
        panel.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
        if mode:
            old, new = (display_slice(a, axis, index) for a in (prior, added))
            rgba = np.zeros((*old.shape, 4))
            rgba[..., 1], rgba[..., 3] = 1, old*.45
            panel.imshow(rgba, origin="lower")
            if mode == 2:
                rgba = np.zeros((*new.shape, 4))
                rgba[..., 1], rgba[..., 2], rgba[..., 3] = 1, 1, new*.55
                panel.imshow(rgba, origin="lower")
                draw_contour(panel, display_slice(domain, axis, index).astype(float), .5, "white", "dotted")
            draw_contour(panel, display_slice(native, axis, index).astype(float), .5, "yellow", "solid")
        if index == profiles[axis]["panel"]["index"]:
            panel.scatter(profiles[axis]["x"], profiles[axis]["y"], s=.5, c="red", linewidths=0)
        remaining = [n for i, n in enumerate(names) if i != axis]
        panel.set_xlabel(remaining[1]+" voxel")
        panel.set_ylabel(remaining[0]+" voxel")

    legend = ("Same-basis ACS RSS | yellow: original support | green: retained v2 | cyan: new v3\n"
              "Red: open annotation at its actual slice | white dotted: local patch domains (not tissue segmentation)\n"
              f"Common window 0..{window:.6g}, gamma=0.5 | logical axes | experimental, no reconstruction")
    fig, panels = plt.subplots(3, 3, figsize=(13, 14), layout="constrained")
    try:
        for axis in range(3):
            index = profiles[axis]["panel"]["index"]
            for mode, title in enumerate(("ACS + open annotation", "Accepted parent v2", "Local patches v3")):
                draw(panels[axis, mode], axis, index, mode)
                panels[axis, mode].set_title(f"{title}: {names[axis]}={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"mouth_support.png", dpi=150)
    finally:
        plt.close(fig)
    fig, panels = plt.subplots(len(patches), 5, figsize=(18, 5*len(patches)), layout="constrained", squeeze=False)
    try:
        spacing = [f/n for f, n in zip(source["geometry"]["physical_fov_mm_xyz"][::-1], native.shape)]
        for row, patch in enumerate(patches):
            axis = patch["trace_axis"]
            center = profiles[axis]["panel"]["index"]
            half = patch["outer_half_width_mm"]/spacing[axis]
            indices = np.clip(np.rint(center+np.linspace(-half, half, 5)), 0, native.shape[axis]-1).astype(int)
            for col, index in enumerate(indices):
                draw(panels[row, col], axis, int(index), 2)
                panels[row, col].set_title(f"{patch['name']}: {names[axis]}={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"adjacent_slices.png", dpi=150)
    finally:
        plt.close(fig)
