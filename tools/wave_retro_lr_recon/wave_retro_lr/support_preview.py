"""Geometry-only review of local CSM support proposals; never generate coil maps."""

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .bart_io import sha256_file
from .csm_cache import file_record, read_json, write_json
from .custom_csm import load_custom_csm, locked_output
from .metal_runner import memory_guard
from .slice_local_csm import local_patch_domains

SCHEMA = "mprage-custom-csm-support-preview-v1"


def preview_masks(native: np.ndarray, additions: np.ndarray, domain: np.ndarray) -> dict:
    """Return existing/proposed full support and strictly additive local proposal masks.

    All inputs are boolean masks on one native grid. No confidence filtering,
    coil-vector extrapolation or image reconstruction is performed. Invalid or
    overlapping native/addition masks raise ValueError. Inputs remain unchanged.
    """
    if (native.ndim != 3 or any(a.dtype != bool or a.shape != native.shape for a in
                              (native, additions, domain)) or (native & additions).any()):
        raise ValueError("Require disjoint native/addition masks and a common boolean grid")
    existing = native | additions
    proposal = domain & ~existing
    return dict(native=native, existing_additions=additions, existing_full=existing,
                proposed_additions=proposal, proposed_full=existing | proposal, patch_domain=domain)


def validate_preview(path: Path, request: dict) -> dict:
    """Validate a completed geometry-only preview and all bound inputs/outputs.

    Returns the preview record; rejects non-preview schemas or mismatched hashes.
    This intentionally is not a custom-CSM completion consumable by reconstruction.
    """
    record = read_json(path)
    if record.get("schema") != SCHEMA or record.get("status") != "preview_complete" or record.get("request") != request:
        raise ValueError("Support preview identity mismatch")
    for item in record["provenance_files"]:
        if file_record(Path(item["path"])) != item:
            raise ValueError("Support preview provenance changed")
    return record


def prepare_support_preview(config_path: Path, output: Path) -> Path:
    """Save only support masks, ACS QC and metadata at an approved isolated directory.

    Configuration selects a validated custom CSM parent, bound open traces, local
    patches, margin and QC slice indices. Parent arrays are never edited. Preview
    masks preserve parent support exactly but do not assert that new CSM vectors
    or signal are recoverable. Strict resume rejects changed inputs/partial output.
    Returns metadata/support_preview.json, not a reconstruction-ready manifest.
    """
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    if set(config) != {"parent_manifest", "traces", "patches", "margin_mm", "views", "axial_neighbors"}:
        raise ValueError("Unexpected support-preview configuration fields")
    parent_path = (config_path.parent/Path(config["parent_manifest"])).resolve(strict=True)
    traces_path = (config_path.parent/Path(config["traces"])).resolve(strict=True)
    parent_file = file_record(parent_path)
    print("Validating parent maps and provenance; no new CSM will be generated.", flush=True)
    parent = load_custom_csm(parent_path)
    masks_path = parent_path.parent.parent/"masks/support.npz"
    for path in (masks_path, traces_path):
        if file_record(path) not in parent["provenance_files"]:
            raise ValueError("Parent does not bind preview masks/traces")
    with np.load(masks_path) as masks:
        native, prior = masks["native"], masks["added"]
    source = parent["source"]
    shape = tuple(source["acs"]["shape"][:3])
    if native.shape != shape:
        raise ValueError("Parent support grid differs from acquisition")
    for view in config["views"]:
        if (len(view) != 2 or any(type(v) is not int for v in view)
                or view[0] not in (0, 1, 2) or not 0 <= view[1] < shape[view[0]]):
            raise ValueError("Invalid logical QC view")
    if (not config["views"] or not config["axial_neighbors"] or
            any(type(i) is not int or not 0 <= i < shape[0] for i in config["axial_neighbors"])):
        raise ValueError("Invalid neighboring axial slices")
    spacing = tuple(f/n for f, n in zip(source["geometry"]["physical_fov_mm_xyz"][::-1], shape))
    profiles = read_json(traces_path)["profiles"]
    memory_guard()
    domain, individual = local_patch_domains(native | prior, spacing, profiles, config["patches"], config["margin_mm"])
    masks = preview_masks(native, prior, domain)
    if not masks["proposed_additions"].any():
        raise ValueError("Empty support proposal")
    request = {"config": file_record(config_path), "settings": config, "source": source,
               "parent_manifest": parent_file, "parent_maps": parent["maps"], "traces": file_record(traces_path),
               "spacing_ro_lin_par_mm": list(spacing), "geometry_only": True,
               "implementation": {name: sha256_file(Path(__file__).with_name(name)) for name in
                    ("support_preview.py", "slice_local_csm.py", "annotated_csm.py", "csm_overlay.py")}}
    inherited = [parent_file, *parent["provenance_files"]]
    output = output.resolve()
    completion = output/"metadata/support_preview.json"
    with locked_output(output, request, ("masks", "figures", "metadata")):
        if completion.exists():
            validate_preview(completion, request)
            return completion
        if any((output/"masks").iterdir()) or any((output/"figures").iterdir()):
            raise ValueError("Partial support preview; refusing overwrite")
        np.savez_compressed(output/"masks/support_preview.npz", **masks,
                            **{f"patch_{i}": m for i, m in enumerate(individual)})
        proposed = masks["proposed_additions"]
        stats = {"existing_full_voxels": int(masks["existing_full"].sum()),
                 "proposed_additions": int(proposed.sum()),
                 "removed_existing_voxels": int((masks["existing_full"] & ~masks["proposed_full"]).sum()),
                 "proposed_outside_patch_domain": int((proposed & ~domain).sum()),
                 "by_ro": {str(i): int(proposed[i].sum()) for i in config["axial_neighbors"]},
                 "uncolored_regions_in_addition_only_display": {}}
        for axis, index in config["views"]:
            addition_slice = np.take(prior, index, axis=axis)
            holes = ndi.binary_fill_holes(addition_slice) & ~addition_slice
            full_slice = np.take(masks["existing_full"], index, axis=axis)
            stats["uncolored_regions_in_addition_only_display"][f"axis{axis}_slice{index}"] = {
                "apparent_hole_voxels": int(holes.sum()), "already_supported_voxels": int((holes & full_slice).sum()),
                "actually_unsupported_voxels": int((holes & ~full_slice).sum())}
        print(f"Rendering full-support preview: {int(proposed.sum())} proposed voxels, no coil vectors calculated.", flush=True)
        plot_support_preview(output/"figures", source, masks, profiles, config["views"], config["axial_neighbors"])
        for item in inherited:
            if file_record(Path(item["path"])) != item:
                raise ValueError("Parent provenance changed during preview")
        load_custom_csm(parent_path)
        artifacts = [output/"metadata/request.json", output/"masks/support_preview.npz",
                     output/"figures/full_support_preview.png", output/"figures/axial_neighbors.png", config_path]
        write_json(completion, {"schema": SCHEMA, "status": "preview_complete", "request": request,
                   "statistics": stats, "coil_vectors_calculated": False, "reconstruction_ready": False,
                   "provenance_files": [*inherited, *(file_record(p) for p in artifacts)]})
    return completion


def plot_support_preview(folder: Path, source: dict, masks: dict, profiles: list[dict],
                         views: list, neighbors: list[int]) -> None:
    """Write full-support and adjacent-slice ACS overlays with proposals clearly separated.

    Green includes ALL existing support, not just earlier additions. Cyan marks
    geometry-only proposals whose coil confidence has not been evaluated. Actual
    red traces appear only on their annotated planes. Both figures use one ACS
    window and logical coordinates and are saved in the confirmed figures folder.
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
        """Render a logical slice with full existing support and optional proposal."""
        bg = np.sqrt(np.clip(display_slice(magnitude, axis, index)/window, 0, 1))
        panel.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
        existing, proposed = (display_slice(masks[k], axis, index) for k in ("existing_full", "proposed_additions"))
        if mode:
            rgba = np.zeros((*existing.shape, 4))
            rgba[..., 1], rgba[..., 3] = 1, existing*.16
            panel.imshow(rgba, origin="lower")
            draw_contour(panel, existing.astype(float), .5, "lime", "solid")
        if mode == 2:
            rgba = np.zeros((*proposed.shape, 4))
            rgba[..., 1], rgba[..., 2], rgba[..., 3] = 1, 1, proposed*.6
            panel.imshow(rgba, origin="lower")
            draw_contour(panel, (existing | proposed).astype(float), .5, "cyan", "dotted")
        if index == profiles[axis]["panel"]["index"]:
            panel.scatter(profiles[axis]["x"], profiles[axis]["y"], s=.5, c="red", linewidths=0)
        remaining = [n for i, n in enumerate(names) if i != axis]
        panel.set_xlabel(remaining[1]+" voxel")
        panel.set_ylabel(remaining[0]+" voxel")

    legend = ("Green: FULL existing v3 support (native + all additions) | cyan fill: proposed v4 additions\n"
              "Geometry preview ONLY: no new CSM, no confidence evaluation, no reconstruction\n"
              f"Same-basis ACS RSS window 0..{window:.6g}, gamma=0.5 | red traces only on annotated slices | logical axes")
    fig, panels = plt.subplots(len(views), 3, figsize=(13, 4.6*len(views)), layout="constrained", squeeze=False)
    try:
        for row, (axis, index) in enumerate(views):
            for mode, title in enumerate(("ACS", "Existing v3 FULL support", "v4 FULL support proposal")):
                draw(panels[row, mode], axis, index, mode)
                panels[row, mode].set_title(f"{title}: {names[axis]}={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"full_support_preview.png", dpi=150)
    finally:
        plt.close(fig)
    fig, panels = plt.subplots(2, len(neighbors), figsize=(3.7*len(neighbors), 9), layout="constrained", squeeze=False)
    try:
        for col, index in enumerate(neighbors):
            for row, mode in enumerate((1, 2)):
                draw(panels[row, col], 0, index, mode)
                panels[row, col].set_title(f"{'v3 existing' if row == 0 else 'v4 proposal'}: RO={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"axial_neighbors.png", dpi=150)
    finally:
        plt.close(fig)
