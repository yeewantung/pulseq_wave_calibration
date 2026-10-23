#!/usr/bin/env python3
"""Compare nested conservative support previews without CSM prediction or reconstruction."""

import argparse
import copy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

import numpy as np

from wave_retro_lr.bart_io import sha256_file
from wave_retro_lr.csm_cache import file_record, read_json, write_json
from wave_retro_lr.custom_csm import load_custom_csm, locked_output
from wave_retro_lr.metal_runner import configure_idle_process, memory_guard
from wave_retro_lr.slice_local_csm import local_patch_domains
from wave_retro_lr.support_preview import validate_preview

SCHEMA = "mprage-support-preview-comparison-v1"


def conservative_variants(existing, previous, spacing, profiles, settings, config):
    """Return nested geometry proposals within a prior preview, preserving existing support.

    Inputs are boolean native-grid masks, mapped traces, previous single-patch
    settings and a common tightened ROI/slab plus increasing distance caps.
    Invalid settings or any growth outside the earlier preview raise ValueError.
    Returns proposed masks and explicit settings for each cap; no vectors are fit.
    """
    caps = config["caps_mm"]
    if existing.dtype != bool or previous.dtype != bool or existing.shape != previous.shape:
        raise ValueError("Require matching boolean support/proposal grids")
    if (not caps or not np.isfinite(caps).all() or min(caps) <= 0
            or caps != sorted(set(caps)) or len(settings["patches"]) != 1
            or len({f"{cap:g}" for cap in caps}) != len(caps)):
        raise ValueError("Require increasing positive caps and one reviewed patch")
    result, configurations = [], []
    for cap in caps:
        patch = copy.deepcopy(settings["patches"][0])
        patch.update(roi_voxels=config["roi_voxels"], core_half_width_mm=config["core_half_width_mm"],
                     outer_half_width_mm=config["outer_half_width_mm"], extension_cap_mm=cap)
        domain, _ = local_patch_domains(existing, spacing, profiles, [patch], settings["margin_mm"])
        proposal = domain & ~existing
        if (proposal & ~previous).any() or (result and (result[-1] & ~proposal).any()):
            raise ValueError("Conservative variants must be nested subsets of the prior preview")
        if not proposal.any():
            raise ValueError("Empty conservative proposal")
        result.append(proposal)
        configurations.append(patch)
    return result, configurations


def validate_comparison(path, request):
    """Validate exact preview-comparison identity and provenance; return its record."""
    record = read_json(path)
    if record.get("schema") != SCHEMA or record.get("request") != request or record.get("status") != "preview_complete":
        raise ValueError("Preview comparison identity mismatch")
    for item in record["provenance_files"]:
        if file_record(Path(item["path"])) != item:
            raise ValueError("Preview comparison provenance changed")
    return record


def prepare_comparison(config_path, output):
    """Write masks/QC/metadata for a confirmed comparison root, with strict resume.

    Config selects a hash-bound earlier preview, common narrowed ROI/slab and
    distance caps. All existing parent support is retained. No output CFL files,
    coil-vector confidence estimation, BART invocation or winner selection occurs.
    Returns metadata/support_comparison.json, deliberately not a custom CSM manifest.
    """
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    if set(config) != {"baseline_preview", "roi_voxels", "core_half_width_mm", "outer_half_width_mm", "caps_mm"}:
        raise ValueError("Unexpected conservative comparison configuration")
    prior_path = (config_path.parent/Path(config["baseline_preview"])).resolve(strict=True)
    prior_root = prior_path.parent.parent
    print("Validating previous preview and accepted parent maps.", flush=True)
    prior = validate_preview(prior_path, read_json(prior_root/"metadata/request.json"))
    settings = prior["request"]["settings"]
    parent_path = Path(prior["request"]["parent_manifest"]["path"])
    parent = load_custom_csm(parent_path)
    if parent["source"] != prior["request"]["source"]:
        raise ValueError("Preview and parent source differ")
    mask_path = prior_root/"masks/support_preview.npz"
    if file_record(mask_path) not in prior["provenance_files"]:
        raise ValueError("Prior preview does not bind its mask")
    with np.load(mask_path) as masks:
        existing, previous = masks["existing_full"], masks["proposed_additions"]
    spacing = tuple(prior["request"]["spacing_ro_lin_par_mm"])
    profiles = read_json(Path(prior["request"]["traces"]["path"]))["profiles"]
    memory_guard()
    variants, parameters = conservative_variants(existing, previous, spacing, profiles, settings, config)
    request = {"config": file_record(config_path), "settings": config, "baseline_preview": file_record(prior_path),
               "source": parent["source"], "parameters": parameters, "geometry_only": True,
               "implementation": {"compare_custom_csm_support.py": sha256_file(Path(__file__)),
                   "slice_local_csm.py": sha256_file(Path(__file__).resolve().parents[1]/"wave_retro_lr/slice_local_csm.py")}}
    inherited = [file_record(prior_path), *prior["provenance_files"]]
    output = output.resolve()
    completion = output/"metadata/support_comparison.json"
    with locked_output(output, request, ("masks", "figures", "metadata")):
        if completion.exists():
            validate_comparison(completion, request)
            return completion
        if any((output/"masks").iterdir()) or any((output/"figures").iterdir()):
            raise ValueError("Partial comparison; refusing overwrite")
        stats = {}
        artifacts = [output/"metadata/request.json", config_path]
        for cap, proposed in zip(config["caps_mm"], variants):
            path = output/f"masks/cap_{cap:g}mm.npz"
            np.savez_compressed(path, existing_full=existing, proposed_additions=proposed, proposed_full=existing | proposed)
            artifacts.append(path)
            stats[f"{cap:g}mm"] = {"proposed_voxels": int(proposed.sum()), "removed_existing_voxels": 0,
                "outside_previous_preview": int((proposed & ~previous).sum()),
                "by_ro": {str(i): int(proposed[i].sum()) for i in settings["axial_neighbors"]}}
        print(f"Rendering proposals: {stats}; no CSM generation.", flush=True)
        plot_comparison(output/"figures", parent["source"], existing, variants, config["caps_mm"],
                        profiles, settings["axial_neighbors"])
        for item in inherited:
            if file_record(Path(item["path"])) != item:
                raise ValueError("Ancestor provenance changed during comparison")
        load_custom_csm(parent_path)
        artifacts.extend([output/"figures/cap_comparison.png", output/"figures/axial_neighbors.png"])
        write_json(completion, {"schema": SCHEMA, "status": "preview_complete", "request": request,
            "statistics": stats, "previous_proposed_voxels": int(previous.sum()),
            "coil_vectors_calculated": False, "reconstruction_ready": False, "selected_variant": None,
            "provenance_files": [*inherited, *(file_record(p) for p in artifacts)]})
    return completion


def plot_comparison(folder, source, existing, variants, caps, profiles, neighbors):
    """Save shared-window ACS/full-support comparisons; cyan denotes unvalidated proposals.

    Produces cap_comparison.png for central reviewed axial slices and
    axial_neighbors.png including unchanged control slices. No anatomy is synthesized.
    """
    from wave_retro_lr.csm_overlay import acs_rss_magnitude, draw_contour
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    magnitude = acs_rss_magnitude(Path(source["acs"]["base"]))
    window = float(np.percentile(magnitude, 99.5))
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Invalid ACS display window")

    def draw(panel, index, proposed=None, plain=False):
        """Render one RO slice without treating uncolored native support as a hole."""
        bg = np.sqrt(np.clip(magnitude[index]/window, 0, 1))
        panel.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1)
        if not plain:
            old = existing[index]
            rgba = np.zeros((*old.shape, 4))
            rgba[..., 1], rgba[..., 3] = 1, old*.16
            panel.imshow(rgba, origin="lower")
            draw_contour(panel, old.astype(float), .5, "lime", "solid")
            if proposed is not None:
                new = proposed[index]
                rgba = np.zeros((*new.shape, 4))
                rgba[..., 1], rgba[..., 2], rgba[..., 3] = 1, 1, new*.6
                panel.imshow(rgba, origin="lower")
        if index == profiles[0]["panel"]["index"]:
            panel.scatter(profiles[0]["x"], profiles[0]["y"], s=.5, c="red", linewidths=0)
        panel.set_xlabel("PAR voxel")
        panel.set_ylabel("LIN voxel")

    legend = ("Green: FULL existing v3 support | cyan: proposed additions | common narrowed lower-RO slab\n"
              f"Geometry ONLY: no CSM/confidence evaluation or reconstruction | ACS window 0..{window:.6g}, gamma=0.5\n"
              "Distance caps are measured from existing support, not outward margins from tissue; no winner selected")
    # Interior neighbor slices expose the tightened region without hiding unchanged controls.
    focus = neighbors[1:4] if len(neighbors) >= 4 else neighbors
    fig, panels = plt.subplots(len(focus), 2+len(caps), figsize=(4*(2+len(caps)), 4.5*len(focus)), layout="constrained", squeeze=False)
    try:
        for row, index in enumerate(focus):
            draw(panels[row, 0], index, plain=True)
            panels[row, 0].set_title(f"ACS: RO={index}")
            draw(panels[row, 1], index)
            panels[row, 1].set_title(f"Existing v3: RO={index}")
            for col, (cap, proposed) in enumerate(zip(caps, variants), 2):
                draw(panels[row, col], index, proposed)
                panels[row, col].set_title(f"{cap:g} mm cap: RO={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"cap_comparison.png", dpi=150)
    finally:
        plt.close(fig)
    fig, panels = plt.subplots(1+len(caps), len(neighbors), figsize=(3.7*len(neighbors), 4.5*(1+len(caps))), layout="constrained", squeeze=False)
    try:
        for col, index in enumerate(neighbors):
            for row, proposed in enumerate([None, *variants]):
                draw(panels[row, col], index, proposed)
                label = "Existing v3" if row == 0 else f"{caps[row-1]:g} mm cap"
                panels[row, col].set_title(f"{label}: RO={index}")
        fig.suptitle(legend)
        fig.savefig(folder/"axial_neighbors.png", dpi=150)
    finally:
        plt.close(fig)


def main():
    """Parse confirmed config/output paths and run an idle single-thread CPU preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_idle_process()
    memory_guard()
    print(prepare_comparison(args.config, args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, MemoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
