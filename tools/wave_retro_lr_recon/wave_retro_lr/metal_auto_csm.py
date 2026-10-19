"""Conservative calibration-guided CSM support completion, enabled only by explicit metal mode."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy import fft, ndimage as ndi

from .bart_io import create_cfl, open_cfl
from .metal_csm import convex_envelope


AUTO_POLICY = {
    "version": "calibration-guided-metal-support-v1",
    "coordinate_system": "logical_RO_LIN_PAR_mm",
    "envelope_axis": 0,
    "maximum_distance_mm": 8.0,
    "near_sigma_mm": 2.0,
    "far_sigma_mm": 4.0,
    "minimum_weight": 0.05,
    "minimum_reference_fraction": 0.05,
    "minimum_vector_coherence": 0.5,
    "minimum_scale_agreement": 0.98,
    "background_mad_multiplier": 4.0,
    "minimum_native_signal_fraction": 0.01,
    "minimum_component_mm3": 8.0,
    "maximum_component_mm3": 20000.0,
    "maximum_added_native_fraction": 0.03,
    "minimum_adjacent_planes": 2,
    "experimental_not_population_validated": True,
}


def calibration_rss(calibration: np.ndarray, checkpoint: Callable[[], None]) -> np.ndarray:
    """Return per-voxel RSS from centered ACS IFFT, one coil at a time to bound RAM."""
    if calibration.ndim != 4:
        raise ValueError("Expected single-map logical 3-D calibration with a coil axis")
    energy = np.zeros(calibration.shape[:3], np.float32)
    for coil in range(calibration.shape[3]):
        checkpoint()
        field = np.asarray(calibration[..., coil])
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite calibration input")
        image = fft.fftshift(fft.ifftn(fft.ifftshift(field), norm="ortho", workers=1))
        energy += abs(image)**2
    result = np.sqrt(energy)
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError("Calibration RSS is nonfinite or empty")
    return result


def automatic_vectors(maps: np.ndarray, rss: np.ndarray, spacing: tuple[float, ...],
                      checkpoint: Callable[[], None]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Propose bounded gaps, gate by ACS signal/phase-aligned consistency, and return sparse vectors.

    Inputs are logical RO/LIN/PAR/coil hard-cropped maps, matching unmasked ACS
    RSS and logical voxel sizes in mm. Original supported vectors are never
    changed. Missing evidence returns a documented no-op, not inferred anatomy.
    Invalid geometry/nonfinite/soft maps raise ValueError. This is not metal
    detection, tissue segmentation or a validated noise/SNR estimator.
    """
    policy = AUTO_POLICY
    spacing = np.asarray(spacing, float)
    if (maps.ndim != 4 or rss.shape != maps.shape[:3] or spacing.shape != (3,)
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(rss).all() or np.any(rss < 0)):
        raise ValueError("Invalid automatic CSM geometry/calibration/spacing")
    energy = np.zeros(rss.shape, np.float32)
    coil_energy = []
    for coil in range(maps.shape[3]):
        checkpoint()
        field = np.asarray(maps[..., coil])
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite CSM")
        e = abs(field)**2
        energy += e
        coil_energy.append(float(e.sum(dtype=np.float64)))
    norms = np.sqrt(energy)
    if not np.all((norms < 1e-8) | (abs(norms-1) < 1e-5)):
        raise ValueError("Automatic completion requires hard-cropped unit-RSS single maps")
    native = norms > .5
    if not native.any():
        raise ValueError("Empty native CSM support")
    hull = np.zeros_like(native)
    for plane in range(native.shape[0]):
        checkpoint()
        hull[plane] = convex_envelope(native[plane])
    distance = ndi.distance_transform_edt(~native, sampling=spacing)
    proposal = hull & ~native & (distance <= policy["maximum_distance_mm"])
    # A background proxy is usable only where enough genuinely outside-envelope
    # voxels exist. With no such evidence, fail conservatively to a no-op.
    outside = ~ndi.binary_dilation(hull, iterations=2)
    background = rss[outside]
    report = {"native_support_voxels": int(native.sum()), "geometric_candidate_voxels": int(proposal.sum()),
              "background_proxy_voxels": int(background.size), "background_is_noise_scan": False}
    added = np.zeros_like(native)
    confidence = np.zeros(rss.shape, np.float32)
    empty = np.empty((0, maps.shape[3]), np.complex64)
    if background.size < 64 or not proposal.any():
        report.update(status="no_op", reason="no_geometric_candidate_or_insufficient_background", added_voxels=0)
        return native, added, empty, report, confidence
    median = float(np.median(background))
    mad = float(np.median(abs(background-median)))
    threshold = max(median + policy["background_mad_multiplier"]*1.4826*mad,
                    float(np.median(rss[native]))*policy["minimum_native_signal_fraction"])
    proposal &= rss > threshold
    # Require evidence on an adjacent SI plane; isolated single-slice additions
    # are deliberately omitted rather than interpreted as recoverable tissue.
    adjacent = np.zeros_like(proposal)
    adjacent[1:] |= proposal[:-1]
    adjacent[:-1] |= proposal[1:]
    proposal &= adjacent
    report.update(signal_threshold=threshold, signal_candidate_voxels=int(proposal.sum()))
    if not proposal.any():
        report.update(status="no_op", reason="no_calibration_supported_candidate", added_voxels=0)
        return native, added, empty, report, confidence

    # Align the otherwise arbitrary common phase using a well-supported coil.
    # Donors with a weak reference channel are excluded, never phase-divided.
    reference_coil = int(np.argmax(coil_energy))
    reference = np.asarray(maps[..., reference_coil])
    donors = native & (abs(reference) >= policy["minimum_reference_fraction"])
    gauge = np.ones(rss.shape, np.complex64)
    gauge[donors] = reference[donors]/abs(reference[donors])
    predictions, weights = [], []
    for width in (policy["near_sigma_mm"], policy["far_sigma_mm"]):
        weights.append(ndi.gaussian_filter(donors.astype(np.float32), width/spacing, mode="constant")[proposal])
        predictions.append(np.empty((int(proposal.sum()), maps.shape[3]), np.complex64))
    for coil in range(maps.shape[3]):
        checkpoint()
        aligned = np.asarray(maps[..., coil])*np.conj(gauge)*donors
        for index, width in enumerate((policy["near_sigma_mm"], policy["far_sigma_mm"])):
            predictions[index][:, coil] = ndi.gaussian_filter(aligned, width/spacing, mode="constant")[proposal]/np.maximum(weights[index], 1e-12)
    coherence = [np.linalg.norm(v, axis=-1) for v in predictions]
    unit = [v/np.maximum(c[:, None], 1e-12) for v, c in zip(predictions, coherence)]
    inner = np.sum(unit[0].conj()*unit[1], axis=-1)
    agreement = abs(inner)
    # Bring the far prediction into the near prediction's common phase before blending.
    unit[1] *= np.exp(-1j*np.angle(inner))[:, None]
    mixture = unit[0] + unit[1]
    norm = np.linalg.norm(mixture, axis=-1)
    accepted = ((weights[0] >= policy["minimum_weight"]) & (weights[1] >= policy["minimum_weight"])
                & (coherence[0] >= policy["minimum_vector_coherence"])
                & (coherence[1] >= policy["minimum_vector_coherence"])
                & (agreement >= policy["minimum_scale_agreement"]) & (norm > .5))
    added[proposal] = accepted
    labels, count = ndi.label(added)
    sizes = np.bincount(labels.ravel())*float(np.prod(spacing))
    keep = (sizes >= policy["minimum_component_mm3"]) & (sizes <= policy["maximum_component_mm3"])
    keep[0] = False
    added = keep[labels]
    if added.sum() > native.sum()*policy["maximum_added_native_fraction"]:
        added[:] = False
        reason = "automatic_safety_volume_cap"
    else:
        reason = "accepted" if added.any() else "no_reliable_bounded_component"
    confidence[proposal] = np.clip(agreement, 0, 1)
    confidence[~added] = 0
    # Restore a nearby original gauge; do not alter any original supported voxel.
    indices = ndi.distance_transform_edt(~donors, return_distances=False, return_indices=True)
    phase = gauge[tuple(indices[:, added])]
    vectors = (mixture/np.maximum(norm[:, None], 1e-12))[added[proposal]]*phase[:, None]
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite automatic continuation")
    report.update(status="extended" if added.any() else "no_op", reason=reason,
                  reference_coil=reference_coil, donor_voxels=int(donors.sum()),
                  added_voxels=int(added.sum()), added_fraction_of_native=float(added.sum()/native.sum()),
                  maximum_added_distance_mm=float(distance[added].max()) if added.any() else 0.,
                  connected_candidates=count)
    return native, added, vectors, report, confidence


def write_automatic_candidate(maps_base: Path, calibration_base: Path, destination: Path,
                              spacing: tuple[float, ...], checkpoint: Callable[[], None]) -> tuple[dict, dict]:
    """Write exclusive candidate CFL and logical-grid support/confidence NPZ; return stats and arrays.

    The destination is a prepared shared-CSM directory with intermediate/ and
    figures/ children. Original inputs remain untouched, including no-op cases.
    """
    maps_raw = open_cfl(maps_base)
    calibration_raw = open_cfl(calibration_base)
    if any(n != 1 for n in maps_raw.shape[4:]) or any(n != 1 for n in calibration_raw.shape[4:]):
        raise ValueError("Multiple maps/echoes are not supported")
    maps = maps_raw.reshape(maps_raw.shape[:4], order="F")
    calibration = calibration_raw.reshape(calibration_raw.shape[:4], order="F")
    if maps.shape != calibration.shape:
        raise ValueError("CSM/calibration shape mismatch")
    rss = calibration_rss(calibration, checkpoint)
    native, added, vectors, stats, confidence = automatic_vectors(maps, rss, spacing, checkpoint)
    candidate = create_cfl(destination/"intermediate/coil_sens", maps.shape)
    for coil in range(maps.shape[3]):
        checkpoint()
        field = np.asarray(maps[..., coil]).copy()
        field[added] = vectors[:, coil]
        if not np.array_equal(field[~added], maps[..., coil][~added]):
            raise ValueError("Original supported CSM changed")
        candidate[..., coil] = field
    candidate.flush()
    np.savez_compressed(destination/"intermediate/support.npz", native=native, added=added,
                        confidence=confidence, calibration_rss=rss, spacing_mm=np.asarray(spacing))
    stats["unchanged_vectors_max_error"] = 0.
    return stats, {"native": native, "added": added, "rss": rss, "confidence": confidence}


def write_auto_figure(path: Path, arrays: dict) -> None:
    """Write a logical-axis QC overlay at maximum addition (or center for no-op), not a clinical view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    added, rss, native = arrays["added"], arrays["rss"], arrays["native"]
    plane = int(np.argmax(added.sum(axis=(1, 2)))) if added.any() else added.shape[0]//2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = max(float(np.percentile(rss[native], 99)), 1e-12)
    for axis, mask, title in zip(axes, (native, added, arrays["confidence"]),
                                 ("Original CSM support", "Added support", "Accepted consistency")):
        axis.imshow(rss[plane].T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        axis.imshow(np.ma.masked_where(mask[plane].T == 0, mask[plane].T), origin="lower", cmap="autumn", alpha=.5, vmin=0, vmax=1)
        axis.set_title(title)
    fig.suptitle(f"Experimental automatic support; logical RO plane {plane}; not anatomical ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
