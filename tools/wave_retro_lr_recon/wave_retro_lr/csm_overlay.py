"""ACS magnitude backgrounds with matched, hash-bound CSM crop overlays."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from .bart_io import open_cfl, sha256_file
from .core import centered_fftn
from .csm_cache import file_record, prepared_source, read_json, validate_cfl, write_json
from .csm_crop import crop_weights, validate_map_shapes


def acs_rss_magnitude(base: Path) -> np.ndarray:
    """Return native-grid RSS of coil-wise centered orthonormal 3D ACS inverse FFTs.

    Args:
        base: Validated native-grid, single-map ACS CFL basename in the same
            physical/PCA/ROVir coil basis as the diagnostic CSM cache.
    Returns:
        Float32 magnitude volume in unchanged RO/LIN/PAR order. No CSM, bias
        correction, interpolation or per-coil intensity normalization is used.
        Only one complex coil image is transformed at a time on one CPU worker.
    Raises:
        ValueError: Non-finite values, incompatible dimensions or zero magnitude.
    """
    source = open_cfl(base)
    if source.ndim < 4 or any(n != 1 for n in source.shape[4:]):
        raise ValueError("ACS background requires one shared native-grid coil volume")
    source = source.reshape(source.shape[:4], order="F")
    squared = np.zeros(source.shape[:3], np.float64)
    for coil in range(source.shape[3]):
        field = np.asarray(source[..., coil])
        if not np.isfinite(field).all():
            raise ValueError("Non-finite ACS values")
        image = centered_fftn(field, axes=(0, 1, 2), inverse=True, workers=1)
        squared += np.square(image.real, dtype=np.float64) + np.square(image.imag, dtype=np.float64)
    magnitude = np.sqrt(squared).astype(np.float32)
    if not np.isfinite(magnitude).all() or not np.any(magnitude > 0):
        raise ValueError("ACS magnitude is non-finite or identically zero")
    return magnitude


def display_slice(values: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a logical slice with its first remaining axis vertical, without resampling.

    This makes RO vertical on MPRAGE LIN/PAR-fixed views. Labels remain logical
    indices, not inferred RAS or laterality labels. ACS and weights use the
    identical operation, including on non-MPRAGE geometry.
    """
    return np.take(values, index, axis=axis)


def draw_contour(panel, values: np.ndarray, level: float, color: str, style: str) -> None:
    """Draw one labelled-by-legend contour only when the level crosses this 2D field."""
    if min(values.shape) > 1 and float(values.min()) < level < float(values.max()):
        panel.contour(values, levels=[level], colors=[color], linestyles=[style], linewidths=1.)


def plot_acs_overlay(root: Path, *, slices: list[int] | None = None,
                     percentile: float = 99.5, gamma: float = .5, alpha: float = .3) -> dict:
    """Add an ACS/CSM comparison PNG and provenance JSON to an approved diagnostic root.

    Args:
        root: Existing completed cache with calibration/, csm_variants/, figures/
            and metadata/; never creates another data root or re-runs BART.
        slices: Three zero-based logical RO/LIN/PAR slice positions, or centers.
        percentile: One whole-volume ACS magnitude percentile defining the
            upper display window shared by every image panel.
        gamma: Positive display exponent, shared by all panels (not data scaling).
        alpha: Constant opacity of the soft-weight color layer in [0, 1].
    Returns:
        JSON-ready figure record with exact source hashes and display parameters.
    Raises:
        ValueError: Invalid geometry, crop provenance, display settings, or partial
            existing output. Completed PNG/JSON pairs are hash-validated on reuse.
        BlockingIOError: Another diagnostic process holds the shared lock.
    """
    if (not np.isfinite([percentile, gamma, alpha]).all() or not 0 < percentile <= 100
            or gamma <= 0 or not 0 <= alpha <= 1):
        raise ValueError("Require percentile in (0,100], gamma >0 and alpha in [0,1]")
    root = root.resolve()
    for name in ("calibration", "csm_variants", "figures", "metadata"):
        if not (root/name).is_dir() or (root/name).is_symlink():
            raise ValueError("Overlay requires an existing complete diagnostic directory")
    if (root/".lock").is_symlink():
        raise ValueError("Diagnostic lock must not be a symlink")
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calibration_path = root/"metadata/calibration.json"
        cached = read_json(calibration_path)
        original = cached["identity"]["source"]
        source = prepared_source(Path(original["manifest"]["path"]))
        if source != original:
            raise ValueError("ACS source no longer matches the completed calibration cache")
        if cached["identity"].get("maps") != 1 or cached["identity"].get("calibration_crop") != 0:
            raise ValueError("Overlay requires a completed single-map uncropped calibration")
        validate_cfl(Path(cached["maps"]["base"]), cached["maps"])
        validate_cfl(Path(cached["eigenvalues"]["base"]), cached["eigenvalues"])
        maps = open_cfl(cached["maps"]["base"])
        eigen = open_cfl(cached["eigenvalues"]["base"])
        shape = validate_map_shapes(maps.shape, eigen.shape)
        if list(shape) != source["acs"]["shape"][:4]:
            raise ValueError("ACS and CSM native geometry/coil count differ")
        indices = [n//2 for n in shape[:3]] if slices is None else slices
        if len(indices) != 3 or any(type(i) is not int or not 0 <= i < n for i, n in zip(indices, shape)):
            raise ValueError("Slices must be valid zero-based RO LIN PAR indices")
        variants = []
        for path in (root/"csm_variants").glob("*/variant.json"):
            record = read_json(path)
            if record["identity"]["calibration_sha256"] != sha256_file(calibration_path):
                raise ValueError("Crop variant is bound to another calibration")
            validate_cfl(path.parent/"weights", record["weights"])
            weights = open_cfl(path.parent/"weights")
            if weights.shape != shape[:3]:
                raise ValueError("Crop weights differ from the ACS/CSM grid")
            mode, crop = record["identity"]["mode"], record["identity"]["crop"]
            eig3 = eigen.reshape(shape[:3], order="F")
            for k in range(shape[2]):
                expected = crop_weights(eig3[:, :, k], crop, mode)
                if not np.allclose(weights[:, :, k], expected, rtol=1e-6, atol=1e-7):
                    raise ValueError("Crop weight array disagrees with its recorded policy")
            variants.append((mode, crop, path, weights))
        if not variants:
            raise ValueError("No completed CSM crop variants found")
        variants.sort(key=lambda item: (item[0] != "hard", -item[1]))
        identity = {"calibration": file_record(calibration_path), "source": source,
                    "variants": [file_record(v[2]) for v in variants],
                    "slices_ro_lin_par": indices, "percentile": percentile, "gamma": gamma,
                    "soft_alpha": alpha, "background": "sqrt(sum_coils(abs(centered_ifft3(ACS))**2))",
                    "background_basis": source["kind"], "csm_used_for_background": False,
                    "reference_contour": "hard 0.6 (eigenvalue magnitude >= 0.6)",
                    "orientation": "logical axes; first remaining axis vertical; no inferred RAS flips",
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "fft_implementation_sha256": sha256_file(Path(__file__).with_name("core.py")),
                    "numpy_version": np.__version__}
        token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        png = root/"figures"/f"acs_csm_overlay_{token}.png"
        metadata = root/"metadata"/f"acs_csm_overlay_{token}.json"
        if metadata.exists():
            previous = read_json(metadata)
            if previous.get("identity") != identity or previous.get("figure") != file_record(png):
                raise ValueError("Existing ACS overlay provenance/hash mismatch")
            return previous
        if png.exists() or png.is_symlink() or metadata.is_symlink():
            raise ValueError("Partial ACS overlay exists; refusing overwrite")
        magnitude = acs_rss_magnitude(Path(source["acs"]["base"]))
        window = float(np.percentile(magnitude, percentile))
        if window <= 0:
            raise ValueError("Selected whole-volume display percentile is zero; increase percentile")
        summary = _render_overlay(png, magnitude, maps.reshape(shape, order="F"),
                                  eigen.reshape(shape[:3], order="F"), variants, indices,
                                  window=window, gamma=gamma, alpha=alpha, basis=source["kind"])
        result = {"identity": identity, "display_window_raw_magnitude": [0., window],
                  "figure": file_record(png), "panels": summary,
                  "interpretation": "ACS is low-resolution calibration magnitude, not reconstructed tissue/SNR."}
        write_json(metadata, result)
        return result


def _render_overlay(png: Path, magnitude: np.ndarray, maps: np.ndarray, eigen: np.ndarray,
                    variants: list, indices: list[int], *, window: float, gamma: float,
                    alpha: float, basis: str) -> list[dict]:
    """Render matched panels to one new PNG; return slice labels and per-panel weight ranges.

    Caller validates and locks the destination. Grey ACS contrast is common to
    all panels. Hard support is cyan; the hard-0.6 reference is dashed yellow.
    Soft maps use a fixed [0,1] color scale and explicit 0.1/0.5/0.9 contours;
    weights above one are reported rather than silently interpreted as unity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(3, 1+len(variants), figsize=(4*(1+len(variants)), 12), squeeze=False)
    names = ("RO", "LIN", "PAR")
    summary = []
    try:
        for axis, index in enumerate(indices):
            background = np.power(np.clip(display_slice(magnitude, axis, index)/window, 0, 1), gamma)
            coil_slice = display_slice(maps, axis, index)
            has_maps = np.sum(np.abs(coil_slice)**2, axis=-1) > 0
            reference = (crop_weights(display_slice(eigen, axis, index), .6) > 0) & has_maps
            remaining = [name for i, name in enumerate(names) if i != axis]
            for column in range(1+len(variants)):
                panel = axes[axis, column]
                panel.imshow(background, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="nearest")
                panel.set_xlabel(f"{remaining[1]} voxel")
                panel.set_ylabel(f"{remaining[0]} voxel")
                title = "ACS RSS only"
                if column:
                    mode, crop, _, weights = variants[column-1]
                    field = display_slice(weights, axis, index).real*has_maps
                    title = f"{mode.replace('_bart_1_0', '')} {crop:g}"
                    if mode == "hard":
                        draw_contour(panel, (field > 0).astype(float), .5, "cyan", "solid")
                    else:
                        panel.imshow(np.ma.masked_where(field <= 0, field), cmap="magma", origin="lower",
                                     vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
                        for level, style in ((.1, "dotted"), (.5, "solid"), (.9, "dashdot")):
                            draw_contour(panel, field, level, "cyan", style)
                    draw_contour(panel, reference.astype(float), .5, "yellow", "dashed")
                    summary.append({"axis": names[axis], "index": index, "mode": mode, "crop": crop,
                                    "maximum_weight": float(field.max()),
                                    "weight_above_one_pixels": int(np.count_nonzero(field > 1))})
                panel.set_title(f"{title}\n{names[axis]}={index}")
        fig.suptitle(f"{basis} ACS RSS background (no CSM combination) | common window 0..{window:.5g}, gamma={gamma:g}\n"
                     "Yellow dashed: hard 0.6 reference | Cyan: hard support or soft w=0.1 dotted / 0.5 solid / 0.9 dash-dot\n"
                     "Logical orientation, not RAS labels | Soft display clipped to [0,1]; see JSON for overshoot", fontsize=12)
        fig.subplots_adjust(top=.87, bottom=.08, left=.035, right=.95, wspace=.18, hspace=.55)
        color_axis = fig.add_axes((.965, .24, .008, .5))
        fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="magma"), cax=color_axis,
                     label=f"Soft crop weight (overlay alpha={alpha:g})")
        fig.savefig(png, dpi=150)
    finally:
        plt.close(fig)
    return summary
