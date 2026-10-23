"""Single-map ESPIRiT crop derivation without recalibration or RSS normalization."""

from __future__ import annotations

import numpy as np

METHOD_VERSION = "single_map_crop_v1"
MODES = ("hard", "soft_bart_1_0")


def crop_weights(eigenvalues: np.ndarray, crop: float, mode: str = "hard") -> np.ndarray:
    """Return float32 weights from finite complex eigenvalues and a crop threshold.

    Hard mode reproduces BART's inclusive magnitude threshold. The explicitly
    versioned soft mode reproduces the reviewed BART 1.0 md_scurve implementation,
    including its discontinuity at normalized t=1 (weights can exceed one).
    This is single-map soft cropping, not multi-map Soft-SENSE. Invalid modes,
    thresholds, or non-finite eigenvalues raise ValueError; inputs are unchanged.
    """
    if mode not in MODES or not np.isfinite(crop) or not 0 <= crop <= 1:
        raise ValueError("Expected hard/soft_bart_1_0 and a finite crop in [0, 1]")
    threshold = np.float32(crop)
    if mode != "hard" and threshold >= 1:
        raise ValueError("Soft cropping requires a float32 threshold strictly below one")
    values = np.asarray(eigenvalues, dtype=np.complex64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite eigenvalues")
    magnitude = np.abs(values)
    if mode == "hard":
        return (magnitude >= threshold).astype(np.float32)
    # Preserve BART's square root, float32 arithmetic and literal S-curve.
    t = (np.sqrt(magnitude) - threshold) * np.float32(1. / (1. - float(threshold)))
    return ((t >= 1).astype(np.float32)
            + (t >= -1) * (t / (1 + t*t) + np.float32(.5)))


def validate_map_shapes(maps_shape: tuple, eigen_shape: tuple) -> tuple[int, int, int, int]:
    """Validate native spatial/coil dimensions and MAPS=1; return RO, LIN, PAR, coils.

    Eigenvalues must have singleton coil and all trailing axes. Multiple ESPIRiT
    maps or echo dimensions are rejected rather than silently squeezed together.
    """
    if (len(maps_shape) < 4 or len(eigen_shape) < 3
            or tuple(maps_shape[:3]) != tuple(eigen_shape[:3])
            or any(n != 1 for n in maps_shape[4:])
            or any(n != 1 for n in eigen_shape[3:])):
        raise ValueError("Expected matched native grids, one eigenmap and MAPS=1")
    return tuple(maps_shape[:4])
