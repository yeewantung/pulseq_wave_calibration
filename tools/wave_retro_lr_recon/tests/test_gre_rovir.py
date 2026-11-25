"""Tests for the experimental GRE ROVir workflow (synthetic data, real BART if present)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import create_cfl, open_cfl  # noqa: E402
from wave_retro_lr.gre_rovir import (  # noqa: E402
    PHYSICAL_DIRECTORY,
    _check_projection,
    _nifti_suffix,
    bart_to,
    comparison_metrics,
    local_intensity_ratio,
    prepare_roi,
)
from wave_retro_lr.rovir import rovir_matrix_view  # noqa: E402

BART = shutil.which("bart")


def _synthetic_physical_tree(root: Path, shape=(24, 20, 8), coils=8, seed=3) -> np.ndarray:
    """Write a minimal ``physical_inputs`` tree with a separable nuisance source.

    Args:
        root: Output root.
        shape: Native grid.
        coils: Physical coils.
        seed: Random seed.

    Returns:
        Coil images in native coil-last order.
    """
    rng = np.random.default_rng(seed)
    ro, lin, par = shape
    grid = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    # Smooth, distinct coil profiles; "brain" in the posterior/superior part,
    # "face" in the anterior/inferior corner.
    brain = ((grid[1] > 8) & (grid[2] > 2)).astype(np.float32)
    face = ((grid[1] < 6) & (grid[2] < 3)).astype(np.float32)
    images = np.zeros((*shape, coils), dtype=np.complex64)
    for coil in range(coils):
        centre = rng.uniform(0, 1, 3) * np.asarray(shape)
        distance = sum(((grid[k] - centre[k]) / shape[k]) ** 2 for k in range(3))
        sensitivity = np.exp(-12 * distance) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        images[..., coil] = (brain + 3 * face) * sensitivity
    images += 0.01 * (rng.standard_normal(images.shape) + 1j * rng.standard_normal(images.shape))
    bart_dir = root / PHYSICAL_DIRECTORY / "bart"
    out = create_cfl(bart_dir / "acs_coil_images", images.shape)
    out[...] = images
    out.flush()
    rss = create_cfl(bart_dir / "acs_rss", shape)
    rss[...] = np.sqrt(np.sum(np.abs(images) ** 2, axis=-1))
    rss.flush()
    manifest = {
        "matrix_ro_lin_par": list(shape),
        "voxel_mm_ro_lin_par": [1.0, 1.0, 2.5],
        "physical_coils": coils,
        "source": {"standard_root": str(root / "missing_standard")},
    }
    (root / PHYSICAL_DIRECTORY / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return images


class MetricTests(unittest.TestCase):
    """Validate pure helper functions."""

    def test_comparison_metrics_match_scale_and_measure_suppression(self) -> None:
        """Remove a global scale and report the null-region ratio.

        Returns:
            None.
        """
        standard = np.ones((4, 4, 4))
        null = np.zeros_like(standard, dtype=bool)
        null[0] = True
        head = ~null
        rovir = 2.0 * standard
        rovir[0] = 0.2  # 0.1 after head matching
        metrics = comparison_metrics(standard, rovir, null, head)
        self.assertAlmostEqual(metrics["head_median_ratio_rovir_over_standard"], 2.0)
        self.assertAlmostEqual(metrics["null_region_rovir_over_standard"], 0.1)
        self.assertAlmostEqual(metrics["head_relative_rms_difference"], 0.0)

    def test_local_intensity_ratio_recovers_smooth_shading(self) -> None:
        """Recover a smooth multiplicative shading inside the support.

        Returns:
            None.
        """
        rng = np.random.default_rng(0)
        standard = 1.0 + 0.2 * rng.random((30, 30, 10))
        ramp = np.linspace(0.3, 0.9, 30)[:, None, None] * np.ones((30, 30, 10))
        support = np.ones_like(standard, dtype=bool)
        ratio = local_intensity_ratio(ramp * standard, standard, support, (1.0, 1.0, 2.5), sigma_mm=3.0)
        np.testing.assert_allclose(ratio[5:25], ramp[5:25], rtol=0.03)

    def test_nifti_suffix_is_alphanumeric(self) -> None:
        """Keep NIfTI suffixes free of separators.

        Returns:
            None.
        """
        suffix = _nifti_suffix("native_r3x3", "fista_r0", "null_0123456789ab", 24)
        self.assertEqual(suffix, "BARTWaveGRERovirNativeR3x3FISTAR0VCC24null0123456789ab")
        self.assertTrue(suffix.isalnum())


@unittest.skipIf(BART is None, "bart is not on PATH")
class BartRoiTests(unittest.TestCase):
    """Run the ROI stage and the ccapply convention check with real BART."""

    def setUp(self) -> None:
        """Create a temporary output root.

        Returns:
            None.
        """
        self.root = Path(tempfile.mkdtemp(prefix="gre_rovir_test_"))

    def tearDown(self) -> None:
        """Remove the temporary output root.

        Returns:
            None.
        """
        shutil.rmtree(self.root, ignore_errors=True)

    def test_roi_stage_suppresses_null_region_and_is_resumable(self) -> None:
        """Estimate a ROVir transform that keeps brain and drops face energy.

        Returns:
            None.
        """
        _synthetic_physical_tree(self.root)
        ellipsoid = "ro=12,lin=0,par=0,lin_radius_mm=9,ro_radius_mm=30,par_radius_mm=9"
        manifest = prepare_roi(self.root, ellipsoids=[ellipsoid], bart=BART)
        curve = {entry["virtual_coils"]: entry for entry in manifest["energy_curve"]["channel_counts"]}
        # Localized synthetic coils separate the regions: half of the coils keep
        # most outside-region energy and almost none of the null region.
        self.assertGreater(curve[4]["signal_retention_fraction"], 0.7)
        self.assertLess(curve[4]["interference_remaining_fraction"], 0.02)
        self.assertAlmostEqual(curve[8]["signal_retention_fraction"], 1.0, places=4)
        roi_dir = self.root / "roi" / manifest["region"]["region_id"]
        for name in ("rovir_energy_curve.png", "acs_rss_with_null_region_sagittal.png"):
            self.assertTrue((roi_dir / "figures" / name).is_file())
        again = prepare_roi(self.root, ellipsoids=[ellipsoid], bart=BART)
        self.assertEqual(again["created_at_utc"], manifest["created_at_utc"])

    def test_ccapply_matches_conjugate_transform_convention(self) -> None:
        """Confirm ``bart ccapply -p N`` equals ``X @ conj(T[:, :N])``.

        Returns:
            None.
        """
        images = _synthetic_physical_tree(self.root)
        manifest = prepare_roi(self.root, boxes=["ro=all,lin=0:4,par=0:2"], bart=BART)
        transform_base = self.root / "roi" / manifest["region"]["region_id"] / "transform" / "rovir_transform"
        transform = rovir_matrix_view(np.asarray(open_cfl(transform_base)))
        data = create_cfl(self.root / "kspace", images.shape)
        data[...] = np.fft.fftn(images, axes=(0, 1, 2))
        data.flush()
        projected = self.root / "projected"
        bart_to(
            projected,
            lambda out: [BART, "ccapply", "-p", "3", str(self.root / "kspace"), str(transform_base), str(out)],
            self.root / "test.log",
        )
        error = _check_projection(self.root / "kspace", projected, transform, 3, readout=5)
        self.assertLess(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
