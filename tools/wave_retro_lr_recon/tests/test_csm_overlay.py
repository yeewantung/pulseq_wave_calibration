"""Synthetic ACS overlay numerics, geometry and immutable provenance tests."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_csm_cache import fixture, write_array
from wave_retro_lr.bart_io import cfl_record, open_cfl, sha256_file
from wave_retro_lr.core import centered_fftn
from wave_retro_lr.csm_cache import derive_variant, prepared_source, write_json
from wave_retro_lr.csm_overlay import acs_rss_magnitude, display_slice, plot_acs_overlay


class OverlayTests(unittest.TestCase):
    """Use tiny same-basis fixtures, never invoking ecalib or reconstruction."""

    def setUp(self):
        """Create a synthetic completed ROVir cache with two weight variants."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        manifest = fixture(self.base, "rovir")
        self.root = self.base/"diagnostics"
        for name in ("calibration", "csm_variants", "figures", "metadata"):
            (self.root/name).mkdir(parents=True)
        maps = np.ones((8, 6, 4, 2), np.complex64)/np.sqrt(2)
        maps[0] = 0
        eigen = np.linspace(.1, .99, 192).reshape(8, 6, 4).astype(np.complex64)
        write_array(self.root/"calibration/coil_sens_uncropped", maps)
        write_array(self.root/"calibration/eigenvalues", eigen)
        self.cached = {"identity": {"source": prepared_source(manifest), "maps": 1, "calibration_crop": 0},
                       "maps": cfl_record(self.root/"calibration/coil_sens_uncropped"),
                       "eigenvalues": cfl_record(self.root/"calibration/eigenvalues")}
        write_json(self.root/"metadata/calibration.json", self.cached)
        for mode in ("hard", "soft_bart_1_0"):
            derive_variant(self.root, self.cached, .6, mode, False)

    def test_rss_without_csm_and_slice_alignment(self):
        """Recover a complex coil phantom via IFFT/RSS with identical logical slices."""
        rng = np.random.default_rng(42)
        coils = (rng.normal(size=(8, 6, 4, 2)) + 1j*rng.normal(size=(8, 6, 4, 2))).astype(np.complex64)
        write_array(self.base/"phantom", centered_fftn(coils, axes=(0, 1, 2)))
        rss = acs_rss_magnitude(self.base/"phantom")
        np.testing.assert_allclose(rss, np.sqrt(np.sum(np.abs(coils)**2, axis=3)), rtol=1e-6)
        self.assertTrue(np.all(rss[0] > 0))
        for axis in range(3):
            np.testing.assert_array_equal(display_slice(rss, axis, 1), np.take(rss, 1, axis=axis))
        write_array(self.base/"zero", np.zeros_like(coils))
        with self.assertRaisesRegex(ValueError, "identically zero"):
            acs_rss_magnitude(self.base/"zero")

    def test_additive_figure_resume_and_exact_hashes(self):
        """Create a real PNG, resume exactly and add another display without BART."""
        calibration = self.root/"metadata/calibration.json"
        original_hash = sha256_file(calibration)
        with patch("wave_retro_lr.csm_cache.run_ecalib") as bart:
            result = plot_acs_overlay(self.root, percentile=100)
            self.assertEqual(result["identity"]["background_basis"], "rovir")
            self.assertFalse(result["identity"]["csm_used_for_background"])
            self.assertEqual(result["identity"]["slices_ro_lin_par"], [4, 3, 2])
            figure = Path(result["figure"]["path"])
            self.assertEqual(sha256_file(figure), result["figure"]["sha256"])
            stamp = figure.stat().st_mtime_ns
            self.assertEqual(plot_acs_overlay(self.root, percentile=100), result)
            self.assertEqual(figure.stat().st_mtime_ns, stamp)
            changed = plot_acs_overlay(self.root, percentile=100, gamma=1.)
            self.assertNotEqual(changed["figure"]["path"], str(figure))
            bart.assert_not_called()
        self.assertEqual(sha256_file(calibration), original_hash)
        figure.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            plot_acs_overlay(self.root, percentile=100)

    def test_invalid_display_and_corrupt_weights(self):
        """Reject invalid indices/display parameters and altered array payloads."""
        for options in ({"slices": [-1, 0, 0]}, {"slices": [8, 0, 0]},
                        {"gamma": 0}, {"alpha": 2}, {"percentile": np.nan}):
            with self.assertRaises(ValueError):
                plot_acs_overlay(self.root, **options)
        weights = open_cfl(self.root/"csm_variants/hard_0p6/weights", mode="r+")
        weights[0, 0, 0] = 9
        weights.flush()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            plot_acs_overlay(self.root)
        self.assertFalse(list((self.root/"figures").iterdir()))

    def test_changed_source_rejected(self):
        """Even plausible manifest-only edits must not relabel an existing calibration."""
        path = Path(self.cached["identity"]["source"]["manifest"]["path"])
        record = json.loads(path.read_text())
        record["source"]["changed"] = True
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            plot_acs_overlay(self.root)


if __name__ == "__main__":
    unittest.main()
