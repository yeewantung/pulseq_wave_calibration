"""Open annotations, local continuation and immutable provenance without BART."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from test_custom_csm import custom_fixture
from wave_retro_lr.annotated_csm import (
    open_trace_profiles, trace_halfspace, annotation_domain, annotated_vectors,
    prepare_annotated_csm,
)
from wave_retro_lr.csm_cache import read_json, write_json
from wave_retro_lr.custom_csm import load_custom_csm
from wave_retro_lr.bart_io import open_cfl, sha256_file


def draw_annotations(path: Path, shape: tuple) -> list[dict]:
    """Write three open synthetic red traces with exactly known image/grid mapping."""
    image = Image.new("RGB", (400, 120), "black")
    draw = ImageDraw.Draw(image)
    panels = []
    for axis in range(3):
        plane = [n for i, n in enumerate(shape) if i != axis]
        left = 10+130*axis
        policy = "y_lower" if axis == 1 else "y_upper"
        height = 90 if policy == "y_lower" else 25
        draw.line([(left+8, height), (left+92, height)], fill="red", width=2)
        panels.append(dict(axis=axis, index=shape[axis]//2, shape=plane,
                           pixel_bounds=[left, 10, 100, 100], policy=policy))
    image.save(path)
    return panels


class AnnotationTests(unittest.TestCase):
    """Check that partial red lines never become full anatomical segmentations."""

    def test_pixel_mapping_open_endpoints_and_margin(self):
        """Digitized endpoints stay open; physical margin follows anisotropic spacing."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"annotation.png"
            panels = draw_annotations(path, (20, 20, 20))
            profiles = open_trace_profiles(path, panels)
            self.assertEqual(len(profiles), 3)
            self.assertFalse(profiles[0]["endpoint_connection"])
            self.assertAlmostEqual(profiles[0]["boundary"][0], 16.4, places=5)
            zero = trace_halfspace(profiles[0], (2., 1.), 0)
            one = trace_halfspace(profiles[0], (2., 1.), 1)
            two = trace_halfspace(profiles[0], (2., 1.), 2)
            np.testing.assert_array_equal(zero, one)
            self.assertGreater(two.sum(), one.sum())
            self.assertFalse(two[:, 0].any())
            self.assertFalse(two[:, -1].any())
            domain, roi = annotation_domain((20, 20, 20), (1., 1., 1.), profiles,
                                            [[2, 18]]*3, 1)
            self.assertTrue(domain.any())
            self.assertFalse((domain & ~roi).any())

    def test_invalid_trace_and_roi(self):
        """Reject empty traces, mapping mismatch, missing constraints and invalid margins."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"annotation.png"
            panels = draw_annotations(path, (20, 20, 20))
            profiles = open_trace_profiles(path, panels)
            with self.assertRaises(ValueError):
                trace_halfspace(profiles[0], (1, 1), -1)
            with self.assertRaises(ValueError):
                annotation_domain((20, 20, 20), (1, 1, 1), profiles[:2], [[0, 20]]*3, 1)
            with self.assertRaises(ValueError):
                annotation_domain((20, 20, 20), (1, 1, 1), profiles, [[0, 21]]*3, 1)
            Image.new("RGB", (400, 120), "black").save(path)
            with self.assertRaisesRegex(ValueError, "no usable"):
                open_trace_profiles(path, panels)

    @patch("wave_retro_lr.annotated_csm.memory_guard")
    def test_sparse_extension_and_coherence(self, guard):
        """Extend coherent complex vectors beyond a short cap without touching native maps."""
        maps = np.zeros((48, 40, 40, 2), np.complex64)
        maps[:8, ..., 0] = 1/np.sqrt(2)
        maps[:8, ..., 1] = 1j/np.sqrt(2)
        original = maps.copy()
        domain = np.zeros(maps.shape[:3], bool)
        domain[8:32, 18:22, 18:22] = True
        native, added, vectors, stats = annotated_vectors(maps, domain, (1, 1, 1), [2, 8, 12], 28)
        self.assertGreater(stats["added_beyond_16mm_voxels"], 0)
        self.assertFalse((added & ~domain).any())
        np.testing.assert_array_equal(maps, original)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(vectors[:, 1], vectors[:, 0]*1j, atol=1e-6)
        short = annotated_vectors(maps, domain, (1, 1, 1), [2, 8, 12], 4)[1]
        self.assertLess(short.sum(), added.sum())
        maps[:, ::2] *= -1
        result = annotated_vectors(maps, domain, (1, 1, 1), [8, 12], 28)
        self.assertEqual(result[1].sum(), 0)
        maps[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            annotated_vectors(maps, domain, (1, 1, 1), [2], 28)

    @patch("wave_retro_lr.annotated_csm.memory_guard")
    def test_complete_provenance_resume_and_unchanged_source(self, guard):
        """Generate tiny real CFL/QC files, reuse exactly and reject annotation tampering."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal, old = custom_fixture(root)
            config = read_json(old)
            annotation = root/"annotated.png"
            panels = draw_annotations(annotation, (24, 24, 20))
            request = root/"open.local.json"
            write_json(request, dict(diagnostic_root=config["diagnostic_root"],
                baseline_csm=config["baseline_csm"], annotation=str(annotation), panels=panels,
                roi_voxels=[[4, 19], [4, 19], [4, 17]], margin_mm=1,
                sigmas_mm=[1, 2, 3], max_distance_mm=5))
            original_hash = sha256_file(Path(config["baseline_csm"]+".cfl"))
            completion = prepare_annotated_csm(request, root/"candidate")
            record = load_custom_csm(completion, normal)
            self.assertGreater(record["statistics"]["added_voxels"], 0)
            self.assertEqual(record["statistics"]["unchanged_vectors_max_error"], 0)
            self.assertEqual(record["statistics"]["added_outside_roi"], 0)
            masks = np.load(root/"candidate/masks/support.npz")
            original = open_cfl(config["baseline_csm"])
            result = open_cfl(record["maps"]["base"])
            np.testing.assert_array_equal(result[~masks["added"]], original[~masks["added"]])
            self.assertEqual(sha256_file(Path(config["baseline_csm"]+".cfl")), original_hash)
            self.assertEqual(prepare_annotated_csm(request, root/"candidate"), completion)
            annotation.write_bytes(annotation.read_bytes()+b"changed")
            with self.assertRaises(ValueError):
                load_custom_csm(completion)
            with self.assertRaises(ValueError):
                prepare_annotated_csm(request, root/"candidate")


if __name__ == "__main__":
    unittest.main()
