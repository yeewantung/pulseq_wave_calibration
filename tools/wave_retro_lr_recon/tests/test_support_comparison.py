"""Nested conservative mask comparisons, immutable previews and no map generation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_annotated_csm import draw_annotations
from test_custom_csm import custom_fixture
from test_slice_local_csm import patch_settings
from wave_retro_lr.annotated_csm import prepare_annotated_csm
from wave_retro_lr.csm_cache import read_json, write_json
from wave_retro_lr.custom_csm import load_custom_csm
from wave_retro_lr.support_preview import prepare_support_preview
from scripts.compare_custom_csm_support import conservative_variants, prepare_comparison


class ComparisonTests(unittest.TestCase):
    """Exercise numerical nesting and complete synthetic preview-only lineage."""

    def test_caps_are_nested_and_bounded(self):
        """Increasing caps add only permitted voxels; invalid/nonconservative variants fail."""
        existing = np.zeros((32, 32, 32), bool)
        existing[:, :10] = True
        profiles = [dict(panel=dict(axis=i, shape=[32, 32], index=16, policy="x_upper"),
                         independent=[0., 31.], boundary=[22., 22.]) for i in range(3)]
        settings = dict(patches=[patch_settings()], margin_mm=1.)
        config = dict(caps_mm=[2., 4.], roi_voxels=[[5, 27], [8, 26], [10, 23]],
                      core_half_width_mm=2., outer_half_width_mm=6.)
        variants, _ = conservative_variants(existing, ~existing, (1, 1, 1), profiles, settings, config)
        self.assertFalse((variants[0] & ~variants[1]).any())
        self.assertLess(variants[0].sum(), variants[1].sum())
        self.assertFalse((variants[1] & existing).any())
        with self.assertRaises(ValueError):
            conservative_variants(existing, variants[0], (1, 1, 1), profiles, settings, config)
        with self.assertRaises(ValueError):
            conservative_variants(existing, ~existing, (1, 1, 1), profiles, settings, dict(config, caps_mm=[4., 2.]))

    @patch("scripts.compare_custom_csm_support.memory_guard")
    @patch("wave_retro_lr.support_preview.memory_guard")
    @patch("wave_retro_lr.annotated_csm.memory_guard")
    def test_preview_lineage_no_maps_resume_and_tamper(self, *guards):
        """Write real temporary QC/masks; reject reconstruction use, tampering and overwrite."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, initial = custom_fixture(root)
            old = read_json(initial)
            annotation = root/"traces.png"
            panels = draw_annotations(annotation, (24, 24, 20))
            c2 = root/"v2.local.json"
            write_json(c2, dict(diagnostic_root=old["diagnostic_root"], baseline_csm=old["baseline_csm"],
                annotation=str(annotation), panels=panels, roi_voxels=[[4, 19], [4, 19], [4, 17]],
                margin_mm=1., sigmas_mm=[1, 2, 3], max_distance_mm=5.))
            parent = prepare_annotated_csm(c2, root/"v2")
            c4 = root/"v4.local.json"
            setting = dict(patch_settings(0), roi_voxels=[[4, 20], [0, 9], [3, 17]], edge_taper_mm=.5)
            write_json(c4, dict(parent_manifest=str(parent), traces=str(root/"v2/metadata/open_traces.json"),
                patches=[setting], margin_mm=1., views=[[0, 12]], axial_neighbors=[6, 12, 18]))
            prior = prepare_support_preview(c4, root/"v4")
            c5 = root/"v5.local.json"
            write_json(c5, dict(baseline_preview=str(prior), roi_voxels=setting["roi_voxels"],
                core_half_width_mm=2., outer_half_width_mm=6., caps_mm=[2., 4.]))
            with patch("wave_retro_lr.annotated_csm.annotated_vectors", side_effect=AssertionError("No fitting")):
                completion = prepare_comparison(c5, root/"v5")
                self.assertEqual(prepare_comparison(c5, root/"v5"), completion)
            record = read_json(completion)
            self.assertFalse(record["reconstruction_ready"])
            self.assertFalse(record["coil_vectors_calculated"])
            self.assertIsNone(record["selected_variant"])
            self.assertFalse(list((root/"v5").rglob("*.cfl")))
            with self.assertRaises(ValueError):
                load_custom_csm(completion)
            file = root/"v5/figures/cap_comparison.png"
            file.write_bytes(file.read_bytes()+b"changed")
            with self.assertRaises(ValueError):
                prepare_comparison(c5, root/"v5")


if __name__ == "__main__":
    unittest.main()
