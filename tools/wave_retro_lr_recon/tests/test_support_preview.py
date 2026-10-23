"""Geometry-only previews preserve accepted support and cannot be used as CSMs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_annotated_csm import draw_annotations
from test_custom_csm import custom_fixture
from test_slice_local_csm import patch_settings
from wave_retro_lr.annotated_csm import prepare_annotated_csm
from wave_retro_lr.bart_io import sha256_file
from wave_retro_lr.csm_cache import read_json, write_json
from wave_retro_lr.custom_csm import load_custom_csm
from wave_retro_lr.support_preview import preview_masks, prepare_support_preview


class SupportPreviewTests(unittest.TestCase):
    """Test complete support display, pure mask output and strict immutable reuse."""

    def test_full_support_has_no_false_native_hole(self):
        """A native island surrounded by added support is not an actual support hole."""
        native = np.zeros((8, 8, 8), bool)
        native[3:5, 3:5, 3:5] = True
        added = np.zeros_like(native)
        added[2:6, 2:6, 2:6] = True
        added &= ~native
        domain = np.zeros_like(native)
        domain[1:7, 1:7, 1:7] = True
        result = preview_masks(native, added, domain)
        self.assertTrue(result["existing_full"][2:6, 2:6, 2:6].all())
        self.assertFalse((result["existing_full"] & ~result["proposed_full"]).any())
        self.assertFalse((result["proposed_additions"] & ~domain).any())
        self.assertEqual(int(native.sum()), 8)
        with self.assertRaises(ValueError):
            preview_masks(native, native, domain)

    @patch("wave_retro_lr.support_preview.memory_guard")
    @patch("wave_retro_lr.annotated_csm.memory_guard")
    def test_no_maps_outputs_and_resume_provenance(self, guard1, guard2):
        """Write real small masks/QC, prohibit vector prediction and reject tampering."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, old_config = custom_fixture(root)
            old = read_json(old_config)
            annotation = root/"traces.png"
            panels = draw_annotations(annotation, (24, 24, 20))
            v2_config = root/"v2.local.json"
            write_json(v2_config, dict(diagnostic_root=old["diagnostic_root"], baseline_csm=old["baseline_csm"],
                annotation=str(annotation), panels=panels, roi_voxels=[[4, 19], [4, 19], [4, 17]],
                margin_mm=1., sigmas_mm=[1, 2, 3], max_distance_mm=5.))
            parent = prepare_annotated_csm(v2_config, root/"v2")
            before = sha256_file(parent)
            config = root/"preview.local.json"
            settings = dict(patch_settings(0), roi_voxels=[[4, 20], [0, 9], [3, 17]], edge_taper_mm=.5)
            write_json(config, dict(parent_manifest=str(parent), traces=str(root/"v2/metadata/open_traces.json"),
                patches=[settings], margin_mm=1., views=[[0, 12], [2, 10]], axial_neighbors=[6, 12, 18]))
            with patch("wave_retro_lr.annotated_csm.annotated_vectors", side_effect=AssertionError("No CSM prediction allowed")):
                result = prepare_support_preview(config, root/"preview")
                self.assertEqual(prepare_support_preview(config, root/"preview"), result)
            record = read_json(result)
            self.assertFalse(record["reconstruction_ready"])
            self.assertFalse(record["coil_vectors_calculated"])
            self.assertGreater(record["statistics"]["proposed_additions"], 0)
            self.assertEqual(record["statistics"]["removed_existing_voxels"], 0)
            self.assertFalse(list((root/"preview").rglob("*.cfl")))
            self.assertFalse((root/"preview/csm").exists())
            self.assertEqual(before, sha256_file(parent))
            with self.assertRaises(ValueError):
                load_custom_csm(result)
            figure = root/"preview/figures/full_support_preview.png"
            figure.write_bytes(figure.read_bytes()+b"edited")
            with self.assertRaises(ValueError):
                prepare_support_preview(config, root/"preview")


if __name__ == "__main__":
    unittest.main()
