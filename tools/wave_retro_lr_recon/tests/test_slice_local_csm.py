"""Regression tests for partial annotations wrongly excluding other slice anatomy."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_annotated_csm import draw_annotations
from test_custom_csm import custom_fixture
from wave_retro_lr.annotated_csm import prepare_annotated_csm, trace_halfspace
from wave_retro_lr.bart_io import open_cfl, sha256_file
from wave_retro_lr.csm_cache import read_json, write_json
from wave_retro_lr.custom_csm import load_custom_csm
from wave_retro_lr.slice_local_csm import local_patch_domains, prepare_slice_local_csm


def patch_settings(axis=2) -> dict:
    """Return a small bounded slab with a physical parent-distance cap for tests."""
    return dict(name="local_patch", trace_axis=axis, roi_voxels=[[5, 27], [8, 26], [10, 23]],
                core_half_width_mm=2., outer_half_width_mm=6., edge_taper_mm=2., extension_cap_mm=6.)


class SliceLocalTests(unittest.TestCase):
    """Keep accepted vectors exact while preventing unrelated slices from acting as vetoes."""

    def test_locality_no_cross_slice_veto_and_taper(self):
        """Restore a locally permitted voxel even when two other slice traces exclude it."""
        support = np.zeros((32, 32, 32), bool)
        support[:, :10] = True
        profiles = []
        for axis, policy, boundary in ((0, "y_upper", 5), (1, "y_lower", 25), (2, "x_upper", 22)):
            profiles.append(dict(panel=dict(axis=axis, shape=[32, 32], index=16, policy=policy),
                                 independent=[0., 31.], boundary=[boundary, boundary]))
        settings = patch_settings()
        domain, individual = local_patch_domains(support, (1, 1, 1), profiles, [settings], 1.)
        self.assertTrue(domain[16, 14, 16])
        self.assertFalse(trace_halfspace(profiles[0], (1, 1), 1)[14, 16])
        self.assertFalse(trace_halfspace(profiles[1], (1, 1), 1)[16, 16])
        self.assertFalse(domain[16, 14, 11])
        self.assertFalse(domain[16, 14, 10])
        self.assertFalse(domain[16, 25, 16])
        self.assertFalse(domain[:5].any())
        self.assertFalse(domain[27:].any())
        np.testing.assert_array_equal(domain, individual[0])
        anisotropic = local_patch_domains(support, (1, 2, 1), profiles, [settings], 1.)[0]
        self.assertFalse(anisotropic[16, 14, 16])
        self.assertTrue(anisotropic[16, 11, 16])
        for invalid in (dict(settings, outer_half_width_mm=1), dict(settings, edge_taper_mm=0),
                        dict(settings, extension_cap_mm=float("nan")), dict(settings, trace_axis=4)):
            with self.assertRaises(ValueError):
                local_patch_domains(support, (1, 1, 1), profiles, [invalid], 1)
        with self.assertRaises(ValueError):
            local_patch_domains(support, (1, 1, 1), profiles, [settings, settings], 1)

    @patch("wave_retro_lr.slice_local_csm.memory_guard")
    @patch("wave_retro_lr.annotated_csm.memory_guard")
    def test_full_parent_preservation_provenance_and_resume(self, guard1, guard2):
        """Generate actual tiny CFL/QC outputs; bind ancestors and reject changed input."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal, old_config = custom_fixture(root)
            old = read_json(old_config)
            annotation = root/"traces.png"
            panels = draw_annotations(annotation, (24, 24, 20))
            v2_config = root/"v2.local.json"
            write_json(v2_config, dict(diagnostic_root=old["diagnostic_root"], baseline_csm=old["baseline_csm"],
                annotation=str(annotation), panels=panels, roi_voxels=[[4, 19], [4, 19], [4, 17]],
                margin_mm=1., sigmas_mm=[1, 2, 3], max_distance_mm=5.))
            parent_path = prepare_annotated_csm(v2_config, root/"v2")
            parent_hash = sha256_file(parent_path)
            parent = load_custom_csm(parent_path)
            config = root/"v3.local.json"
            settings = dict(patch_settings(0), roi_voxels=[[4, 20], [0, 9], [3, 17]], edge_taper_mm=.5)
            write_json(config, dict(parent_manifest=str(parent_path), patches=[settings], margin_mm=1.))
            result_path = prepare_slice_local_csm(config, root/"v3")
            result = load_custom_csm(result_path, normal)
            masks = np.load(root/"v3/masks/support.npz")
            added = masks["new_added"]
            self.assertTrue(added.any())
            self.assertFalse((added & ~masks["patch_domain"]).any())
            self.assertFalse((added & masks["parent_added"]).any())
            new_maps = open_cfl(result["maps"]["base"])
            old_maps = open_cfl(parent["maps"]["base"])
            np.testing.assert_array_equal(new_maps[~added], old_maps[~added])
            self.assertEqual(result["statistics"]["parent_vectors_max_error"], 0.)
            self.assertEqual(sha256_file(parent_path), parent_hash)
            self.assertEqual(prepare_slice_local_csm(config, root/"v3"), result_path)
            self.assertTrue((root/"v3/figures/adjacent_slices.png").is_file())
            original_hash = sha256_file(result_path)
            config_bytes = config.read_bytes()
            config.write_bytes(config_bytes+b"\n")
            with self.assertRaises(ValueError):
                prepare_slice_local_csm(config, root/"v3")
            with self.assertRaises(ValueError):
                load_custom_csm(result_path)
            self.assertEqual(sha256_file(result_path), original_hash)
            config.write_bytes(config_bytes)
            annotation.write_bytes(annotation.read_bytes()+b"changed")
            with self.assertRaises(ValueError):
                load_custom_csm(result_path)


if __name__ == "__main__":
    unittest.main()
