"""Tests for retrospective low-resolution quantitative tradeoff analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_retrospective_low_resolution import (  # noqa: E402
    gradient_components_per_mm,
    map_fixed_mask_to_reconstruction,
    matched_fidelity_metrics,
    run,
    sha256_file,
)


def _affine(shape: tuple[int, int, int], zooms: tuple[float, float, float]) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.diag(zooms)
    affine[:3, 3] = -np.asarray(zooms) * (np.asarray(shape) - 1.0) / 2.0
    return affine


def _registration(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    return {
        "rigid": {
            "rotation_matrix": np.eye(3).tolist(),
            "parameters": {
                "rotation_degrees_ras_xyz": [0.0, 0.0, 0.0],
                "translation_mm_ras_xyz": list(translation),
            },
        }
    }


class MetricPrimitiveTests(unittest.TestCase):
    def test_gradient_uses_physical_spacing(self) -> None:
        fine = np.indices((16, 8, 8), dtype=np.float32)[0]
        coarse = 2.0 * np.indices((8, 8, 8), dtype=np.float32)[0]
        fine_x, *_ = gradient_components_per_mm(fine, (1.0, 1.0, 1.0), smoothing_sigma_mm=0.0)
        coarse_x, *_ = gradient_components_per_mm(coarse, (2.0, 1.0, 1.0), smoothing_sigma_mm=0.0)
        np.testing.assert_allclose(fine_x, 1.0, atol=1e-6)
        np.testing.assert_allclose(coarse_x, 1.0, atol=1e-6)

    def test_fixed_mask_is_transferred_without_changing_target_data(self) -> None:
        shape = (12, 12, 12)
        affine = _affine(shape, (1.0, 1.0, 1.0))
        fixed = np.zeros(shape, dtype=np.uint8)
        fixed[5:7, 3:9, 3:9] = 1
        fixed_image = nib.Nifti1Image(fixed, affine)
        target_image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
        identity = map_fixed_mask_to_reconstruction(
            fixed_image, target_image, _registration()
        )
        np.testing.assert_array_equal(identity, fixed > 0)
        shifted = map_fixed_mask_to_reconstruction(
            fixed_image, target_image, _registration((1.0, 0.0, 0.0))
        )
        self.assertTrue(np.all(shifted[4:6, 3:9, 3:9]))

    def test_identity_matched_metrics_are_exact(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.random((12, 12, 12), dtype=np.float32)
        brain = np.zeros(data.shape, dtype=bool)
        brain[2:10, 2:10, 2:10] = True
        edge = np.zeros(data.shape, dtype=bool)
        edge[3:9, 3:9, 3:9] = True
        metrics = matched_fidelity_metrics(data, data, brain, edge, (1.0, 1.0, 1.0))
        self.assertLess(metrics["nrmse_brain"], 1e-10)
        self.assertAlmostEqual(metrics["ncc_brain"], 1.0, places=7)
        self.assertAlmostEqual(metrics["ssim_axial_brain_bbox_mean"], 1.0, places=7)


class AnalysisRunTests(unittest.TestCase):
    def test_complete_analysis_is_descriptive_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specifications = [
                ("grappa", (64, 64, 64), (1.0, 1.0, 1.0)),
                ("full_resolution_llr", (64, 64, 64), (1.0, 1.0, 1.0)),
                ("lower_x", (32, 64, 64), (2.0, 1.0, 1.0)),
                ("lower_y", (64, 32, 64), (1.0, 2.0, 1.0)),
                ("lower_xy", (32, 32, 64), (2.0, 2.0, 1.0)),
            ]
            inputs = []
            for index, (key, shape, zooms) in enumerate(specifications):
                affine = _affine(shape, zooms)
                coordinates = np.indices(shape, dtype=np.float32)
                world = [
                    coordinates[axis] * zooms[axis] + affine[axis, 3]
                    for axis in range(3)
                ]
                radius_squared = sum(component * component for component in world)
                anatomy = np.exp(-radius_squared / (2.0 * 13.0**2))
                texture = 1.0 + 0.12 * np.cos(world[0] / 2.5) * np.cos(world[1] / 3.0)
                background = 0.001 * (
                    2.0 + np.sin(world[0] * 0.7 + index) + np.cos(world[2] * 0.5)
                )
                data = (anatomy * texture + background).astype(np.float32)
                path = root / f"{key}.nii.gz"
                nib.save(nib.Nifti1Image(data, affine), path)
                inputs.append(
                    {
                        "key": key,
                        "title": key,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "case_manifest": None,
                        "geometry": {
                            "shape_xyz": list(shape),
                            "voxel_size_mm_xyz": list(zooms),
                        },
                    }
                )
            review_path = root / "review_manifest.json"
            review_path.write_text(
                json.dumps({"status": "complete", "inputs": inputs}), encoding="utf-8"
            )

            mask_shape = (64, 64, 64)
            mask_affine = _affine(mask_shape, (1.0, 1.0, 1.0))
            mask_coordinates = np.indices(mask_shape, dtype=np.float32)
            mask_world = [
                mask_coordinates[axis] + mask_affine[axis, 3] for axis in range(3)
            ]
            brain = sum(component * component for component in mask_world) <= 18.0**2
            mask_path = root / "approved_mask.nii.gz"
            nib.save(nib.Nifti1Image(brain.astype(np.uint8), mask_affine), mask_path)
            registration_path = root / "registration.json"
            registration_path.write_text(json.dumps(_registration()), encoding="utf-8")

            output = root / "analysis"
            manifest = run(
                argparse.Namespace(
                    review_manifest=review_path,
                    approved_bet_mask=mask_path,
                    shared_registration=registration_path,
                    output_dir=output,
                )
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["scientific_scope"]["automatic_selection_performed"])
            self.assertFalse(manifest["scientific_scope"]["true_snr_or_cnr_claimed"])
            self.assertTrue(manifest["scientific_scope"]["approved_bet_used_for_metrics_only"])
            with (output / "native_resolution_metrics.csv").open(newline="") as stream:
                native_rows = list(csv.DictReader(stream))
            with (output / "matched_fidelity_metrics.csv").open(newline="") as stream:
                matched_rows = list(csv.DictReader(stream))
            self.assertEqual(len(native_rows), 5)
            self.assertEqual(len(matched_rows), 10)
            self.assertTrue((output / "resolution_tradeoff_summary.png").is_file())
            self.assertTrue((output / "analysis_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
