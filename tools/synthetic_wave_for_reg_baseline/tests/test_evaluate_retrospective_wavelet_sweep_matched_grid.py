"""Tests for the independent matched-grid retrospective sweep evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_retrospective_wavelet_sweep_matched_grid import (  # noqa: E402
    _original_context,
    metric_leaders,
)
from bart_cfl import sha256_file  # noqa: E402


class OriginalMatchedGridTests(unittest.TestCase):
    def test_original_context_reuses_hash_bound_references_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            affine = np.eye(4)
            reference_paths = {}
            for key, scale in (("full_resolution_fista_lambda0", 1.0), ("direct_fft_rss", 2.0)):
                path = root / f"{key}.nii.gz"
                nib.save(
                    nib.Nifti1Image(
                        np.full((8, 8, 8), scale, dtype=np.float32), affine
                    ),
                    path,
                )
                reference_paths[key] = path
            brain = np.zeros((8, 8, 8), dtype=np.uint8)
            brain[1:7, 1:7, 1:7] = 1
            edge = np.zeros((8, 8, 8), dtype=np.uint8)
            edge[2:6, 2:6, 2:6] = 1
            brain_path = root / "approved_bet_on_reconstruction_grid.nii.gz"
            edge_path = root / "fixed_reference_edge_mask.nii.gz"
            nib.save(nib.Nifti1Image(brain, affine), brain_path)
            nib.save(nib.Nifti1Image(edge, affine), edge_path)

            review_path = root / "review_manifest.json"
            review_path.write_text(
                json.dumps(
                    {
                        "inputs": [
                            {
                                "key": key,
                                "path": str(path),
                                "sha256": sha256_file(path),
                            }
                            for key, path in reference_paths.items()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            analysis_path = root / "analysis_manifest.json"
            analysis_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "inputs": {
                            "review_manifest": {
                                "path": str(review_path),
                                "sha256": sha256_file(review_path),
                            }
                        },
                        "fixed_masks": {
                            "brain_voxel_count": int(brain.sum()),
                            "edge_voxel_count": int(edge.sum()),
                            "outputs": [
                                {"path": str(brain_path), "sha256": sha256_file(brain_path)},
                                {"path": str(edge_path), "sha256": sha256_file(edge_path)},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            context = _original_context(analysis_path)

            self.assertEqual(int(context["brain"].sum()), int(brain.sum()))
            self.assertEqual(int(context["edge"].sum()), int(edge.sum()))
            self.assertEqual(context["full_image"].shape, (8, 8, 8))

    def test_leaders_are_reported_per_metric_without_one_selection(self) -> None:
        rows = []
        for value, nrmse, ssim, gradient, edge in (
            (1e-3, 0.09, 0.92, 0.96, 0.99),
            (1e-2, 0.08, 0.91, 0.97, 1.03),
        ):
            rows.append(
                {
                    "reference": "direct_fft_rss",
                    "case_name": "case",
                    "lambda": value,
                    "nrmse_brain": nrmse,
                    "ssim_axial_brain_bbox_mean": ssim,
                    "gradient_ncc_fixed_edge": gradient,
                    "edge_gradient_preservation_ratio": edge,
                }
            )
        leaders = metric_leaders(rows)["case"]
        self.assertEqual(leaders["nrmse_brain"]["lambda"], 1e-2)
        self.assertEqual(leaders["ssim_axial_brain_bbox_mean"]["lambda"], 1e-3)


if __name__ == "__main__":
    unittest.main()
