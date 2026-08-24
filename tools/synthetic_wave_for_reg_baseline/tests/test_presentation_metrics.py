"""Tests for exact-grid presentation metrics against direct FFT R1."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bart_cfl import sha256_file  # noqa: E402
from presentation_metrics import evaluate_against_direct_fft  # noqa: E402


class PresentationMetricsTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        shape = (48, 48, 48)
        coordinates = np.indices(shape, dtype=np.float32)
        rng = np.random.default_rng(42)
        reference = (
            1.0
            + 0.5 * coordinates[0]
            + 0.25 * coordinates[1]
            + 0.125 * coordinates[2]
            + rng.uniform(0.0, 2.0, size=shape)
        ).astype(np.float32)
        brain = np.zeros(shape, dtype=np.uint8)
        brain[10:38, 10:38, 10:38] = 1
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

        reference_path = root / "direct_fft_rss.nii.gz"
        mask_path = root / "approved_brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(reference, affine), reference_path)
        nib.save(nib.Nifti1Image(brain, affine), mask_path)
        manifest = {
            "status": "approved_for_metrics",
            "ranking_reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "brain_mask": {
                "path": str(mask_path),
                "sha256": sha256_file(mask_path),
            },
        }
        manifest_path = root / "metrics_reference_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        percentile = float(np.percentile(reference[reference > 0], 99.0))
        candidate_path = root / "candidate.nii.gz"
        nib.save(nib.Nifti1Image(reference / percentile, affine), candidate_path)
        candidate_path.with_name("candidate.json").write_text(
            json.dumps(
                {
                    "MagnitudeNormalization": {
                        "Method": "positive-finite-percentile",
                        "Percentile": 99.0,
                        "InputPercentileValue": percentile,
                        "OutputPercentileValue": 1.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        return candidate_path, manifest_path

    def test_exact_candidate_has_presentation_ready_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, manifest = self._fixture(Path(temporary))
            result = evaluate_against_direct_fft(candidate, manifest)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["geometry_policy"]["registration_performed"])
        self.assertFalse(result["geometry_policy"]["interpolation_performed"])
        self.assertLess(result["metrics"]["nrmse_brain"], 1e-6)
        self.assertAlmostEqual(result["metrics"]["ssim_3d_brain_bbox"], 1.0, places=6)

    def test_affine_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, manifest = self._fixture(root)
            image = nib.load(str(candidate))
            changed_affine = image.affine.copy()
            changed_affine[0, 3] = 1.0
            nib.save(
                nib.Nifti1Image(np.asarray(image.dataobj), changed_affine),
                candidate,
            )
            with self.assertRaisesRegex(ValueError, "exact direct-FFT reference grid"):
                evaluate_against_direct_fft(candidate, manifest)


if __name__ == "__main__":
    unittest.main()
