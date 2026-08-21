"""Focused tests for direct-FFT regularization evaluation helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from bart_cfl import sha256_file  # noqa: E402
from evaluate_direct_fft_regularization import (  # noqa: E402
    _load_candidate,
    _metric_leaders,
    _plot_llr_heatmaps,
)


class CandidateLoadingTests(unittest.TestCase):
    def test_restores_export_percentile_normalization(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            nifti_path = root / "magnitude.nii.gz"
            sidecar_path = root / "magnitude.json"
            normalized = np.linspace(0.0, 2.0, 27, dtype=np.float32).reshape(3, 3, 3)
            nib.save(nib.Nifti1Image(normalized, np.eye(4)), nifti_path)
            sidecar_path.write_text(
                json.dumps(
                    {
                        "MagnitudeNormalization": {
                            "Method": "positive-finite-percentile",
                            "Percentile": 99.0,
                            "InputPercentileValue": 0.25,
                            "OutputPercentileValue": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            restored, record = _load_candidate(
                {
                    "case_id": "wavelet:1e-4",
                    "magnitude_nifti": str(nifti_path),
                    "magnitude_nifti_sha256": sha256_file(nifti_path),
                    "magnitude_sidecar": str(sidecar_path),
                    "magnitude_sidecar_sha256": sha256_file(sidecar_path),
                }
            )

            np.testing.assert_allclose(restored, normalized * 0.25)
            self.assertEqual(record["restoration_multiplier"], 0.25)


class MetricLeaderTests(unittest.TestCase):
    def test_controls_are_excluded_from_positive_lambda_leaders(self) -> None:
        records = []
        for regularizer in ("wavelet", "llr"):
            for value, nrmse, ssim in ((0.0, 0.01, 0.99), (1e-4, 0.2, 0.8), (2e-4, 0.1, 0.9)):
                records.append(
                    {
                        "case_id": f"{regularizer}:{value}",
                        "regularizer": regularizer,
                        "lambda": value,
                        "block_size": 8 if regularizer == "llr" else "",
                        "nrmse_brain": nrmse,
                        "ssim_3d_brain_bbox": ssim,
                        "ncc_brain": ssim,
                        "gradient_ncc_brain_edge": ssim,
                        "edge_preservation_ratio": 1.0 + nrmse,
                    }
                )

        leaders = _metric_leaders(records)

        self.assertEqual(leaders["wavelet"]["lowest_nrmse_brain"]["lambda"], 2e-4)
        self.assertEqual(leaders["llr"]["highest_ssim_3d_brain_bbox"]["lambda"], 2e-4)
        self.assertEqual(
            leaders["llr_by_block"]["8"]["highest_gradient_ncc_brain_edge"]["lambda"],
            2e-4,
        )

    def test_ragged_llr_grid_heatmap_marks_unrun_cases(self) -> None:
        records = []
        for block_size, lambda_value in ((4, 3e-3), (8, 2e-2)):
            records.append(
                {
                    "regularizer": "llr",
                    "block_size": block_size,
                    "lambda": lambda_value,
                    "nrmse_brain": 0.04,
                    "ssim_3d_brain_bbox": 0.97,
                    "gradient_ncc_brain_edge": 0.99,
                    "edge_preservation_ratio": 1.01,
                }
            )

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            output = Path(temporary) / "ragged_grid.png"
            _plot_llr_heatmaps(records, output)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
