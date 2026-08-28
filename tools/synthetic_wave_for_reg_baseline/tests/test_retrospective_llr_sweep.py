"""Tests for retrospective corrected-LLR sweep and matched-grid evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_retrospective_llr_sweep_matched_grid import metric_leaders  # noqa: E402
from run_retrospective_llr_sweep import (  # noqa: E402
    _command_setting,
    build_llr_config,
)


class LlrReconstructionTests(unittest.TestCase):
    def test_command_contract_requires_corrected_split_complex_gpu_llr(self) -> None:
        command = [
            "bart",
            "wave",
            "-l",
            "-v",
            "-b",
            "8",
            "-f",
            "-r",
            "0.01",
            "-i",
            "100",
            "-g",
            "maps",
            "psf",
            "kspace",
            "image",
        ]
        self.assertEqual(_command_setting(command), (8, 0.01))
        with self.assertRaisesRegex(ValueError, "corrected GPU"):
            _command_setting([value for value in command if value != "-v"])

    def test_llr_config_is_a_thin_pipeline_specialization(self) -> None:
        base = {
            "source": {"subject": "base"},
            "output_root": "/old",
            "reconstruction": {
                "regularizer": "wavelet",
                "lambda": 0.0,
                "iterations": 100,
            },
        }
        found = build_llr_config(
            base,
            block_size=16,
            lambda_value=0.005,
            output_root=Path("/new"),
            prepared_root=Path("/prepared"),
            subject="llr-sweep",
        )
        self.assertEqual(found["reconstruction"]["regularizer"], "llr")
        self.assertEqual(found["reconstruction"]["block_size"], 16)
        self.assertEqual(found["reconstruction"]["lambda"], 0.005)
        self.assertEqual(found["prepared_cases_root"], "/prepared")
        self.assertEqual(base["reconstruction"]["regularizer"], "wavelet")


class LlrEvaluationTests(unittest.TestCase):
    def test_leaders_are_separate_for_each_block_case_and_metric(self) -> None:
        rows = []
        for block in (4, 8):
            for value, nrmse, ssim, gradient, edge in (
                (0.002, 0.09, 0.93, 0.96, 0.99),
                (0.01, 0.08, 0.92, 0.97, 1.03),
            ):
                rows.append(
                    {
                        "reference": "direct_fft_rss",
                        "method": "llr",
                        "block_size": block,
                        "case_name": "case",
                        "lambda": value,
                        "nrmse_brain": nrmse,
                        "ssim_axial_brain_bbox_mean": ssim,
                        "gradient_ncc_fixed_edge": gradient,
                        "edge_gradient_preservation_ratio": edge,
                    }
                )
        leaders = metric_leaders(rows)
        self.assertEqual(leaders["4"]["case"]["nrmse_brain"]["lambda"], 0.01)
        self.assertEqual(
            leaders["8"]["case"]["ssim_axial_brain_bbox_mean"]["lambda"],
            0.002,
        )


if __name__ == "__main__":
    unittest.main()
