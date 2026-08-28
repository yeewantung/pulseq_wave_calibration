"""Focused tests for the lambda-zero visual approval gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from record_lambda0_visual_approval import run  # noqa: E402


class LambdaZeroVisualApprovalTests(unittest.TestCase):
    def test_records_and_reuses_hash_bound_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = [root / name for name in ("recon.png", "maps_mag.png", "maps_phase.png")]
            for index, path in enumerate(artifacts):
                path.write_bytes(f"image-{index}".encode())
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "lambda0_complete_awaiting_visual_review",
                        "config": {
                            "ecalib_crop": 0.6,
                            "gpu_wave_reconstruction": True,
                        },
                        "wave_lambda0": {"command": ["bart", "wave", "-g"]},
                        "nifti": {"central_slice_quicklook": str(artifacts[0])},
                        "ecalib": {
                            "diagnostic_montages": [
                                str(artifacts[1]),
                                str(artifacts[2]),
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                lambda_zero_manifest=manifest_path,
                output=root / "approval.json",
                confirm_reconstruction_and_maps_reviewed=True,
                notes="Looks good.",
            )

            approval = run(args)

            self.assertEqual(approval["status"], "approved_for_regularization_sweep")
            self.assertEqual(len(approval["reviewed_artifacts"]), 3)
            self.assertEqual(run(args), approval)


if __name__ == "__main__":
    unittest.main()
