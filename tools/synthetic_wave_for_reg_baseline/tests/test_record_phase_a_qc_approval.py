"""Tests for the explicit Phase A visual-QC approval gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from record_phase_a_qc_approval import run  # noqa: E402


class ApprovalTests(unittest.TestCase):
    def test_requires_explicit_confirmation(self) -> None:
        args = argparse.Namespace(
            brain_mask_manifest=Path("mask.json"),
            orientation_report=Path("orientation.json"),
            confirm_reviewed_mask_and_lr=False,
            notes="test",
        )
        with self.assertRaisesRegex(ValueError, "Explicit"):
            run(args)

    def test_records_approved_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = root / "mask.json"
            orientation = root / "orientation.json"
            mask.write_text(json.dumps({"status": "visual_review_required"}), encoding="utf-8")
            orientation.write_text(
                json.dumps(
                    {
                        "status": "orientation_approved",
                        "decision_fields": {
                            "user_approved_best_signed_axis_mapping": True,
                            "approved_mapping": {"permutation": [0, 1, 2], "flips_ras_grid_axes": [True, False, True]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = run(
                argparse.Namespace(
                    brain_mask_manifest=mask,
                    orientation_report=orientation,
                    confirm_reviewed_mask_and_lr=True,
                    notes="reviewed",
                )
            )
            self.assertEqual(result["status"], "approved")
            self.assertTrue(result["approval"]["mask_boundary_visually_approved"])


if __name__ == "__main__":
    unittest.main()
