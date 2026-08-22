"""Tests for explicit approval of the qualitative R3 transfer review."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from bart_cfl import sha256_file  # noqa: E402
from record_r3_transfer_approval import run  # noqa: E402


class TransferApprovalTests(unittest.TestCase):
    def test_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Explicit"):
            run(
                argparse.Namespace(
                    review_manifest=Path("review.json"),
                    confirm_qualitative_transfer_reviewed=False,
                    notes="test",
                )
            )

    def test_records_approved_frozen_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def record(name: str) -> dict[str, str]:
                path = root / name
                path.write_text(name, encoding="utf-8")
                return {"path": str(path), "sha256": sha256_file(path)}

            source = record("lambda0.json")
            comparison = record("comparison.png")
            relative_l2 = record("relative_l2.png")
            cases = []
            for label, value in (("0", 0.0), ("1.5e-2", 0.015)):
                run_manifest = record(f"run-{label}.json")
                nifti = record(f"image-{label}.nii.gz")
                sidecar = record(f"image-{label}.json")
                cases.append(
                    {
                        "lambda": value,
                        "lambda_label": label,
                        "run_manifest": run_manifest["path"],
                        "run_manifest_sha256": run_manifest["sha256"],
                        "nifti": {
                            "nifti": nifti["path"],
                            "nifti_sha256": nifti["sha256"],
                            "sidecar": sidecar["path"],
                            "sidecar_sha256": sidecar["sha256"],
                        },
                    }
                )
            manifest_path = root / "review.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "awaiting_qualitative_transfer_assessment",
                        "qualitative_transfer_only": True,
                        "regularizer": "wavelet",
                        "lambda_zero_manifest": source,
                        "cases": cases,
                        "outputs": {
                            "comparison": comparison["path"],
                            "comparison_sha256": comparison["sha256"],
                            "relative_l2_plot": relative_l2["path"],
                            "relative_l2_plot_sha256": relative_l2["sha256"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            approved = run(
                argparse.Namespace(
                    review_manifest=manifest_path,
                    confirm_qualitative_transfer_reviewed=True,
                    notes="looks good",
                )
            )

            self.assertEqual(approved["status"], "qualitative_transfer_approved")
            self.assertFalse(approved["approval"]["lambda_retuned_on_r3"])
            self.assertEqual(approved["approval"]["notes"], "looks good")


if __name__ == "__main__":
    unittest.main()
