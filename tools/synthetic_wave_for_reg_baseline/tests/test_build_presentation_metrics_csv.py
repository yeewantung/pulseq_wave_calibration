"""Tests for presentation-ordered metric aggregation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bart_cfl import sha256_file  # noqa: E402
from build_presentation_metrics_csv import build_rows  # noqa: E402


class PresentationMetricsCsvTests(unittest.TestCase):
    def _manifest_record(self, root: Path, name: str, payload: dict) -> dict:
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": str(path), "sha256": sha256_file(path)}

    def test_combines_exact_retrospective_and_nonmetric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = self._manifest_record(root, "empty.json", {})
            direct = self._manifest_record(
                root,
                "direct.json",
                {
                    "direct_fft_metrics": {
                        "status": "complete",
                        "metrics": {"nrmse_brain": 0.1, "ncc_brain": 0.9},
                    }
                },
            )
            retrospective = self._manifest_record(
                root,
                "retrospective.json",
                {"case": {"case_name": "retro_case"}},
            )
            entries = [
                {
                    "display_order": 1,
                    "key": "dicom_context",
                    "label": "DICOM",
                    "status": "available",
                    "collection_file": "dicom.nii.gz",
                },
                {
                    "display_order": 2,
                    "key": "direct_fft_rss",
                    "label": "Direct FFT",
                    "status": "available",
                    "collection_file": "fft.nii.gz",
                    "source_manifest": empty,
                },
                {
                    "display_order": 3,
                    "key": "new_recon",
                    "label": "New recon",
                    "status": "available",
                    "collection_file": "new.nii.gz",
                    "source_manifest": direct,
                },
                {
                    "display_order": 4,
                    "key": "regularization_recon",
                    "label": "Regularization recon",
                    "status": "available",
                    "collection_file": "regularization.nii.gz",
                    "source_sha256": "regularization-hash",
                    "source_manifest": empty,
                },
                {
                    "display_order": 5,
                    "key": "retro_recon",
                    "label": "Retro recon",
                    "status": "available",
                    "collection_file": "retro.nii.gz",
                    "source_manifest": retrospective,
                },
                {
                    "display_order": 6,
                    "key": "pending",
                    "label": "Pending",
                    "status": "placeholder",
                    "collection_file": "pending.placeholder.json",
                    "reason": "Not reconstructed",
                },
            ]
            rows = build_rows(
                {"entries": entries},
                [
                    {
                        "source_nifti_sha256": "regularization-hash",
                        "nrmse_brain": "0.2",
                    }
                ],
                [
                    {
                        "reference": "direct_fft_rss",
                        "candidate": "direct_fft_rss",
                        "nrmse_brain": "0",
                    },
                    {
                        "reference": "direct_fft_rss",
                        "candidate": "retro_case",
                        "nrmse_brain": "0.3",
                        "ssim_axial_brain_bbox_mean": "0.8",
                    },
                ],
                [
                    {
                        "case": "retro_case",
                        "smooth_region_signal_to_residual_proxy": "12.5",
                    }
                ],
            )
        self.assertEqual([row["display_order"] for row in rows], list(range(1, 7)))
        self.assertEqual(rows[0]["metric_status"], "not_evaluated_qualitative_only")
        self.assertEqual(rows[1]["metric_status"], "reference_identity")
        self.assertEqual(rows[2]["nrmse_brain"], 0.1)
        self.assertEqual(rows[3]["nrmse_brain"], "0.2")
        self.assertEqual(rows[4]["comparison_grid"], "matched_full_resolution_grid")
        self.assertEqual(rows[4]["ssim_axial_brain_mean"], "0.8")
        self.assertEqual(rows[5]["metric_status"], "pending_reconstruction")


if __name__ == "__main__":
    unittest.main()
