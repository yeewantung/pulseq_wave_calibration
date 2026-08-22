"""Tests for manifested retrospective visual-review approval."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from record_retrospective_low_resolution_review_approval import run, sha256_file  # noqa: E402


class ApprovalRecordTests(unittest.TestCase):
    def test_records_both_hash_matched_figures(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outputs = []
            for name in ("native_grid_comparison.png", "matched_1mm_grid_comparison.png"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                outputs.append({"path": str(path), "sha256": sha256_file(path)})
            review = root / "review_manifest.json"
            review.write_text(
                json.dumps({"status": "complete", "outputs": outputs}),
                encoding="utf-8",
            )
            output = root / "visual_approval.json"
            record = run(
                argparse.Namespace(
                    review_manifest=review,
                    approval_statement="Looks good.",
                    approval_source="test",
                    output=output,
                )
            )
            self.assertEqual(record["status"], "approved")
            self.assertEqual(len(record["figures"]), 2)
            self.assertEqual(record["review_manifest"]["sha256"], sha256_file(review))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
