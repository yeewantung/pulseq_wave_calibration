"""Tests for non-destructive presentation NIfTI collection building."""

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

from build_presentation_nifti_collection import run, sha256_file  # noqa: E402


class PresentationCollectionTests(unittest.TestCase):
    def test_builds_copy_and_placeholder_then_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)), source)
            config = root / "config.json"
            output = root / "collection"
            payload = {
                "format_version": 1,
                "output_dir": str(output),
                "entries": [
                    {
                        "display_order": 1,
                        "key": "available",
                        "label": "Available",
                        "status": "available",
                        "source_nifti": str(source),
                    },
                    {
                        "display_order": 2,
                        "key": "pending",
                        "label": "Pending",
                        "status": "placeholder",
                        "reason": "not run",
                    },
                ],
            }
            config.write_text(json.dumps(payload), encoding="utf-8")
            manifest = run(config, refresh=False)
            self.assertEqual(manifest["status"], "complete_with_placeholders")
            copied = output / "available.nii.gz"
            self.assertEqual(sha256_file(copied), sha256_file(source))
            self.assertTrue((output / "pending.placeholder.json").is_file())

            payload["entries"][1] = {
                "display_order": 2,
                "key": "pending",
                "label": "Pending",
                "status": "available",
                "source_nifti": str(source),
            }
            config.write_text(json.dumps(payload), encoding="utf-8")
            manifest = run(config, refresh=True)
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((output / "pending.nii.gz").is_file())
            self.assertFalse((output / "pending.placeholder.json").exists())

    def test_rejects_non_ras_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "las.nii.gz"
            affine = np.diag([-1.0, 1.0, 1.0, 1.0])
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine), source)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "output_dir": str(root / "output"),
                        "entries": [
                            {
                                "display_order": 1,
                                "key": "las",
                                "label": "LAS",
                                "status": "available",
                                "source_nifti": str(source),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not canonical RAS"):
                run(config, refresh=False)

    def test_refresh_removes_only_hash_valid_stale_owned_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)),
                source,
            )
            config = root / "config.json"
            output = root / "collection"
            payload = {
                "format_version": 1,
                "output_dir": str(output),
                "entries": [
                    {
                        "display_order": 1,
                        "key": "old-key",
                        "label": "Old",
                        "status": "available",
                        "source_nifti": str(source),
                    }
                ],
            }
            config.write_text(json.dumps(payload), encoding="utf-8")
            run(config, refresh=False)
            self.assertTrue((output / "old-key.nii.gz").is_file())

            payload["entries"][0]["key"] = "new-key"
            payload["entries"][0]["label"] = "New"
            config.write_text(json.dumps(payload), encoding="utf-8")
            run(config, refresh=True)

            self.assertFalse((output / "old-key.nii.gz").exists())
            self.assertTrue((output / "new-key.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
