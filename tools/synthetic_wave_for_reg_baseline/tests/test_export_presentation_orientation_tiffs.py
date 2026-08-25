"""Tests for fixed-index presentation TIFF export."""

from __future__ import annotations

import argparse
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
from export_presentation_orientation_tiffs import (  # noqa: E402
    orientation_slices,
    run,
)


class PresentationOrientationTiffTests(unittest.TestCase):
    def test_orientation_slice_mapping(self) -> None:
        x, y, z = np.indices((3, 3, 3))
        volume = 100 * x + 10 * y + z
        slices = orientation_slices(volume, 1)
        self.assertEqual(slices["sagittal"][0, 0], volume[1, 0, 2])
        self.assertEqual(slices["coronal"][0, 0], volume[0, 1, 2])
        self.assertEqual(slices["axial"][0, 0], volume[0, 2, 1])

    def test_run_exports_three_tiffs_per_available_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nifti_path = root / "example.nii.gz"
            data = np.arange(4**3, dtype=np.float32).reshape((4, 4, 4)) + 1
            nib.save(nib.Nifti1Image(data, np.eye(4)), nifti_path)
            collection = {
                "status": "complete_with_placeholders",
                "entries": [
                    {
                        "display_order": 1,
                        "key": "example",
                        "label": "Example",
                        "status": "available",
                        "collection_file": nifti_path.name,
                        "collection_sha256": sha256_file(nifti_path),
                    },
                    {
                        "display_order": 2,
                        "key": "pending",
                        "label": "Pending",
                        "status": "placeholder",
                        "collection_file": "pending.placeholder.json",
                    },
                ],
            }
            collection_path = root / "collection_manifest.json"
            collection_path.write_text(json.dumps(collection), encoding="utf-8")
            output = root / "tiffs"
            result = run(
                argparse.Namespace(
                    collection_manifest=collection_path,
                    output_dir=output,
                    index=2,
                    display_percentile=99.5,
                    refresh=False,
                )
            )
            self.assertEqual(result["entry_count"], 1)
            self.assertEqual(result["tiff_count"], 3)
            self.assertEqual(len(list(output.glob("*.tiff"))), 3)


if __name__ == "__main__":
    unittest.main()
