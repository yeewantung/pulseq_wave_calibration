"""Focused tests for explicit MPRAGE ROVir selection records."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import cfl_record, create_cfl, sha256_file  # noqa: E402
from wave_retro_lr.rovir_selection import record_mprage_rovir_selection  # noqa: E402


class RovirSelectionTests(unittest.TestCase):
    """Verify explicit confirmation and hash-bound selection provenance."""

    def test_records_validated_manual_selection(self) -> None:
        """Bind one reviewed choice and preserve nonselected comparisons.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feasibility = root / "feasibility"
            reconstruction = root / "reconstruction"
            candidate_id = "negative_ro000_020"

            candidate_path = feasibility / "masks" / "candidates" / "manifest.json"
            _write_json(
                candidate_path,
                {
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "parameters": {
                                "negative_human_inclusive_bounds": {"RO": [0, 20]}
                            },
                        }
                    ]
                },
            )
            approved_root = feasibility / "masks" / "approved"
            for name in ("positive_estimation_mask", "negative_estimation_mask"):
                values = create_cfl(approved_root / name, (4, 3, 2))
                values[...] = 1
                values.flush()
                del values
            approved_path = approved_root / "manifest.json"
            _write_json(
                approved_path,
                {
                    "status": "mprage_rovir_masks_approved",
                    "candidate_id": candidate_id,
                    "candidate_manifest_sha256": sha256_file(candidate_path),
                    "positive_estimation_mask": cfl_record(
                        approved_root / "positive_estimation_mask"
                    ),
                    "negative_estimation_mask": cfl_record(
                        approved_root / "negative_estimation_mask"
                    ),
                },
            )

            transform_base = feasibility / "transforms" / "rovir_full" / "transform"
            transform = create_cfl(transform_base, (1, 1, 1, 2, 2))
            transform[...] = 0
            transform[0, 0, 0, 0, 0] = 1
            transform[0, 0, 0, 1, 1] = 1
            transform.flush()
            del transform
            transform_qc_path = feasibility / "manifests" / "rovir_transform_qc.json"
            _write_json(
                transform_qc_path,
                {
                    "status": "mprage_bart_rovir_transform_qc_ready",
                    "automatic_selection": False,
                    "solver_backend": "bart rovir only",
                    "selected_virtual_coils": None,
                    "transform": cfl_record(transform_base),
                },
            )

            branch_id = "rovir_ncc2"
            branch_path = reconstruction / branch_id / "bart_inputs" / "manifest.json"
            _write_json(
                branch_path,
                {
                    "status": "measured_wave_mprage_rovir_control_ready",
                    "rovir": {
                        "virtual_coils": 2,
                        "solver_backend": "bart rovir only",
                        "basis_applied_identically_to_image_and_acs": True,
                        "transform_qc_manifest": _file_record(transform_qc_path),
                        "transform_source": cfl_record(transform_base),
                    },
                    "sampling_validation": {
                        "zero_outside_sampling_mask": True,
                        "acs_merged_into_wave_image_kspace": False,
                    },
                    "psf_calibration": {"reused_without_recalibration": True},
                },
            )
            shared_path = reconstruction / "shared" / "manifest.json"
            _write_json(
                shared_path,
                {
                    "status": "measured_wave_mprage_rovir_coil_count_comparison_ready",
                    "automatic_winner_selected": False,
                    "feasibility_root": str(feasibility.resolve()),
                    "channel_counts": [2],
                    "branches": {branch_id: _file_record(branch_path)},
                },
            )

            nifti_root = reconstruction / branch_id / "nifti" / "fista_r0" / "sub-test"
            nifti_root.mkdir(parents=True)
            selected_paths = {}
            for part in ("mag", "phase"):
                path = nifti_root / f"sub-test_part-{part}_ROVir.nii.gz"
                nib.save(nib.Nifti1Image(np.ones((4, 3, 2), dtype=np.float32), np.eye(4)), path)
                _write_json(
                    path.with_suffix("").with_suffix(".json"),
                    {"PreparedInputManifest": str(branch_path.resolve())},
                )
                selected_paths[part] = path
            other_path = root / "comparison_only.nii.gz"
            nib.save(
                nib.Nifti1Image(np.full((4, 3, 2), 2, dtype=np.float32), np.eye(4)),
                other_path,
            )

            bart_output = reconstruction / branch_id / "bart_output"
            ecalib_path = bart_output / "ecalib_command.txt"
            wave_path = bart_output / "fista_r0" / "wave_command.txt"
            ecalib_path.parent.mkdir(parents=True)
            wave_path.parent.mkdir(parents=True)
            ecalib_path.write_text("bart ecalib -m 1 -c 0.1 calib sens\n", encoding="utf-8")
            wave_path.write_text("bart wave -g -w -f -r 0 sens psf ksp out\n", encoding="utf-8")

            qc_path = root / "qc" / "manifest.json"
            _write_json(
                qc_path,
                {
                    "status": "mprage_rovir_negative_roi_series_qc_ready",
                    "automatic_winner_selected": False,
                    "display_window": {"shared_between_rows": True},
                    "candidates": [
                        {"label": "selected", "nifti": _file_record(selected_paths["mag"])},
                        {"label": "comparison only", "nifti": _file_record(other_path)},
                    ],
                },
            )
            manifest = record_mprage_rovir_selection(
                feasibility,
                reconstruction,
                qc_path,
                selection_scope="test dataset",
                candidate_id=candidate_id,
                virtual_coils=2,
                ecalib_crop=0.1,
                reviewer_note="Selected after manual fixed-window review.",
                confirm_user_selection=True,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["selection"]["virtual_coils"], 2)
            self.assertEqual(len(manifest["evidence"]["comparison_only_candidates"]), 1)
            self.assertTrue(
                (reconstruction / "selection" / "selection_manifest.json").is_file()
            )

    def test_requires_explicit_confirmation(self) -> None:
        """Reject an otherwise requested selection without user confirmation.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "confirmation"):
            record_mprage_rovir_selection(
                "missing-feasibility",
                "missing-reconstruction",
                "missing-qc.json",
                selection_scope="test",
                candidate_id="candidate",
                virtual_coils=1,
                ecalib_crop=0.1,
                reviewer_note="reviewed",
                confirm_user_selection=False,
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one test JSON object.

    Args:
        path: Destination file.
        payload: JSON-compatible object.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, object]:
    """Create one strict ordinary-file identity for a test artifact.

    Args:
        path: Existing test file.

    Returns:
        Path, size, and SHA-256 record.
    """
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


if __name__ == "__main__":
    unittest.main()
