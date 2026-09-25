"""Focused tests for matched MPRAGE ROVir coil-count controls."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import cfl_record, create_cfl, open_cfl, sha256_file  # noqa: E402
from wave_retro_lr.rovir_control import (  # noqa: E402
    prepare_mprage_rovir_comparison,
    write_mprage_rovir_comparison_qc,
    write_mprage_rovir_mask_comparison_qc,
    write_mprage_rovir_mask_series_qc,
)
from wave_retro_lr.sampling import SamplingPattern  # noqa: E402


class RovirControlTests(unittest.TestCase):
    """Verify projection, provenance, layout, and review behavior."""

    def test_preparation_uses_one_bart_transform_for_both_counts(self) -> None:
        """Project image and ACS with the same conjugated BART transform.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = root / "accepted"
            feasibility = root / "feasibility"
            output = root / "comparison"
            source_inputs = accepted / "normal" / "bart_inputs"
            source_inputs.mkdir(parents=True)
            (feasibility / "manifests").mkdir(parents=True)
            (feasibility / "manifests" / "physical_calibration.json").write_text(
                "{}\n", encoding="utf-8"
            )
            rovir_inputs = feasibility / "manifests" / "rovir_inputs.json"
            rovir_inputs.write_text("{}\n", encoding="utf-8")

            sampling = SamplingPattern(
                name="R1",
                acceleration_lin_par=(1, 1),
                lin_residue=None,
                matrix_lin_par=(6, 4),
                acquired_lin=tuple(range(6)),
                acquired_par=tuple(range(4)),
                measurement_index=1,
                skip_lin_par=(0, 0),
            )
            (source_inputs / "manifest.json").write_text(
                json.dumps({"sampling": sampling.to_json()}), encoding="utf-8"
            )
            psf = create_cfl(source_inputs / "psf", (8, 6, 4, 1, 1))
            psf[:] = 1
            psf.flush()
            del psf

            physical_directory = feasibility / "inputs" / "physical_calibration"
            physical_directory.mkdir(parents=True)
            physical_values = np.zeros((4, 4, 4, 3), dtype=np.complex64)
            packed = (
                np.arange(4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3)
                + 1j
            ).astype(np.complex64)
            physical_values[:, 1:3, 1:3, :] = packed
            physical = create_cfl(
                physical_directory / "physical_set4_kspace", physical_values.shape
            )
            physical[:] = physical_values
            physical.flush()
            del physical

            scale = np.float32(1 / np.sqrt(2))
            stored_transform = np.array(
                [
                    [scale, 1j * scale, 0],
                    [1j * scale, scale, 0],
                    [0, 0, 1],
                ],
                dtype=np.complex64,
            )
            transform_base = feasibility / "transforms" / "rovir_full" / "transform"
            transform = create_cfl(transform_base, (1, 1, 1, 3, 3))
            transform[0, 0, 0, :, :] = stored_transform
            transform.flush()
            del transform
            bart_record = feasibility / "logs" / "bart_version.txt"
            curve_csv = feasibility / "diagnostics" / "curve.csv"
            curve_plot = feasibility / "diagnostics" / "curve.png"
            bart_record.parent.mkdir(parents=True)
            curve_csv.parent.mkdir(parents=True)
            bart_record.write_text("v1.0-test\n", encoding="utf-8")
            curve_csv.write_text("virtual_coils,signal\n1,1\n", encoding="utf-8")
            curve_plot.write_bytes(b"synthetic plot")

            def file_record(path: Path) -> dict[str, object]:
                """Return a strict identity record for one test artifact."""
                return {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

            qc = {
                "status": "mprage_bart_rovir_transform_qc_ready",
                "rovir_input_manifest_sha256": sha256_file(rovir_inputs),
                "selected_virtual_coils": None,
                "transform": cfl_record(transform_base),
                "transform_validation": {"orthogonality_tolerance": 1e-4},
                "bart": file_record(bart_record),
                "region_curve_csv": file_record(curve_csv),
                "region_curve_plot": file_record(curve_plot),
            }
            qc_path = feasibility / "manifests" / "rovir_transform_qc.json"
            qc_path.write_text(
                json.dumps(qc), encoding="utf-8"
            )

            image_values = (
                np.arange(8 * 6 * 4 * 3, dtype=np.float32).reshape(8, 6, 4, 3)
                + 2j
            ).astype(np.complex64)
            helper = SimpleNamespace(
                load_img=lambda _: torch.from_numpy(image_values.copy()),
            )
            source = {
                "geometry": {
                    "logical_matrix_ro_lin_par": [4, 6, 4],
                    "physical_fov_mm_xyz": [4.0, 6.0, 4.0],
                    "readout_oversampling_factor": 2,
                    "readout_oversampled": 8,
                },
                "coil_compression": {"physical_coils": 3},
                "psf_calibration": {"ncalib": 4, "nacs": 2},
            }
            with (
                patch(
                    "wave_retro_lr.rovir_control._validated_source_contract",
                    return_value=source,
                ),
                patch(
                    "wave_retro_lr.rovir_control._validate_physical_calibration_export"
                ),
                patch(
                    "wave_retro_lr.rovir_control.inspect_twix_sampling",
                    return_value=(sampling, {}),
                ),
                patch(
                    "wave_retro_lr.rovir_control.load_wave_mprage_helpers",
                    return_value=helper,
                ),
            ):
                shared = prepare_mprage_rovir_comparison(
                    root / "source.dat",
                    root / "source.seq",
                    accepted,
                    feasibility,
                    output,
                    channel_counts=(2, 3),
                    partition_progress_interval=2,
                )
                wave_before = sha256_file(
                    output / "rovir_ncc2" / "bart_inputs" / "wave_kspace.cfl"
                )
                qc["created_at_utc"] = "regenerated-wrapper-metadata"
                qc_path.write_text(json.dumps(qc), encoding="utf-8")
                resumed = prepare_mprage_rovir_comparison(
                    root / "source.dat",
                    root / "source.seq",
                    accepted,
                    feasibility,
                    output,
                    channel_counts=(2, 3),
                    partition_progress_interval=2,
                )
                self.assertEqual(
                    sha256_file(
                        output / "rovir_ncc2" / "bart_inputs" / "wave_kspace.cfl"
                    ),
                    wave_before,
                )
                refreshed = json.loads(
                    (
                        output
                        / "rovir_ncc2"
                        / "bart_inputs"
                        / "manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertTrue(
                    refreshed["provenance_refresh"][
                        "scientific_arrays_reused_without_modification"
                    ]
                )
                self.assertEqual(resumed["channel_counts"], [2, 3])

            expected = image_values.reshape(-1, 3) @ stored_transform.conj()
            expected = expected.reshape(image_values.shape)
            for count in (2, 3):
                inputs = output / f"rovir_ncc{count}" / "bart_inputs"
                actual = np.asarray(open_cfl(inputs / "wave_kspace"))[:, :, :, :, 0]
                np.testing.assert_allclose(actual, expected[..., :count], rtol=1e-6)
                calibration = np.asarray(open_cfl(inputs / "kspace_calib"))
                expected_acs = packed.reshape(-1, 3) @ stored_transform.conj()[:, :count]
                expected_acs = expected_acs.reshape(4, 2, 2, count)
                np.testing.assert_allclose(
                    calibration[:, 2:4, 1:3, :], expected_acs, rtol=1e-6
                )
                outside = calibration.copy()
                outside[:, 2:4, 1:3, :] = 0
                self.assertEqual(np.count_nonzero(outside), 0)
                manifest = json.loads((inputs / "manifest.json").read_text())
                self.assertEqual(manifest["rovir"]["virtual_coils"], count)
                self.assertTrue(
                    manifest["rovir"]["basis_applied_identically_to_image_and_acs"]
                )
                self.assertFalse(manifest["scientific_actions"]["psf_recalibrated"])
            self.assertEqual(shared["channel_counts"], [2, 3])
            self.assertTrue(shared["same_transform_image_source_acs_psf_and_sampling"])
            self.assertFalse(shared["automatic_winner_selected"])

    def test_qc_uses_one_neutral_window(self) -> None:
        """Use one shared display scale and avoid automatic selection.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
            second = first * 1.2
            affine = np.eye(4)
            first_path = root / "rovir24.nii.gz"
            second_path = root / "rovir48.nii.gz"
            nib.save(nib.Nifti1Image(first, affine), first_path)
            nib.save(nib.Nifti1Image(second, affine), second_path)
            normalization = {
                "MagnitudeNormalization": {
                    "Method": "positive-finite-percentile",
                    "Percentile": 99.0,
                    "InputPercentileValue": 2.0,
                    "OutputPercentileValue": 1.0,
                    "Clipped": False,
                }
            }
            (root / "rovir24.json").write_text(json.dumps(normalization))
            normalization["MagnitudeNormalization"]["InputPercentileValue"] = 3.0
            (root / "rovir48.json").write_text(json.dumps(normalization))
            manifest = write_mprage_rovir_comparison_qc(
                first_path, second_path, root / "qc"
            )
            self.assertTrue(
                (root / "qc" / "rovir_ncc24_vs_ncc48_fixed_window.png").is_file()
            )
            self.assertTrue(manifest["display_window"]["shared_between_rows"])
            self.assertEqual(
                [entry["display_to_restored_scale"] for entry in manifest["magnitude_normalization_restoration"]],
                [2.0, 3.0],
            )
            self.assertFalse(manifest["automatic_winner_selected"])

            mask_manifest = write_mprage_rovir_mask_comparison_qc(
                first_path, second_path, root / "mask_qc"
            )
            self.assertTrue(
                (root / "mask_qc" / "rovir24_ro000_030_vs_ro000_020_fixed_window.png").is_file()
            )
            self.assertIn("ro000_030", mask_manifest)
            self.assertIn("ro000_020", mask_manifest)
            self.assertFalse(mask_manifest["automatic_winner_selected"])

            series_manifest = write_mprage_rovir_mask_series_qc(
                (
                    ("negative RO 0--30", first_path),
                    ("negative RO 0--20", second_path),
                    ("negative RO 0--10", first_path),
                ),
                root / "series_qc",
                figure_filename="three_masks.png",
            )
            self.assertTrue((root / "series_qc" / "three_masks.png").is_file())
            self.assertEqual(len(series_manifest["candidates"]), 3)
            percentiles = series_manifest["display_window"][
                "per_branch_restored_positive_p99_5"
            ]
            self.assertAlmostEqual(percentiles[2], percentiles[0])
            self.assertGreater(percentiles[1], percentiles[0])
            self.assertEqual(
                [entry["label"] for entry in series_manifest["candidates"]],
                ["negative RO 0--30", "negative RO 0--20", "negative RO 0--10"],
            )
            self.assertFalse(series_manifest["automatic_winner_selected"])

            single_manifest = write_mprage_rovir_mask_series_qc(
                (("ROVir-24 FISTA lambda=0", first_path),),
                root / "single_qc",
                figure_filename="rovir_only.png",
            )
            self.assertTrue((root / "single_qc" / "rovir_only.png").is_file())
            self.assertEqual(
                single_manifest["status"],
                "mprage_rovir_single_reconstruction_qc_ready",
            )
            self.assertEqual(len(single_manifest["candidates"]), 1)
            self.assertFalse(single_manifest["display_window"]["shared_between_rows"])

    def test_sample_script_keeps_bart_commands_explicit(self) -> None:
        """Keep ecalib and CPU/GPU FISTA-r0 commands visible in Bash.

        Returns:
            None.
        """
        script = TOOL_ROOT / "scripts" / "sample_mprage_rovir_comparison.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn('CHANNEL_COUNTS_CSV="24,48"', source)
        self.assertIn('for count in "${CHANNEL_COUNTS[@]}"', source)
        self.assertIn("bart ecalib -m 1 -c", source)
        self.assertIn("bart wave -g -w -f -r 0 -i 100 -t 1e-6", source)
        self.assertIn("bart wave -w -f -r 0 -i 100 -t 1e-6", source)
        self.assertNotIn("optimal_wavelet", source)


if __name__ == "__main__":
    unittest.main()
