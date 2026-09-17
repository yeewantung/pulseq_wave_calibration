"""Focused tests for the higher-channel MPRAGE PCA control."""

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

from wave_retro_lr.bart_io import create_cfl, open_cfl  # noqa: E402
from wave_retro_lr.pca_control import (  # noqa: E402
    prepare_mprage_pca_control,
    write_mprage_pca_control_qc,
)
from wave_retro_lr.sampling import SamplingPattern  # noqa: E402


class PcaControlTests(unittest.TestCase):
    """Verify PCA-control scientific and command contracts."""

    def test_preparation_applies_one_basis_and_copies_psf_exactly(self) -> None:
        """Apply the same basis to image/ACS and preserve PSF bytes.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = root / "accepted"
            feasibility = root / "feasibility"
            output = root / "control"
            source_inputs = accepted / "normal" / "bart_inputs"
            source_inputs.mkdir(parents=True)
            (feasibility / "manifests").mkdir(parents=True)
            (feasibility / "manifests" / "physical_calibration.json").write_text(
                "{}\n", encoding="utf-8"
            )

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
            psf_values = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
            psf = create_cfl(source_inputs / "psf", (8, 6, 4, 1, 1))
            psf[:, :, :, 0, 0] = psf_values
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
            physical[...] = physical_values
            physical.flush()
            del physical

            image_values = (
                np.arange(8 * 6 * 4 * 3, dtype=np.float32).reshape(8, 6, 4, 3)
                + 2j
            ).astype(np.complex64)
            basis = np.eye(3, 2, dtype=np.complex64)

            def apply_cc(
                values: torch.Tensor, transform: np.ndarray, x_chunk: int
            ) -> torch.Tensor:
                """Apply the fixture coil transform.

                Args:
                    values: Coil-last complex tensor.
                    transform: Physical-to-virtual coil matrix.
                    x_chunk: Unused fixture chunk size.

                Returns:
                    Coil-compressed tensor.
                """
                del x_chunk
                return values @ torch.from_numpy(transform)

            helper = SimpleNamespace(
                estimate_cc_matrix_coillast=lambda *args, **kwargs: (
                    basis,
                    np.array([3.0, 2.0, 1.0]),
                    np.array([0.6, 0.9, 1.0]),
                ),
                apply_cc_coillast_torch=apply_cc,
                load_img=lambda _: torch.from_numpy(image_values.copy()),
            )
            source = {
                "geometry": {
                    "logical_matrix_ro_lin_par": [4, 6, 4],
                    "physical_fov_mm_xyz": [4.0, 6.0, 4.0],
                    "readout_oversampling_factor": 2,
                    "readout_oversampled": 8,
                },
                "coil_compression": {
                    "physical_coils": 3,
                    "leading_singular_values": [3.0, 2.0],
                },
                "psf_calibration": {"ncalib": 4, "nacs": 2},
            }
            with (
                patch(
                    "wave_retro_lr.pca_control._validated_source_contract",
                    return_value=source,
                ),
                patch(
                    "wave_retro_lr.pca_control._validate_physical_calibration_export"
                ),
                patch(
                    "wave_retro_lr.pca_control.inspect_twix_sampling",
                    return_value=(sampling, {}),
                ),
                patch(
                    "wave_retro_lr.pca_control.load_wave_mprage_helpers",
                    return_value=helper,
                ),
            ):
                manifest = prepare_mprage_pca_control(
                    root / "source.dat",
                    root / "source.seq",
                    accepted,
                    feasibility,
                    output,
                    virtual_coils=2,
                )

            inputs = output / "normal" / "bart_inputs"
            np.testing.assert_array_equal(
                np.asarray(open_cfl(inputs / "wave_kspace"))[:, :, :, :, 0],
                image_values[..., :2],
            )
            calibration = np.asarray(open_cfl(inputs / "kspace_calib"))
            np.testing.assert_array_equal(calibration[:, 2:4, 1:3, :], packed[..., :2])
            outside = calibration.copy()
            outside[:, 2:4, 1:3, :] = 0
            self.assertEqual(np.count_nonzero(outside), 0)
            self.assertEqual(
                (inputs / "psf.cfl").read_bytes(), (source_inputs / "psf.cfl").read_bytes()
            )
            self.assertEqual(manifest["coil_compression"]["virtual_coils"], 2)
            self.assertTrue(
                manifest["coil_compression"][
                    "basis_applied_identically_to_image_and_acs"
                ]
            )
            self.assertTrue(manifest["psf_calibration"]["source_and_copy_hashes_equal"])
            self.assertFalse(manifest["scientific_actions"]["psf_recalibrated"])

    def test_qc_uses_one_baseline_anchored_window(self) -> None:
        """Use one fixed window for both images without selecting a winner.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
            control = baseline * 1.1
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            baseline_path = root / "baseline.nii.gz"
            control_path = root / "control.nii.gz"
            nib.save(nib.Nifti1Image(baseline, affine), baseline_path)
            nib.save(nib.Nifti1Image(control, affine), control_path)
            manifest = write_mprage_pca_control_qc(
                baseline_path, control_path, root / "qc"
            )
            self.assertTrue((root / "qc" / "fixed_window_ncc12_vs_ncc24.png").is_file())
            self.assertTrue(manifest["display_window"]["shared_between_rows"])
            self.assertFalse(manifest["automatic_winner_selected"])

    def test_sample_script_keeps_commands_explicit(self) -> None:
        """Keep ecalib and CPU/GPU FISTA-r0 commands visible in Bash.

        Returns:
            None.
        """
        script = TOOL_ROOT / "scripts" / "sample_mprage_pca_control.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn('VIRTUAL_COILS=24', source)
        self.assertIn('ECALIB_CROP=0.1', source)
        self.assertIn("bart ecalib -m 1 -c", source)
        self.assertIn("bart wave -g -w -f -r 0 -i 100 -t 1e-6", source)
        self.assertIn("bart wave -w -f -r 0 -i 100 -t 1e-6", source)
        self.assertNotIn("optimal_wavelet", source)


if __name__ == "__main__":
    unittest.main()
