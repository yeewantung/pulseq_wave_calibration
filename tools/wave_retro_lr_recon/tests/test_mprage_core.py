"""Focused scientific-contract tests for Wave retrospective preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import create_cfl, open_cfl, read_shape  # noqa: E402
from wave_retro_lr.core import CaseSpec, Geometry, build_wave_options, resolve_case  # noqa: E402
from wave_retro_lr.core import build_case_mask  # noqa: E402
from wave_retro_lr.psf import (  # noqa: E402
    PSF_COEFFICIENT_PLOT_NAME,
    PSF_COEFFICIENT_REJECTED_DIAGNOSTICS_NAME,
    PSF_COEFFICIENT_REJECTED_PLOT_NAME,
    evaluate_calibrated_psf,
    write_psf_coefficient_plot,
)
from wave_retro_lr.retrospective import (  # noqa: E402
    resample_sensitivity_maps,
    synthesize_wave_from_no_wave_crop,
    write_measured_wave_crop,
)
from wave_retro_lr.sampling import (  # noqa: E402
    classify_mprage_sampling,
    pure_cartesian_image_lattice_mask,
    validate_pure_cartesian_image_lattice,
)
from wave_retro_lr.mprage import (  # noqa: E402
    AutomaticPsfFitRejected,
    _calibrated_psf_inputs,
    _embed_image_stream,
    _ensure_r3x1_psf_coefficient_plot,
    _normalize_psf_coefficient_settings,
    _recover_c_only_automatic_rejection,
    _write_automatic_psf_rejection_diagnostics,
)
from wave_retro_lr.mprage import prepare_normal_mprage, prepare_retro_mprage  # noqa: E402
from wave_retro_lr.sampling import SamplingPattern  # noqa: E402


class SamplingTests(unittest.TestCase):
    def test_pure_cartesian_masks_have_exact_counts_and_hashes(self) -> None:
        """Verify all five rerun masks are exact image lattices without ACS.

        Returns:
            None.
        """
        expected = {
            ((256, 256), (3, 1), (1, 0)): (
                21760,
                "ea4e03688efd6458eab1470c6a422ea08d2f7ffc377a4f86c72bbaf6c5e1418b",
            ),
            ((256, 256), (3, 2), (1, 0)): (
                10880,
                "22c680851a8799e602ef3bdf8c0e0edc0eace80cc766a0210a8bc31fdca3926e",
            ),
            ((256, 172), (3, 2), (1, 0)): (
                7310,
                "aaa8dc49115a6c9baeb7b5f18e43d4b89e968e4d3270496b8ea815f0ca3c63d4",
            ),
            ((172, 256), (3, 2), (1, 0)): (
                7296,
                "2ffc344bb02f01f5b0d77019a380d33abd3f4228461824575e4c52eae81e3774",
            ),
            ((204, 204), (3, 2), (2, 0)): (
                6936,
                "908e4cfa7161aabcb6cc20c1f41c9cabaacc99e6dc5a1b3637ab122349736e04",
            ),
        }
        for (shape, acceleration, residue), (count, digest) in expected.items():
            mask, metadata = pure_cartesian_image_lattice_mask(
                shape,
                acceleration_lin_par=acceleration,
                residue_lin_par=residue,
            )
            self.assertEqual(int(mask.sum()), count)
            self.assertEqual(metadata["logical_sha256"], digest)
            self.assertFalse(metadata["acs_coordinates_included"])
            self.assertEqual(validate_pure_cartesian_image_lattice(mask, metadata), metadata)

    def test_pure_mask_rejects_historical_acs_union(self) -> None:
        """Verify an old lattice-plus-central-band contract fails hard.

        Returns:
            None.
        """
        mask, metadata = pure_cartesian_image_lattice_mask(
            (8, 8), acceleration_lin_par=(3, 2), residue_lin_par=(1, 0)
        )
        metadata["mask_kind"] = "cartesian_with_full_pe1_acs"
        mask[3, :] = True
        with self.assertRaisesRegex(ValueError, "Historical ACS-union"):
            validate_pure_cartesian_image_lattice(mask, metadata)

    def test_accepts_only_r1_or_regular_lin_r3x1(self) -> None:
        """Verify the two supported image-stream sampling classes.

        Returns:
            None.
        """
        r1_coordinates = [(line, par) for par in range(4) for line in range(8)]
        r1 = classify_mprage_sampling(
            [item[0] for item in r1_coordinates],
            [item[1] for item in r1_coordinates],
            matrix_lin_par=(8, 4),
        )
        self.assertEqual(r1.name, "R1")
        self.assertEqual(r1.acceleration_lin_par, (1, 1))

        r3_coordinates = [(line, par) for par in range(4) for line in (1, 4, 7)]
        r3 = classify_mprage_sampling(
            [item[0] for item in r3_coordinates],
            [item[1] for item in r3_coordinates],
            matrix_lin_par=(8, 4),
        )
        self.assertEqual(r3.name, "R3x1")
        self.assertEqual(r3.lin_residue, 1)
        self.assertEqual(int(r3.mask().sum()), 12)

    def test_rejects_duplicate_and_irregular_coordinates(self) -> None:
        """Verify ambiguous or non-Cartesian MDH coordinate sets fail.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            classify_mprage_sampling([0, 0], [0, 0], matrix_lin_par=(4, 4))
        with self.assertRaisesRegex(ValueError, "factor-three"):
            classify_mprage_sampling(
                [0, 2, 0, 2], [0, 0, 1, 1], matrix_lin_par=(4, 2)
            )

    def test_r3_image_residue_need_not_contain_center_when_acs_is_separate(self) -> None:
        """Verify valid R3 image sampling may omit the separately calibrated center.

        Returns:
            None.
        """
        coordinates = [(line, par) for par in range(4) for line in (1, 4)]
        pattern = classify_mprage_sampling(
            [item[0] for item in coordinates],
            [item[1] for item in coordinates],
            matrix_lin_par=(6, 4),
        )
        self.assertEqual(pattern.name, "R3x1")
        self.assertEqual(pattern.lin_residue, 1)
        self.assertFalse(pattern.to_json()["image_kspace_center_acquired"])

    def test_compact_mapvbvd_payload_uses_mdh_skip_and_mask(self) -> None:
        """Verify a bounded mapVBVD payload is embedded on its logical grid.

        Returns:
            None.
        """
        pattern = classify_mprage_sampling(
            [1, 4, 7] * 4,
            [partition for partition in range(4) for _ in range(3)],
            matrix_lin_par=(8, 4),
            skip_lin_par=(1, 0),
        )
        compact = np.zeros((2, 7, 4, 2), dtype=np.complex64)
        compact[:, (0, 3, 6), :, :] = 1
        full = _embed_image_stream(
            compact, pattern, readout_oversampled=2, physical_coils=2
        ).numpy()
        self.assertEqual(full.shape, (2, 8, 4, 2))
        self.assertTrue(np.all(full[:, pattern.mask(), :] == 1))
        self.assertTrue(np.all(full[:, ~pattern.mask(), :] == 0))

    def test_full_grid_embedding_reuses_storage_without_changing_samples(self) -> None:
        """Verify full-grid masking avoids a second physical-coil allocation.

        Returns:
            None.
        """
        import torch

        pattern = classify_mprage_sampling(
            [1, 4, 1, 4],
            [0, 0, 1, 1],
            matrix_lin_par=(6, 2),
        )
        # Match the non-contiguous coil-last view returned by upstream load_img.
        loaded = torch.zeros((4, 3, 6, 2), dtype=torch.complex64).permute(0, 2, 3, 1)
        loaded[:, pattern.mask(), :] = 2 + 3j
        storage_pointer = loaded.data_ptr()

        full = _embed_image_stream(
            loaded,
            pattern,
            readout_oversampled=4,
            physical_coils=3,
        )

        self.assertEqual(full.data_ptr(), storage_pointer)
        self.assertTrue(torch.all(full[:, pattern.mask(), :] == 2 + 3j))
        self.assertTrue(torch.all(full[:, ~pattern.mask(), :] == 0))

    def test_full_grid_embedding_rejects_unmeasured_nonzero_samples(self) -> None:
        """Verify bounded lattice validation still detects invalid payloads.

        Returns:
            None.
        """
        import torch

        pattern = classify_mprage_sampling(
            [1, 4, 1, 4],
            [0, 0, 1, 1],
            matrix_lin_par=(6, 2),
        )
        loaded = torch.zeros((17, 6, 2, 3), dtype=torch.complex64)
        loaded[:, pattern.mask(), :] = 1
        loaded[16, 0, 0, 0] = 1

        with self.assertRaisesRegex(ValueError, "outside its MDH sampling mask"):
            _embed_image_stream(
                loaded,
                pattern,
                readout_oversampled=17,
                physical_coils=3,
            )


class PsfAndGeometryTests(unittest.TestCase):
    def test_only_r3x1_preparation_requests_the_visible_diagnostic(self) -> None:
        """Verify the visible coefficient diagnostic is specific to R3x1.

        Returns:
            None.
        """
        settings = {
            "coefficient_processing": "smooth",
            "fit_kx_range": None,
        }
        vectors = tuple(np.zeros(16, dtype=np.float64) for _ in range(3))
        with tempfile.TemporaryDirectory() as folder:
            normal = Path(folder) / "normal"
            self.assertIsNone(
                _ensure_r3x1_psf_coefficient_plot(
                    normal,
                    "R1",
                    vectors,
                    settings,
                )
            )
            self.assertFalse((normal / PSF_COEFFICIENT_PLOT_NAME).exists())

            result = _ensure_r3x1_psf_coefficient_plot(
                normal,
                "R3x1",
                vectors,
                settings,
            )
            self.assertEqual(result, normal / PSF_COEFFICIENT_PLOT_NAME)
            self.assertTrue(result.is_file())

    def test_psf_coefficient_plot_records_processed_vectors_and_fit_range(self) -> None:
        """Verify the diagnostic overlays raw samples on fitted curves.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / PSF_COEFFICIENT_PLOT_NAME
            kx = np.arange(32, dtype=np.float64)
            fitted = (np.sin(kx / 5.0), np.cos(kx / 6.0), 0.01 * kx)
            raw = tuple(value + 0.05 * np.sin(kx) for value in fitted)
            from matplotlib.axes import Axes

            original_scatter = Axes.scatter
            with patch.object(
                Axes,
                "scatter",
                autospec=True,
                side_effect=original_scatter,
            ) as scatter:
                result = write_psf_coefficient_plot(
                    *fitted,
                    destination,
                    processing="sine-line",
                    fit_kx_range=(4, 28),
                    raw_coefficients=raw,
                )
            self.assertEqual(scatter.call_count, 3)
            for index, call in enumerate(scatter.call_args_list):
                np.testing.assert_array_equal(call.args[1], kx)
                np.testing.assert_allclose(call.args[2], raw[index])
            self.assertEqual(result, destination.resolve())
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(destination.stat().st_size, 1000)

            with self.assertRaisesRegex(ValueError, "within the readout"):
                write_psf_coefficient_plot(
                    kx,
                    kx,
                    kx,
                    destination,
                    processing="sine-line",
                    fit_kx_range=(4, 40),
                )
            with self.assertRaisesRegex(ValueError, "match the processed"):
                write_psf_coefficient_plot(
                    *fitted,
                    destination,
                    processing="sine-line",
                    raw_coefficients=(kx, kx, kx[:-1]),
                )

    def test_hybrid_plot_labels_smooth_c_and_rejected_constrained_candidate(
        self,
    ) -> None:
        """Verify a smooth-c fallback is explicit in the accepted PSF PNG.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            normal = Path(folder) / "normal"
            kx = np.arange(32, dtype=np.float64)
            vectors = (np.sin(kx), np.cos(kx), 0.01 * kx)
            diagnostics = {
                "effective_coefficient_processing": "sine_line_ab_smooth_c",
                "automatic_c_recovery": {
                    "outcome": "smooth_c_fallback",
                    "constrained_c_fit": {
                        "A": 0.1,
                        "w": 0.2,
                        "phi": 0.3,
                        "C1": 0.001,
                        "C2": 0.0,
                        "validation_passed": False,
                    },
                },
            }
            from matplotlib.axes import Axes

            original_plot = Axes.plot
            with patch.object(
                Axes,
                "plot",
                autospec=True,
                side_effect=original_plot,
            ) as plot:
                result = _ensure_r3x1_psf_coefficient_plot(
                    normal,
                    "R3x1",
                    vectors,
                    {
                        "coefficient_processing": "sine-line",
                        "fit_range_selection": "automatic",
                        "fit_kx_range": [4, 28],
                    },
                    raw_coefficient_vectors=vectors,
                    processing_diagnostics=diagnostics,
                    overwrite=True,
                )

            labels = [call.kwargs.get("label") for call in plot.call_args_list]
            self.assertIn("accepted 9-point smooth fallback", labels)
            self.assertIn("rejected constrained c fit", labels)
            self.assertIsNotNone(result)
            self.assertTrue(result.is_file())

    def test_psf_plot_uses_fixed_phase_range(self) -> None:
        """Verify blow-up values cannot expand the diagnostic phase range.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / PSF_COEFFICIENT_PLOT_NAME
            values = np.zeros(32, dtype=np.float64)
            values[0] = 1.0e6
            from matplotlib.axes import Axes

            original_set_ylim = Axes.set_ylim
            with patch.object(
                Axes,
                "set_ylim",
                autospec=True,
                side_effect=original_set_ylim,
            ) as set_ylim:
                write_psf_coefficient_plot(
                    values,
                    values,
                    values,
                    destination,
                    processing="smooth",
                )

            fixed_calls = [
                call
                for call in set_ylim.call_args_list
                if call.args[1:] == (-2.0 * np.pi, 2.0 * np.pi)
            ]
            self.assertEqual(len(fixed_calls), 3)
            self.assertTrue(destination.is_file())

    def test_psf_coefficient_settings_preserve_upstream_modes(self) -> None:
        """Verify smooth and half-open sine-line settings are validated.

        Returns:
            None.
        """
        self.assertEqual(
            _normalize_psf_coefficient_settings("smooth", None, None),
            {
                "coefficient_processing": "smooth",
                "fit_range_selection": None,
                "requested_fit_kx_range": None,
                "fit_kx_range_convention": "half-open",
            },
        )
        self.assertEqual(
            _normalize_psf_coefficient_settings("sine-line", None, None),
            {
                "coefficient_processing": "sine-line",
                "fit_range_selection": "automatic",
                "requested_fit_kx_range": None,
                "fit_kx_range_convention": "half-open",
            },
        )
        self.assertEqual(
            _normalize_psf_coefficient_settings("sine-line", 12, 180),
            {
                "coefficient_processing": "sine-line",
                "fit_range_selection": "manual",
                "requested_fit_kx_range": [12, 180],
                "fit_kx_range_convention": "half-open",
            },
        )
        with self.assertRaisesRegex(ValueError, "only with sine-line"):
            _normalize_psf_coefficient_settings("smooth", 12, 180)
        with self.assertRaisesRegex(ValueError, "requires both"):
            _normalize_psf_coefficient_settings("sine-line", 12, None)
        with self.assertRaisesRegex(ValueError, "0 <= min < max"):
            _normalize_psf_coefficient_settings("sine-line", 12, 12)

    def test_sine_line_settings_reach_upstream_processing(self) -> None:
        """Verify the selected mode and kx bounds reach the upstream helper.

        Returns:
            None.
        """
        raw_a = np.arange(8, dtype=np.float64)
        raw_b = raw_a + 1
        raw_c = raw_a + 2
        quality = {"combined_support": np.ones(8, dtype=np.float64)}
        diagnostics = {
            "coefficient_processing": "sine-line",
            "fit_range_selection": "manual",
            "kx_range": [1, 7],
            "kx_range_convention": "half-open [min, max)",
        }
        native = Mock()
        native.fit_wave_psf_deviation_from_projection.return_value = (
            raw_a,
            raw_b,
            raw_c,
            320,
            {"projection_quality": quality},
        )
        native._process_psf_coefficients.return_value = (
            raw_a,
            raw_b,
            raw_c,
            diagnostics,
        )
        native.generate_theoretical_wave_trajectory.return_value = (raw_a, raw_b)

        result = _calibrated_psf_inputs(
            native,
            twix_path=Path("input.dat"),
            sequence_path=Path("input.seq"),
            readout_oversampled=8,
            ncalib=4,
            nacs=4,
            coefficient_processing="sine-line",
            fit_kx_min=1,
            fit_kx_max=7,
        )

        self.assertEqual(len(result), 7)
        for observed, expected in zip(result[-2], (raw_a, raw_b, raw_c), strict=True):
            np.testing.assert_array_equal(observed, expected)
        self.assertIs(result[-1], diagnostics)
        calibration_call = native.fit_wave_psf_deviation_from_projection.call_args
        self.assertTrue(calibration_call.kwargs["return_diagnostics"])
        processing_call = native._process_psf_coefficients.call_args
        self.assertEqual(processing_call.kwargs["coefficient_processing"], "sine-line")
        self.assertEqual(processing_call.kwargs["fit_kx_min"], 1)
        self.assertEqual(processing_call.kwargs["fit_kx_max"], 7)
        self.assertIs(processing_call.kwargs["fit_quality"], quality)
        self.assertTrue(processing_call.kwargs["return_diagnostics"])

        native.reset_mock()
        native.fit_wave_psf_deviation_from_projection.return_value = (
            raw_a,
            raw_b,
            raw_c,
            320,
            {"projection_quality": quality},
        )
        native._process_psf_coefficients.return_value = (
            raw_a,
            raw_b,
            raw_c,
            {**diagnostics, "fit_range_selection": "automatic"},
        )
        native.generate_theoretical_wave_trajectory.return_value = (raw_a, raw_b)
        _calibrated_psf_inputs(
            native,
            twix_path=Path("input.dat"),
            sequence_path=Path("input.seq"),
            readout_oversampled=8,
            ncalib=4,
            nacs=4,
            coefficient_processing="sine-line",
            fit_kx_min=None,
            fit_kx_max=None,
        )
        automatic_call = native._process_psf_coefficients.call_args
        self.assertIsNone(automatic_call.kwargs["fit_kx_min"])
        self.assertIsNone(automatic_call.kwargs["fit_kx_max"])
        self.assertIs(automatic_call.kwargs["fit_quality"], quality)

    def test_automatic_sine_line_rejection_retains_candidate_and_raw_samples(
        self,
    ) -> None:
        """Verify failed validation returns the exact upstream candidate evidence.

        Returns:
            None.
        """
        raw = tuple(np.linspace(index, index + 1, 8) for index in range(3))
        diagnostics = {
            "model": "A*sin(w*kx+phi)+C1*kx+C2",
            "kx_range": [1, 7],
            "fit_range_selection": "automatic",
            "validation_passed": False,
            "coefficients": {
                name: {
                    "A": 0.5 + index,
                    "w": 0.2,
                    "phi": 0.1,
                    "C1": 0.01,
                    "C2": -0.2,
                    "validation_passed": False,
                }
                for index, name in enumerate(("a", "b", "c"))
            },
        }

        def reject_candidate(*args: object, **kwargs: object) -> None:
            """Write mock upstream diagnostics and reject the candidate.

            Args:
                args: Unused positional coefficient arrays.
                kwargs: Upstream processing options with diagnostics location.

            Returns:
                None.

            Raises:
                ValueError: Always, to emulate upstream validation rejection.
            """
            destination = Path(str(kwargs["out_folder"])) / (
                f"psf_sine_line_fit_{kwargs['file_tag']}.json"
            )
            destination.write_text(json.dumps(diagnostics), encoding="utf-8")
            raise ValueError("automatic candidate rejected")

        native = Mock()
        native.fit_wave_psf_deviation_from_projection.return_value = (
            *raw,
            320,
            {"projection_quality": {"combined_support": np.ones(8)}},
        )
        native._process_psf_coefficients.side_effect = reject_candidate
        native.generate_theoretical_wave_trajectory.return_value = (raw[0], raw[1])

        with self.assertRaises(AutomaticPsfFitRejected) as context:
            _calibrated_psf_inputs(
                native,
                twix_path=Path("input.dat"),
                sequence_path=Path("input.seq"),
                readout_oversampled=8,
                ncalib=4,
                nacs=4,
                coefficient_processing="sine-line",
                fit_kx_min=None,
                fit_kx_max=None,
            )
        rejection = context.exception
        self.assertEqual(rejection.diagnostics, diagnostics)
        for observed, expected in zip(rejection.raw_coefficients, raw, strict=True):
            np.testing.assert_array_equal(observed, expected)
        self.assertIsNotNone(rejection.candidate_coefficients)
        kx = np.arange(8, dtype=np.float64)
        for index, observed in enumerate(rejection.candidate_coefficients or ()):
            expected = (0.5 + index) * np.sin(0.2 * kx + 0.1) + 0.01 * kx - 0.2
            np.testing.assert_allclose(observed, expected)

    def test_rejected_fit_diagnostics_do_not_populate_bart_inputs(self) -> None:
        """Verify rejected PNG/JSON permit a later manual preparation rerun.

        Returns:
            None.
        """
        raw = tuple(np.linspace(index, index + 1, 8) for index in range(3))
        rejection = AutomaticPsfFitRejected(
            "automatic candidate rejected",
            raw_coefficients=raw,
            candidate_coefficients=None,
            diagnostics={
                "kx_range": [1, 7],
                "fit_range_selection": "automatic",
                "validation_passed": False,
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            normal = root / "normal"
            bart_inputs = normal / "bart_inputs"
            bart_inputs.mkdir(parents=True)
            twix = root / "input.dat"
            sequence = root / "input.seq"
            twix.write_bytes(b"twix")
            sequence.write_text("sequence\n", encoding="utf-8")
            plot_path, json_path = _write_automatic_psf_rejection_diagnostics(
                normal,
                rejection,
                twix_path=twix,
                sequence_path=sequence,
            )

            self.assertEqual(plot_path, normal / PSF_COEFFICIENT_REJECTED_PLOT_NAME)
            self.assertEqual(
                json_path, normal / PSF_COEFFICIENT_REJECTED_DIAGNOSTICS_NAME
            )
            self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            record = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "automatic_sine_line_psf_fit_rejected")
            self.assertFalse(record["accepted_for_reconstruction"])
            self.assertEqual(
                record["manual_override"]["required_arguments"],
                ["--psf-fit-kx-min", "--psf-fit-kx-max"],
            )
            self.assertFalse(any(bart_inputs.iterdir()))

    def test_calibrated_inputs_return_an_accepted_c_only_recovery(self) -> None:
        """Verify an accepted recovery continues the normal PSF input path.

        Returns:
            None.
        """
        raw = tuple(np.linspace(index, index + 1, 8) for index in range(3))
        trajectory = (np.zeros(8), np.ones(8))
        accepted_diagnostics = {
            "validation_passed": True,
            "effective_coefficient_processing": "sine_line_ab_smooth_c",
        }
        native = Mock()
        native.fit_wave_psf_deviation_from_projection.return_value = (
            *raw,
            320,
            {"projection_quality": {}},
        )
        native.generate_theoretical_wave_trajectory.return_value = trajectory
        native._process_psf_coefficients.side_effect = ValueError("c rejected")

        with patch(
            "wave_retro_lr.mprage._recover_c_only_automatic_rejection",
            return_value=(raw, accepted_diagnostics),
        ) as recover:
            result = _calibrated_psf_inputs(
                native,
                twix_path=Path("input.dat"),
                sequence_path=Path("input.seq"),
                readout_oversampled=8,
                ncalib=4,
                nacs=4,
                coefficient_processing="sine-line",
                fit_kx_min=None,
                fit_kx_max=None,
            )

        recover.assert_called_once()
        self.assertIs(result[-1], accepted_diagnostics)
        for observed, expected in zip(result[2:5], raw, strict=True):
            np.testing.assert_array_equal(observed, expected)

    def test_c_only_rejection_uses_common_frequency_when_relaxed_gates_pass(
        self,
    ) -> None:
        """Verify strict a/b fits can anchor an accepted constrained c fit.

        Returns:
            None.
        """
        readout_size = 128
        kx = np.arange(readout_size, dtype=np.float64)
        frequency = 2.0 * np.pi * 8.0 / readout_size
        raw = (
            np.sin(frequency * kx + 0.1),
            0.6 * np.sin(frequency * kx - 0.3),
            0.08 * np.sin(frequency * kx + 0.7) + 0.0002 * kx,
        )
        diagnostics = {
            "kx_range": [8, 120],
            "range_selection_diagnostics": {
                "excluded_sample_indices_within_interval": []
            },
            "validation_passed": False,
            "coefficients": {
                "a": {
                    "A": 1.0,
                    "w": frequency * 0.995,
                    "phi": 0.1,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": True,
                },
                "b": {
                    "A": 0.6,
                    "w": frequency * 1.005,
                    "phi": -0.3,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": True,
                },
                "c": {
                    "A": 0.02,
                    "w": 2.0 * np.pi / readout_size,
                    "phi": 0.0,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": False,
                },
            },
        }
        native = Mock()
        native.AUTO_FIT_PREFILTER_WINDOW = 9
        native.smooth_1d_nan.side_effect = lambda values, window: values
        recovery = _recover_c_only_automatic_rejection(
            native,
            raw_coefficients=raw,
            rejected_diagnostics=diagnostics,
            delta_lin=np.sin(frequency * kx),
            delta_par=np.cos(frequency * kx),
        )

        self.assertIsNotNone(recovery)
        coefficients, accepted = recovery or ((), {})
        self.assertEqual(
            accepted["effective_coefficient_processing"],
            "sine_line_ab_constrained_common_frequency_c",
        )
        self.assertEqual(
            accepted["automatic_c_recovery"]["outcome"],
            "constrained_common_frequency_c",
        )
        self.assertTrue(accepted["validation_passed"])
        np.testing.assert_allclose(coefficients[2], raw[2], atol=1e-12)

    def test_c_only_rejection_falls_back_to_nine_point_smooth(self) -> None:
        """Verify failure of relaxed c gates accepts an explicit smooth hybrid.

        Returns:
            None.
        """
        readout_size = 128
        kx = np.arange(readout_size, dtype=np.float64)
        frequency = 2.0 * np.pi * 8.0 / readout_size
        raw = (
            np.sin(frequency * kx),
            0.5 * np.cos(frequency * kx),
            0.05 * np.sin(frequency * kx + 0.4) + 0.001 * kx,
        )
        diagnostics = {
            "kx_range": [8, 120],
            "range_selection_diagnostics": {
                "excluded_sample_indices_within_interval": []
            },
            "validation_passed": False,
            "coefficients": {
                "a": {
                    "A": 1.0,
                    "w": frequency,
                    "phi": 0.0,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": True,
                },
                "b": {
                    "A": 0.5,
                    "w": frequency,
                    "phi": np.pi / 2.0,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": True,
                },
                "c": {
                    "A": 0.01,
                    "w": 2.0 * np.pi / readout_size,
                    "phi": 0.0,
                    "C1": 0.0,
                    "C2": 0.0,
                    "validation_passed": False,
                },
            },
        }
        native = Mock()
        native.AUTO_FIT_PREFILTER_WINDOW = 9
        native.smooth_1d_nan.side_effect = lambda values, window: values
        with patch(
            "wave_retro_lr.mprage.C_FIXED_FREQUENCY_MAXIMUM_CONDITION_NUMBER",
            -1.0,
        ):
            recovery = _recover_c_only_automatic_rejection(
                native,
                raw_coefficients=raw,
                rejected_diagnostics=diagnostics,
                delta_lin=np.sin(frequency * kx),
                delta_par=np.cos(frequency * kx),
            )

        self.assertIsNotNone(recovery)
        coefficients, accepted = recovery or ((), {})
        self.assertEqual(
            accepted["effective_coefficient_processing"],
            "sine_line_ab_smooth_c",
        )
        recovery_record = accepted["automatic_c_recovery"]
        self.assertEqual(recovery_record["outcome"], "smooth_c_fallback")
        self.assertEqual(recovery_record["smooth_c_window_samples"], 9)
        self.assertFalse(recovery_record["fallback_was_silent"])
        np.testing.assert_array_equal(coefficients[2], raw[2])

    def test_target_matrices_are_nearest_multiple_of_four(self) -> None:
        """Verify requested LR spacings resolve to compatible PE matrices.

        Returns:
            None.
        """
        geometry = Geometry((256.0, 256.0, 256.0), (256, 256, 256))
        expected = {
            (1.5, 1.0, 1.0): (256, 256, 172),
            (1.0, 1.5, 1.0): (256, 172, 256),
            (1.25, 1.25, 1.0): (256, 204, 204),
        }
        for resolution, shape in expected.items():
            case = resolve_case(CaseSpec(resolution, (3, 2)), geometry)
            self.assertEqual(case.target_logical_matrix_ro_lin_par, shape)
            self.assertEqual(shape[1] % 4, 0)
            self.assertEqual(shape[2] % 4, 0)

    def test_psf_uses_trajectory_and_coefficients_on_requested_grid(self) -> None:
        """Verify direct PSF evaluation preserves shape and unit magnitude.

        Returns:
            None.
        """
        delta_lin = np.array([0.0, 0.25])
        delta_par = np.array([0.0, -0.5])
        a_fit = np.array([0.0, 0.1])
        b_fit = np.array([0.0, -0.2])
        c_fit = np.array([0.3, -0.4])
        psf = evaluate_calibrated_psf(
            delta_lin, delta_par, a_fit, b_fit, c_fit, nlin=4, npar=8
        )
        self.assertEqual(psf.shape, (2, 4, 8))
        np.testing.assert_allclose(np.abs(psf), 1.0, atol=2e-7)
        self.assertAlmostEqual(float(np.angle(psf[0, 2, 4])), 0.3, places=6)


class BartInputTests(unittest.TestCase):
    def test_measured_wave_crop_does_not_forward_simulate(self) -> None:
        """Verify measured-Wave LR preparation is a direct centered crop.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = create_cfl(root / "wave", (4, 8, 8, 1, 1))
            values = np.arange(4 * 8 * 8, dtype=np.float32).reshape((4, 8, 8), order="F")
            source[:, :, :, 0, 0] = values
            source.flush()
            del source
            geometry = Geometry((8.0, 8.0, 4.0), (4, 8, 8))
            case = resolve_case(CaseSpec((2.0, 1.0, 1.0), (1, 1)), geometry)
            metrics = write_measured_wave_crop(
                root / "wave", root / "cropped", case, np.ones((8, 8), bool), (1, 1)
            )
            result = np.asarray(open_cfl(root / "cropped"))[:, :, :, 0, 0]
            np.testing.assert_array_equal(result, values[:, :, 2:6])
            self.assertGreater(float(metrics["wave_kspace_norm"]), 0.0)

    def test_measured_wave_crop_preserves_r3_residue_and_adds_par_two(self) -> None:
        """Verify measured LIN residue is retained while PARx2 is added.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = create_cfl(root / "wave", (4, 8, 8, 1, 1))
            source[...] = 0
            source[:, (0, 3, 6), :, 0, 0] = 1
            source.flush()
            del source
            geometry = Geometry((8.0, 8.0, 4.0), (4, 8, 8))
            case = resolve_case(CaseSpec((1.0, 1.0, 1.0), (3, 2)), geometry)
            source_mask = np.zeros((8, 8), dtype=bool)
            source_mask[(0, 3, 6), :] = True
            metrics = write_measured_wave_crop(
                root / "wave", root / "r3x2", case, source_mask, (3, 1)
            )
            result = np.asarray(open_cfl(root / "r3x2"))[:, :, :, 0, 0]
            expected = np.zeros((8, 8), dtype=bool)
            expected[np.ix_([0, 3, 6], [0, 2, 4, 6])] = True
            np.testing.assert_array_equal(np.any(result != 0, axis=0), expected)
            self.assertEqual(metrics["sampled_coordinate_count"], 12)
            self.assertFalse(metrics["image_kspace_center_acquired"])

    def test_legacy_mask_does_not_infer_acs_from_fully_sampled_rows(self) -> None:
        """Verify legacy masking never treats full image rows as ACS.

        Returns:
            None.
        """
        geometry = Geometry((8.0, 8.0, 4.0), (4, 8, 8))
        case = resolve_case(CaseSpec((1.0, 1.0, 1.0), (3, 2)), geometry)
        result = build_case_mask(np.ones((8, 8), bool), case, (1, 1))
        expected = np.zeros((8, 8), dtype=bool)
        expected[np.ix_([1, 4, 7], [0, 2, 4, 6])] = True
        np.testing.assert_array_equal(result, expected)

    def test_lr_maps_keep_readout_and_are_rss_normalized(self) -> None:
        """Verify same-FOV CSM resizing changes PE only and normalizes coils.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bart_shape = (4, 8, 8, 2, 1) + (1,) * 11
            maps = create_cfl(root / "maps", bart_shape)
            maps[...] = np.complex64(1 / np.sqrt(2))
            maps.flush()
            del maps
            resample_sensitivity_maps(root / "maps", root / "small", target_lin_par=(4, 4))
            self.assertEqual(read_shape(root / "small"), (4, 4, 4, 2, 1))
            result = np.asarray(open_cfl(root / "small"))[:, :, :, :, 0]
            rss = np.sqrt(np.sum(np.abs(result) ** 2, axis=3))
            np.testing.assert_allclose(rss, 1.0, atol=1e-6)

    def test_no_wave_operation_is_explicitly_separate(self) -> None:
        """Verify synthetic no-Wave encoding remains an explicit utility.

        Returns:
            None.
        """
        no_wave = np.ones((4, 4, 4), dtype=np.complex64)
        psf = np.ones((4, 4, 4), dtype=np.complex64)
        result = synthesize_wave_from_no_wave_crop(
            no_wave, psf, readout_oversampled=4, target_mask=np.eye(4, dtype=bool)
        )
        self.assertEqual(result.shape, (4, 4, 4))
        self.assertTrue(np.all(result[:, ~np.eye(4, dtype=bool)] == 0))

    def test_legacy_bart_options_still_require_gpu(self) -> None:
        """Verify compatibility BART Wave options retain GPU execution.

        Returns:
            None.
        """
        options = build_wave_options(
            "wavelet", 0.0, block_size=8, iterations=100, tolerance=1e-6, maximum_eigenvalue=None
        )
        self.assertEqual(options, ["-w", "-f", "-r", "0", "-i", "100", "-t", "1e-06", "-g"])


class PreparationIntegrationTests(unittest.TestCase):
    def test_mock_twix_preparation_writes_native_and_four_retro_contracts(self) -> None:
        """Verify mocked raw preparation writes all required BART contracts.

        Returns:
            None.
        """
        import torch

        class Helpers:
            @staticmethod
            def _resolve_mprage_wave_mode(*args, **kwargs):
                """Return the expected Wave acquisition mode for the mock.

                Args:
                    *args: Ignored positional upstream arguments.
                    **kwargs: Ignored keyword upstream arguments.

                Returns:
                    The literal ``"wave"`` mode.
                """
                return "wave"

            @staticmethod
            def load_img(path):
                """Return a finite R3x1 mock image stream on the full grid.

                Args:
                    path: Ignored mock TWIX path.

                Returns:
                    A finite complex image-stream tensor with R3x1 support.
                """
                image = torch.zeros((8, 16, 16, 12), dtype=torch.complex64)
                image[:, (1, 4, 7, 10, 13), :, :] = 1
                return image

            @staticmethod
            def load_ref(path):
                """Return a five-set mock integrated reference stream.

                Args:
                    path: Ignored mock TWIX path.

                Returns:
                    A finite five-set complex reference tensor.
                """
                return torch.ones((8, 4, 4, 5, 12), dtype=torch.complex64)

            @staticmethod
            def _check_integrated_refscan_shape(reference, **kwargs):
                """Validate the expected mock integrated-reference shape.

                Args:
                    reference: Mock integrated-reference tensor.
                    **kwargs: Ignored upstream validation settings.

                Returns:
                    None.
                """
                if tuple(reference.shape) != (8, 4, 4, 5, 12):
                    raise ValueError("bad mock reference")

            @staticmethod
            def estimate_cc_matrix_coillast(*args, **kwargs):
                """Return an identity mock coil-compression basis and spectra.

                Args:
                    *args: Ignored positional upstream arguments.
                    **kwargs: Ignored keyword upstream arguments.

                Returns:
                    Identity basis, singular values, and retained-energy arrays.
                """
                return (
                    np.eye(12, dtype=np.complex64),
                    np.ones(12, dtype=np.float64),
                    np.ones(12, dtype=np.float64),
                )

            @staticmethod
            def apply_cc_coillast_torch(kspace, basis, x_chunk=8):
                """Apply the supplied mock coil-compression matrix.

                Args:
                    kspace: Input complex k-space tensor.
                    basis: Coil-compression matrix.
                    x_chunk: Ignored upstream chunk size.

                Returns:
                    The coil-compressed tensor.
                """
                return kspace @ torch.as_tensor(basis)

        pattern = SamplingPattern(
            name="R3x1",
            acceleration_lin_par=(3, 1),
            lin_residue=1,
            matrix_lin_par=(16, 16),
            acquired_lin=(1, 4, 7, 10, 13),
            acquired_par=tuple(range(16)),
            measurement_index=0,
        )
        definitions = {
            "ReadoutOversamplingFactor": 2,
            "Calibration_Ncalib1": 4,
            "Calibration_Nacs": 4,
        }
        upstream_geometry = {
            "Nro": 4,
            "Nlin": 16,
            "Npar": 16,
            "FOVxyz": (0.016, 0.016, 0.004),
        }
        zero_vectors = tuple(np.zeros(8, dtype=np.float64) for _ in range(5))
        processing_diagnostics = {
            "coefficient_processing": "smooth",
            "fit_range_selection": None,
            "kx_range": None,
            "kx_range_convention": "half-open [min, max)",
        }

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            twix = root / "input.dat"
            sequence = root / "input.seq"
            twix.write_bytes(b"mock twix")
            sequence.write_text("mock sequence\n", encoding="utf-8")
            output = root / "output"
            with (
                patch("wave_retro_lr.mprage.load_wave_mprage_helpers", return_value=Helpers()),
                patch(
                    "wave_retro_lr.mprage._read_sequence",
                    return_value=(definitions, upstream_geometry),
                ),
                patch(
                    "wave_retro_lr.mprage.inspect_twix_sampling",
                    return_value=(pattern, object()),
                ),
                patch(
                    "wave_retro_lr.mprage._calibrated_psf_inputs",
                    return_value=(
                        *zero_vectors,
                        zero_vectors[2:],
                        processing_diagnostics,
                    ),
                ),
            ):
                manifest = prepare_normal_mprage(twix, output, sequence)
                diagnostic = output / "normal" / PSF_COEFFICIENT_PLOT_NAME
                self.assertTrue(diagnostic.is_file())
                diagnostic.unlink()
                retro = prepare_retro_mprage(twix, output, sequence)

            normal = output / "normal" / "bart_inputs"
            self.assertEqual(manifest["sampling"]["name"], "R3x1")
            self.assertEqual(
                manifest["psf_calibration"]["coefficient_processing"], "smooth"
            )
            self.assertIsNone(manifest["psf_calibration"]["fit_kx_range"])
            self.assertIsNone(manifest["psf_calibration"]["fit_range_selection"])
            self.assertIsNone(
                manifest["psf_calibration"]["requested_fit_kx_range"]
            )
            self.assertEqual(
                manifest["psf_calibration"]["processing_diagnostics"],
                processing_diagnostics,
            )
            self.assertEqual(
                manifest["psf_calibration"][
                    "visual_assessment_plot_relative_to_output_root"
                ],
                f"normal/{PSF_COEFFICIENT_PLOT_NAME}",
            )
            self.assertTrue(diagnostic.is_file())
            self.assertEqual(read_shape(normal / "wave_kspace"), (8, 16, 16, 12, 1))
            self.assertEqual(read_shape(normal / "kspace_calib"), (4, 16, 16, 12))
            self.assertEqual(read_shape(normal / "psf"), (8, 16, 16, 1, 1))
            self.assertEqual(read_shape(normal / "psf_coefficients_raw"), (8, 3))
            self.assertEqual(
                manifest["psf_calibration"]["raw_psf_coefficients"],
                "psf_coefficients_raw",
            )
            self.assertEqual(len(retro), 4)
            for case in retro:
                shape = tuple(case["case"]["target_logical_matrix_ro_lin_par"])
                self.assertEqual(shape[1] % 4, 0)
                self.assertEqual(shape[2] % 4, 0)
                case_inputs = output / "retro" / case["case_directory"] / "bart_inputs"
                self.assertEqual(
                    read_shape(case_inputs / "wave_kspace")[:3],
                    (8, shape[1], shape[2]),
                )
                self.assertEqual(read_shape(case_inputs / "psf")[:3], (8, shape[1], shape[2]))

            with self.assertRaisesRegex(ValueError, "PSF coefficient-processing"):
                prepare_normal_mprage(
                    twix,
                    output,
                    sequence,
                    psf_coefficient_processing="sine-line",
                    psf_fit_kx_min=1,
                    psf_fit_kx_max=7,
                )


if __name__ == "__main__":
    unittest.main()
