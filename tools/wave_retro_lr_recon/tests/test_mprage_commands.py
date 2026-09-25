"""Static interface tests for the readable MPRAGE sample workflows."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.nifti_collection import HeadMaskParameters  # noqa: E402
from wave_retro_lr.mprage import (  # noqa: E402
    R3X3_WAVELET_LAMBDA,
    prepare_normal_mprage,
    prepare_retro_mprage,
    prepare_retro_mprage_r3x3,
)
from wave_retro_lr.rovir_workflow import (  # noqa: E402
    _single_magnitude,
    finalize_existing_normal_rovir,
    normal_rovir_context,
    validate_normal_rovir_invocation,
)
from scripts.prepare_mprage_normal import _parser as normal_parser  # noqa: E402
from scripts.prepare_mprage_retro import _parser as retro_parser  # noqa: E402
from scripts.prepare_mprage_retro_r3x3 import _parser as r3x3_parser  # noqa: E402

SCRIPTS = TOOL_ROOT / "scripts"


class SampleCommandTests(unittest.TestCase):
    def test_normal_script_keeps_bart_commands_explicit(self) -> None:
        """Verify normal ecalib and explicit dual-branch Wave commands.

        Returns:
            None.
        """
        source = (SCRIPTS / "sample_mprage_normal_recon.sh").read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertEqual(sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1)
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 3)
        self.assertEqual(sum(line.startswith("bart wave -w ") for line in commands), 3)
        self.assertIn('ECALIB_CROP="0.6"', source)
        self.assertIn('R3_LAMBDA="3.5e-2"', source)
        self.assertIn("USE_GPU=false", source)
        self.assertIn("-g) USE_GPU=true; shift ;;", source)
        self.assertIn('PSF_COEFFICIENT_PROCESSING="sine-line"', source)
        self.assertIn("--psf-coefficient-processing sine-line", source)
        self.assertIn('--psf-fit-kx-min "$PSF_FIT_KX_MIN"', source)
        self.assertIn('--psf-fit-kx-max "$PSF_FIT_KX_MAX"', source)
        self.assertIn('--psf-fit-y-min "$PSF_FIT_Y_MIN"', source)
        self.assertIn('--psf-fit-y-max "$PSF_FIT_Y_MAX"', source)
        self.assertIn('--psf-fit-z-min "$PSF_FIT_Z_MIN"', source)
        self.assertIn('--psf-fit-z-max "$PSF_FIT_Z_MAX"', source)
        self.assertEqual(sum(line.startswith("bart wave -g -w -f -r 0 ") for line in commands), 2)
        self.assertEqual(sum(line.startswith("bart wave -w -f -r 0 ") for line in commands), 2)
        self.assertIn('bart wave -g -w -f -r "$R3_LAMBDA" ', source)
        self.assertIn('bart wave -w -f -r "$R3_LAMBDA" ', source)
        self.assertIn('ECALIB_RECORD="$BART_OUTPUT_ROOT/ecalib_command.txt"', source)
        self.assertIn('$BART_OUTPUT_ROOT/fista_r0/wave_command.txt"', source)
        self.assertIn('$BART_OUTPUT_ROOT/optimal_wavelet/wave_command.txt"', source)
        self.assertNotIn("build_mprage_nifti_collection.py", source)
        self.assertIn("PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png", source)
        self.assertIn("TROUBLESHOOTING.md", source)

    def test_retro_script_has_one_ecalib_and_eight_wave_commands(self) -> None:
        """Verify one ecalib and explicit CPU/GPU Wave branches per case.

        Returns:
            None.
        """
        source = (SCRIPTS / "sample_mprage_retro_lr_recon.sh").read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertEqual(sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1)
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 8)
        self.assertEqual(sum(line.startswith("bart wave -w ") for line in commands), 8)
        self.assertIn("USE_GPU=false", source)
        self.assertIn("-g) USE_GPU=true; shift ;;", source)
        self.assertIn('PSF_COEFFICIENT_PROCESSING="sine-line"', source)
        self.assertIn("--psf-coefficient-processing sine-line", source)
        self.assertIn('--psf-fit-kx-min "$PSF_FIT_KX_MIN"', source)
        self.assertIn('--psf-fit-kx-max "$PSF_FIT_KX_MAX"', source)
        self.assertIn('--psf-fit-y-min "$PSF_FIT_Y_MIN"', source)
        self.assertIn('--psf-fit-y-max "$PSF_FIT_Y_MAX"', source)
        self.assertIn('--psf-fit-z-min "$PSF_FIT_Z_MIN"', source)
        self.assertIn('--psf-fit-z-max "$PSF_FIT_Z_MAX"', source)
        self.assertEqual(sum(line.startswith("bart wave -g -w -f -r 0 ") for line in commands), 4)
        self.assertEqual(sum(line.startswith("bart wave -w -f -r 0 ") for line in commands), 4)
        self.assertEqual(sum("bart wave -g -w -f -r 3.5e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -w -f -r 3.5e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -g -w -f -r 2.5e-2 " in line for line in commands), 2)
        self.assertEqual(sum("bart wave -w -f -r 2.5e-2 " in line for line in commands), 2)
        self.assertEqual(sum("bart wave -g -w -f -r 2.2e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -w -f -r 2.2e-2 " in line for line in commands), 1)
        self.assertNotIn("bart ecalib -g", source)
        self.assertEqual(source.count('wave_command.txt"'), 8)
        self.assertEqual(source.count("bart_output/fista_r0/image_wave"), 20)
        self.assertEqual(source.count("bart_output/optimal_wavelet/image_wave"), 20)
        self.assertNotIn("sample_mprage_normal_recon.sh", source)
        self.assertNotIn("build_mprage_nifti_collection.py", source)
        self.assertEqual(source.count("sample_mprage_retro_r3x3_recon.sh"), 1)
        self.assertIn(
            'R3X3_ARGS+=(--psf-fit-kx-min "$PSF_FIT_KX_MIN"', source
        )
        self.assertIn('R3X3_ARGS+=(-g)', source)
        self.assertIn("PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png", source)
        self.assertIn("TROUBLESHOOTING.md", source)

    def test_r3x3_script_is_independent_and_uses_locked_wavelet(self) -> None:
        """Verify the tracked R3x3 entry point runs only two explicit branches.

        Returns:
            None.
        """
        source = (SCRIPTS / "sample_mprage_retro_r3x3_recon.sh").read_text(
            encoding="utf-8"
        )
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertEqual(R3X3_WAVELET_LAMBDA, 0.045)
        self.assertIn('R3X3_LAMBDA="4.5e-2"', source)
        self.assertEqual(sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1)
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 2)
        self.assertEqual(sum(line.startswith("bart wave -w ") for line in commands), 2)
        self.assertEqual(sum("-r 0 " in line for line in commands), 2)
        self.assertEqual(sum('-r "$R3X3_LAMBDA" ' in line for line in commands), 2)
        self.assertIn("prepare_mprage_retro_r3x3.py", source)
        self.assertNotIn("prepare_mprage_retro.py", source)
        self.assertNotIn("prepare_mprage_retro_maps.py", source)
        self.assertNotIn("native_r3x2", source)

    def test_samples_parse_and_offer_dataset_independent_help(self) -> None:
        """Verify all Bash samples parse and expose path-agnostic help.

        Returns:
            None.
        """
        samples = {
            "sample_mprage_normal_recon.sh": "TWIX.dat OUTPUT_ROOT SEQUENCE.seq",
            "sample_mprage_retro_lr_recon.sh": "TWIX.dat OUTPUT_ROOT SEQUENCE.seq",
            "sample_mprage_retro_r3x3_recon.sh": "TWIX.dat OUTPUT_ROOT SEQUENCE.seq",
            "sample_mprage_nifti_collection.sh": "OUTPUT_ROOT",
        }
        for name, expected_help in samples.items():
            path = SCRIPTS / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            completed = subprocess.run(
                ["bash", str(path), "--help"], check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected_help, completed.stdout)
            if name != "sample_mprage_nifti_collection.sh":
                self.assertIn("[-g]", completed.stdout)
                self.assertIn("--psf-coefficient-processing", completed.stdout)
                self.assertIn("--psf-fit-kx-min", completed.stdout)
                self.assertIn("--psf-fit-y-min", completed.stdout)
                self.assertIn("--psf-fit-z-min", completed.stdout)

        defaults = HeadMaskParameters()
        self.assertEqual(defaults.relative_threshold, 0.02)
        self.assertEqual(defaults.core_relative_threshold, 0.05)
        self.assertEqual(defaults.maximum_growth_distance_mm, 12.0)
        self.assertEqual(defaults.smoothing_mm, 1.0)
        self.assertEqual(defaults.opening_radius_mm, 0.0)
        self.assertEqual(defaults.closing_radius_mm, 1.5)
        self.assertEqual(defaults.dilation_radius_mm, 0.0)

    def test_public_rovir_cli_treats_explicit_roi_as_authorization(self) -> None:
        """Keep the ROVir interface compact, explicit, and nonredundant.

        Returns:
            None.
        """
        public = SCRIPTS / "sample_mprage_rovir_recon.sh"
        retro = SCRIPTS / "sample_mprage_rovir_retro_recon.sh"
        for script in (public, retro):
            subprocess.run(["bash", "-n", str(script)], check=True)
        completed = subprocess.run(
            ["bash", str(public), "--help"], check=False, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("inspect TWIX.dat OUTPUT_ROOT SEQUENCE.seq", completed.stdout)
        self.assertIn("run TWIX.dat OUTPUT_ROOT SEQUENCE.seq", completed.stdout)
        self.assertNotIn("defaults to the current directory", completed.stdout)
        self.assertIn("--null-box", completed.stdout)
        self.assertNotIn("--confirm-roi-id", completed.stdout)
        self.assertNotIn("--config", completed.stdout)
        source = public.read_text(encoding="utf-8")
        self.assertNotIn("read -r -p", source)
        self.assertNotIn("exact candidate ID mismatch", source)
        self.assertIn("choose exactly one ROI input mode", source)
        self.assertIn("bart fft -iu 7", source)
        self.assertIn("bart rss 8", source)
        self.assertIn("bart rovir", source)
        self.assertIn("bart ecalib -m 1 -c", source)
        self.assertIn('WAVE_ARGS=(-w -f -r 0 -i 100 -t 1e-6)', source)
        self.assertIn('validate-invocation "$TWIX_FILE" "$RECONSTRUCTION_ROOT"', source)
        self.assertIn("Reusing complete nested ROVir magnitude/phase NIfTI outputs", source)
        self.assertIn("existing ROVir NIfTI output is incomplete or ambiguous", source)
        retro_source = (SCRIPTS / "sample_mprage_retro_lr_recon.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--rovir", retro_source)
        self.assertIn("sample_mprage_rovir_retro_recon.sh", retro_source)
        self.assertIn("finalize-existing", retro_source)
        self.assertLess(
            retro_source.index('if [[ "$USE_ROVIR" == true ]]'),
            retro_source.index('python "$SCRIPT_DIR/prepare_mprage_retro.py"'),
        )
        self.assertIn("--rovir reuses its canonical CSM", retro_source)
        self.assertIn("--rovir reuses its canonical PSF", retro_source)

    def test_public_rovir_sources_are_validated_against_normal_manifest(self) -> None:
        """Bind the explicit public arguments to the normal source contract.

        Returns:
            None.
        """
        expected = {"output_root": "/reconstruction"}
        with patch(
            "wave_retro_lr.rovir_workflow._validated_source_contract"
        ) as validate, patch(
            "wave_retro_lr.rovir_workflow.normal_rovir_context",
            return_value=expected,
        ) as context:
            actual = validate_normal_rovir_invocation(
                "input.dat", "/reconstruction", "input.seq"
            )
        validate.assert_called_once_with(
            "input.dat",
            "input.seq",
            "/reconstruction",
            include_twix_hash=False,
        )
        context.assert_called_once_with("/reconstruction")
        self.assertEqual(actual, expected)

    def test_rovir_context_does_not_require_standard_reconstruction(self) -> None:
        """Use prepared inputs alone and treat nested normal NIfTI as optional.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            twix = root / "input.dat"
            sequence = root / "input.seq"
            manifest = root / "normal" / "bart_inputs" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            twix.write_bytes(b"twix")
            sequence.write_bytes(b"sequence")
            manifest.write_text(
                json.dumps(
                    {
                        "source": {
                            "twix": {"path": str(twix)},
                            "sequence": {"path": str(sequence)},
                        }
                    }
                ),
                encoding="utf-8",
            )

            context = normal_rovir_context(root)
            self.assertFalse(context["normal_reconstruction_required"])
            self.assertEqual(context["standard_comparison_status"], "not_available")
            self.assertIsNone(context["normal_magnitude_nifti"])
            self.assertEqual(context["ecalib_crop"], 0.6)
            self.assertEqual(context["ecalib_crop_source"], "mprage_launcher_default")

            nested = root / "normal" / "nifti" / "fista_r0" / "sub-normal"
            nested.mkdir(parents=True)
            magnitude = nested / "sub-normal_part-mag_BARTWaveMPRAGENormalFISTAR0.nii.gz"
            magnitude.touch()
            output = root / "normal" / "bart_output"
            output.mkdir(parents=True)
            (output / "ecalib_command.txt").write_text(
                "bart ecalib -m 1 -c 0.1 input output\n", encoding="utf-8"
            )

            context = normal_rovir_context(root)
            self.assertEqual(context["standard_comparison_status"], "available")
            self.assertEqual(context["normal_magnitude_nifti"], str(magnitude.resolve()))
            self.assertEqual(context["ecalib_crop"], 0.1)
            self.assertEqual(
                context["ecalib_crop_source"], "standard_normal_ecalib_command"
            )
            self.assertEqual(
                _single_magnitude(
                    root / "normal" / "nifti" / "fista_r0", recursive=True
                ),
                magnitude.resolve(),
            )

    def test_finalize_existing_rovir_infers_ncc_or_reuses_contract(self) -> None:
        """Finalize interrupted output without asking the user for Ncc.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            inputs = root / "normal" / "rovir" / "bart_inputs" / "manifest.json"
            inputs.parent.mkdir(parents=True)
            inputs.write_text(
                json.dumps({"rovir": {"virtual_coils": 24}}), encoding="utf-8"
            )
            context = {
                "output_root": str(root),
                "rovir_root": str(root / "normal" / "rovir"),
                "normal_manifest_sha256": "normal-hash",
            }
            completed = {
                "status": "mprage_normal_rovir_complete",
                "retro_consumable": True,
            }
            with patch(
                "wave_retro_lr.rovir_workflow.normal_rovir_context",
                return_value=context,
            ), patch(
                "wave_retro_lr.rovir_workflow.finalize_normal_rovir",
                return_value=completed,
            ) as finalize:
                self.assertEqual(finalize_existing_normal_rovir(root), completed)
            finalize.assert_called_once_with(root, 24)

            canonical = root / "normal" / "rovir" / "manifest.json"
            canonical.write_text(
                json.dumps(
                    {
                        **completed,
                        "normal_source_manifest": {"sha256": "normal-hash"},
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "wave_retro_lr.rovir_workflow.normal_rovir_context",
                return_value=context,
            ), patch(
                "wave_retro_lr.rovir_workflow.finalize_normal_rovir"
            ) as finalize:
                reused = finalize_existing_normal_rovir(root)
            finalize.assert_not_called()
            self.assertTrue(reused["retro_consumable"])

    def test_mprage_preparation_defaults_to_automatic_sine_line(self) -> None:
        """Keep the sample, preparation CLI, and Python API defaults aligned.

        Returns:
            None.
        """
        arguments = ["input.dat", "output", "input.seq"]
        for parser in (normal_parser, retro_parser, r3x3_parser):
            parsed = parser().parse_args(arguments)
            self.assertEqual(parsed.psf_coefficient_processing, "sine-line")
            self.assertIsNone(parsed.psf_fit_kx_min)
            self.assertIsNone(parsed.psf_fit_kx_max)
        for function in (
            prepare_normal_mprage,
            prepare_retro_mprage,
            prepare_retro_mprage_r3x3,
        ):
            parameter = inspect.signature(function).parameters[
                "psf_coefficient_processing"
            ]
            self.assertEqual(parameter.default, "sine-line")

    def test_new_python_workflow_does_not_launch_bart(self) -> None:
        """Verify measured-data Python modules never launch BART processes.

        Returns:
            None.
        """
        measured_sources = [
            TOOL_ROOT / "wave_retro_lr" / "mprage.py",
            TOOL_ROOT / "wave_retro_lr" / "projection_psf.py",
            SCRIPTS / "prepare_mprage_normal.py",
            SCRIPTS / "prepare_mprage_retro.py",
            SCRIPTS / "prepare_mprage_retro_r3x3.py",
            SCRIPTS / "prepare_mprage_retro_maps.py",
            SCRIPTS / "convert_mprage_bart_to_nifti.py",
            SCRIPTS / "build_mprage_nifti_collection.py",
            TOOL_ROOT / "wave_retro_lr" / "nifti_collection.py",
            TOOL_ROOT / "wave_retro_lr" / "rovir_workflow.py",
            TOOL_ROOT / "wave_retro_lr" / "rovir_retro.py",
            SCRIPTS / "mprage_rovir_workflow.py",
            SCRIPTS / "prepare_mprage_rovir_retro.py",
        ]
        for path in measured_sources:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("subprocess", source, path.name)
            self.assertNotIn("Popen", source, path.name)

    def test_mprage_converter_keeps_dicom_validated_orientation(self) -> None:
        """Verify the exporter retains the DICOM-validated SI/LR correction.

        Returns:
            None.
        """
        source = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)", source)
        self.assertIn(
            "twix_array_axis_flips=MPRAGE_BART_ARRAY_AXIS_FLIPS", source
        )

    def test_mprage_orientation_policy_is_not_replaced_by_gre(self) -> None:
        """Verify GRE integration leaves the validated MPRAGE flips unchanged.

        Returns:
            None.
        """

        source = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)", source)


if __name__ == "__main__":
    unittest.main()
