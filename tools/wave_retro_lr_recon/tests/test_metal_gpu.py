"""Synthetic GPU dispatch, execution provenance and collection tests; never launch BART."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test_metal_csm import fixture, parameters, write_array
from test_metal_collection import collection_fixture, mocked_science
from test_metal_pipeline import arguments
from scripts.mprage_metal_pipeline import main
from wave_retro_lr.bart_io import open_cfl, sha256_file
from wave_retro_lr.metal_pipeline import run_pipeline
from wave_retro_lr.metal_runner import (
    configure_gpu_process, configure_idle_process, run_metal_workflow, run_reconstruction,
)
from wave_retro_lr.metal_workflow import file_record, read_json, write_json
from wave_retro_lr.nifti_collection import build_mprage_nifti_collection


class MetalGPUTests(unittest.TestCase):
    """Verify optional GPU execution without CUDA hardware, patient data or subprocess jobs."""

    def test_entry_preserves_gpu_visibility_and_cpu_default(self):
        """Keep the caller's device selection for -g and hide CUDA only on the CPU path."""
        for gpu in (False, True):
            with self.subTest(gpu=gpu), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                 patch("wave_retro_lr.metal_runner.configure_idle_process") as policy, \
                 patch("wave_retro_lr.metal_pipeline.run_pipeline") as run, \
                 patch("scripts.mprage_metal_pipeline.signal.signal"), redirect_stdout(io.StringIO()):
                flags = ["-g"] if gpu else []
                result = main(["--mode", "normal", "raw.dat", "output", "scan.seq",
                               "--metal-csm-extrapolation", *flags])
                self.assertEqual(result, 0)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2" if gpu else "")
                self.assertEqual(run.call_args.args[0].g, gpu)
                policy.assert_called_once_with(gpu=gpu)

    def test_gpu_policy_omits_only_our_address_space_cap(self):
        """Do not impose an RLIMIT_AS cap on CUDA; keep idle priority and disabled core dumps."""
        for gpu in (False, True):
            with self.subTest(gpu=gpu), patch("os.getpriority", return_value=19), \
                 patch("os.sched_setscheduler") as schedule, \
                 patch("resource.getrlimit", return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)), \
                 patch("resource.setrlimit") as limit:
                configure_idle_process(gpu=gpu)
                schedule.assert_called_once()
                limits = {call.args[0]: call.args[1] for call in limit.call_args_list}
                self.assertEqual(limits[resource.RLIMIT_CORE], (0, 0))
                self.assertEqual(resource.RLIMIT_AS in limits, not gpu)

    def test_gpu_execution_environment_resume_and_backend_mismatch(self):
        """Bind -g, caller visibility and output hashes; reject backend/device changes on resume."""
        for visibility in (None, "2"):
            with self.subTest(visibility=visibility), tempfile.TemporaryDirectory() as temporary, \
                 patch.dict(os.environ), patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
                if visibility is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
                root = Path(temporary)
                manifest, reference = fixture(root)
                out = root/"diagnostics/metal"
                preparation = run_metal_workflow(manifest, reference, out, parameters(),
                                                enabled=True, prepare_only=True)
                baseline = read_json(manifest)["outputs"][0]["base"]

                def fake_bart(command, **kwargs):
                    """Emulate successful GPU BART with a tiny finite baseline image."""
                    self.assertEqual(command[:3], [str(Path(sys.executable).resolve()), "wave", "-g"])
                    self.assertEqual(kwargs["env"].get("CUDA_VISIBLE_DEVICES"), visibility)
                    self.assertIs(kwargs["preexec_fn"], configure_gpu_process)
                    write_array(Path(command[-1]), open_cfl(baseline))
                    process = MagicMock(pid=12345, returncode=0)
                    process.poll.return_value = process.wait.return_value = 0
                    return process

                with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), \
                     patch("wave_retro_lr.metal_runner.subprocess.Popen", side_effect=fake_bart) as launch:
                    first = run_reconstruction(out, preparation, gpu=True)
                    second = run_reconstruction(out, preparation, gpu=True)
                    self.assertEqual(first, second)
                    self.assertIs(first["identity"]["gpu"], True)
                    self.assertIsNone(first["identity"]["address_space_gib"])
                    self.assertEqual(first["identity"]["cuda_visible_devices"], visibility)
                    with self.assertRaisesRegex(ValueError, "mismatched"):
                        run_reconstruction(out, preparation, gpu=False)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "another-device"
                    with self.assertRaisesRegex(ValueError, "mismatched"):
                        run_reconstruction(out, preparation, gpu=True)
                    self.assertEqual(launch.call_count, 1)

    def test_gpu_failure_never_falls_back_to_cpu(self):
        """Retain a failed GPU attempt and reject retries without silently rerunning on CPU."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(temporary)
            manifest, reference = fixture(root)
            out = root/"diagnostics/metal"
            preparation = run_metal_workflow(manifest, reference, out, parameters(),
                                            enabled=True, prepare_only=True)
            process = MagicMock(pid=12345, returncode=1)
            process.poll.return_value = process.wait.return_value = 1
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), \
                 patch("wave_retro_lr.metal_runner.subprocess.Popen", return_value=process) as launch:
                with self.assertRaisesRegex(RuntimeError, "BART returned 1"):
                    run_reconstruction(out, preparation, gpu=True)
                self.assertEqual(read_json(out/"metadata/reconstruction.json")["status"], "failed_or_stopped")
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    run_reconstruction(out, preparation, gpu=True)
                launch.assert_called_once()

    def test_gpu_ram_guard_remains_active(self):
        """Reject insufficient host RAM before GPU BART is launched."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(temporary)
            manifest, reference = fixture(root)
            out = root/"diagnostics/metal"
            preparation = run_metal_workflow(manifest, reference, out, parameters(),
                                            enabled=True, prepare_only=True)
            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), \
                 patch("wave_retro_lr.metal_runner.available_gib", return_value=100), \
                 patch("wave_retro_lr.metal_runner.subprocess.Popen") as launch:
                with self.assertRaises(MemoryError):
                    run_reconstruction(out, preparation, gpu=True)
                launch.assert_not_called()

    def test_isolated_ecalib_uses_guarded_cpu_runner(self):
        """Run synthetic ecalib through the real guard/manifest code without a real BART process."""
        from test_metal_csm import synthetic_maps

        with tempfile.TemporaryDirectory() as temporary, patch("wave_retro_lr.metal_runner.available_gib", return_value=200):
            root = Path(temporary)
            fixture(root)
            out = root/"calibration"
            for name in ("metadata", "logs", "reconstruction"):
                (out/name).mkdir(parents=True)
            calibration = root/"vcc2/normal/bart_inputs/kspace_calib"
            from wave_retro_lr.bart_io import cfl_record
            prep = {"request": {"source": {"records": {"calibration": cfl_record(calibration)}}, "ecalib_crop": .6}}
            write_json(out/"metadata/preparation.json", prep)

            def fake_calibration(command, **kwargs):
                """Write tiny single-map CSMs and return a completed fake process."""
                self.assertEqual(command[1:6], ["ecalib", "-m", "1", "-c", "0.6"])
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                write_array(Path(command[-1]), synthetic_maps())
                child = MagicMock(pid=12345, returncode=0)
                child.poll.return_value = child.wait.return_value = 0
                return child

            with patch("wave_retro_lr.metal_runner.shutil.which", return_value=sys.executable), \
                 patch("wave_retro_lr.metal_runner.subprocess.Popen", side_effect=fake_calibration) as launch:
                result = run_reconstruction(out, prep, operation="ecalib")
                self.assertEqual(result, run_reconstruction(out, prep, operation="ecalib"))
                self.assertEqual(launch.call_count, 1)
                with self.assertRaisesRegex(ValueError, "Only wave"):
                    run_reconstruction(out, prep, operation="ecalib", gpu=True)

    def test_gpu_normal_retro_collection_and_tampered_backend(self):
        """Collect all GPU metal cases and reject mismatched GPU provenance atomically."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            destination = run_pipeline(arguments(root, "normal", "-g"))
            run_pipeline(arguments(root, "retro", "-g"))
            run_pipeline(arguments(root, "r3x3", "-g"))
            run_pipeline(arguments(root, "normal", "-g"))
            with self.assertRaisesRegex(ValueError, "request mismatch"):
                run_pipeline(arguments(root))
            exports = sorted(destination.rglob("metadata/export.json"))
            self.assertEqual(len(exports), 4)
            for export in exports:
                branch = export.parent.parent
                request = read_json(branch/"metadata/request.json")
                self.assertIs(request["gpu"], True)
                self.assertIn(" -g ", request["standard_request"]["expected_commands"][0])
                for sidecar in (branch/"nifti").rglob("*.json"):
                    self.assertIs(read_json(sidecar)["MetalCSM"]["GPU"], True)
            result = build_mprage_nifti_collection(root, require_retro=True)
            self.assertEqual(len([c for c in result["cases"] if c["collection_id"] == "vcc2"]), 4)
            before = {str(p): sha256_file(p) for p in (root/"nifti_collection").rglob("*") if p.is_file()}
            export = exports[0]
            state_path = export.parent/"reconstruction.json"
            state = read_json(state_path)
            state["identity"]["gpu"] = False
            state_path.write_text(json.dumps(state))
            payload = read_json(export)
            payload["execution"] = file_record(state_path)
            export.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                build_mprage_nifti_collection(root, require_retro=True)
            self.assertEqual(before, {str(p): sha256_file(p) for p in (root/"nifti_collection").rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
