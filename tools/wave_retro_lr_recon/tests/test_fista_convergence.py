"""Tests for the logged FISTA-r0 convergence control (Arm 1)."""

from __future__ import annotations

import copy
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))
sys.path.insert(0, str(TOOL_ROOT / "tests"))

from wave_retro_lr import csm_consistency, fista_convergence  # noqa: E402
from test_csm_consistency import build_synthetic_example  # noqa: E402

SHELL_SCRIPT = TOOL_ROOT / "scripts" / "sample_mprage_fista_convergence.sh"
PYTHON_CLI = TOOL_ROOT / "scripts" / "mprage_fista_convergence.py"
STUB_BART = """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    version) echo "v1.0.00-stub" ;;
    wave)
        args=("$@")
        iterations=""
        for ((i = 0; i < ${#args[@]}; i++)); do
            [[ "${args[$i]}" == -i ]] && iterations="${args[$((i + 1))]}"
        done
        out="${args[-1]}"
        cp "$STUB_IMAGE.hdr" "$out.hdr"
        cp "$STUB_IMAGE.cfl" "$out.cfl"
        printf '\\tMax eval: 4.95e+07\\n' >&2
        for ((k = 0; k < iterations; k++)); do
            printf '#It %03d: 0.%06d   \\n' "$k" "$((500000 / (k + 1)))" >&2
        done
        printf '\\t\\tFISTA iterations: %d\\n' "$iterations" >&2
        printf 'Done.\\nReconstruction time: 1.000000 seconds.\\nTotal time: 2.000000 seconds.\\n' >&2
        ;;
    *) echo "stub bart: unsupported $1" >&2; exit 1 ;;
esac
"""


def debug_log(lines: int, final: int, *, timestamps: bool = True) -> str:
    """Build a BART debug-level-5 log as the shell records it.

    Args:
        lines: Number of ``#It`` lines.
        final: Value of the ``FISTA iterations`` line.
        timestamps: Whether each line carries the shell's timestamp prefix.

    Returns:
        Log text.
    """
    body = ["\tMax eval: 4.95e+07", "lsqr: add GPU wrapper"]
    body += [f"#It {k:03d}: {0.5 / (k + 1):.6f}   " for k in range(lines)]
    body += [f"\t\tFISTA iterations: {final}", "Reconstruction time: 130.2 seconds.", "Total time: 157.0 seconds."]
    if timestamps:
        body = [f"1790800000.{index:06d}\t{line}" for index, line in enumerate(body)]
    return "\n".join(body) + "\n"


TIME_LOG = """\tCommand being timed: "bart wave -w -f -r 0 -i 300 -t 1e-6 a b c d"
\tPercent of CPU this job got: 99%
\tElapsed (wall clock) time (h:mm:ss or m:ss): 2:10.53
\tMaximum resident set size (kbytes): 15728640
\tExit status: 0
"""


def write_run_outputs(example_accepted: Path, output: Path, iterations: int, *, exit_status: int = 0) -> None:
    """Write the files the shell leaves after a successful BART run.

    Args:
        example_accepted: Accepted root whose image is copied as the result.
        output: Convergence-control output root.
        iterations: Iteration cap of the run.
        exit_status: Exit status written into the time record.
    """
    paths = fista_convergence.layout(output, iterations)
    paths["image"].parent.mkdir(parents=True)
    for suffix in (".hdr", ".cfl"):
        shutil.copyfile(
            (example_accepted / "normal/bart_output/fista_r0/image_wave").with_suffix(suffix),
            paths["image"].with_suffix(suffix),
        )
    argv = fista_convergence.convergence_argv(example_accepted, output, iterations)
    paths["command_record"].write_text(csm_consistency.format_command(argv) + "\n", encoding="utf-8")
    (output / "logs" / "environment").mkdir(parents=True)
    paths["debug_log"].write_text(debug_log(iterations, iterations), encoding="utf-8")
    paths["time_log"].write_text(TIME_LOG.replace("Exit status: 0", f"Exit status: {exit_status}"), encoding="utf-8")
    paths["bart_version"].write_text("v1.0.00-stub\n", encoding="utf-8")
    paths["bart_binary_hash"].write_text("0" * 64 + "  /stub/bart\n", encoding="utf-8")
    (output / "logs" / "environment" / "reconstruct_test.txt").write_text("stage: reconstruct\n", encoding="utf-8")


class ConvergenceControlTests(unittest.TestCase):
    """Derive, validate, record, and verify the convergence control."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one synthetic accepted root shared by every test.

        Returns:
            None.
        """
        cls._temporary = tempfile.TemporaryDirectory()
        cls.example = build_synthetic_example(Path(cls._temporary.name) / "fixture")
        cls.accepted = cls.example.accepted

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the shared fixture.

        Returns:
            None.
        """
        cls._temporary.cleanup()

    def setUp(self) -> None:
        """Give each test its own scratch directory.

        Returns:
            None.
        """
        self._scratch = tempfile.TemporaryDirectory()
        self.scratch = Path(self._scratch.name)

    def tearDown(self) -> None:
        """Remove the per-test scratch directory.

        Returns:
            None.
        """
        self._scratch.cleanup()

    def test_argv_changes_only_the_iteration_cap_and_image(self) -> None:
        """Keep every accepted flag and input; change -i and the output only.

        Returns:
            None.
        """
        output = self.scratch / "arm1"
        argv = fista_convergence.convergence_argv(self.accepted, output, 300)
        paths = csm_consistency.accepted_artifact_paths(self.accepted)
        accepted = shlex.split(paths["fista_r0_command"].read_text(encoding="utf-8"))
        self.assertEqual(argv[:-4][: argv.index("-i") + 1], accepted[:-4][: accepted.index("-i") + 1])
        self.assertEqual(argv[argv.index("-i") + 1], "300")
        self.assertEqual(argv[argv.index("-i") + 2 : -4], accepted[accepted.index("-i") + 2 : -4])
        self.assertEqual(argv[-4:-1], [str(paths["coil_sens"]), str(paths["psf"]), str(paths["wave_kspace"])])
        self.assertEqual(argv[-1], str(output.resolve() / "normal/bart_output/fista_r0_i300/image_wave"))
        self.assertEqual(fista_convergence.accepted_iteration_cap(accepted), 100)

    def test_output_root_must_not_overlap_source_roots(self) -> None:
        """Refuse equal, nested, containing, protected, and process-dependent roots.

        Returns:
            None.
        """
        protected = self.scratch / "stage3"
        protected.mkdir()
        refused = {
            "the accepted root": self.accepted,
            "inside the accepted root": self.accepted / "arm1",
            "containing the accepted root": self.accepted.parent,
            "inside a protected root": protected / "arm1",
            "containing a protected root": self.scratch,
            "a /proc spelling": "/proc/self/cwd/arm1",
        }
        for name, root in refused.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                fista_convergence.require_separate_root(root, self.accepted, [protected])
        self.assertEqual(
            fista_convergence.require_separate_root(self.scratch / "arm1", self.accepted),
            (self.scratch / "arm1").resolve(),
        )

    def test_shell_command_must_match_token_by_token(self) -> None:
        """Accept only the derived command, including the device flag.

        Returns:
            None.
        """
        output = self.scratch / "arm1"
        argv = fista_convergence.convergence_argv(self.accepted, output, 300)
        text = csm_consistency.format_command(argv)
        self.assertEqual(fista_convergence.check_shell_command(text, self.accepted, output, 300), argv)
        position = argv.index("-i")
        variants = {
            "accepted cap": argv[: position + 1] + ["100"] + argv[position + 2 :],
            "regularized": [token if token != "0" else "0.1" for token in argv],
            "added GPU flag": argv[:2] + ["-g"] + argv[2:],
            "other image": argv[:-1] + [str(self.scratch / "other" / "image_wave")],
        }
        for name, tokens in variants.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                fista_convergence.check_shell_command(csm_consistency.format_command(tokens), self.accepted, output, 300)

    def test_debug_log_parsing(self) -> None:
        """Parse runs to the cap and tolerance stops; refuse broken traces.

        Returns:
            None.
        """
        full = fista_convergence.parse_debug_log(debug_log(300, 300), 300)
        self.assertFalse(full["stopped_by_tolerance"])
        self.assertEqual((full["final_iteration_count"], len(full["trace"])), (300, 300))
        self.assertEqual(full["trace"][0], [0, 0.5])
        self.assertEqual(full["max_eigenvalue"], 4.95e7)
        self.assertEqual((full["reconstruction_time_s"], full["total_time_s"]), (130.2, 157.0))
        early = fista_convergence.parse_debug_log(debug_log(151, 150, timestamps=False), 300)
        self.assertTrue(early["stopped_by_tolerance"])
        self.assertEqual(early["final_iteration_count"], 150)
        zero = fista_convergence.parse_debug_log(debug_log(3, 3).replace("0.166667", "0.000000"), 3)
        self.assertEqual(zero["last_residual"], 0.0)
        broken = {
            "missing iteration": debug_log(300, 300).replace("#It 010:", "#It 011:", 1),
            "no final count": debug_log(300, 300).replace("FISTA iterations: 300", ""),
            "two final counts": debug_log(300, 300) + "FISTA iterations: 300\n",
            "length mismatch": debug_log(299, 300),
            "count above cap": debug_log(300, 301),
        }
        for name, text in broken.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                fista_convergence.parse_debug_log(text, 300)

    def test_time_and_gpu_logs(self) -> None:
        """Parse GNU time and nvidia-smi samples, skipping headers and notices.

        Returns:
            None.
        """
        timing = fista_convergence.parse_time_log(TIME_LOG)
        self.assertAlmostEqual(timing["wall_clock_s"], 130.53)
        self.assertEqual((timing["max_rss_kib"], timing["exit_status"]), (15728640, 0))
        hours = fista_convergence.parse_time_log(TIME_LOG.replace("2:10.53", "1:02:03"))
        self.assertAlmostEqual(hours["wall_clock_s"], 3723.0)
        with self.assertRaises(ValueError):
            fista_convergence.parse_time_log("Exit status: 0\n")
        device = (
            "timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
            "2026/10/01 10:00:00.000, 0, 54022 MiB, 97887 MiB, 100 %\n"
            "timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
            "2026/10/01 10:00:01.000, 0, 76022 MiB, 97887 MiB, 100 %\n"
        )
        processes = (
            "timestamp, pid, process_name, used_memory [MiB]\n"
            "No running processes found\n"
            "2026/10/01 10:00:01.000, 42, /opt/bart/bart, 21500 MiB\n"
            "2026/10/01 10:00:01.000, 7, /usr/bin/python, 49164 MiB\n"
        )
        gpu = fista_convergence.parse_gpu_logs(device, processes)
        self.assertEqual((gpu["device_samples"], gpu["peak_device_memory_used_mib"]), (2, 76022.0))
        self.assertEqual((gpu["peak_bart_process_memory_mib"], gpu["device_memory_total_mib"]), (21500.0, 97887.0))
        with self.assertRaises(ValueError):
            fista_convergence.parse_gpu_logs("timestamp, index\n", processes)

    def test_record_verify_and_refuse_partial_or_failed_runs(self) -> None:
        """Record once, verify on reuse, refuse tampering, leftovers, and failures.

        Returns:
            None.
        """
        output = self.scratch / "arm1"
        write_run_outputs(self.accepted, output, 300)
        log = output / "logs" / "environment" / "reconstruct_test.txt"
        manifest = fista_convergence.record_control(self.accepted, output, 300, environment_log=log)
        self.assertEqual(manifest["changed_variable"], {"name": "FISTA iteration cap (-i)", "accepted": 100, "control": 300})
        self.assertEqual(manifest["convergence"]["final_iteration_count"], 300)
        self.assertFalse(manifest["flags"]["csm_changed"] or manifest["flags"]["accepted_root_written"])
        self.assertEqual(fista_convergence.record_control(self.accepted, output, 300), manifest)
        text = csm_consistency.format_command(fista_convergence.convergence_argv(self.accepted, output, 300))
        self.assertEqual(fista_convergence.validate_control(self.accepted, output, 300, text)["state"], "recorded")
        image = fista_convergence.layout(output, 300)["image"].with_suffix(".cfl")
        original = image.read_bytes()
        tampered = bytearray(original)
        tampered[len(tampered) // 2] ^= 0xFF
        image.write_bytes(bytes(tampered))
        try:
            with self.assertRaises(ValueError):
                fista_convergence.verify_control(manifest, self.accepted, output, 300)
        finally:
            image.write_bytes(original)
        partial = self.scratch / "partial"
        fista_convergence.layout(partial, 300)["image"].parent.mkdir(parents=True)
        fista_convergence.layout(partial, 300)["image"].with_suffix(".hdr").write_text("", encoding="utf-8")
        partial_text = csm_consistency.format_command(fista_convergence.convergence_argv(self.accepted, partial, 300))
        with self.assertRaises(FileExistsError):
            fista_convergence.validate_control(self.accepted, partial, 300, partial_text)
        failed = self.scratch / "failed"
        write_run_outputs(self.accepted, failed, 300, exit_status=1)
        with self.assertRaisesRegex(ValueError, "exited with status 1"):
            fista_convergence.record_control(
                self.accepted, failed, 300, environment_log=failed / "logs/environment/reconstruct_test.txt"
            )
        self.assertFalse(fista_convergence.layout(failed, 300)["manifest"].exists())
        unlogged = self.scratch / "unlogged"
        write_run_outputs(self.accepted, unlogged, 300)
        with self.assertRaisesRegex(ValueError, "environment log"):
            fista_convergence.record_control(self.accepted, unlogged, 300)

    def _recorded(self, name: str) -> tuple[Path, dict[str, Any]]:
        """Write a finished run in the scratch directory and record it.

        Args:
            name: Output-root name.

        Returns:
            Output root and its stored manifest.
        """
        output = self.scratch / name
        write_run_outputs(self.accepted, output, 300)
        log = output / "logs" / "environment" / "reconstruct_test.txt"
        return output, fista_convergence.record_control(self.accepted, output, 300, environment_log=log)

    def test_fixed_contract_refuses_coordinated_removal_and_additions(self) -> None:
        """Refuse a record removed together with its file, and any added file.

        Returns:
            None.
        """
        output, manifest = self._recorded("arm1")
        paths = fista_convergence.layout(output, 300)
        environment_file = output / manifest["environment"]["shell_environment_log"]["relative_path"]
        removals = {f"log {key}": (("logs", key), (paths[key],)) for key in fista_convergence.REQUIRED_LOG_KEYS}
        removals.update({
            "environment log": (("environment", "shell_environment_log"), (environment_file,)),
            "command record": (("command", "record"), (paths["command_record"],)),
            "image": (("image",), (paths["image"].with_suffix(".hdr"), paths["image"].with_suffix(".cfl"))),
        })
        for name, (keys, files) in removals.items():
            with self.subTest(removed=name):
                edited = copy.deepcopy(manifest)
                node = edited
                for key in keys[:-1]:
                    node = node[key]
                del node[keys[-1]]
                for file in files:
                    file.rename(file.with_name(file.name + ".aside"))
                try:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        fista_convergence.verify_control(edited, self.accepted, output, 300)
                finally:
                    for file in files:
                        file.with_name(file.name + ".aside").rename(file)
        gpu_claim = copy.deepcopy(manifest)
        paths["gpu_device_log"].write_text("timestamp, index\n", encoding="utf-8")
        gpu_claim["logs"]["gpu_device_log"] = fista_convergence.rovir_feasibility._file_record(paths["gpu_device_log"])
        with self.assertRaises(ValueError):
            fista_convergence.verify_control(gpu_claim, self.accepted, output, 300)
        paths["gpu_device_log"].unlink()
        additions = {
            "image directory": paths["image"].parent / "extra.cfl",
            "logs": output / "logs" / "extra.log",
            "environment logs": output / "logs" / "environment" / "reconstruct_other.txt",
            "manifests": output / "manifests" / "extra.json",
            "output root": output / "extra.txt",
            "another branch": output / "normal" / "bart_output" / "fista_r0_i100",
        }
        for name, extra in additions.items():
            with self.subTest(added=name):
                if name == "another branch":
                    extra.mkdir()
                else:
                    extra.write_text("unexpected\n", encoding="utf-8")
                try:
                    with self.assertRaises(ValueError):
                        fista_convergence.verify_control(manifest, self.accepted, output, 300)
                finally:
                    extra.rmdir() if extra.is_dir() else extra.unlink()
        fista_convergence.verify_control(manifest, self.accepted, output, 300)

    def test_tampered_records_are_refused(self) -> None:
        """Refuse edited implementation, baseline, command, summaries, and flags.

        Returns:
            None.
        """
        output, manifest = self._recorded("arm1")
        argv = manifest["command"]["argv"]
        position = argv.index("-i")
        cases = {
            "implementation hash": (("implementation", "wave_retro_lr/fista_convergence.py"), "0" * 64),
            "accepted CSM hash": (("accepted_baseline", "coil_sens", "payload_sha256"), "0" * 64),
            "accepted command text": (("accepted_baseline", "fista_r0_command", "text"), "bart wave -w"),
            "command argv": (("command", "argv"), argv[: position + 1] + ["100"] + argv[position + 2 :]),
            "command text": (("command", "text"), manifest["command"]["text"].replace("-i 300", "-i 100")),
            "debug level": (("command", "debug_level"), 4),
            "changed variable": (("changed_variable", "accepted"), 300),
            "final count": (("convergence", "final_iteration_count"), 299),
            "residual trace": (("convergence", "trace"), manifest["convergence"]["trace"][:-1]),
            "peak memory": (("resources", "time", "max_rss_kib"), 1),
            "flags": (("flags", "csm_changed"), True),
            "status": (("status",), "other"),
            "format version": (("format_version",), 2),
        }
        for name, (keys, value) in cases.items():
            with self.subTest(tampered=name):
                edited = copy.deepcopy(manifest)
                node = edited
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                with self.assertRaises((ValueError, FileExistsError)):
                    fista_convergence.verify_control(edited, self.accepted, output, 300)
        with patch.object(fista_convergence, "implementation_identity", return_value={"changed": "0" * 64}):
            with self.assertRaisesRegex(FileExistsError, "different implementation"):
                fista_convergence.verify_control(manifest, self.accepted, output, 300)
        fista_convergence.verify_control(manifest, self.accepted, output, 300)

    def test_fresh_accepted_baseline_binding_is_required(self) -> None:
        """Refuse reuse after the accepted CSM or FISTA-r0 record changed.

        Returns:
            None.
        """
        output, manifest = self._recorded("arm1")
        paths = csm_consistency.accepted_artifact_paths(self.accepted)
        for name, target in (("accepted CSM payload", paths["coil_sens"].with_suffix(".cfl")), ("accepted FISTA-r0 record", paths["fista_r0_command"])):
            original = target.read_bytes()
            changed = bytearray(original)
            if target.suffix == ".cfl":
                changed[len(changed) // 2] ^= 0xFF
            else:
                changed = bytearray(original.replace(b"-i 100", b"-i 200"))
            target.write_bytes(bytes(changed))
            try:
                with self.subTest(changed=name), self.assertRaises(ValueError):
                    fista_convergence.verify_control(manifest, self.accepted, output, 300)
            finally:
                target.write_bytes(original)
        fista_convergence.verify_control(manifest, self.accepted, output, 300)

    def _shell_environment(self) -> dict[str, str]:
        """Put a stub BART on PATH for real-shell runs.

        Returns:
            Process environment.
        """
        stub = self.scratch / "stub" / "bart"
        if not stub.exists():
            stub.parent.mkdir()
            stub.write_text(STUB_BART, encoding="utf-8")
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        return {
            **os.environ,
            "PATH": f"{stub.parent}{os.pathsep}{os.environ['PATH']}",
            "STUB_IMAGE": str(self.accepted / "normal/bart_output/fista_r0/image_wave"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def _run_shell(self, stage: str, output: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        """Run one stage of the real shell entry point.

        Args:
            stage: Stage name.
            output: Convergence-control output root.
            environment: Process environment with the stub BART.

        Returns:
            Completed process with captured output.
        """
        arguments = [str(self.example.twix), str(self.example.sequence), str(self.accepted), str(output)]
        options = ["--iterations", "300", "--protected-root", str(self.example.output), "--min-free-gb", "0"]
        return subprocess.run(
            ["bash", str(SHELL_SCRIPT), stage, *arguments, *options],
            env=environment, capture_output=True, text=True, timeout=600,
        )

    def test_interruption_after_each_shell_step_refuses_rerun(self) -> None:
        """Refuse a rerun after an interruption at every shell step, writing nothing.

        Each state holds exactly what the reconstruct stage has written once
        the given step completes. A rerun must exit with code 2 and leave every
        entry and byte unchanged.

        Returns:
            None.
        """
        environment = self._shell_environment()
        steps = ("directories", "environment log", "BART version", "binary hash", "GPU samplers",
                 "BART started", "BART finished", "command record")

        def write(path: Path, content: str | bytes) -> None:
            """Write one file, creating its directory.

            Args:
                path: Destination.
                content: Text or bytes.
            """
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")

        accepted_image = self.accepted / "normal/bart_output/fista_r0/image_wave"
        for count in range(1, len(steps) + 1):
            output = self.scratch / f"interrupted_{count}"
            paths = fista_convergence.layout(output, 300)
            done = steps[:count]
            paths["image"].parent.mkdir(parents=True)
            paths["debug_log"].parent.mkdir(parents=True, exist_ok=True)
            if "environment log" in done:
                write(output / "logs/environment/reconstruct_20261001T000000Z_abc123.txt", "stage: reconstruct\n")
            if "BART version" in done:
                write(paths["bart_version"], "v1.0.00-stub\n")
            if "binary hash" in done:
                write(paths["bart_binary_hash"], "0" * 64 + "  /stub/bart\n")
            if "GPU samplers" in done:
                write(paths["gpu_device_log"], "timestamp, index, memory.used [MiB]\n")
                write(paths["gpu_process_log"], "timestamp, pid, process_name, used_memory [MiB]\n")
            if "BART started" in done:
                write(paths["debug_log"], debug_log(10, 10)[:120])
            if "BART finished" in done:
                for suffix in (".hdr", ".cfl"):
                    write(paths["image"].with_suffix(suffix), accepted_image.with_suffix(suffix).read_bytes())
                write(paths["debug_log"], debug_log(300, 300))
                write(paths["time_log"], TIME_LOG)
            if "command record" in done:
                argv = fista_convergence.convergence_argv(self.accepted, output, 300)
                write(paths["command_record"], csm_consistency.format_command(argv) + "\n")
            before = {entry: entry.read_bytes() if entry.is_file() else None for entry in output.rglob("*")}
            with self.subTest(interrupted_after=steps[count - 1]):
                rerun = self._run_shell("reconstruct", output, environment)
                self.assertEqual(rerun.returncode, 2, rerun.stdout + rerun.stderr)
                self.assertIn("without a manifest", rerun.stderr)
                after = {entry: entry.read_bytes() if entry.is_file() else None for entry in output.rglob("*")}
                self.assertEqual(after, before)

    def test_print_command_and_stub_run(self) -> None:
        """Match Python's command, run once with a stub BART, reuse, and refuse leftovers.

        Returns:
            None.
        """
        environment = self._shell_environment()
        output = self.scratch / "interventions" / "arm1"
        shell = self._run_shell("print-command", output, environment)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        python = subprocess.run(
            [sys.executable, str(PYTHON_CLI), "print-command", str(self.accepted), str(output), "--iterations", "300"],
            env=environment, capture_output=True, text=True, check=True,
        )
        self.assertEqual(shlex.split(shell.stdout), shlex.split(python.stdout))
        self.assertFalse(output.exists())
        initial = self._run_shell("reconstruct", output, environment)
        self.assertEqual(initial.returncode, 0, initial.stderr)
        self.assertIn("Free space on the output file system", initial.stdout)
        paths = fista_convergence.layout(output, 300)
        self.assertTrue(paths["manifest"].is_file())
        self.assertEqual(shlex.split(paths["command_record"].read_text(encoding="utf-8")), shlex.split(shell.stdout))
        self.assertEqual(len(list((output / "logs" / "environment").glob("reconstruct_*.txt"))), 1)
        recorded = {entry: entry.read_bytes() for entry in output.rglob("*") if entry.is_file()}
        repeated = self._run_shell("reconstruct", output, environment)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertIn("nothing was run", repeated.stdout)
        self.assertEqual({entry: entry.read_bytes() for entry in output.rglob("*") if entry.is_file()}, recorded)
        paths["manifest"].rename(self.scratch / "aside.json")
        interrupted = self._run_shell("reconstruct", output, environment)
        self.assertEqual(interrupted.returncode, 2)
        self.assertIn("without a manifest", interrupted.stderr)

    def test_modules_launch_no_process(self) -> None:
        """Keep the module and the CLI free of process launching.

        Returns:
            None.
        """
        for relative in ("wave_retro_lr/fista_convergence.py", "scripts/mprage_fista_convergence.py"):
            source = (TOOL_ROOT / relative).read_text(encoding="utf-8")
            for forbidden in ("subprocess", "os.system", "Popen", "execv"):
                self.assertNotIn(forbidden, source, relative)


class ShellEntryPointTests(unittest.TestCase):
    """Keep the single BART command explicit and run the real shell with a stub."""

    def test_static_command_contract(self) -> None:
        """Allow exactly one BART computation, bart wave with the fixed flags.

        Returns:
            None.
        """
        subprocess.run(["bash", "-n", str(SHELL_SCRIPT)], check=True)
        script = SHELL_SCRIPT.read_text(encoding="utf-8")
        # Scan the code only, after the usage text.
        code = script.split("EOF\n}", 1)[1]
        executable = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]
        self.assertEqual(sum('WAVE_ARGV=(bart wave "${GPU_FLAG[@]}" -w -f -r 0 -i "$ITERATIONS" -t 1e-6' in line for line in executable), 1)
        self.assertIn('BART_DEBUG_LEVEL=5 /usr/bin/time -v -o "$LOG_DIR/bart_wave.time.txt" "${WAVE_ARGV[@]}"', script)
        joined = "\n".join(executable)
        for forbidden in ("ecalib", "bart pics", "bart fft", "bart rss", "bart -d", "BART_GPU_GLOBAL_MEMORY"):
            self.assertNotIn(forbidden, joined, msg=f"forbidden {forbidden!r}")
        # BART's -d waits for a debugger; it must never reach the wave command.
        wave_definition = code[code.index("WAVE_ARGV=(") : code.index('"$IMAGE_BASE")')]
        self.assertNotIn(" -d ", wave_definition)
        bart_lines = [line for line in executable if line.startswith("bart ") or " bart " in f" {line} "]
        for line in bart_lines:
            self.assertTrue(
                line.startswith(("WAVE_ARGV=(bart wave", "bart version", "echo", "command -v bart", "sha256sum"))
                or "command -v bart" in line,
                line,
            )


if __name__ == "__main__":
    unittest.main()
