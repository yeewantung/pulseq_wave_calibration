"""Logged FISTA-r0 convergence control for the accepted Wave-MPRAGE baseline.

The control repeats the accepted PCA-12 one-map FISTA-r0 reconstruction with a
larger iteration cap and records the residual of every iteration. Only the
``-i`` value and the output image differ from the accepted command record: the
CSM, PSF, wave k-space, device flag, and every other flag are the accepted
ones, read in place. The shell entry point runs the single ``bart wave``
command; this module validates and records and never launches a process.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import csm_consistency, rovir_feasibility
from .bart_io import cfl_record

FORMAT_VERSION = 1
STATUS = "mprage_fista_convergence_recorded"
TOOL_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "wave_retro_lr/fista_convergence.py",
    "scripts/sample_mprage_fista_convergence.sh",
    "scripts/mprage_fista_convergence.py",
)
# Values below 5e-7 print as 0.000000 because BART prints with "%f".
RESIDUAL_PRINT_RESOLUTION = 1e-6
RESIDUAL_DEFINITION = (
    "||A^H y - A^H A x_k|| / ||A^H y|| at the FISTA extrapolated point (lambda = 0), "
    "the quantity BART tests against -t; printed with six decimals"
)
_ITERATION_LINE = re.compile(r"#It (\d+): (\d+\.\d+)")
_FINAL_COUNT = re.compile(r"FISTA iterations: (\d+)")
_MAX_EVAL = re.compile(r"Max eval: ([0-9.]+(?:e[+-]?\d+)?)")
_RECONSTRUCTION_TIME = re.compile(r"Reconstruction time: ([0-9.]+) seconds")
_TOTAL_TIME = re.compile(r"Total time: ([0-9.]+) seconds")
# Fixed required-output contract of the reconstruct stage. Reuse checks this
# list, never the entries a manifest happens to hold.
REQUIRED_LOG_KEYS = ("debug_log", "time_log", "bart_version", "bart_binary_hash")
GPU_LOG_KEYS = ("gpu_device_log", "gpu_process_log")
BASELINE_KEYS = ("root", "manifest", "coil_sens", "psf", "wave_kspace", "fista_r0_image", "fista_r0_command")
FIXED_FLAGS = {
    "bart_launched_by_python": False,
    "csm_changed": False,
    "psf_recalibrated": False,
    "inputs_copied": False,
    "accepted_root_written": False,
}
# Top-level entries the later stages may add: convert writes normal/nifti and
# its log, and the review writes qc/.
_ROOT_ENTRIES = {"normal", "logs", "manifests", "qc"}


def branch_name(iterations: int) -> str:
    """Return the branch directory name of one iteration cap.

    Args:
        iterations: FISTA iteration cap.

    Returns:
        ``fista_r0_i<iterations>``.

    Raises:
        ValueError: If the cap is not a positive integer.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"The FISTA iteration cap must be a positive integer, not {iterations!r}.")
    return f"fista_r0_i{iterations}"


def layout(output_root: str | Path, iterations: int) -> dict[str, Path]:
    """Return the fixed output paths of one convergence control.

    Args:
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.

    Returns:
        Absolute paths of the image, command record, NIfTI directory,
        manifest, logs, and residual figure.
    """
    root = Path(output_root).expanduser().resolve()
    branch = branch_name(iterations)
    logs = root / "logs"
    return {
        "root": root,
        "image": root / "normal" / "bart_output" / branch / "image_wave",
        "command_record": root / "normal" / "bart_output" / branch / "wave_command.txt",
        "nifti_directory": root / "normal" / "nifti" / branch,
        "manifest": root / "manifests" / "fista_convergence.json",
        "debug_log": logs / "bart_wave.debug5.log",
        "time_log": logs / "bart_wave.time.txt",
        "gpu_device_log": logs / "gpu_device.csv",
        "gpu_process_log": logs / "gpu_processes.csv",
        "bart_version": logs / "bart_version.txt",
        "bart_binary_hash": logs / "bart_binary.sha256",
        "residual_figure": root / "qc" / "figures" / "residual_trace.png",
    }


def require_separate_root(
    output_root: str | Path,
    accepted_root: str | Path,
    protected_roots: Sequence[str | Path] = (),
) -> Path:
    """Refuse an output root that overlaps the accepted or a protected root.

    Args:
        output_root: Proposed convergence-control output root.
        accepted_root: Accepted normal root read by the control.
        protected_roots: Further roots that must never be written, such as
            the Stage 3 diagnostic root.

    Returns:
        Resolved output root.

    Raises:
        ValueError: If the output path depends on the process, or equals,
            lies inside, or contains the accepted or a protected root.
    """
    if csm_consistency._process_dependent(str(output_root)):
        raise ValueError(f"The output root path depends on the process: {output_root}")
    output = Path(output_root).expanduser().resolve()
    others = [("accepted root", accepted_root), *(("protected root", root) for root in protected_roots)]
    for label, root in others:
        other = Path(root).expanduser().resolve()
        if output == other or other in output.parents or output in other.parents:
            raise ValueError(f"The output root {output} overlaps the {label} {other}.")
    return output


def accepted_iteration_cap(argv: Sequence[str]) -> int:
    """Return the ``-i`` value of a recorded FISTA command.

    Args:
        argv: Parsed ``bart wave`` argument list.

    Returns:
        Iteration cap.

    Raises:
        ValueError: If the command does not hold exactly one ``-i`` value.
    """
    positions = [index for index, token in enumerate(argv) if token == "-i"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError("The FISTA command must hold exactly one -i value.")
    return int(argv[positions[0] + 1])


def convergence_argv(accepted_root: str | Path, output_root: str | Path, iterations: int) -> list[str]:
    """Build the convergence command from the accepted FISTA-r0 record.

    The accepted record is parsed by the reviewed parser, which binds every
    token to this root's files by identity. Only the ``-i`` value and the
    output image change; inputs are spelled by their resolved accepted paths.
    Nothing is hashed here.

    Args:
        accepted_root: Accepted normal root.
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap of the control.

    Returns:
        ``bart wave [-g] -w -f -r 0 -i N -t 1e-6 COIL_SENS PSF WAVE_KSPACE IMAGE``.

    Raises:
        FileNotFoundError: If the accepted command record is absent.
        ValueError: If the accepted record is not the reviewed FISTA-r0 form.
    """
    paths = csm_consistency.accepted_artifact_paths(accepted_root)
    text = paths["fista_r0_command"].read_text(encoding="utf-8").strip()
    argv = list(csm_consistency._accepted_fista_r0_command(text, paths)["argv"])
    argv[argv.index("-i") + 1] = str(iterations)
    argv[-4:] = [
        str(paths["coil_sens"]),
        str(paths["psf"]),
        str(paths["wave_kspace"]),
        str(layout(output_root, iterations)["image"]),
    ]
    return argv


def check_shell_command(
    command_text: str, accepted_root: str | Path, output_root: str | Path, iterations: int
) -> list[str]:
    """Require the shell's command to equal the derived command token by token.

    Args:
        command_text: Command printed by the shell entry point.
        accepted_root: Accepted normal root.
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.

    Returns:
        Validated argument list.

    Raises:
        ValueError: If any token differs, including the device flag.
    """
    expected = convergence_argv(accepted_root, output_root, iterations)
    tokens = shlex.split(command_text)
    if tokens != expected:
        raise ValueError(
            "The shell command differs from the command derived from the accepted record:\n"
            f"  shell:    {command_text}\n  expected: {csm_consistency.format_command(expected)}"
        )
    return expected


def required_log_keys(gpu: bool) -> tuple[str, ...]:
    """Return the fixed set of logs a recorded run must hold.

    Args:
        gpu: Whether the command ran with ``-g``.

    Returns:
        Layout keys of the required logs.
    """
    return REQUIRED_LOG_KEYS + (GPU_LOG_KEYS if gpu else ())


def output_state(output_root: str | Path, iterations: int) -> str:
    """Classify the convergence outputs of one root.

    Without a manifest, any entry under the output root, even an empty
    directory or a single log, comes from an interrupted run: the root is then
    ``"partial"``, so nothing is ever written over it.

    Args:
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.

    Returns:
        ``"recorded"`` when the manifest exists, ``"partial"`` when anything
        else exists, and ``"absent"`` for a missing or empty root.
    """
    root = Path(output_root).expanduser().resolve()
    if layout(root, iterations)["manifest"].is_file():
        return "recorded"
    if not root.exists():
        return "absent"
    if not root.is_dir() or any(root.iterdir()):
        return "partial"
    return "absent"


def validate_control(
    accepted_root: str | Path,
    output_root: str | Path,
    iterations: int,
    command_text: str,
    protected_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Validate everything the reconstruct stage needs before BART runs.

    Args:
        accepted_root: Accepted normal root.
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.
        command_text: Command the shell is about to run.
        protected_roots: Roots that must never be written.

    Returns:
        ``state`` (``"absent"`` or ``"recorded"``) and the validated ``argv``.

    Raises:
        FileExistsError: If outputs exist without a manifest.
        ValueError: If the roots overlap, the accepted baseline or the shell
            command is not the reviewed one, or a recorded control changed.
    """
    output = require_separate_root(output_root, accepted_root, protected_roots)
    # Full baseline binding, including CSM and image hashes.
    csm_consistency.validate_accepted_baseline(accepted_root)
    argv = check_shell_command(command_text, accepted_root, output, iterations)
    state = output_state(output, iterations)
    if state == "partial":
        entries = sorted(str(path.relative_to(output)) for path in output.rglob("*"))[:10] if output.is_dir() else [str(output)]
        raise FileExistsError(
            f"Convergence outputs exist without a manifest under {output} (interrupted run: "
            f"{', '.join(entries)}); inspect and move them aside before rerunning."
        )
    if state == "recorded":
        verify_control(
            rovir_feasibility._read_json(layout(output, iterations)["manifest"]),
            accepted_root,
            output,
            iterations,
            protected_roots,
        )
    return {"state": state, "argv": argv}


def parse_debug_log(text: str, iterations: int) -> dict[str, Any]:
    """Parse the per-iteration residual trace of a BART debug-level-5 log.

    BART prints ``#It k`` for k = 0, 1, ... after each update and tests the
    tolerance after printing. A run to the cap prints ``iterations`` lines and
    reports ``FISTA iterations: <cap>``; a tolerance stop at k prints k + 1
    lines and reports ``k``.

    Args:
        text: Debug log, optionally with a timestamp before each line.
        iterations: Iteration cap of the run.

    Returns:
        Trace ``[[k, residual], ...]``, final count, whether the tolerance
        stopped the run, and BART's eigenvalue and timing lines.

    Raises:
        ValueError: If the trace is not contiguous from 0, its length does not
            match the final count, or the final count is missing or repeated.
    """
    trace = [(int(index), float(value)) for index, value in _ITERATION_LINE.findall(text)]
    if not trace or [index for index, _ in trace] != list(range(len(trace))):
        raise ValueError("The FISTA residual trace is empty or not contiguous from iteration 0.")
    counts = _FINAL_COUNT.findall(text)
    if len(counts) != 1:
        raise ValueError(f"The debug log must report the final FISTA count once; found {len(counts)}.")
    final = int(counts[0])
    stopped_by_tolerance = final < iterations
    expected_lines = final + 1 if stopped_by_tolerance else iterations
    if final > iterations or len(trace) != expected_lines:
        raise ValueError(
            f"The FISTA trace has {len(trace)} lines for a final count of {final} and a cap of {iterations}."
        )

    def number(pattern: re.Pattern[str]) -> float | None:
        """Return the single value of an optional BART summary line."""
        found = pattern.findall(text)
        return float(found[-1]) if found else None

    return {
        "residual_definition": RESIDUAL_DEFINITION,
        "residual_print_resolution": RESIDUAL_PRINT_RESOLUTION,
        "iteration_cap": iterations,
        "final_iteration_count": final,
        "stopped_by_tolerance": stopped_by_tolerance,
        "last_residual": trace[-1][1],
        "trace": [[index, value] for index, value in trace],
        "max_eigenvalue": number(_MAX_EVAL),
        "reconstruction_time_s": number(_RECONSTRUCTION_TIME),
        "total_time_s": number(_TOTAL_TIME),
    }


def parse_time_log(text: str) -> dict[str, Any]:
    """Parse the GNU ``/usr/bin/time -v`` record of one command.

    Args:
        text: ``time -v`` output.

    Returns:
        Wall-clock seconds, peak RSS in KiB, CPU share, and exit status.

    Raises:
        ValueError: If a required field is missing or malformed.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator:
            fields[key] = value.strip()
    try:
        clock = [float(part) for part in fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"].split(":")]
        seconds = 0.0
        for part in clock:
            seconds = seconds * 60.0 + part
        return {
            "wall_clock_s": seconds,
            "max_rss_kib": int(fields["Maximum resident set size (kbytes)"]),
            "percent_cpu": fields.get("Percent of CPU this job got"),
            "exit_status": int(fields["Exit status"]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"The time -v record is incomplete or malformed: {exc}") from exc


def parse_gpu_logs(device_text: str, process_text: str) -> dict[str, Any]:
    """Summarize the once-per-second ``nvidia-smi`` samples of one run.

    Args:
        device_text: CSV of ``timestamp, index, memory.used, memory.total,
            utilization.gpu``.
        process_text: CSV of ``timestamp, pid, process_name, used_memory``.

    Returns:
        Sample counts, peak device memory (all users), device total, and the
        peak memory of BART processes.

    Raises:
        ValueError: If the device log holds no sample.
    """

    def rows(text: str, columns: int, memory_column: int) -> list[list[str]]:
        """Return the sample rows of one nvidia-smi CSV.

        Headers, which loop mode may repeat, and notices such as "No running
        processes found" are skipped because their memory cell is not numeric.
        """
        samples = []
        for row in csv.reader(io.StringIO(text)):
            cells = [cell.strip() for cell in row]
            memory = cells[memory_column].split() if len(cells) >= columns else []
            if memory and memory[0].isdigit():
                samples.append(cells)
        return samples

    def mib(cell: str) -> float:
        """Convert an ``N MiB`` cell to a number."""
        return float(cell.split()[0])

    device = rows(device_text, 5, 2)
    if not device:
        raise ValueError("The GPU device log holds no sample.")
    bart = [mib(row[3]) for row in rows(process_text, 4, 3) if "bart" in Path(row[2]).name]
    return {
        "device_samples": len(device),
        "peak_device_memory_used_mib": max(mib(row[2]) for row in device),
        "device_memory_total_mib": mib(device[0][3]),
        "bart_process_samples": len(bart),
        "peak_bart_process_memory_mib": max(bart) if bart else None,
    }


def implementation_identity() -> dict[str, str]:
    """Hash the files that define this control.

    Returns:
        Tool-relative path mapped to SHA-256.
    """
    return {
        relative: hashlib.sha256((TOOL_ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }


def record_control(
    accepted_root: str | Path,
    output_root: str | Path,
    iterations: int,
    *,
    environment_log: str | Path | None = None,
    protected_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Record a completed convergence run after verifying its outputs.

    Args:
        accepted_root: Accepted normal root.
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.
        environment_log: Shell log of this invocation under ``logs/environment``;
            required, because the environment log is part of the contract.
        protected_roots: Roots that must never be written.

    Returns:
        Convergence manifest in its stored form; an existing manifest is
        returned only after :func:`verify_control` has verified it.

    Raises:
        FileNotFoundError: If an output or log is missing.
        ValueError: If no environment log is given, the roots overlap, the
            command record differs from the reviewed command, BART did not
            exit cleanly, or a log is malformed.

    Side Effects:
        Writes ``manifests/fista_convergence.json``.
    """
    output = require_separate_root(output_root, accepted_root, protected_roots)
    paths = layout(output, iterations)
    if paths["manifest"].is_file():
        existing = rovir_feasibility._read_json(paths["manifest"])
        verify_control(existing, accepted_root, output, iterations, protected_roots)
        return existing
    if environment_log is None:
        raise ValueError("The convergence record requires the environment log of this invocation.")
    baseline, _ = csm_consistency.validate_accepted_baseline(accepted_root)
    argv = convergence_argv(accepted_root, output, iterations)
    command_text = paths["command_record"].read_text(encoding="utf-8").strip()
    if shlex.split(command_text) != argv:
        raise ValueError(f"The recorded command is not the reviewed command: {command_text}")
    image = cfl_record(paths["image"])
    if list(image["shape"]) != list(baseline["fista_r0_image"]["shape"]):
        raise ValueError(
            f"The control image shape {image['shape']} differs from the accepted "
            f"{baseline['fista_r0_image']['shape']}."
        )
    timing = parse_time_log(paths["time_log"].read_text(encoding="utf-8"))
    if timing["exit_status"] != 0:
        raise ValueError(f"BART exited with status {timing['exit_status']}.")
    convergence = parse_debug_log(paths["debug_log"].read_text(encoding="utf-8"), iterations)
    gpu = _gpu_summary(paths, argv)
    logs = {key: rovir_feasibility._file_record(paths[key]) for key in required_log_keys(argv[2:3] == ["-g"])}
    accepted_argv = baseline["fista_r0_command"]["argv"]
    payload = {
        "format_version": FORMAT_VERSION,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": implementation_identity(),
        "environment": csm_consistency.environment_record(output, environment_log),
        "accepted_baseline": _baseline_block(baseline),
        "changed_variable": {
            "name": "FISTA iteration cap (-i)",
            "accepted": accepted_iteration_cap(accepted_argv),
            "control": iterations,
        },
        "command": {
            "argv": argv,
            "text": command_text,
            "record": rovir_feasibility._file_record(paths["command_record"]),
            "debug_level": 5,
        },
        "image": image,
        "convergence": convergence,
        "resources": {"time": timing, "gpu": gpu},
        "logs": logs,
        "flags": dict(FIXED_FLAGS),
    }
    rovir_feasibility._write_json(paths["manifest"], payload)
    # Return the stored form, so a reused manifest compares equal to this one.
    return rovir_feasibility._read_json(paths["manifest"])


def _baseline_block(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return the accepted-baseline records a convergence manifest keeps.

    Args:
        baseline: Record from :func:`csm_consistency.validate_accepted_baseline`.

    Returns:
        JSON-native copy of the bound accepted artifacts and command.
    """
    return json.loads(json.dumps({key: baseline[key] for key in BASELINE_KEYS}))


def _gpu_summary(paths: Mapping[str, Path], argv: Sequence[str]) -> dict[str, Any] | None:
    """Summarize the GPU logs of a run, or return ``None`` for a CPU run.

    Args:
        paths: Output layout.
        argv: Recorded ``bart wave`` arguments.

    Returns:
        :func:`parse_gpu_logs` summary, or ``None`` without ``-g``.
    """
    if argv[2:3] != ["-g"]:
        return None
    return parse_gpu_logs(
        paths["gpu_device_log"].read_text(encoding="utf-8"),
        paths["gpu_process_log"].read_text(encoding="utf-8"),
    )


def _require_entries(directory: Path, allowed: set[str], label: str, *, required: set[str] | None = None) -> None:
    """Require a workflow directory to hold only known entries.

    Args:
        directory: Directory to inspect.
        allowed: Entry names that may exist.
        label: Directory name for error messages.
        required: Entry names that must exist; defaults to none.

    Raises:
        ValueError: If an unknown entry exists or a required one is missing.
    """
    present = {entry.name for entry in directory.iterdir()} if directory.is_dir() else set()
    unknown = sorted(present - allowed)
    missing = sorted((required or set()) - present)
    if unknown or missing:
        raise ValueError(f"{label} does not match the fixed output contract: unknown {unknown}, missing {missing}.")


def verify_control(
    manifest: Mapping[str, Any],
    accepted_root: str | Path,
    output_root: str | Path,
    iterations: int,
    protected_roots: Sequence[str | Path] = (),
) -> None:
    """Verify a recorded convergence control against its fixed contract.

    Every check derives what must exist from the implementation and the
    accepted root, never from the manifest's own entries, so removing a record
    together with its file is detected. The accepted baseline is bound again,
    the command is derived again, and the summaries are re-parsed from the
    verified logs.

    Args:
        manifest: Stored convergence manifest.
        accepted_root: Accepted normal root, bound again here.
        output_root: Convergence-control output root.
        iterations: FISTA iteration cap.
        protected_roots: Roots that must never be written.

    Raises:
        FileExistsError: If the control was recorded by another implementation
            and must be moved aside to be recomputed.
        FileNotFoundError: If a required file is missing.
        ValueError: If any recorded field, file, binding, command, or summary
            differs from the fixed contract or the current inputs.
    """
    output = require_separate_root(output_root, accepted_root, protected_roots)
    paths = layout(output, iterations)
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("status") != STATUS:
        raise ValueError("The manifest is not a convergence record of this format.")
    if manifest.get("implementation") != implementation_identity():
        raise FileExistsError(
            "The convergence control was recorded by a different implementation; move "
            f"{output} aside before recomputing."
        )
    baseline, _ = csm_consistency.validate_accepted_baseline(accepted_root)
    if manifest.get("accepted_baseline") != _baseline_block(baseline):
        raise ValueError("The accepted baseline differs from the one bound when the control was recorded.")
    changed = {
        "name": "FISTA iteration cap (-i)",
        "accepted": accepted_iteration_cap(baseline["fista_r0_command"]["argv"]),
        "control": iterations,
    }
    if manifest.get("changed_variable") != changed:
        raise ValueError("The recorded changed variable is not the fixed convergence contract.")
    argv = convergence_argv(accepted_root, output, iterations)
    command = manifest.get("command")
    if not isinstance(command, Mapping) or command.get("argv") != argv:
        raise ValueError("The recorded command differs from the command derived from the accepted record.")
    text = csm_consistency.format_command(argv)
    if command.get("text") != text or command.get("debug_level") != 5:
        raise ValueError("The recorded command text or debug level is not the reviewed one.")
    csm_consistency._verify_file(command.get("record"), paths["command_record"], "Command record")
    if paths["command_record"].read_text(encoding="utf-8").strip() != text:
        raise ValueError("The command record file differs from the reviewed command.")
    csm_consistency._verify_cfl(manifest.get("image"), paths["image"], "Convergence image")
    if list(manifest["image"]["shape"]) != list(baseline["fista_r0_image"]["shape"]):
        raise ValueError("The convergence image shape differs from the accepted image.")
    gpu = argv[2:3] == ["-g"]
    logs = manifest.get("logs")
    required = required_log_keys(gpu)
    if not isinstance(logs, Mapping) or set(logs) != set(required):
        raise ValueError(f"The recorded logs are not the fixed set {sorted(required)}.")
    for key in required:
        csm_consistency._verify_file(logs[key], paths[key], f"Log {key}")
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("shell_environment_log"), Mapping):
        raise ValueError("The convergence record has no environment log.")
    csm_consistency._verify_environment(manifest, output, "The convergence manifest")
    # Recorded summaries must equal what the verified files say.
    timing = parse_time_log(paths["time_log"].read_text(encoding="utf-8"))
    if timing["exit_status"] != 0:
        raise ValueError(f"BART exited with status {timing['exit_status']}.")
    expected_resources = {"time": timing, "gpu": _gpu_summary(paths, argv)}
    convergence = parse_debug_log(paths["debug_log"].read_text(encoding="utf-8"), iterations)
    if manifest.get("convergence") != json.loads(json.dumps(convergence)):
        raise ValueError("The recorded convergence summary differs from the verified debug log.")
    if manifest.get("resources") != json.loads(json.dumps(expected_resources)):
        raise ValueError("The recorded resources differ from the verified time and GPU logs.")
    if manifest.get("flags") != FIXED_FLAGS:
        raise ValueError("The recorded flags are not the fixed convergence flags.")
    # Closed set of files per workflow directory, so nothing can be added or swapped in.
    branch = branch_name(iterations)
    _require_entries(output, _ROOT_ENTRIES, "The output root", required={"normal", "logs", "manifests"})
    _require_entries(output / "normal", {"bart_output", "nifti"}, "normal/", required={"bart_output"})
    _require_entries(output / "normal" / "bart_output", {branch}, "normal/bart_output/", required={branch})
    _require_entries(output / "normal" / "nifti", {branch}, "normal/nifti/")
    image_files = {"image_wave.hdr", "image_wave.cfl", "wave_command.txt"}
    _require_entries(paths["image"].parent, image_files, f"normal/bart_output/{branch}/", required=image_files)
    _require_entries(output / "manifests", {"fista_convergence.json"}, "manifests/", required={"fista_convergence.json"})
    log_names = {paths[key].name for key in required}
    _require_entries(
        output / "logs", log_names | {"environment", f"convert_{branch}.log"}, "logs/", required=log_names | {"environment"}
    )
    recorded_log = Path(environment["shell_environment_log"]["relative_path"]).name
    _require_entries(output / "logs" / "environment", {recorded_log}, "logs/environment/", required={recorded_log})


def write_residual_figure(manifest: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Plot the recorded residual against iteration on a logarithmic axis.

    Printed zeros lie below BART's six-decimal resolution, so they are drawn at
    half that resolution and marked.

    Args:
        manifest: Convergence manifest.
        path: PNG destination, which must not exist yet.

    Returns:
        File record of the figure.

    Raises:
        FileExistsError: If the figure already exists.

    Side Effects:
        Writes one PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"The residual figure already exists: {destination}")
    convergence = manifest["convergence"]
    floor = RESIDUAL_PRINT_RESOLUTION / 2
    iterations = [index for index, _ in convergence["trace"]]
    values = [max(value, floor) for _, value in convergence["trace"]]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.semilogy(iterations, values, color="#1a6a6f", linewidth=1.4)
    axis.axhline(1e-6, color="#8c5500", linestyle="--", linewidth=1.0, label="tolerance 1e-6")
    axis.axvline(99, color="#53636f", linestyle=":", linewidth=1.0, label="accepted cap (100 iterations)")
    axis.set_xlabel("FISTA iteration k")
    axis.set_ylabel("relative normal-equation residual")
    axis.set_title(
        f"Final count {convergence['final_iteration_count']} of {convergence['iteration_cap']}; "
        f"stopped by tolerance: {convergence['stopped_by_tolerance']}",
        fontsize=10,
    )
    axis.legend(fontsize=8)
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)
    return rovir_feasibility._file_record(destination)
