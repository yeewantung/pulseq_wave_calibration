"""Explicit CPU execution for isolated metal CSM experiments, never a default pipeline stage."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import nibabel
import numpy
import scipy

from .bart_io import cfl_record, sha256_file
from .metal_csm import METHOD_VERSION, MetalCSMParameters
from .metal_workflow import (
    export_result, file_record, prepare_candidate, read_json, source_contract,
    validate_cfl, validate_destination, write_json,
)

START_AVAILABLE_GIB = 112
STOP_AVAILABLE_GIB = 64
STOP_RSS_GIB = 64
ADDRESS_SPACE_GIB = 96


def available_gib() -> float:
    """Return Linux host MemAvailable in GiB; fail closed if unavailable."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])/1024**2
    raise RuntimeError("Linux MemAvailable is required for this CPU-only experimental runner")


def memory_guard() -> None:
    """Abort our preparation when host available RAM is below the fixed 64-GiB safety reserve."""
    if available_gib() < STOP_AVAILABLE_GIB:
        raise MemoryError("Yielding to other workloads: host MemAvailable <64 GiB")


def configure_idle_process() -> None:
    """Apply nice 19, Linux SCHED_IDLE, a 96-GiB address-space cap and disabled core dumps.

    Affects only the caller (or our BART child through preexec_fn), never another
    user's task. Unsupported scheduling APIs or permission failures propagate.
    """
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 19:
        os.nice(19-current)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = ADDRESS_SPACE_GIB*1024**3
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def update_state(path: Path, value: dict) -> None:
    """Atomically update only this run's mutable status JSON; no scientific artifacts are replaced."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        import json
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def cpu_command(preparation: dict, output: Path, bart: str) -> list[str]:
    """Return an explicit native CPU wave command retaining the baseline lambda and fixed solver settings."""
    source = preparation["request"]["source"]
    return [bart, "wave", "-w", "-f", "-r", format(source["regularization"], ".17g"),
            "-i", "100", "-t", "1e-6", preparation["candidate"]["base"],
            source["records"]["psf"]["base"], source["records"]["kspace"]["base"],
            str(output/"reconstruction/image_wave")]


def run_cpu_reconstruction(output: Path, preparation: dict, threads: int = 8) -> dict:
    """Run or strictly reuse one CPU BART reconstruction; stop only our child on memory pressure.

    Args:
        output: Explicit, initialized and locked derivative directory.
        preparation: Validated candidate and immutable source request.
        threads: Positive OpenMP thread cap, at most eight; BLAS remains single-threaded.
    Returns:
        Completed execution record with input/output/tool hashes and resource usage.
    Raises:
        ValueError: Reuse mismatch or partial prior attempt.
        MemoryError: Insufficient headroom before/during execution.
        RuntimeError: BART unavailable or unsuccessful. Partial files are preserved.
    """
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("CPU threads must be an integer between 1 and 8")
    bart = shutil.which("bart")
    if bart is None:
        raise RuntimeError("Source the compatible host BART startup before reconstruction")
    bart = str(Path(bart).resolve())
    command = cpu_command(preparation, output, bart)
    identity = {"command": command, "bart": file_record(bart), "gpu": False,
                "omp_threads": threads, "blas_threads": 1, "scheduler": "SCHED_IDLE",
                "preparation_sha256": sha256_file(output/"metadata/preparation.json"),
                "start_available_gib": START_AVAILABLE_GIB, "stop_available_gib": STOP_AVAILABLE_GIB,
                "stop_rss_gib": STOP_RSS_GIB, "address_space_gib": ADDRESS_SPACE_GIB}
    state_path = output/"metadata/reconstruction.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("status") != "complete" or state.get("identity") != identity:
            raise ValueError("Incomplete/mismatched reconstruction attempt; use a fresh output directory")
        validate_cfl(state["output"])
        for record in state["logs"]:
            if file_record(record["path"]) != record:
                raise ValueError("Reconstruction log hash mismatch")
        return state
    if any((output/"reconstruction").iterdir()) or any((output/"logs").iterdir()):
        raise ValueError("Unbound/partial reconstruction outputs exist; refusing overwrite")
    if available_gib() < START_AVAILABLE_GIB:
        raise MemoryError("Need at least 112 GiB available RAM; no BART job launched. Retry later.")
    for record in (preparation["candidate"], *preparation["request"]["source"]["records"].values()):
        validate_cfl(record)
    # Recheck admission after network reads; other workloads may have allocated RAM meanwhile.
    if available_gib() < START_AVAILABLE_GIB:
        raise MemoryError("RAM headroom decreased during validation; no BART job launched")
    if shutil.disk_usage(output).free < 2*1024**3:
        raise RuntimeError("Need at least 2 GiB free disk space before BART execution")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BART_DEBUG_LEVEL="3")
    state = {"status": "starting", "identity": identity, "started_unix": time.time(),
             "peak_rss_gib": 0., "minimum_available_gib": available_gib()}
    log_path, memory_path = output/"logs/bart.log", output/"logs/memory.tsv"
    process = None
    update_state(state_path, state)
    try:
        with log_path.open("x") as log, memory_path.open("x") as memory:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       preexec_fn=configure_idle_process)
            state.update(status="running", pid=process.pid)
            update_state(state_path, state)
            memory.write("elapsed_seconds\trss_gib\tavailable_gib\n")
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss = next(int(line.split()[1])/1024**2 for line in status.splitlines() if line.startswith("VmRSS:"))
                except (FileNotFoundError, StopIteration):
                    time.sleep(1)
                    continue
                available = available_gib()
                state["peak_rss_gib"] = max(state["peak_rss_gib"], rss)
                state["minimum_available_gib"] = min(state["minimum_available_gib"], available)
                memory.write(f"{time.time()-state['started_unix']:.1f}\t{rss:.3f}\t{available:.3f}\n")
                memory.flush()
                if rss > STOP_RSS_GIB or available < STOP_AVAILABLE_GIB:
                    raise MemoryError("Yielding memory: own RSS >64 GiB or host MemAvailable <64 GiB")
                update_state(state_path, state)
                time.sleep(5)
            if process.wait() != 0:
                raise RuntimeError(f"BART returned {process.returncode}; partial outputs preserved")
        state["output"] = cfl_record(output/"reconstruction/image_wave")
        validate_cfl(state["output"])
        for record in (preparation["candidate"], *preparation["request"]["source"]["records"].values()):
            validate_cfl(record)
        state["logs"] = [file_record(log_path), file_record(memory_path)]
        state["status"] = "complete"
        return state
    except BaseException as exc:
        state.update(status="failed_or_stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state["returncode"] = process.returncode
        state["elapsed_seconds"] = time.time()-state["started_unix"]
        update_state(state_path, state)


def run_metal_workflow(
    source_manifest: str | Path, reference_nifti: str | Path, output: str | Path,
    parameters: MetalCSMParameters, *, enabled: bool = False, prepare_only: bool = False,
    threads: int = 8,
) -> dict:
    """Prepare/reconstruct an explicitly enabled native MPRAGE metal derivative.

    Args:
        source_manifest: Completed standard normal R3x1 branch manifest.
        reference_nifti: Hash-identical original magnitude export/collection copy.
        output: User-chosen independent output directory, never inferred here.
        parameters: Explicit reviewed spatial parameters; no automatic ROI selection.
        enabled: Mandatory explicit opt-in; default false rejects before any I/O.
        prepare_only: When true, do not locate or launch BART.
        threads: CPU OpenMP cap (1--8); part of the immutable request.
    Returns:
        Preparation or comparison manifest. No existing canonical output is changed.
    Raises:
        ValueError: Disabled feature, unsafe output, mismatch or incomplete reuse.
        BlockingIOError: Another process owns this output's advisory lock.
    """
    if enabled is not True:
        raise ValueError("Metal CSM extrapolation is disabled by default; explicit opt-in is required")
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("CPU threads must be an integer between 1 and 8")
    parameters = MetalCSMParameters.from_dict(parameters.to_dict())
    source = source_contract(source_manifest, reference_nifti)
    output = Path(output).expanduser().resolve()
    validate_destination(output, source)
    software = {name: sha256_file(Path(__file__).with_name(name)) for name in ("metal_csm.py", "metal_workflow.py", "metal_runner.py")}
    request = {"method_version": METHOD_VERSION, "enabled": True, "default_enabled": False,
               "source": source, "parameters": parameters.to_dict(), "cpu_threads": threads,
               "software_sha256": software,
               "libraries": {"numpy": numpy.__version__, "scipy": scipy.__version__, "nibabel": nibabel.__version__}}
    if output.exists() and any(output.iterdir()) and not (output/"metadata/request.json").is_file():
        raise ValueError("Output directory is nonempty and is not a recognized metal derivative")
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Symlinks inside a metal derivative output are not supported")
    output.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(output/".metal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = output/"metadata/request.json"
        if request_path.exists():
            if read_json(request_path) != request:
                raise ValueError("Metal request mismatch; changing inputs, Ncc, ROI or settings needs a fresh output")
        else:
            if set(p.name for p in output.iterdir()) != {".metal.lock"}:
                raise ValueError("Partial output initialization; refusing overwrite")
            for folder in ("intermediate", "reconstruction", "nifti", "figures", "metadata", "logs"):
                (output/folder).mkdir()
            write_json(request_path, request)
        memory_guard()
        maps_bytes = source["records"]["maps"]["payload_bytes"]
        if not (output/"metadata/preparation.json").exists() and shutil.disk_usage(output).free < maps_bytes+2*1024**3:
            raise RuntimeError("Insufficient free space for a new CSM derivative")
        preparation = prepare_candidate(output, request, checkpoint=memory_guard)
        if prepare_only:
            return preparation
        execution = run_cpu_reconstruction(output, preparation, threads)
        # Revalidate all metadata and original exports after execution, not just the large arrays.
        if source_contract(source_manifest, reference_nifti) != source:
            raise ValueError("Source contract changed during reconstruction")
        return export_result(output, preparation, execution)
    finally:
        os.close(lock_fd)
