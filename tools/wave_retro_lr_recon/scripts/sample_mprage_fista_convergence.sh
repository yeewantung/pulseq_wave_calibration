#!/usr/bin/env bash
set -euo pipefail

# Logged FISTA-r0 convergence control for one accepted Wave-MPRAGE normal root.
# Only the iteration cap and the output image differ from the accepted FISTA-r0
# record; the CSM, PSF, wave k-space, device flag, and other flags are the
# accepted ones, read in place. The single BART command of this workflow is the
# bart wave call in the reconstruct stage; Python validates and records only.

usage() {
    cat <<'EOF'
Usage: sample_mprage_fista_convergence.sh STAGE TWIX.dat SEQUENCE.seq ACCEPTED_NORMAL_ROOT OUTPUT_ROOT [options]

Stages:
  print-command   print the exact BART command without running it
  reconstruct     validate, check free disk (and GPU memory), run bart wave with full logging, record
  convert         export the recorded image to NIfTI with the existing converter

Options:
  --iterations N        FISTA iteration cap (default 300)
  --protected-root DIR  root the output must not overlap, such as the Stage 3 root; repeatable
  --min-free-gb G       required free space on the output file system (default 5)
  --min-gpu-free-mib M  required free GPU memory with -g (default 32768)
  -g                    run on the GPU; must match the accepted FISTA-r0 record

Confirm the exact OUTPUT_ROOT before reconstruct creates it. No stage runs coil
calibration, Soft-SENSE, PSF calibration, or any regularized reconstruction.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 5 ]] || { usage >&2; exit 2; }
STAGE="$1"
case "$STAGE" in
    print-command|reconstruct|convert) ;;
    *) echo "Error: unknown stage $STAGE" >&2; usage >&2; exit 2 ;;
esac
# Resolve paths exactly as Python does, so commands compare token by token.
TWIX_FILE="$(realpath -m -- "$2")"
SEQUENCE_FILE="$(realpath -m -- "$3")"
ACCEPTED_NORMAL_ROOT="$(realpath -m -- "$4")"
OUTPUT_ROOT="$(realpath -m -- "$5")"
shift 5

ITERATIONS=300
MIN_FREE_GB=5
MIN_GPU_FREE_MIB=32768
GPU_FLAG=()
PROTECTED_ARGS=()
while (($#)); do
    case "$1" in
        --iterations) [[ $# -ge 2 ]] || { echo "Error: --iterations needs a value." >&2; exit 2; }; ITERATIONS="$2"; shift 2 ;;
        --protected-root) [[ $# -ge 2 ]] || { echo "Error: --protected-root needs a path." >&2; exit 2; }; PROTECTED_ARGS+=(--protected-root "$(realpath -m -- "$2")"); shift 2 ;;
        --min-free-gb) [[ $# -ge 2 ]] || { echo "Error: --min-free-gb needs a value." >&2; exit 2; }; MIN_FREE_GB="$2"; shift 2 ;;
        --min-gpu-free-mib) [[ $# -ge 2 ]] || { echo "Error: --min-gpu-free-mib needs a value." >&2; exit 2; }; MIN_GPU_FREE_MIB="$2"; shift 2 ;;
        -g) GPU_FLAG=(-g); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --iterations must be a positive integer." >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CLI="$SCRIPT_DIR/mprage_fista_convergence.py"
BRANCH="fista_r0_i${ITERATIONS}"
BRANCH_DIR="$OUTPUT_ROOT/normal/bart_output/$BRANCH"
IMAGE_BASE="$BRANCH_DIR/image_wave"
COMMAND_RECORD="$BRANCH_DIR/wave_command.txt"
NIFTI_DIR="$OUTPUT_ROOT/normal/nifti/$BRANCH"
LOG_DIR="$OUTPUT_ROOT/logs"
MANIFEST="$OUTPUT_ROOT/manifests/fista_convergence.json"

# The only BART computation: the accepted FISTA-r0 command with a new cap and image.
WAVE_ARGV=(bart wave "${GPU_FLAG[@]}" -w -f -r 0 -i "$ITERATIONS" -t 1e-6
    "$ACCEPTED_NORMAL_ROOT/normal/bart_output/coil_sens"
    "$ACCEPTED_NORMAL_ROOT/normal/bart_inputs/psf"
    "$ACCEPTED_NORMAL_ROOT/normal/bart_inputs/wave_kspace"
    "$IMAGE_BASE")
printf -v WAVE_COMMAND '%q ' "${WAVE_ARGV[@]}"
WAVE_COMMAND="${WAVE_COMMAND% }"

record_environment() {
    local stage="$1"
    local directory="$LOG_DIR/environment"
    mkdir -p "$directory"
    # Every invocation writes its own uniquely named log; a recorded log is never rewritten.
    ENVIRONMENT_LOG="$(mktemp --suffix=.txt "$directory/${stage}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    chmod a+r "$ENVIRONMENT_LOG"
    {
        echo "stage: $stage"
        echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host: $(hostname -s)"
        echo "python: $(command -v python) $(python --version 2>&1)"
        echo "bart: $(command -v bart)"
        echo "repository_commit: $(git -C "$REPOSITORY_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo "repository_status:"
        git -C "$REPOSITORY_ROOT" status --porcelain=v1 2>/dev/null || true
    } >| "$ENVIRONMENT_LOG"  # the fresh mktemp file; every other log is written with noclobber
}

existing_ancestor() {
    local path="$1"
    while [[ ! -e "$path" ]]; do path="$(dirname -- "$path")"; done
    printf '%s' "$path"
}

if [[ "$STAGE" == print-command ]]; then
    printf '%s\n' "$WAVE_COMMAND"
    exit 0
fi

command -v python >/dev/null || { echo "Error: python is not on PATH; activate the conda environment." >&2; exit 2; }
case "$STAGE" in
    reconstruct)
        command -v bart >/dev/null || { echo "Error: bart is not on PATH; source the host BART startup script." >&2; exit 2; }
        [[ -x /usr/bin/time ]] || { echo "Error: GNU time (/usr/bin/time) is required to record runtime and peak memory." >&2; exit 2; }
        if ((${#GPU_FLAG[@]})); then
            command -v nvidia-smi >/dev/null || { echo "Error: nvidia-smi is required with -g." >&2; exit 2; }
        fi
        echo "Validating the accepted baseline, the command, and existing outputs."
        STATE_LINE="$(python "$CLI" validate "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --iterations "$ITERATIONS" \
            "${PROTECTED_ARGS[@]}" --command-text "$WAVE_COMMAND" | tail -n 1)"
        if [[ "$STATE_LINE" == STATE=recorded ]]; then
            echo "Verified the recorded convergence control; nothing was run."
            python "$CLI" summary "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --iterations "$ITERATIONS" "${PROTECTED_ARGS[@]}"
            exit 0
        fi
        [[ "$STATE_LINE" == STATE=absent ]] || { echo "Error: unexpected validation result: $STATE_LINE" >&2; exit 2; }
        # Recheck free space immediately before every run.
        FREE_BYTES="$(df -B1 --output=avail "$(existing_ancestor "$OUTPUT_ROOT")" | tail -n 1 | tr -d ' ')"
        echo "Free space on the output file system: $((FREE_BYTES / 1000000000)) GB (required ${MIN_FREE_GB} GB)."
        ((FREE_BYTES >= MIN_FREE_GB * 1000000000)) || { echo "Error: not enough free disk space for this run." >&2; exit 2; }
        if ((${#GPU_FLAG[@]})); then
            GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
            echo "Free GPU memory: ${GPU_FREE_MIB} MiB (required ${MIN_GPU_FREE_MIB} MiB)."
            ((GPU_FREE_MIB >= MIN_GPU_FREE_MIB)) || { echo "Error: not enough free GPU memory; rerun later rather than switching device." >&2; exit 2; }
        fi
        # From here on no existing file may be overwritten: validation required an
        # absent or empty output root, and noclobber guards every redirection.
        set -o noclobber
        mkdir -p "$BRANCH_DIR" "$LOG_DIR"
        record_environment reconstruct
        bart version > "$LOG_DIR/bart_version.txt" 2>&1
        sha256sum "$(realpath -e -- "$(command -v bart)")" > "$LOG_DIR/bart_binary.sha256"
        if ((${#GPU_FLAG[@]})); then
            # Sample device and per-process GPU memory once per second during the run.
            nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -lms 1000 > "$LOG_DIR/gpu_device.csv" 2>&1 &
            DEVICE_SAMPLER=$!
            nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv -lms 1000 > "$LOG_DIR/gpu_processes.csv" 2>&1 &
            PROCESS_SAMPLER=$!
            trap 'kill "$DEVICE_SAMPLER" "$PROCESS_SAMPLER" 2>/dev/null || true' EXIT
        fi
        [[ ! -e "$LOG_DIR/bart_wave.time.txt" ]] || { echo "Error: a time record already exists; refusing to overwrite it." >&2; exit 2; }
        echo "Running: $WAVE_COMMAND"
        # Debug level 5 prints the residual of every FISTA iteration; it changes logging only.
        BART_DEBUG_LEVEL=5 /usr/bin/time -v -o "$LOG_DIR/bart_wave.time.txt" "${WAVE_ARGV[@]}" 2>&1 \
            | while IFS= read -r line; do printf '%s\t%s\n' "$(date +%s.%N)" "$line"; done > "$LOG_DIR/bart_wave.debug5.log"
        if ((${#GPU_FLAG[@]})); then
            kill "$DEVICE_SAMPLER" "$PROCESS_SAMPLER" 2>/dev/null || true
            wait "$DEVICE_SAMPLER" "$PROCESS_SAMPLER" 2>/dev/null || true
            trap - EXIT
        fi
        printf '%s\n' "$WAVE_COMMAND" > "$COMMAND_RECORD"
        python "$CLI" record "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --iterations "$ITERATIONS" \
            "${PROTECTED_ARGS[@]}" --environment-log "$ENVIRONMENT_LOG"
        ;;
    convert)
        [[ -f "$MANIFEST" ]] || { echo "Error: run the reconstruct stage before convert." >&2; exit 2; }
        python "$CLI" summary "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --iterations "$ITERATIONS" "${PROTECTED_ARGS[@]}" > /dev/null
        if [[ -e "$NIFTI_DIR" ]] && [[ -n "$(ls -A -- "$NIFTI_DIR")" ]]; then
            echo "Error: $NIFTI_DIR is not empty; inspect and move it aside before converting again." >&2
            exit 2
        fi
        set -o noclobber
        mkdir -p "$NIFTI_DIR" "$LOG_DIR"
        python "$SCRIPT_DIR/convert_mprage_bart_to_nifti.py" --bart-inputs "$ACCEPTED_NORMAL_ROOT/normal/bart_inputs" \
            --image "$IMAGE_BASE" --twix "$TWIX_FILE" --seq "$SEQUENCE_FILE" --output "$NIFTI_DIR" \
            --suffix "BARTWaveMPRAGENormalFISTAR0I${ITERATIONS}" > "$LOG_DIR/convert_${BRANCH}.log" 2>&1
        echo "NIfTI: $NIFTI_DIR"
        ;;
esac
