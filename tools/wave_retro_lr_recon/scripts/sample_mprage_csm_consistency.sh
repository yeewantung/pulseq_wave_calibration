#!/usr/bin/env bash
set -euo pipefail

# Calibration-only set-4 coil-sensitivity consistency diagnostics for one
# accepted Wave-MPRAGE normal root. Python validates, records, and computes the
# metrics; the only BART command is the diagnostic two-map ESPIRiT calibration
# in the calibrate stage, whose maps are never used for Wave reconstruction.

usage() {
    cat <<'EOF'
Usage: sample_mprage_csm_consistency.sh STAGE TWIX.dat SEQUENCE.seq ACCEPTED_NORMAL_ROOT OUTPUT_ROOT [options]

Stages:
  prepare                    validate sources, export the physical set-4 ACS, record noise and channel identity
  print-calibration-command  print the exact diagnostic BART command without running it
  calibrate                  run bart ecalib -m 2 -c 0 on the accepted PCA calibration k-space, then record it
  roi-template               export the canonical-RAS reference and an empty five-label template
  diagnose                   compute diagnostics, figures, and the report from reviewed ROIs

Options for diagnose (exactly one kind):
  --roi-labels PATH          reviewed five-label NIfTI drawn on OUTPUT_ROOT/rois/template
  --roi-box SPEC             inclusive LABEL=ro=a:b,lin=c:d,par=e:f box; repeatable

Confirm the exact OUTPUT_ROOT before the prepare stage creates it. No stage runs
Wave reconstruction, Soft-SENSE, PSF calibration, or refscan sets 0-3.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ $# -ge 5 ]] || { usage >&2; exit 2; }
STAGE="$1"
case "$STAGE" in
    prepare|print-calibration-command|calibrate|roi-template|diagnose) ;;
    *) echo "Error: unknown stage $STAGE" >&2; usage >&2; exit 2 ;;
esac
# Resolve paths exactly as Python does, so recorded commands compare token by token.
TWIX_FILE="$(realpath -m -- "$2")"
SEQUENCE_FILE="$(realpath -m -- "$3")"
ACCEPTED_NORMAL_ROOT="$(realpath -m -- "$4")"
OUTPUT_ROOT="$(realpath -m -- "$5")"
shift 5

ROI_ARGS=()
while (($#)); do
    case "$1" in
        --roi-labels)
            [[ $# -ge 2 ]] || { echo "Error: --roi-labels needs a path." >&2; exit 2; }
            ROI_ARGS+=(--roi-labels "$(realpath -m -- "$2")"); shift 2 ;;
        --roi-box)
            [[ $# -ge 2 ]] || { echo "Error: --roi-box needs a box specification." >&2; exit 2; }
            ROI_ARGS+=(--roi-box "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$STAGE" != diagnose && ${#ROI_ARGS[@]} -gt 0 ]]; then
    echo "Error: ROI options apply only to the diagnose stage." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CLI="$SCRIPT_DIR/mprage_csm_consistency.py"
KSPACE_CALIB="$ACCEPTED_NORMAL_ROOT/normal/bart_inputs/kspace_calib"
MAP_BASE="$OUTPUT_ROOT/csm/map2_uncropped/coil_sens"
EIGENVALUE_BASE="$OUTPUT_ROOT/csm/eigenvalues/ev_m2_c0"
COMMAND_RECORD="$OUTPUT_ROOT/csm/map2_uncropped/ecalib_command.txt"
INPUT_RECORD="$OUTPUT_ROOT/csm/map2_uncropped/ecalib_input.sha256"
CALIBRATION_MANIFEST="$OUTPUT_ROOT/manifests/two_map_calibration.json"
LOG_DIR="$OUTPUT_ROOT/logs"

printf -v ECALIB_COMMAND '%q ' bart ecalib -m 2 -c 0 "$KSPACE_CALIB" "$MAP_BASE" "$EIGENVALUE_BASE"
ECALIB_COMMAND="${ECALIB_COMMAND% }"

record_environment() {
    local stage="$1"
    local directory="$LOG_DIR/environment"
    mkdir -p "$directory"
    # Every invocation writes its own uniquely named log, so a log recorded by
    # a manifest is never rewritten; Python records it only for a new manifest.
    ENVIRONMENT_LOG="$(mktemp --suffix=.txt "$directory/${stage}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    chmod a+r "$ENVIRONMENT_LOG"
    {
        echo "stage: $stage"
        echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host: $(hostname -s)"
        echo "python: $(command -v python) $(python --version 2>&1)"
        if command -v bart >/dev/null; then echo "bart: $(command -v bart)"; fi
        echo "repository_commit: $(git -C "$REPOSITORY_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo "repository_status:"
        git -C "$REPOSITORY_ROOT" status --porcelain=v1 2>/dev/null || true
        echo "submodules:"
        git -C "$REPOSITORY_ROOT" submodule status 2>/dev/null || true
    } > "$ENVIRONMENT_LOG"
}

pair_state() {
    local base="$1"
    if [[ -f "$base.hdr" && -f "$base.cfl" ]]; then
        echo complete
    elif [[ -e "$base.hdr" || -e "$base.cfl" ]]; then
        echo partial
    else
        echo absent
    fi
}

if [[ "$STAGE" == print-calibration-command ]]; then
    printf '%s\n' "$ECALIB_COMMAND"
    exit 0
fi

command -v python >/dev/null || { echo "Error: python is not on PATH; follow SETUP.md." >&2; exit 2; }

case "$STAGE" in
    prepare)
        record_environment prepare
        python "$CLI" prepare "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --environment-log "$ENVIRONMENT_LOG"
        ;;
    calibrate)
        command -v bart >/dev/null || { echo "Error: bart is not on PATH; source the host BART startup script." >&2; exit 2; }
        [[ -x /usr/bin/time ]] || { echo "Error: GNU time (/usr/bin/time) is required to record runtime and peak memory." >&2; exit 2; }
        [[ -f "$OUTPUT_ROOT/manifests/csm_consistency_prepare.json" ]] || { echo "Error: run the prepare stage before calibrate." >&2; exit 2; }
        echo "Revalidating sources and accepted arrays before calibration."
        python "$CLI" validate "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" >/dev/null
        record_environment calibrate
        MAP_STATE="$(pair_state "$MAP_BASE")"
        EIGENVALUE_STATE="$(pair_state "$EIGENVALUE_BASE")"
        # Existing outputs are reused only through their calibration manifest,
        # which record-calibration verifies; a matching command text does not
        # show which k-space the maps were computed from.
        if [[ -f "$CALIBRATION_MANIFEST" ]]; then
            echo "Verifying the recorded diagnostic two-map calibration."
        elif [[ "$MAP_STATE" != absent || "$EIGENVALUE_STATE" != absent || -e "$COMMAND_RECORD" || -e "$INPUT_RECORD" ]]; then
            echo "Error: two-map outputs exist without a calibration manifest (interrupted or superseded run); inspect and move them aside before rerunning." >&2
            exit 2
        else
            mkdir -p "$(dirname -- "$MAP_BASE")" "$(dirname -- "$EIGENVALUE_BASE")"
            bart version > "$LOG_DIR/bart_version.txt" 2>&1
            # Name the canonical executable, independent of relative PATH entries.
            sha256sum "$(realpath -e -- "$(command -v bart)")" > "$LOG_DIR/bart_binary.sha256"
            # Bind the outputs to the exact calibration k-space that ecalib reads.
            sha256sum "$KSPACE_CALIB.hdr" "$KSPACE_CALIB.cfl" > "$INPUT_RECORD"
            echo "Running: $ECALIB_COMMAND"
            # The only BART command of this workflow: diagnostic two-map ESPIRiT
            # without cropping. Its maps are never used for Wave reconstruction.
            BART_DEBUG_LEVEL=3 /usr/bin/time -v bart ecalib -m 2 -c 0 "$KSPACE_CALIB" "$MAP_BASE" "$EIGENVALUE_BASE" > "$LOG_DIR/ecalib_m2_c0.log" 2>&1
            printf '%s\n' "$ECALIB_COMMAND" > "$COMMAND_RECORD"
        fi
        python "$CLI" record-calibration "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --environment-log "$ENVIRONMENT_LOG"
        ;;
    roi-template)
        record_environment roi-template
        python "$CLI" roi-template "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" --environment-log "$ENVIRONMENT_LOG"
        echo "Review: $OUTPUT_ROOT/rois/template/README.txt"
        ;;
    diagnose)
        [[ ${#ROI_ARGS[@]} -gt 0 ]] || { echo "Error: diagnose needs --roi-labels or --roi-box." >&2; exit 2; }
        record_environment diagnose
        python "$CLI" diagnose "$TWIX_FILE" "$SEQUENCE_FILE" "$ACCEPTED_NORMAL_ROOT" "$OUTPUT_ROOT" "${ROI_ARGS[@]}" --environment-log "$ENVIRONMENT_LOG"
        echo "Report: $OUTPUT_ROOT/reports/csm_consistency_report.md"
        ;;
esac
