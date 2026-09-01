#!/usr/bin/env bash
# Dispatch one review-gated pure-mask rerun operation to its Python entry point.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    cat >&2 <<'EOF'
Usage: run_pure_mask_rerun_stage.sh ACTION CONFIG OUTPUT_ROOT [REVIEWER_NOTE]

Actions:
  validate-sources      Validate rebuildable source inputs without writing.
  materialize-sources   Rebuild hash-identical accepted sources in the new tree.
  validate-inputs       Validate immutable inputs without writing outputs.
  prepare               Prepare the five pure-mask cases and references.
  validate-coarse       Validate the coarse candidate pool without BART.
  run-coarse            Run the coarse GPU BART sweep.
  evaluate-coarse       Evaluate the completed coarse sweep.
  refresh-coarse-evaluation
                        Refresh only manifest-owned coarse metrics and figures.
  validate-fine         Validate the explicitly configured fine candidate pool.
  run-fine              Run the fine GPU BART sweep.
  evaluate-fine         Evaluate the completed fine sweep.
  refresh-fine-evaluation
                        Refresh only manifest-owned fine metrics and figures.
  validate-shortlist    Validate the explicit manual-review shortlist.
  render-shortlist      Render the explicit manual-review shortlist.
  record-selections     Record reviewed choices; REVIEWER_NOTE is required.

This dispatcher never chains actions. The caller must activate the required
Conda environment and source the host-compatible BART startup script before a
run-coarse or run-fine action.
EOF
    exit 2
fi

ACTION="$1"
CONFIG="$2"
OUTPUT_ROOT="$3"
REVIEWER_NOTE="${4:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$ACTION" in
    validate-sources)
        exec python "$SCRIPT_DIR/materialize_pure_mask_sources.py" \
            --config "$CONFIG" \
            --validate-only
        ;;
    materialize-sources)
        exec python "$SCRIPT_DIR/materialize_pure_mask_sources.py" \
            --config "$CONFIG" \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    validate-inputs)
        exec python "$SCRIPT_DIR/prepare_pure_mask_rerun.py" \
            --config "$CONFIG" \
            --validate-only
        ;;
    prepare)
        exec python "$SCRIPT_DIR/prepare_pure_mask_rerun.py" \
            --config "$CONFIG" \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    validate-coarse)
        exec python "$SCRIPT_DIR/run_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage coarse \
            --validate-only
        ;;
    run-coarse)
        command -v bart >/dev/null
        echo "Backend: run_pure_mask_sweeps.py -> GPU BART wave -g"
        exec python "$SCRIPT_DIR/run_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage coarse \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    evaluate-coarse)
        exec python "$SCRIPT_DIR/evaluate_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage coarse \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    refresh-coarse-evaluation)
        exec python "$SCRIPT_DIR/evaluate_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage coarse \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume \
            --refresh-derived-outputs
        ;;
    validate-fine)
        exec python "$SCRIPT_DIR/run_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage fine \
            --validate-only
        ;;
    run-fine)
        command -v bart >/dev/null
        echo "Backend: run_pure_mask_sweeps.py -> GPU BART wave -g"
        exec python "$SCRIPT_DIR/run_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage fine \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    evaluate-fine)
        exec python "$SCRIPT_DIR/evaluate_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage fine \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    refresh-fine-evaluation)
        exec python "$SCRIPT_DIR/evaluate_pure_mask_sweeps.py" \
            --config "$CONFIG" \
            --stage fine \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume \
            --refresh-derived-outputs
        ;;
    validate-shortlist)
        exec python "$SCRIPT_DIR/render_pure_mask_shortlist.py" \
            --config "$CONFIG" \
            --stage fine \
            --validate-only
        ;;
    render-shortlist)
        exec python "$SCRIPT_DIR/render_pure_mask_shortlist.py" \
            --config "$CONFIG" \
            --stage fine \
            --confirm-output-root "$OUTPUT_ROOT" \
            --resume
        ;;
    record-selections)
        if [[ -z "$REVIEWER_NOTE" ]]; then
            echo "Error: record-selections requires a nonempty REVIEWER_NOTE." >&2
            exit 2
        fi
        exec python "$SCRIPT_DIR/record_pure_mask_selections.py" \
            --config "$CONFIG" \
            --confirm-output-root "$OUTPUT_ROOT" \
            --confirm-manual-visual-review \
            --reviewer-note "$REVIEWER_NOTE"
        ;;
    *)
        echo "Error: unknown action: $ACTION" >&2
        exit 2
        ;;
esac
