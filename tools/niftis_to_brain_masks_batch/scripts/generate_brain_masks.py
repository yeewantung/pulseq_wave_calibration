#!/usr/bin/env python3
"""Generate HD-BET brain masks for every subject of a reconstruction share.

Each mask is cut from that subject/contrast's fully sampled R3x1 baseline and
written beside the head mask already shipped with the data, so it can be reused
by any later analysis instead of being recomputed per run. A subject that
already has a complete mask is skipped unless ``--force`` is given.

Masks land as ``visual_review_required``. Review each ``brain_mask_hdbet_qc.png``
and run ``approve_brain_masks.py`` before anything scores against them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from brain_mask_batch.generate import (  # noqa: E402
    DEVICES,
    MaskResult,
    generate_mask,
    resolve_device,
    resolve_executable,
    summarize,
)
from brain_mask_batch.layout import (  # noqa: E402
    baseline_matches,
    discover_targets,
    is_complete,
    read_sidecar,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Mask every discovered baseline that does not already have one.

    Args:
        argv: Optional command-line argument vector.

    Returns:
        Zero once every target is either masked or deliberately skipped.

    Raises:
        SystemExit: If HD-BET is absent or nothing could be discovered.
    """
    args = _parser().parse_args(argv)
    executable = resolve_executable(args.hd_bet)
    device = resolve_device(args.device)

    targets = discover_targets(
        args.reconstruction_root,
        subjects=args.subjects,
        contrasts=args.contrasts,
        output_root=args.output_root,
    )
    if not targets:
        raise SystemExit(
            f"No baselines discovered under {args.reconstruction_root}. "
            "Check the share root, --subjects and --contrasts."
        )

    print(f"Discovered {len(targets)} baselines.\n")

    results: list[MaskResult] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, target in enumerate(targets, start=1):
        label = f"{target.subject} {target.contrast}"
        header = f"[{index}/{len(targets)}] {label}"

        if not args.force and is_complete(target):
            if args.verify_baseline and not baseline_matches(
                target, read_sidecar(target.sidecar_path)
            ):
                print(f"{header}: baseline changed since masking; regenerating")
            else:
                skipped.append(label)
                print(f"{header}: already masked, skipping")
                continue

        print(f"{header}: masking")
        try:
            results.append(
                generate_mask(
                    target,
                    executable=executable,
                    device=device,
                    disable_tta=not args.enable_tta,
                    verbose=args.verbose,
                )
            )
            print(f"{header}: {results[-1].volume_ml:.1f} mL -> {target.mask_path}")
        except (RuntimeError, OSError) as error:
            # One bad subject must not cost the whole batch.
            failed.append((label, str(error)))
            print(f"{header}: FAILED — {error}")

    print(
        f"\nGenerated {len(results)}, skipped {len(skipped)}, failed {len(failed)}."
    )
    if results:
        print(summarize(results))
    if failed:
        print("\nFailures:")
        for label, message in failed:
            print(f"  {label}: {message}")
    if results:
        print(
            "\nReview every brain_mask_hdbet_qc.png, then record approval:\n"
            f"  python scripts/approve_brain_masks.py "
            f"{args.reconstruction_root} --approved-by <name>"
        )
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reconstruction_root",
        type=Path,
        help="Share holding FMP_* subjects with fully sampled R3x1 baselines.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to mask; defaults to every FMP_* found.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=None,
        help="Contrasts to mask; defaults to every contrast holding a baseline.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Mirror masks under this root instead of writing beside the data. "
            "Use it to stage a run without touching a shared share."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate masks that already exist, discarding their approval.",
    )
    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help=(
            "Also re-hash each baseline when deciding to skip, catching a mask "
            "left over from an earlier reconstruction. Forces a download of "
            "online-only files."
        ),
    )
    parser.add_argument("--hd-bet", default="hd-bet", help="HD-BET executable name.")
    parser.add_argument(
        "--device",
        choices=list(DEVICES),
        default="auto",
        help=(
            "Torch device HD-BET predicts on. 'auto' takes the best this "
            "machine has (cuda, then mps, then cpu); an explicit device the "
            "machine cannot honour falls back with a warning rather than "
            "failing. nnU-Net still does resampling and export on CPU when "
            "predicting on Metal, which is expected."
        ),
    )
    parser.add_argument(
        "--enable-tta",
        action="store_true",
        help=(
            "Enable test-time augmentation: eight mirrored passes, slightly "
            "better masks and roughly eight times the compute."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Ask HD-BET for verbose progress."
    )
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
