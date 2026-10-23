#!/usr/bin/env python3
"""Write reviewed mask/QC proposals only; never produce CSM arrays or reconstruction."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    """Parse the local configuration and approved preview directory and print completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    from wave_retro_lr.metal_runner import configure_idle_process, memory_guard
    from wave_retro_lr.support_preview import prepare_support_preview
    configure_idle_process()
    memory_guard()
    print(prepare_support_preview(args.config, args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, MemoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
