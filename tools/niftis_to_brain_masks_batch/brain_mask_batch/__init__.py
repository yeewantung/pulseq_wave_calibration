"""Batch HD-BET brain masking, stored beside the reconstructions it describes."""

from __future__ import annotations

from .generate import MaskResult, generate_mask, resolve_executable
from .layout import MaskTarget, approve, discover_targets, is_complete, read_sidecar

__all__ = [
    "MaskResult",
    "MaskTarget",
    "approve",
    "discover_targets",
    "generate_mask",
    "is_complete",
    "read_sidecar",
    "resolve_executable",
]
