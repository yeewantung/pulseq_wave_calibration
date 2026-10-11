"""Resolve and immutably bind MPRAGE ROVir source inputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .bart_io import sha256_file
from .standard import (
    DEFAULT_VIRTUAL_COILS,
    STANDARD_VARIANT_PATTERN,
)

ROVIR_SOURCE_ANCHOR = Path("normal") / "rovir" / "source_inputs"
ROVIR_SOURCE_BINDING = Path("normal") / "rovir" / "source_binding.json"


def discover_rovir_source_inputs(output_root: str | Path) -> tuple[Path, ...]:
    """Discover manifest-bearing source-input trees in deterministic priority.

    Args:
        output_root: Existing top-level reconstruction root.

    Returns:
        Legacy normal inputs, VCC24, other descending ``vccN`` inputs, and a
        ROVir-only source anchor when each corresponding manifest exists.
    """

    root = Path(output_root).expanduser().resolve()
    candidates: list[Path] = []
    legacy = root / "normal" / "bart_inputs"
    if (legacy / "manifest.json").is_file():
        candidates.append(legacy)
    variants: list[tuple[int, Path]] = []
    if root.is_dir():
        for child in root.iterdir():
            match = STANDARD_VARIANT_PATTERN.fullmatch(child.name)
            if match is None or not child.is_dir():
                continue
            inputs = child / "normal" / "bart_inputs"
            if (inputs / "manifest.json").is_file():
                variants.append((int(match.group(1)), inputs))
    variants.sort(
        key=lambda item: (item[0] == DEFAULT_VIRTUAL_COILS, item[0]),
        reverse=True,
    )
    candidates.extend(inputs for _, inputs in variants)
    anchor = root / ROVIR_SOURCE_ANCHOR
    if (anchor / "manifest.json").is_file():
        candidates.append(anchor)
    return tuple(candidates)


def incomplete_rovir_source_layouts(output_root: str | Path) -> tuple[Path, ...]:
    """Report legacy or ``vccN`` layouts that lack a usable manifest.

    Args:
        output_root: Existing top-level reconstruction root.

    Returns:
        Candidate directories whose presence prevents an implicit bootstrap.
    """

    root = Path(output_root).expanduser().resolve()
    incomplete: list[Path] = []
    legacy = root / "normal" / "bart_inputs"
    if legacy.exists() and not (legacy / "manifest.json").is_file():
        incomplete.append(legacy)
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if STANDARD_VARIANT_PATTERN.fullmatch(child.name) is None:
                continue
            inputs = child / "normal" / "bart_inputs"
            if not (inputs / "manifest.json").is_file():
                incomplete.append(inputs)
    return tuple(incomplete)


def resolve_rovir_source_inputs(output_root: str | Path) -> Path:
    """Resolve the immutable binding or the highest-priority available source.

    Args:
        output_root: Existing top-level reconstruction root.

    Returns:
        Selected source-input directory.

    Raises:
        FileNotFoundError: If no bound or discoverable manifest exists.
        ValueError: If an existing binding is malformed, escaped, or changed.
    """

    root = Path(output_root).expanduser().resolve()
    binding_path = root / ROVIR_SOURCE_BINDING
    if binding_path.is_file():
        binding = _read_json(binding_path)
        if binding.get("status") != "mprage_rovir_source_bound":
            raise ValueError("ROVir source binding has an unsupported status.")
        manifest_record = binding.get("manifest")
        if not isinstance(manifest_record, Mapping) or not isinstance(
            manifest_record.get("path"), str
        ):
            raise ValueError("ROVir source binding lacks its manifest identity.")
        manifest = Path(manifest_record["path"]).expanduser().resolve()
        allowed = {
            (inputs / "manifest.json").resolve()
            for inputs in discover_rovir_source_inputs(root)
        }
        if manifest not in allowed:
            raise ValueError("ROVir source binding references an unavailable source tree.")
        if manifest_record.get("sha256") != sha256_file(manifest):
            raise ValueError("ROVir source manifest changed after it was bound.")
        return manifest.parent
    candidates = discover_rovir_source_inputs(root)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No legacy, vccN, or ROVir-only source manifest exists below {root}."
    )


def bind_rovir_source_inputs(output_root: str | Path, source_inputs: str | Path) -> dict[str, Any]:
    """Persist one immutable ROVir source-manifest selection.

    Args:
        output_root: Existing top-level reconstruction root.
        source_inputs: Selected manifest-bearing input directory.

    Returns:
        Newly written or strictly reused binding manifest.

    Raises:
        FileNotFoundError: If the selected manifest is absent.
        ValueError: If the path is not discoverable or another source is bound.

    Side Effects:
        Atomically writes ``normal/rovir/source_binding.json``.
    """

    root = Path(output_root).expanduser().resolve()
    inputs = Path(source_inputs).expanduser().resolve()
    manifest = inputs / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if inputs not in discover_rovir_source_inputs(root):
        raise ValueError("Selected ROVir source inputs are outside supported layouts.")
    record = {
        "path": str(manifest),
        "size_bytes": manifest.stat().st_size,
        "sha256": sha256_file(manifest),
    }
    payload = {
        "format_version": 1,
        "status": "mprage_rovir_source_bound",
        "output_root": str(root),
        "source_inputs": str(inputs),
        "manifest": record,
        "selection_policy": (
            "legacy normal, then vcc24, then other descending vccN; "
            "ROVir-only anchor only when no compatible standard source exists"
        ),
    }
    binding_path = root / ROVIR_SOURCE_BINDING
    if binding_path.is_file():
        existing = _read_json(binding_path)
        if existing != payload:
            raise ValueError("A different immutable ROVir source is already bound.")
        return existing
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".source_binding-", suffix=".json", dir=binding_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(binding_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON mapping.

    Raises:
        ValueError: If the file does not contain an object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
