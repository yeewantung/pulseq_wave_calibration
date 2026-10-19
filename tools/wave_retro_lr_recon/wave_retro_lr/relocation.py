"""Explicit, hash-bound read-only views of relocated MPRAGE source records.

Original documents are never rewritten. A user-created receipt authorizes only
listed historical documents; fresh outputs and unlisted files are unchanged.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import shlex

from .bart_io import sha256_file


RECEIPT = Path("metadata/path_relocation.local.json")


@lru_cache(maxsize=512)
def _cached_hash(path: str, signature: tuple) -> str:
    """Hash a file for a particular device/inode/size/mtime/ctime signature."""
    return sha256_file(Path(path))


def checked_hash(path: Path) -> str:
    """Return a content hash, invalidating cached checks when file identity changes."""
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    result = _cached_hash(str(path.resolve()), signature)
    after = path.stat()
    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"File changed during relocation validation: {path}")
    return result


def receipt_for(path: Path) -> tuple[Path, dict] | None:
    """Find and validate the nearest relocation receipt without creating files.

    The receipt is restricted to its reconstruction root and its dataset parent.
    Raw hashes are new migration-time identities, not invented historical hashes.
    """
    path = path.resolve()
    for root in path.parents:
        candidate = root / RECEIPT
        if not candidate.is_file():
            continue
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if (value.get("schema") != "mprage-path-relocation-v1"
                or value.get("root") != str(root)
                or value.get("target_dataset") != str(root.parent)
                or not value.get("documents") or not value.get("raw_sources")
                or not value.get("source_prefixes")):
            raise ValueError(f"Invalid relocation receipt: {candidate}")
        for prefix in value["source_prefixes"]:
            if (not Path(prefix).is_absolute() or ".." in Path(prefix).parts
                    or len(Path(prefix).parts) < 4 or prefix.rstrip("/") != prefix
                    or prefix == value["target_dataset"]):
                raise ValueError("Invalid relocation source prefix")
        for raw in value["raw_sources"]:
            actual = Path(raw["path"]).resolve(strict=True)
            if (actual.parent != root.parent or raw.get("historical_sha256_available") is not False
                    or raw.get("verification") != "user-confirmed-relocation-plus-size-mtime; current-sha256"):
                raise ValueError("Invalid raw relocation attestation")
            stat = actual.stat()
            if (stat.st_size != raw["size_bytes"] or stat.st_mtime_ns != raw["mtime_ns"]
                    or checked_hash(actual) != raw["sha256"]):
                raise ValueError("Relocated raw identity changed")
        return candidate, value
    return None


def relocation_record(path: Path) -> dict | None:
    """Return the receipt identity for inclusion in new derivative provenance."""
    found = receipt_for(path)
    if found is None:
        return None
    receipt, _ = found
    return {"path": str(receipt), "sha256": checked_hash(receipt)}


def _path_view(value: str, receipt: dict) -> str:
    """Translate an exact absolute prefix, never substrings or relative paths."""
    for prefix in receipt["source_prefixes"]:
        if value == prefix or value.startswith(prefix + "/"):
            suffix = Path(value).relative_to(prefix)
            target = Path(receipt["target_dataset"]) / suffix
            if ".." in suffix.parts or not target.resolve().is_relative_to(receipt["target_dataset"]):
                raise ValueError("Relocation path escapes the confirmed dataset")
            return str(target)
    return value


def _command_view(value: str, receipt: dict) -> str:
    """Rebase individual command arguments without executing shell text."""
    tokens = shlex.split(value)
    mapped = [_path_view(token, receipt) for token in tokens]
    return shlex.join(mapped) if mapped != tokens else value


def _view(value, receipt: dict, key: str = ""):
    """Build an in-memory JSON view while preserving scientific values and hashes."""
    if isinstance(value, dict):
        return {name: _view(item, receipt, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_view(item, receipt, key) for item in value]
    if isinstance(value, str):
        if key in ("expected_commands", "BARTWaveCommand", "BARTEcalibCommand"):
            return _command_view(value, receipt)
        return _path_view(value, receipt)
    return value


def _document_receipt(path: Path) -> dict | None:
    """Authorize a listed historical document by its original byte-level hash."""
    found = receipt_for(path)
    if found is None:
        return None
    _, receipt = found
    relative = str(path.resolve().relative_to(receipt["root"]))
    expected = receipt["documents"].get(relative)
    if expected is None:
        return None
    if checked_hash(path) != expected:
        raise ValueError(f"Historical relocation document changed: {path}")
    return receipt


def read_json(path: str | Path) -> dict:
    """Load a JSON object with an authorized relocation view; reject modified originals."""
    path = Path(path)
    receipt = _document_receipt(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return _view(value, receipt) if receipt is not None else value


def read_command(path: str | Path) -> str:
    """Read a command record, applying only its hash-authorized argument relocation."""
    path = Path(path)
    receipt = _document_receipt(path)
    value = path.read_text(encoding="utf-8").strip()
    return _command_view(value, receipt) if receipt is not None else value


def is_relocated_document(path: Path) -> bool:
    """Return whether a document has an immutable authorized relocation view."""
    return _document_receipt(path) is not None
