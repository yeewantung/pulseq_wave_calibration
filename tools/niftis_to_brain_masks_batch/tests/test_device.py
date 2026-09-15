"""Device selection falls back rather than failing on an unequipped machine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from brain_mask_batch import generate  # noqa: E402


@pytest.fixture()
def machine(monkeypatch):
    """Pretend this machine has exactly the given devices."""

    def configure(*devices: str) -> None:
        ordered = tuple(
            name for name in generate.DEVICE_PREFERENCE if name in set(devices)
        )
        monkeypatch.setattr(generate, "available_devices", lambda: ordered)

    return configure


def _messages() -> tuple[list[str], object]:
    recorded: list[str] = []
    return recorded, recorded.append


def test_available_devices_always_offers_cpu() -> None:
    available = generate.available_devices()
    assert "cpu" in available
    # Reported in preference order, best first.
    assert list(available) == [
        name for name in generate.DEVICE_PREFERENCE if name in available
    ]


def test_auto_prefers_cuda_then_mps_then_cpu(machine) -> None:
    machine("cuda", "mps", "cpu")
    assert generate.resolve_device("auto", log=lambda _: None) == "cuda"

    machine("mps", "cpu")
    assert generate.resolve_device("auto", log=lambda _: None) == "mps"

    machine("cpu")
    assert generate.resolve_device("auto", log=lambda _: None) == "cpu"


def test_mps_falls_back_to_cpu_when_undetected(machine) -> None:
    machine("cpu")
    recorded, log = _messages()
    assert generate.resolve_device("mps", log=log) == "cpu"
    assert any("not available" in message and "cpu" in message for message in recorded)


def test_cuda_falls_back_to_the_best_available(machine) -> None:
    machine("mps", "cpu")
    recorded, log = _messages()
    assert generate.resolve_device("cuda", log=log) == "mps"
    assert any("falling back" in message for message in recorded)


def test_an_available_device_is_honoured_silently(machine) -> None:
    machine("mps", "cpu")
    recorded, log = _messages()
    assert generate.resolve_device("cpu", log=log) == "cpu"
    assert recorded == []


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        generate.resolve_device("tpu", log=lambda _: None)


def test_auto_is_the_command_line_default() -> None:
    assert "auto" in generate.DEVICES
    assert generate.DEVICES[0] == "auto"
