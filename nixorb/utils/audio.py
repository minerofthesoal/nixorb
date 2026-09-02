"""nixorb/utils/audio.py — Audio device utilities.

The recurring failure mode this module exists to prevent: NixOrb opens an
``sd.InputStream`` on whatever ``sd.default.device`` resolves to, records for
up to 30s, and reports "no speech detected" — with no indication of *which*
device it just listened to. On PipeWire/PulseAudio systems the resolved
default is frequently a monitor source (a loopback of what your speakers are
playing, not your microphone), an inactive virtual sink, or simply not the
device you plugged in. That listens successfully and hears nothing, forever,
with no error to act on.

Everything here is about making device selection explicit and diagnosable:
resolve a real input device up front, log which one was picked and why, and
prefer physical microphones over monitors/loopbacks when nothing is
configured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import sounddevice as sd

log = logging.getLogger(__name__)

# Devices whose names match these are loopbacks of *output* audio, not
# microphones — PipeWire/PulseAudio expose them as "input" devices because
# you can technically record from them, which is exactly what makes them a
# silent trap: recording your speaker output while you talk into a mic that
# was never opened.
_MONITOR_MARKERS = ("monitor of", ".monitor", "loopback")


@dataclass
class ResolvedDevice:
    index: int | None  # None means "let the backend pick the OS default"
    name: str
    reason: str  # human-readable explanation, for logging


def _is_monitor(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _MONITOR_MARKERS)


def list_input_devices() -> list[dict]:
    """All devices with at least one input channel, annotated for the UI."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.error("Audio: could not query devices: %s", exc)
        return []

    try:
        default_idx = default_input_index()
    except Exception:
        default_idx = None

    return [
        {
            "index": i,
            "name": d["name"],
            "channels": d["max_input_channels"],
            "sample_rate": int(d["default_samplerate"]),
            "is_monitor": _is_monitor(d["name"]),
            "is_default": i == default_idx,
        }
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def default_input_index() -> int | None:
    try:
        idx = sd.default.device[0]
        return int(idx) if idx is not None and idx >= 0 else None
    except Exception:
        return None


def describe_devices() -> str:
    """Multi-line human-readable device list, for startup/diagnostic logs."""
    devices = list_input_devices()
    if not devices:
        return "  (no input devices found — is a microphone connected?)"
    lines = []
    for d in devices:
        tags = []
        if d["is_default"]:
            tags.append("default")
        if d["is_monitor"]:
            tags.append("MONITOR — not a mic")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"  [{d['index']}] {d['name']} "
            f"({d['channels']}ch, {d['sample_rate']}Hz){tag_str}"
        )
    return "\n".join(lines)


def resolve_input_device(
    microphone_index: int | None,
    microphone_name: str = "",
) -> ResolvedDevice:
    """Pick the actual input device to record from, with a clear reason.

    Priority:
      1. An explicit, still-valid ``microphone_index``.
      2. A device whose name contains ``microphone_name`` (case-insensitive).
      3. The OS default input — *if* it isn't a monitor/loopback device.
      4. The first non-monitor input device with channels, as a last resort.
      5. Whatever the OS default is, monitor or not — better than crashing.
    """
    devices = list_input_devices()

    if microphone_index is not None:
        match = next((d for d in devices if d["index"] == microphone_index), None)
        if match is not None:
            return ResolvedDevice(
                match["index"], match["name"], "configured microphone_index"
            )
        log.warning(
            "Audio: configured microphone_index=%d no longer exists "
            "(devices can be renumbered by PipeWire/ALSA on reboot or "
            "replug) — falling back to auto-detection",
            microphone_index,
        )

    name_filter = microphone_name.strip().lower() if isinstance(microphone_name, str) else ""
    if name_filter:
        match = next((d for d in devices if name_filter in d["name"].lower()), None)
        if match is not None:
            return ResolvedDevice(
                match["index"], match["name"], f"name matches '{microphone_name}'"
            )
        log.warning(
            "Audio: no input device name matches '%s' — falling back to "
            "auto-detection", microphone_name,
        )

    default_idx = default_input_index()
    default_match = next((d for d in devices if d["index"] == default_idx), None)
    if default_match is not None and not default_match["is_monitor"]:
        return ResolvedDevice(
            default_match["index"], default_match["name"], "OS default input"
        )

    non_monitor = next((d for d in devices if not d["is_monitor"]), None)
    if non_monitor is not None:
        if default_match is not None and default_match["is_monitor"]:
            log.warning(
                "Audio: OS default input '%s' is a monitor/loopback device "
                "(records speaker output, not a mic) — using '%s' instead. "
                "Set microphone_index or microphone_name in settings to "
                "override.", default_match["name"], non_monitor["name"],
            )
        return ResolvedDevice(
            non_monitor["index"], non_monitor["name"], "first non-monitor input"
        )

    if default_match is not None:
        return ResolvedDevice(
            default_match["index"], default_match["name"],
            "OS default (only monitor devices available)",
        )

    return ResolvedDevice(None, "(unknown)", "no input devices detected")
