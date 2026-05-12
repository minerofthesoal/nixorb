"""nixorb/utils/audio.py — Audio device utilities."""
from __future__ import annotations
import sounddevice as sd

def list_input_devices() -> list[dict]:
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"],
         "sample_rate": int(d["default_samplerate"])}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]

def default_input_index() -> int | None:
    try:
        idx = sd.default.device[0]
        return int(idx) if idx is not None else None
    except Exception:
        return None
