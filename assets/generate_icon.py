#!/usr/bin/env python3
"""Generate placeholder PNG icons for NixOrb when no real artwork exists."""
import struct, zlib, pathlib

def _make_png(size: int, r: int, g: int, b: int) -> bytes:
    """Minimal valid PNG — solid colour square."""
    raw = b""
    for _ in range(size):
        row = b"\x00" + bytes([r, g, b, 255] * size)
        raw += row
    compressed = zlib.compress(raw)
    def chunk(tag: bytes, data: bytes) -> bytes:
        c  = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )

here = pathlib.Path(__file__).parent
here.mkdir(exist_ok=True)
(here / "nixorb_256.png").write_bytes(_make_png(256, 74, 144, 217))
(here / "tray_icon.png").write_bytes(_make_png(48, 74, 144, 217))
print("Icons generated.")
