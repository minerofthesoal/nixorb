"""Read a GGUF file's own header, so a load failure can say what is wrong.

llama.cpp answers every load failure with one line — `Failed to load model
from file: <path>` — whether the file is truncated, is not a GGUF at all,
or is a perfectly good GGUF whose architecture the bundled llama.cpp is too
old to know. Those need three different responses, and the message
distinguishes none of them.

The header is cheap to read and says which it is. Format (little-endian):

    magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
    then kv_count × (key: string, type: u32, value)

where a string is a u64 length followed by that many UTF-8 bytes. Only
enough of the value grammar to skip values is implemented — the goal is to
reach `general.architecture`, not to parse the model.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MAGIC = b"GGUF"

# ggml_type enum for metadata values.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED_WIDTHS = {
    _UINT8: 1, _INT8: 1,
    _UINT16: 2, _INT16: 2,
    _UINT32: 4, _INT32: 4, _FLOAT32: 4,
    _BOOL: 1,
    _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}

# Reading the whole header would mean reading the whole file for a model
# with thousands of metadata entries; the architecture is near the front.
MAX_HEADER_BYTES = 8 * 1024 * 1024
_MAX_STRING = 1 << 20


class NotGGUF(ValueError):
    """The file does not begin with a GGUF header."""


@dataclass(frozen=True)
class GGUFInfo:
    """What the header says about a GGUF file."""

    path: str
    size: int
    version: int
    tensor_count: int
    architecture: str = ""


class _Reader:
    def __init__(self, handle) -> None:
        self._handle = handle

    def take(self, count: int) -> bytes:
        chunk = self._handle.read(count)
        if len(chunk) != count:
            raise NotGGUF("header ends mid-value")
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > _MAX_STRING:
            raise NotGGUF(f"implausible string length {length}")
        return self.take(length).decode("utf-8", "replace")

    def skip_value(self, value_type: int) -> None:
        width = _FIXED_WIDTHS.get(value_type)
        if width is not None:
            self.take(width)
            return
        if value_type == _STRING:
            self.string()
            return
        if value_type == _ARRAY:
            element_type = self.u32()
            count = self.u64()
            element_width = _FIXED_WIDTHS.get(element_type)
            if element_width is not None:
                self._handle.seek(element_width * count, 1)
                return
            if element_type == _STRING:
                for _ in range(count):
                    self.string()
                return
            raise NotGGUF(f"unknown array element type {element_type}")
        raise NotGGUF(f"unknown metadata type {value_type}")


def probe(path: str | Path) -> GGUFInfo:
    """Read `path`'s GGUF header. Raises NotGGUF if it is not one."""
    file_path = Path(path)
    size = file_path.stat().st_size

    with file_path.open("rb") as handle:
        magic = handle.read(4)
        if magic != MAGIC:
            raise NotGGUF(
                f"file starts with {magic!r}, not {MAGIC!r} — "
                "not a GGUF file (a failed download often leaves an HTML "
                "error page or an LFS pointer in its place)"
            )

        reader = _Reader(handle)
        version = reader.u32()
        tensor_count = reader.u64()
        kv_count = reader.u64()

        architecture = ""
        for _ in range(kv_count):
            if handle.tell() > MAX_HEADER_BYTES:
                break
            try:
                key = reader.string()
                value_type = reader.u32()
                if key == "general.architecture" and value_type == _STRING:
                    architecture = reader.string()
                    break
                reader.skip_value(value_type)
            except (NotGGUF, struct.error, OSError):
                # A readable magic and version is already worth reporting;
                # losing the architecture is not worth losing that.
                break

    return GGUFInfo(
        path=str(file_path),
        size=size,
        version=version,
        tensor_count=tensor_count,
        architecture=architecture,
    )


def human_size(count: int) -> str:
    """A byte count someone can read at a glance."""
    scaled = float(count)
    for unit in ("B", "KB", "MB"):
        if abs(scaled) < 1000:
            return f"{scaled:.0f} {unit}" if unit == "B" else f"{scaled:.1f} {unit}"
        scaled /= 1000.0
    return f"{scaled:.2f} GB"


def llama_cpp_version() -> str:
    """Installed llama-cpp-python version, or '' if it is not installed."""
    try:
        from importlib.metadata import version

        return version("llama-cpp-python")
    except Exception:
        return ""


def explain_load_failure(path: str | Path, exc: BaseException) -> str:
    """Turn "Failed to load model from file" into something to act on."""
    try:
        info = probe(path)
    except NotGGUF as bad:
        return f"{exc}\n{Path(path).name} is not a usable GGUF: {bad}"
    except OSError as bad:
        return f"{exc}\nCould not read {path}: {bad}"

    installed = llama_cpp_version()
    have = f" (you have {installed})" if installed else ""
    described = (
        f"architecture '{info.architecture}'" if info.architecture
        else "an architecture its header does not name"
    )
    return (
        f"{exc}\n"
        f"The file itself is fine: a valid GGUF v{info.version}, "
        f"{human_size(info.size)}, {info.tensor_count} tensors, {described}. "
        f"So this is not a bad download — llama.cpp read the header and "
        f"would not build the model. Most often that means the llama.cpp "
        f"bundled in llama-cpp-python{have} predates the architecture; it "
        f"can also be memory. Upgrade first:\n"
        f"  pip install -U --force-reinstall --no-cache-dir llama-cpp-python\n"
        f"(that rebuilds against current llama.cpp — the prebuilt CPU wheel "
        f"index lags upstream badly). If it still fails, the architecture "
        f"is newer than any release and you need a model your build "
        f"supports."
    )
