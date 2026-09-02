"""Shims for Python versions our dependency stack has not caught up to.

Everything here is a stopgap with an expiry condition written next to it.
When the upstream package ships a fix, delete the shim — do not grow this
file into a permanent compatibility layer.
"""
from __future__ import annotations

import typing


def restore_typing_bytestring() -> bool:
    """Put ``typing.ByteString`` back, and report whether it is there now.

    Python 3.14 removed ``typing.ByteString`` (deprecated since 3.9) along
    with ``collections.abc.ByteString``. ``overrides`` reads it at import
    time to build a lookup table, so on 3.14 ``import overrides`` raises
    AttributeError before running a line of its own logic — and ChromaDB
    imports ``overrides`` from ``chromadb/errors.py``, which every other
    ChromaDB module pulls in. The visible symptom is NixOrb starting with
    long-term memory silently switched off.

    There is nothing to upgrade to: ``overrides`` last released 7.7.0 in
    January 2024 and still classifies itself up to Python 3.9, while
    ChromaDB requires ``overrides>=7.3.1``. So restore the name instead.
    It is used as a single dict key mapping the deprecated alias to
    ``bytes``; the union below is what the alias meant, and no annotation
    running on 3.14 can refer to it anyway.
    """
    if hasattr(typing, "ByteString"):
        return True
    typing.ByteString = bytes | bytearray | memoryview  # type: ignore[assignment]
    return hasattr(typing, "ByteString")
