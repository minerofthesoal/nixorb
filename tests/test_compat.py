"""The Python 3.14 shim that keeps ChromaDB importable.

3.14 removed `typing.ByteString`. `overrides` (last released January 2024,
pinned by ChromaDB at >=7.3.1) reads it at import time, so on 3.14
`import chromadb` raises AttributeError from a package neither we nor
ChromaDB wrote, and NixOrb starts with long-term memory silently off.
"""
from __future__ import annotations

import typing

from nixorb.compat import restore_typing_bytestring


def test_the_name_is_available_afterwards():
    assert restore_typing_bytestring()
    assert hasattr(typing, "ByteString")


def test_it_is_idempotent():
    assert restore_typing_bytestring()
    first = typing.ByteString
    assert restore_typing_bytestring()
    assert typing.ByteString is first


def test_it_leaves_an_existing_name_alone(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(typing, "ByteString", sentinel, raising=False)
    assert restore_typing_bytestring()
    assert typing.ByteString is sentinel


def test_it_restores_a_removed_name(monkeypatch):
    # Simulate 3.14 on any interpreter: take the name away, put it back.
    monkeypatch.delattr(typing, "ByteString", raising=False)
    assert not hasattr(typing, "ByteString")

    assert restore_typing_bytestring()
    assert typing.ByteString is not None

    # And what comes back is usable as `overrides` uses it: a dict key.
    assert {typing.ByteString: bytes}[typing.ByteString] is bytes


def test_chromadb_imports_with_the_shim_applied():
    restore_typing_bytestring()
    overrides = __import__("overrides")
    assert hasattr(overrides, "EnforceOverrides")
