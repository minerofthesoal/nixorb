"""tests/test_memory.py — VectorMemory unit tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def memory(tmp_memory):
    from nixorb.memory.vector_store import VectorMemory
    return VectorMemory(tmp_memory)


def test_store_and_query(memory):
    memory.store("I love Python programming", metadata={"tag": "pref"})
    results = memory.query("Python")
    assert len(results) >= 1
    assert any("Python" in r for r in results)


def test_empty_db_query_returns_empty(memory):
    assert memory.query("anything") == []


def test_count_increments(memory):
    assert memory.count() == 0
    memory.store("First entry")
    assert memory.count() == 1
    memory.store("Second entry")
    assert memory.count() == 2


def test_ids_are_unique_within_the_same_millisecond(memory):
    """IDs used to be time-only, so a fast burst overwrote itself."""
    for i in range(5):
        assert memory.store(f"burst entry number {i}")
    assert memory.count() == 5


def test_context_block_empty_when_no_results(memory):
    assert memory.build_context_block("obscure query xyz 12345") == ""


def test_context_block_format(memory):
    memory.store("User likes dark themes")
    block = memory.build_context_block("theme preferences")
    if block:  # embedding match is not guaranteed
        assert "Relevant past conversations" in block
        assert "User likes dark themes" in block


def test_store_ignores_empty_text(memory):
    assert memory.store("") is False
    assert memory.store("   ") is False
    assert memory.count() == 0


def test_metadata_stored(memory):
    memory.store("command: ls -la", metadata={"type": "command"})
    entries = memory.search("list files")
    assert isinstance(entries, list)
    assert entries and entries[0]["metadata"]["type"] == "command"


def test_clear_removes_everything(memory):
    memory.store("something to forget")
    assert memory.count() == 1
    assert memory.clear() is True
    assert memory.count() == 0


def test_disabled_memory_degrades_quietly(tmp_memory, monkeypatch):
    """If ChromaDB cannot start, every call must no-op instead of raising."""
    from nixorb.memory.vector_store import VectorMemory

    monkeypatch.setattr(
        VectorMemory, "_init_chroma", lambda self: None, raising=True
    )
    m = VectorMemory(tmp_memory)

    assert m.store("x") is False
    assert m.query("x") == []
    assert m.search("x") == []
    assert m.build_context_block("x") == ""
    assert m.count() == 0
    assert m.clear() is False
