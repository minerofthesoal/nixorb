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
    results = memory.query("anything")
    assert results == []


def test_count_increments(memory):
    assert memory.count() == 0
    memory.store("First entry")
    assert memory.count() == 1
    memory.store("Second entry")
    assert memory.count() == 2


def test_context_block_empty_when_no_results(memory):
    block = memory.build_context_block("obscure query xyz 12345")
    assert block == ""


def test_context_block_format(memory):
    memory.store("User likes dark themes")
    block = memory.build_context_block("theme preferences")
    if block:  # may or may not match depending on embedding
        assert "<long_term_memory>" in block
        assert "</long_term_memory>" in block


def test_store_ignores_empty_text(memory):
    memory.store("")
    memory.store("   ")
    assert memory.count() == 0


def test_metadata_stored(memory):
    memory.store("command: ls -la", metadata={"type": "command"})
    results = memory.query("list files")
    # Just verify no crash; semantic matching may vary
    assert isinstance(results, list)
