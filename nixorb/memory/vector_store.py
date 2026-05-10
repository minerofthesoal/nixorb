"""nixorb/memory/vector_store.py — ChromaDB long-term vector memory."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class VectorMemory:
    """
    Persistent ChromaDB-backed memory.
    Stores conversation snippets, commands, and preferences.
    Uses cosine similarity for retrieval.
    """

    def __init__(self, memory_dir: str | Path) -> None:
        import chromadb
        path = Path(memory_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col    = self._client.get_or_create_collection(
            name="nixorb_memory",
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorMemory ready (%d entries in %s)", self._col.count(), path)

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        if not text.strip():
            return
        doc_id = hashlib.sha256(f"{text}{time.time()}".encode()).hexdigest()[:20]
        self._col.add(
            documents=[text],
            metadatas=[{**(metadata or {}), "ts": str(time.time())}],
            ids=[doc_id],
        )

    def query(self, text: str, n_results: int = 5) -> list[str]:
        count = self._col.count()
        if count == 0:
            return []
        results = self._col.query(
            query_texts=[text],
            n_results=min(n_results, count),
        )
        return results.get("documents", [[]])[0]

    def build_context_block(self, query: str, n: int = 4) -> str:
        memories = self.query(query, n)
        if not memories:
            return ""
        lines = "\n".join(f"  • {m}" for m in memories)
        return f"\n<long_term_memory>\n{lines}\n</long_term_memory>\n"

    def count(self) -> int:
        return self._col.count()
