"""nixorb/memory/vector_store.py — ChromaDB long-term vector memory."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class _LocalHashEmbedding:
    """Small deterministic embedding function that never downloads models."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def name(self) -> str:
        return "nixorb-local-hash"

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)

    def __call__(self, input):  # ChromaDB's EmbeddingFunction protocol uses this name
        import math
        import re

        vectors = []
        for doc in input:
            vec = [0.0] * self._dimensions
            for token in re.findall(r"[\w']+", str(doc).lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


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
        self._embedding = _LocalHashEmbedding()
        self._col    = self._client.get_or_create_collection(
            name="nixorb_memory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding,
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
