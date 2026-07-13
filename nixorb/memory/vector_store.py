"""NixOrb vector memory — long-term conversation storage with ChromaDB.

Stores conversation history as embeddings for semantic retrieval.
Injects relevant past context into LLM prompts.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class VectorMemory:
    """Vector-based memory for conversation context retrieval."""

    def __init__(self, memory_dir: str | None = None) -> None:
        self._memory_dir = memory_dir or str(
            Path.home() / ".local" / "share" / "nixorb" / "memory"
        )
        self._client = None
        self._collection = None
        self._embedding_available = False

        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client."""
        try:
            import chromadb

            Path(self._memory_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._memory_dir)
            self._collection = self._client.get_or_create_collection(
                name="conversations",
                metadata={"description": "NixOrb conversation memory"},
            )
            self._embedding_available = True
            log.info("Memory: ChromaDB initialized at %s", self._memory_dir)
        except ImportError:
            log.warning("Memory: chromadb not installed — memory disabled")
        except Exception as exc:
            log.error("Memory: ChromaDB init failed: %s", exc)

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> bool:
        """Store a text entry in memory."""
        if not self._embedding_available or not self._collection:
            return False

        try:
            doc_id = f"entry_{int(time.time() * 1000)}"
            meta = metadata or {}
            meta["timestamp"] = time.time()

            self._collection.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[meta],
            )
            return True
        except Exception as exc:
            log.error("Memory: store failed: %s", exc)
            return False

    def build_context_block(self, query: str, max_results: int = 3) -> str:
        """Build a context block from relevant past conversations."""
        if not self._embedding_available or not self._collection:
            return ""

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(max_results, 10),
            )

            if not results or not results["documents"]:
                return ""

            contexts = []
            for docs in results["documents"]:
                for doc in docs:
                    if doc and len(doc) > 10:
                        contexts.append(doc)

            if not contexts:
                return ""

            context_text = "\n\n---\n\n".join(contexts[:max_results])
            return f"\n\n[Relevant past conversations]:\n{context_text}\n\n"

        except Exception as exc:
            log.error("Memory: query failed: %s", exc)
            return ""

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Search memory for relevant entries."""
        if not self._embedding_available or not self._collection:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, 20),
            )

            entries = []
            if results and results["documents"]:
                for i, docs in enumerate(results["documents"]):
                    for j, doc in enumerate(docs):
                        entry = {
                            "text": doc,
                            "metadata": (
                                results["metadatas"][i][j]
                                if results["metadatas"]
                                else {}
                            ),
                            "distance": (
                                results["distances"][i][j]
                                if results["distances"]
                                else None
                            ),
                        }
                        entries.append(entry)
            return entries

        except Exception as exc:
            log.error("Memory: search failed: %s", exc)
            return []

    def clear(self) -> bool:
        """Clear all memory entries."""
        if not self._embedding_available or not self._collection:
            return False

        try:
            # Get all IDs and delete them (ChromaDB doesn't support empty where in newer versions)
            all_data = self._collection.get()
            if all_data and all_data.get("ids"):
                self._collection.delete(ids=all_data["ids"])
            log.info("Memory: all entries cleared")
            return True
        except Exception as exc:
            log.error("Memory: clear failed: %s", exc)
            return False

    def count(self) -> int:
        """Get the number of stored entries."""
        if not self._embedding_available or not self._collection:
            return 0

        try:
            return self._collection.count()
        except Exception:
            return 0
