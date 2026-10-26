"""The running engine: store + embedder + a search index kept in step with the database.

BM25 lives in memory (built from every chunk in Postgres); vectors stay in Postgres
and are searched through pgvector's HNSW index. Any upload or delete marks the
in-memory index stale, and the next search rebuilds it. Rebuilding 62k chunks takes
a few seconds, which is acceptable for a demo; a production system would update
the inverted index incrementally instead.
"""
import threading

from .ingest.records import Chunk
from .ingest.service import Ingestor
from .search.retriever import Retriever


class Engine:
    def __init__(self, store, embedder, strategy: str = "structural", reranker=None):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.ingestor = Ingestor(store, embedder, strategy, on_change=self.invalidate)
        self._retriever = None
        self._lock = threading.Lock()

    def invalidate(self):
        self._retriever = None

    @property
    def retriever(self) -> Retriever:
        with self._lock:
            if self._retriever is None:
                chunks = [Chunk(r["chunk_id"], r["doc_id"], r["text"], r["meta"]) for r in self.store.all_chunks()]
                self._retriever = Retriever(chunks, self.store, self.embedder, self.reranker)
            return self._retriever
