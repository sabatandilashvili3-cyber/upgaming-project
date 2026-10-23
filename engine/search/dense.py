"""Semantic search: embed the text, find the nearest vectors.

Embeddings come from a sentence-embedding model (default BAAI/bge-small-en-v1.5,
384 dimensions). Vectors are L2-normalised, so cosine similarity is a plain dot
product. Two backends share one interface:
  - MemoryIndex: a numpy matrix, exact search (used offline, in tests, in eval);
  - pgvector (engine/store): an HNSW index inside Postgres (used by the deployed app).
"""
import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to put this instruction in front of *queries* (not passages)
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def passages(self, texts, batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=progress, convert_to_numpy=True).astype(np.float32)

    def query(self, text: str) -> np.ndarray:
        return self.passages([QUERY_PREFIX.get(self.name, "") + text])[0]


class MemoryIndex:
    def __init__(self, ids, vectors: np.ndarray):
        self.ids = list(ids)
        self.vectors = vectors

    def search(self, qvec: np.ndarray, k: int = 50, allowed: np.ndarray | None = None):
        sims = self.vectors @ qvec
        if allowed is not None:
            sims = np.where(allowed, sims, -1.0)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]
