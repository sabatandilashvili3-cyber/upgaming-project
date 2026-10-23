"""One search entry point: keyword + semantic -> fusion -> boost -> collapse duplicates.

    query ─┬─ BM25 (own) ─────── top 50 ─┐
           └─ embed ─ dense ───── top 50 ─┴─ RRF (own) ─ id/time boost (own) ─ collapse near-dupes ─ top k

Every hit remembers where it came from (its BM25 rank and dense rank), so the UI
and the eval can show *why* a chunk was retrieved.
"""
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25
from .boost import analyse, boost
from .dense import MemoryIndex
from .fuse import rrf


@dataclass
class SearchConfig:
    mode: str = "hybrid"            # bm25 | dense | hybrid
    boost: bool = True
    collapse: bool = True           # merge near-duplicate documents into one hit
    per_file: int | None = None     # at most this many rows of one file before the rest (None = off)
    k: int = 10
    candidates: int = 50            # how deep each list goes before fusion


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict
    bm25_rank: int | None = None
    dense_rank: int | None = None
    also_in: list = field(default_factory=list)   # near-duplicate docs merged into this hit


class Retriever:
    def __init__(self, chunks, dense, embedder):
        """dense: a vectors matrix (searched in memory) or any object with .search(qvec, k),
        such as the pgvector-backed Store."""
        self.chunks = {c.chunk_id: c for c in chunks}
        ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25().build(ids, [c.text for c in chunks])
        self.dense = MemoryIndex(ids, dense) if isinstance(dense, np.ndarray) else dense
        self.embedder = embedder

    def _meta(self, cid):
        c = self.chunks.get(cid)
        return c.meta if c else {}

    def search(self, query: str, cfg: SearchConfig = SearchConfig(), qvec: np.ndarray | None = None):
        kw, dn = [], []
        if cfg.mode in ("bm25", "hybrid"):
            kw = [cid for cid, _ in self.bm25.search(query, cfg.candidates)]
        if cfg.mode in ("dense", "hybrid"):
            qvec = self.embedder.query(query) if qvec is None else qvec
            dn = [cid for cid, _ in self.dense.search(qvec, cfg.candidates)]
        if cfg.mode == "hybrid":
            ranked = rrf([kw, dn])
        else:
            lst = kw or dn
            ranked = [(cid, 1.0 / (r + 1)) for r, cid in enumerate(lst)]
        if cfg.boost:
            ranked = boost(ranked, self._meta, analyse(query))
        if cfg.per_file:
            ranked = diversify(ranked, cfg.per_file)
        kw_rank = {c: r for r, c in enumerate(kw, 1)}
        dn_rank = {c: r for r, c in enumerate(dn, 1)}
        hits, by_doc = [], {}
        for cid, score in ranked:
            c = self.chunks.get(cid)
            if c is None:                       # deleted since this index was built
                continue
            twin = c.meta.get("near_duplicate_of")
            if cfg.collapse and twin and twin in by_doc:
                by_doc[twin].also_in.append(c.doc_id)
                continue
            h = Hit(cid, c.doc_id, score, c.text, c.meta, kw_rank.get(cid), dn_rank.get(cid))
            hits.append(h)
            by_doc.setdefault(c.doc_id, h)
            if len(hits) >= cfg.k:
                break
        return hits


def diversify(ranked, per_file):
    """Keep at most `per_file` rows of the same file in front; the rest keep their order but go last.

    Five complaint tickets from one tickets_2026-04.json all mention the incident id and
    all get boosted, crowding out the postmortem that actually explains it. A row id
    looks like "file#row", so the file is everything before "#"; standalone documents
    (a postmortem, a Wikipedia paragraph) are their own file and are never affected.
    """
    seen, front, back = {}, [], []
    for cid, score in ranked:
        f = cid.split("#", 1)[0] if "#" in cid else cid.split("::", 1)[0]
        seen[f] = seen.get(f, 0) + 1
        (front if seen[f] <= per_file else back).append((cid, score))
    return front + back


def doc_ranking(hits):
    """Chunk hits -> document ids in first-seen order (several chunks can share a doc)."""
    seen, out = set(), []
    for h in hits:
        for d in [h.doc_id, *h.also_in]:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out
