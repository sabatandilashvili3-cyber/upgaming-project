"""Upload, update and delete documents in the live store.

ingest_file(name, bytes):
    1. load    -> Records (one per document, or one per CSV/JSON row)
    2. for each record:
         unchanged hash        -> skip
         same hash elsewhere   -> exact duplicate: stored, not chunked, points at the original
         MinHash band match    -> verify Jaccard >= 0.8 -> near duplicate: chunked, linked
         otherwise             -> new or updated (version + 1)
    3. chunk + embed everything that needs it in one batch
    4. write each record in its own transaction (document + chunks + LSH bands)
    5. rows that were in the old version of the file but not in the new one are deleted
"""
from dataclasses import dataclass, field

from .chunking import STRATEGIES
from .dedup import THRESHOLD, content_hash, jaccard, lsh_keys, minhash, shingles
from .loaders import load_bytes


@dataclass
class IngestReport:
    source: str
    records: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    exact_duplicates: list = field(default_factory=list)   # [(doc_id, original)]
    near_duplicates: list = field(default_factory=list)    # [(doc_id, original, jaccard)]
    chunks: int = 0


class Ingestor:
    def __init__(self, store, embedder, strategy: str = "structural", on_change=None):
        self.store, self.embedder = store, embedder
        self.chunker = STRATEGIES[strategy]
        self.on_change = on_change or (lambda: None)      # e.g. mark the BM25 index stale

    def _near_duplicate(self, rec, keys, sh):
        candidates = [d for d in self.store.lsh_candidates(keys) if d != rec.doc_id]
        best = (0.0, None)
        for doc_id, text in self.store.texts(candidates).items():
            best = max(best, (jaccard(sh, shingles(text)), doc_id))
        return best if best[0] >= THRESHOLD else (0.0, None)

    def ingest_file(self, name: str, data: bytes) -> IngestReport:
        records = load_bytes(data, name)
        report = IngestReport(source=name, records=len(records))
        old_ids = {r["doc_id"] for r in self.store.docs_of_source(name)}
        plans = []
        for rec in records:
            h = content_hash(rec.text)
            current = self.store.get_document(rec.doc_id)
            if current and current["content_hash"] == h:
                report.unchanged += 1
                continue
            sh = shingles(rec.text)
            keys = lsh_keys(minhash(sh))
            same = self.store.find_by_hash(h)
            plan = {"rec": rec, "hash": h, "keys": keys, "dup": None, "near": None, "is_update": bool(current)}
            if same and same["doc_id"] != rec.doc_id:
                plan["dup"] = same["doc_id"]
                report.exact_duplicates.append((rec.doc_id, same["doc_id"]))
            else:
                sim, near = self._near_duplicate(rec, keys, sh)
                if near:
                    plan["near"] = near
                    report.near_duplicates.append((rec.doc_id, near, round(sim, 3)))
            plans.append(plan)

        # chunk and embed in one batch: one model call for a 600-row CSV, not 600
        to_embed = []
        for p in plans:
            p["chunks"] = [] if p["dup"] else self.chunker(p["rec"])
            for c in p["chunks"]:
                c.meta.update(source=p["rec"].source, format=p["rec"].format)
                if p["near"]:
                    c.meta["near_duplicate_of"] = p["near"]
            to_embed.extend(p["chunks"])
        vectors = iter(self.embedder.passages([c.text for c in to_embed]) if to_embed else [])

        for p in plans:
            vecs = [next(vectors) for _ in p["chunks"]]
            self.store.upsert_document(p["rec"], p["hash"], p["chunks"], vecs,
                                       lsh_keys=[] if p["dup"] else p["keys"],
                                       duplicate_of=p["dup"], near_duplicate_of=p["near"])
            report.chunks += len(p["chunks"])
            if p["is_update"]:
                report.updated += 1
            else:
                report.added += 1

        stale = old_ids - {r.doc_id for r in records}
        if stale:
            report.removed = self.store.delete_documents(stale)
        if plans or stale:
            self.on_change()
        return report

    def delete_file(self, name: str) -> int:
        n = self.store.delete_source(name)
        if n:
            self.on_change()
        return n
