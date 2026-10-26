"""Load the whole prepared corpus (data/cache/<strategy>/<model>) into Postgres in one go.

Uploading 62k records through the API one by one would re-embed everything; the
cache already holds the chunks and vectors, so here we only stream them in with
COPY (binary), which is far faster than row-by-row INSERTs.

Usage: python -m engine.store.bulk [--strategy structural] [--limit-hotpot N]
       DATABASE_URL must be set (or --local to use the bundled pgserver).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from psycopg.types.json import Jsonb

from engine.ingest.dedup import content_hash, lsh_keys, minhash, shingles
from engine.ingest.pipeline import iter_hotpot, iter_world
from engine.search.dense import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


def local_url():
    import pgserver
    d = ROOT / "data" / "pgdata"
    d.mkdir(parents=True, exist_ok=True)
    return pgserver.get_server(str(d), cleanup_mode=None).get_uri()


def copy_chunks(conn, rows):
    """rows: (chunk_id, doc_id, ord, text, meta dict, vector). Binary COPY: fast, typed.

    meta must be wrapped in Jsonb: in binary COPY a plain str is stored as a JSON *string*
    ("{...}" in quotes), not an object, and every later meta.get() fails. Found by
    running the API on the bulk-loaded data; covered by tests/test_store.py.
    """
    with conn.cursor().copy("copy chunks (chunk_id, doc_id, ord, text, meta, embedding) from stdin "
                            "with (format binary)") as cp:
        cp.set_types(["text", "text", "int4", "text", "jsonb", "vector"])
        for chunk_id, doc_id, n, text, meta, vec in rows:
            cp.write_row((chunk_id, doc_id, n, text, Jsonb(meta), vec))


def load(store, strategy="structural", model=DEFAULT_MODEL):
    d = CACHE / strategy / model.split("/")[-1]
    t0 = time.time()
    dupes = json.loads((d / "duplicates.json").read_text())
    records = list(iter_world()) + list(iter_hotpot())
    chunks = [json.loads(line) for line in open(d / "chunks.jsonl")]
    vectors = np.load(d / "vectors.npy")
    with store.tx() as c:
        c.execute("truncate documents, chunks, lsh_bands cascade")
        # building the HNSW graph once at the end is much faster than updating it per row
        c.execute("drop index if exists chunks_embedding")
        with c.cursor().copy("copy documents (doc_id, source, format, title, text, content_hash, version, "
                             "duplicate_of, near_duplicate_of, meta) from stdin") as cp:
            for r in records:
                kept, kind, _ = dupes.get(r.doc_id, (None, None, None))
                cp.write_row((r.doc_id, r.source, r.format, r.title, r.text, content_hash(r.text), 1,
                              kept if kind == "exact" else None, kept if kind == "near" else None,
                              json.dumps(r.meta, default=str)))
        print(f"documents: {len(records)} ({time.time() - t0:.0f}s)")
        ord_of = {}

        def rows():
            for ch, vec in zip(chunks, vectors):
                n = ord_of[ch["doc_id"]] = ord_of.get(ch["doc_id"], -1) + 1
                yield ch["chunk_id"], ch["doc_id"], n, ch["text"], ch["meta"], vec
        copy_chunks(c, rows())
        print(f"chunks: {len(chunks)} ({time.time() - t0:.0f}s)")
        with c.cursor().copy("copy lsh_bands (band, value, doc_id) from stdin") as cp:
            for r in records:
                if dupes.get(r.doc_id, (None, None))[1] == "exact":
                    continue
                for band, key in lsh_keys(minhash(shingles(r.text))):
                    cp.write_row((band, key, r.doc_id))
        print(f"lsh bands done ({time.time() - t0:.0f}s)")
        c.execute("set local maintenance_work_mem = '256MB'")
        c.execute("create index chunks_embedding on chunks using hnsw (embedding vector_cosine_ops)")
        print(f"hnsw index built ({time.time() - t0:.0f}s)")
    with store.pool.connection() as c:
        c.execute("analyze")
    return store.stats()


if __name__ == "__main__":
    from engine.store.db import Store
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structural")
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    s = Store(local_url() if a.local else os.environ["DATABASE_URL"])
    print(load(s, a.strategy))
