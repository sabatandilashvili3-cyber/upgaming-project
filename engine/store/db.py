"""Postgres + pgvector access. All SQL lives here; nothing else talks to the database.

Every write that touches more than one table runs in a single transaction, so a
crash halfway through an upload never leaves a document without its chunks.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = (Path(__file__).parent / "schema.sql").read_text()


class Store:
    def __init__(self, url: str | None = None, min_size: int = 1, max_size: int = 5):
        self.url = url or os.environ["DATABASE_URL"]
        # register the vector type on a throwaway connection first, so the extension exists
        with psycopg.connect(self.url, autocommit=True) as conn:
            conn.execute(SCHEMA)
        # check: Neon's free tier suspends after 5 idle minutes and drops connections, so every
        # connection is tested before use and idle ones are recycled instead of failing a request
        self.pool = ConnectionPool(self.url, min_size=min_size, max_size=max_size, open=True,
                                   configure=lambda c: register_vector(c), kwargs={"row_factory": dict_row},
                                   check=ConnectionPool.check_connection, max_idle=240)

    def close(self):
        self.pool.close()

    @contextmanager
    def tx(self):
        with self.pool.connection() as conn:
            with conn.transaction():
                yield conn

    # ---------------------------------------------------------------- documents
    def find_by_hash(self, content_hash: str):
        with self.pool.connection() as c:
            return c.execute("select doc_id from documents where content_hash = %s and duplicate_of is null limit 1",
                             (content_hash,)).fetchone()

    def lsh_candidates(self, keys):
        """keys: [(band, int key)] -> doc_ids sharing at least one band."""
        if not keys:
            return []
        with self.pool.connection() as c:
            rows = c.execute(
                "select distinct doc_id from lsh_bands where (band, value) in (select * from unnest(%s::smallint[], %s::bigint[]))",
                ([b for b, _ in keys], [v for _, v in keys])).fetchall()
        return [r["doc_id"] for r in rows]

    def texts(self, doc_ids):
        with self.pool.connection() as c:
            rows = c.execute("select doc_id, text from documents where doc_id = any(%s)", (list(doc_ids),)).fetchall()
        return {r["doc_id"]: r["text"] for r in rows}

    def upsert_document(self, rec, content_hash, chunks, vectors, lsh_keys=(), duplicate_of=None,
                        near_duplicate_of=None):
        """Insert or replace one document and all its chunks atomically. Returns the new version."""
        with self.tx() as c:
            old = c.execute("select version, content_hash from documents where doc_id = %s for update",
                            (rec.doc_id,)).fetchone()
            if old and old["content_hash"] == content_hash:
                return old["version"]                                  # unchanged: nothing to do
            version = old["version"] + 1 if old else 1
            c.execute("delete from documents where doc_id = %s", (rec.doc_id,))   # cascades to chunks/bands
            c.execute("""insert into documents (doc_id, source, format, title, text, content_hash, version,
                                                duplicate_of, near_duplicate_of, meta)
                         values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                      (rec.doc_id, rec.source, rec.format, rec.title, rec.text, content_hash, version,
                       duplicate_of, near_duplicate_of, json.dumps(rec.meta, default=str)))
            if chunks:
                with c.cursor() as cur:
                    cur.executemany(
                        "insert into chunks (chunk_id, doc_id, ord, text, meta, embedding) values (%s,%s,%s,%s,%s,%s)",
                        [(ch.chunk_id, ch.doc_id, n, ch.text, json.dumps(ch.meta, default=str), vec)
                         for n, (ch, vec) in enumerate(zip(chunks, vectors))])
            if lsh_keys:
                with c.cursor() as cur:
                    cur.executemany("insert into lsh_bands (band, value, doc_id) values (%s,%s,%s)",
                                    [(b, v, rec.doc_id) for b, v in lsh_keys])
            return version

    def docs_of_source(self, source: str):
        with self.pool.connection() as c:
            return c.execute("select doc_id, version from documents where source = %s", (source,)).fetchall()

    def delete_documents(self, doc_ids) -> int:
        with self.tx() as c:
            return c.execute("delete from documents where doc_id = any(%s)", (list(doc_ids),)).rowcount

    def delete_source(self, source: str) -> int:
        """Delete every document that came from one uploaded file. Returns how many."""
        with self.tx() as c:
            return c.execute("delete from documents where source = %s", (source,)).rowcount

    def list_sources(self, limit: int = 200, offset: int = 0, q: str | None = None):
        with self.pool.connection() as c:
            return c.execute("""
                select source, min(format) as format, count(*) as records, max(version) as version,
                       max(updated_at) as updated_at, count(duplicate_of) as exact_duplicates,
                       count(near_duplicate_of) as near_duplicates
                from documents where (%s::text is null or source ilike '%%' || %s || '%%')
                group by source order by max(updated_at) desc, source limit %s offset %s""",
                             (q, q, limit, offset)).fetchall()

    def get_document(self, doc_id: str):
        with self.pool.connection() as c:
            return c.execute("select * from documents where doc_id = %s", (doc_id,)).fetchone()

    def stats(self):
        with self.pool.connection() as c:
            return c.execute("""select (select count(*) from documents) as documents,
                                       (select count(distinct source) from documents) as sources,
                                       (select count(*) from chunks) as chunks,
                                       pg_database_size(current_database()) as bytes""").fetchone()

    # ---------------------------------------------------------------- search
    def all_chunks(self):
        """(chunk_id, doc_id, text, meta) for every chunk: feeds the in-memory BM25 index."""
        with self.pool.connection() as c:
            return c.execute("select chunk_id, doc_id, text, meta from chunks order by chunk_id").fetchall()

    def chunks_by_id(self, ids):
        with self.pool.connection() as c:
            rows = c.execute("select chunk_id, doc_id, text, meta from chunks where chunk_id = any(%s)",
                             (list(ids),)).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def search(self, qvec: np.ndarray, k: int = 50):
        """Same interface as MemoryIndex.search, so the Retriever can use either."""
        return self.dense_search(qvec, k)

    def dense_search(self, qvec: np.ndarray, k: int = 50, ef_search: int = 100):
        """Nearest chunks by cosine distance through the HNSW index."""
        with self.tx() as c:
            c.execute(f"set local hnsw.ef_search = {int(ef_search)}")
            rows = c.execute("select chunk_id, 1 - (embedding <=> %s) as sim from chunks "
                             "order by embedding <=> %s limit %s", (qvec, qvec, k)).fetchall()
        return [(r["chunk_id"], float(r["sim"])) for r in rows]

    # ---------------------------------------------------------------- traces + cache
    def save_trace(self, t: dict):
        with self.pool.connection() as c:
            c.execute("""insert into traces (trace_id, question, report, steps, model, tokens_in, tokens_out,
                                             cost_usd, latency_ms, error)
                         values (%(trace_id)s, %(question)s, %(report)s, %(steps)s, %(model)s, %(tokens_in)s,
                                 %(tokens_out)s, %(cost_usd)s, %(latency_ms)s, %(error)s)""",
                      {**t, "report": json.dumps(t.get("report"), default=str),
                       "steps": json.dumps(t.get("steps", []), default=str)})

    def recent_traces(self, limit: int = 30):
        with self.pool.connection() as c:
            return c.execute("select trace_id, question, model, tokens_in, tokens_out, cost_usd, latency_ms, error, "
                             "created_at from traces order by created_at desc limit %s", (limit,)).fetchall()

    def get_trace(self, trace_id: str):
        with self.pool.connection() as c:
            return c.execute("select * from traces where trace_id = %s", (trace_id,)).fetchone()

    def cache_get(self, key: str):
        with self.pool.connection() as c:
            row = c.execute("select response from llm_cache where key = %s", (key,)).fetchone()
        return row["response"] if row else None

    def cache_put(self, key: str, response: dict):
        with self.pool.connection() as c:
            c.execute("insert into llm_cache (key, response) values (%s, %s) on conflict (key) do nothing",
                      (key, json.dumps(response)))
