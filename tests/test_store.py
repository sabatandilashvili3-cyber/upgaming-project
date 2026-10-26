"""Integration tests against a real Postgres + pgvector (pgserver locally, a service container in CI).

Skipped when no database is available. A fake embedder keeps them fast: it hashes
words into a 384-dim vector, so texts sharing words are close, which is all the
search assertions need.
"""
import hashlib
import os
import shutil
import tempfile

import numpy as np
import pytest

from engine.ingest.service import Ingestor


class FakeEmbedder:
    name = "fake"
    dim = 384

    def passages(self, texts, **_):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out

    def query(self, text):
        return self.passages([text])[0]


@pytest.fixture(scope="module")
def db_url():
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    d = tempfile.mkdtemp(prefix="pgtest")
    srv = pgserver.get_server(d, cleanup_mode="stop")
    yield srv.get_uri()
    srv.cleanup()
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def store(db_url):
    from engine.store.db import Store
    s = Store(db_url)
    with s.tx() as c:
        c.execute("truncate documents, chunks, lsh_bands, traces, llm_cache cascade")
    yield s
    s.close()


@pytest.fixture()
def ingestor(store):
    changes = []
    ing = Ingestor(store, FakeEmbedder(), on_change=lambda: changes.append(1))
    ing.changes = changes
    return ing


CSV_V1 = b"change_id,service,summary\nCHG-1,ledger,reduce pool 240 -> 60\nCHG-2,auth,rotate keys\n"
CSV_V2 = b"change_id,service,summary\nCHG-1,ledger,reduce pool 240 -> 30\nCHG-3,kyc,scale up\n"
STATUS = (b"<html><head><title>Status</title></head><body><h1>Card payments degraded</h1><p>Started 2026-03-14 "
          b"09:12 UTC. Resolved 11:12 UTC. Some customers saw delays with card payments and payouts. Our engineers "
          b"identified the issue and applied a fix. We apologise for the disruption. Reference INC-2026-016</p>"
          b"</body></html>")


def test_upload_update_and_remove_rows(ingestor, store):
    r1 = ingestor.ingest_file("changes.csv", CSV_V1)
    assert (r1.added, r1.updated, r1.removed) == (2, 0, 0) and r1.chunks == 2
    r2 = ingestor.ingest_file("changes.csv", CSV_V2)
    assert (r2.added, r2.updated, r2.unchanged, r2.removed) == (1, 1, 0, 1)
    doc = store.get_document("changes.csv#CHG-1")
    assert doc["version"] == 2 and "240 -> 30" in doc["text"]
    assert store.get_document("changes.csv#CHG-2") is None
    r3 = ingestor.ingest_file("changes.csv", CSV_V2)
    assert (r3.unchanged, r3.added, r3.updated) == (2, 0, 0)
    assert len(ingestor.changes) == 2                     # the no-op upload did not invalidate search


def test_exact_and_near_duplicates_on_upload(ingestor, store):
    ingestor.ingest_file("status.html", STATUS)
    exact = ingestor.ingest_file("status_copy.html", STATUS)
    assert exact.exact_duplicates == [("status_copy.html", "status.html")] and exact.chunks == 0
    near = ingestor.ingest_file("email.html", STATUS.replace(b"We apologise for", b"We are sorry for"))
    assert near.near_duplicates and near.near_duplicates[0][1] == "status.html"
    assert store.get_document("email.html")["near_duplicate_of"] == "status.html"


def test_delete_file_cascades_to_chunks(ingestor, store):
    ingestor.ingest_file("changes.csv", CSV_V1)
    assert store.stats()["chunks"] == 2
    assert ingestor.delete_file("changes.csv") == 2
    assert store.stats()["chunks"] == 0 and store.stats()["documents"] == 0


def test_dense_search_through_hnsw(ingestor, store):
    ingestor.ingest_file("changes.csv", CSV_V1)
    hits = store.dense_search(FakeEmbedder().query("ledger reduce pool"), k=2)
    assert hits[0][0] == "changes.csv#CHG-1::0"


def test_unsupported_upload_writes_nothing(ingestor, store):
    from engine.ingest.loaders import UnsupportedFormat
    with pytest.raises(UnsupportedFormat):
        ingestor.ingest_file("malware.exe", b"MZ")
    assert store.stats()["documents"] == 0


def test_bulk_copy_stores_metadata_as_an_object(store, ingestor):
    from engine.store.bulk import copy_chunks
    ingestor.ingest_file("changes.csv", CSV_V1)                     # a parent document to point at
    with store.tx() as c:
        copy_chunks(c, [("changes.csv#CHG-1::9", "changes.csv#CHG-1", 9, "text",
                         {"ids": ["CHG-1"], "dates": []}, FakeEmbedder().query("text"))])
    row = store.chunks_by_id(["changes.csv#CHG-1::9"])["changes.csv#CHG-1::9"]
    assert isinstance(row["meta"], dict) and row["meta"]["ids"] == ["CHG-1"]
