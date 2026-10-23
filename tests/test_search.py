import math

import numpy as np

from engine.search.bm25 import BM25, tokenize
from engine.search.boost import analyse, boost
from engine.search.dense import MemoryIndex
from engine.search.fuse import rrf


def test_tokenize_keeps_ids_whole_and_splits_parts():
    toks = tokenize("Root cause: CHG-3340 on fraud-scoring")
    assert "chg-3340" in toks and "fraud-scoring" in toks
    assert "fraud" in toks and "scoring" in toks
    assert "on" not in toks                                  # stopword


def test_bm25_matches_hand_computed_score():
    docs = ["ledger pool exhausted", "ledger ledger ok", "unrelated text here"]
    bm = BM25(k1=1.2, b=0.75).build(["d0", "d1", "d2"], docs)
    # "exhausted": df=1 of N=3; d0 has tf=1, |d0|=3 tokens, avgdl=3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 3 / 3))
    (top_id, score), = bm.search("exhausted", k=5)
    assert top_id == "d0" and math.isclose(score, expected, rel_tol=1e-5)


def test_bm25_rare_term_beats_common_term():
    docs = ["payment delayed", "payment ok", "payment fine", "CHG-3340 payment"]
    bm = BM25().build(list("abcd"), docs)
    assert bm.search("payment CHG-3340", k=1)[0][0] == "d"


def test_bm25_respects_filter_mask():
    bm = BM25().build(["a", "b"], ["ledger", "ledger"])
    assert [i for i, _ in bm.search("ledger", allowed=np.array([False, True]))] == ["b"]


def test_rrf_prefers_agreement_over_a_single_first_place():
    fused = dict(rrf([["x", "both"], ["y", "both"]]))
    assert fused["both"] > fused["x"] and fused["both"] > fused["y"]
    assert math.isclose(fused["both"], 2 / 62)


def test_dense_returns_nearest_by_cosine():
    vecs = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    idx = MemoryIndex(["east", "north", "diag"], vecs)
    assert [i for i, _ in idx.search(np.array([1, 0.1], dtype=np.float32), k=2)] == ["east", "diag"]


def test_analyse_dates_and_ids():
    assert analyse("delays in March 2026").window == ("2026-03-01", "2026-03-31")
    assert analyse("what happened on 14 March 2026").window == ("2026-03-14", "2026-03-14")
    assert analyse("what may have caused the market decline").window is None
    assert analyse("How long did INC-2026-016 last?").ids == ["INC-2026-016"]


def test_boost_promotes_in_window_and_demotes_out_of_window():
    meta = {"in": {"dates": ["2026-03-14"]}, "out": {"dates": ["2026-04-02"]}, "timeless": {}}
    ranked = boost([("out", 1.0), ("timeless", 0.9), ("in", 0.8)], meta.get, analyse("delays in March 2026"))
    assert [c for c, _ in ranked] == ["in", "timeless", "out"]
