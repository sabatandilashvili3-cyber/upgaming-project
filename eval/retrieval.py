"""Retrieval evaluation: no LLM, so it is free, deterministic and runs in CI.

Configs compared (the brief asks for at least two):
  A  fixed chunks,      dense only
  B  fixed chunks,      BM25 only
  C  fixed chunks,      hybrid (RRF)
  D  structural chunks, hybrid (RRF)
  E  structural chunks, hybrid + id/time boost
  F  E + a second hop that follows ids found in the top results (engine/search/chain.py)
  G  F + at most 2 rows of the same file in front (source diversity)   <- the engine's default
  H  G + cross-encoder re-ranking of the top 30, cross-encoder order only (engine/search/rerank.py)
  H-fused  G + cross-encoder order fused with G's order by RRF
  H-routed H-fused only for questions with no ids and no dates; G otherwise
  G-minilm  G with a different embedding model (all-MiniLM-L6-v2 instead of bge-small)

Questions: every Meridian world question that has gold evidence, plus a seeded
sample of HotpotQA questions. All of them search the *whole* corpus (Wikipedia
paragraphs and company documents together, ~62k records).

Usage: python -m eval.retrieval --hotpot 500 [--configs A B C D E]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine.ingest.records import Chunk
from engine.search.dense import DEFAULT_MODEL, Embedder
from engine.search.chain import chained_search
from engine.search.retriever import Retriever, SearchConfig, doc_ranking

from .build_index import CACHE, slug
from .metrics import all_gold_at_k, mrr, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
RESULTS = ROOT / "eval" / "results"

CONFIGS = {
    "A": ("fixed", SearchConfig(mode="dense", boost=False)),
    "B": ("fixed", SearchConfig(mode="bm25", boost=False)),
    "C": ("fixed", SearchConfig(mode="hybrid", boost=False)),
    "D": ("structural", SearchConfig(mode="hybrid", boost=False)),
    "E": ("structural", SearchConfig(mode="hybrid", boost=True)),
    "F": ("structural", SearchConfig(mode="hybrid", boost=True)),      # + chained second hop
    "G": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2)),   # + chained, diversified
    "H": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=30, rerank_mode="replace")),
    "H-fused": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=30, rerank_mode="fuse")),
    "H-routed": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=30, rerank_mode="fuse",
                                            rerank_when="plain")),
    "H-routed-15": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=15, rerank_mode="fuse",
                                               rerank_when="plain")),
    "G-minilm": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2)),
    # smaller cross-encoders of the same family: the re-ranker is ~93% of a re-ranked search's time
    "H-routed-15-L4": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=15,
                                                  rerank_mode="fuse", rerank_when="plain")),
    "H-routed-15-L2": ("structural", SearchConfig(mode="hybrid", boost=True, per_file=2, rerank_top=15,
                                                  rerank_mode="fuse", rerank_when="plain")),
}
MODEL_OF = {"G-minilm": "sentence-transformers/all-MiniLM-L6-v2"}
L6 = "cross-encoder/ms-marco-MiniLM-L-6-v2"           # the H configs were measured with it; pinned so they reproduce
RERANKER_OF = {"H": L6, "H-fused": L6, "H-routed": L6, "H-routed-15": L6,
               "H-routed-15-L4": "cross-encoder/ms-marco-MiniLM-L-4-v2",      # the live default
               "H-routed-15-L2": "cross-encoder/ms-marco-MiniLM-L-2-v2"}
CHAINED = {"F", "G", "H", "H-fused", "H-routed", "H-routed-15", "G-minilm", "H-routed-15-L4", "H-routed-15-L2"}


def load_questions(n_hotpot, seed=13):
    qs = []
    for line in open(CORPUS / "world" / "world_questions.jsonl"):
        q = json.loads(line)
        if q["gold_doc_ids"]:
            qs.append({"qid": q["qid"], "question": q["question"], "gold": q["gold_doc_ids"],
                       "group": "world:" + q["kind"]})
    hp = [json.loads(line) for line in open(CORPUS / "hotpot_questions.jsonl")]
    for q in random.Random(seed).sample(hp, n_hotpot):
        qs.append({"qid": q["qid"], "question": q["question"], "gold": q["gold_doc_ids"],
                   "group": "hotpot:" + q["type"]})
    return qs


def load_index(strategy, model):
    d = CACHE / strategy / slug(model)
    chunks = [Chunk(**json.loads(line)) for line in open(d / "chunks.jsonl")]
    dupes = json.loads((d / "duplicates.json").read_text())
    aliases = {k: v[0] for k, v in dupes.items() if v[1] == "exact"}
    return chunks, np.load(d / "vectors.npy"), aliases


def evaluate(configs, n_hotpot, model=DEFAULT_MODEL, k_values=(5, 10)):
    questions = load_questions(n_hotpot)
    embedders, qvecs, retrievers, report, rows = {}, {}, {}, {}, []
    rerankers = {}
    for name in configs:
        strategy, cfg = CONFIGS[name]
        m = MODEL_OF.get(name, model)
        if m not in embedders:
            embedders[m] = Embedder(m)
            qvecs[m] = embedders[m].passages([embedder_prefix(embedders[m]) + q["question"] for q in questions],
                                             batch_size=64)
        rr_name = None
        if cfg.rerank_top:
            from engine.search.rerank import DEFAULT_RERANKER, Reranker
            rr_name = RERANKER_OF.get(name, DEFAULT_RERANKER)
            if rr_name not in rerankers:
                rerankers[rr_name] = Reranker(rr_name)
        key = (strategy, m, rr_name)
        if key not in retrievers:
            chunks, vecs, aliases = load_index(strategy, m)
            retrievers[key] = (Retriever(chunks, vecs, embedders[m], rerankers.get(rr_name)), aliases)
        retriever, aliases = retrievers[key]
        cfg.k = max(k_values) * 3                       # chunks; several chunks may share a document
        per_group = defaultdict(lambda: defaultdict(list))
        t0 = time.time()
        for q, qv in zip(questions, qvecs[m]):
            gold = [aliases.get(g, g) for g in q["gold"]]
            if name in CHAINED:
                hits, _ = chained_search(retriever, q["question"], cfg, qvec=qv)
            else:
                hits = retriever.search(q["question"], cfg, qvec=qv)
            ranking = doc_ranking(hits)
            m = {f"recall@{k}": recall_at_k(ranking, gold, k) for k in k_values}
            m.update({f"all_gold@{k}": all_gold_at_k(ranking, gold, k) for k in k_values})
            m["mrr"] = mrr(ranking, gold)
            for g in (q["group"], q["group"].split(":")[0], "all"):
                for key, v in m.items():
                    per_group[g][key].append(v)
            rows.append({"config": name, "qid": q["qid"], "group": q["group"], **m,
                         "missed": [g for g in gold if g not in ranking[:max(k_values)]],
                         "top": ranking[:5]})
        report[name] = {g: {key: round(float(np.mean(v)), 3) for key, v in ms.items()} | {"n": len(ms["mrr"])}
                        for g, ms in per_group.items()}
        report[name]["all"]["ms_per_query"] = round(1000 * (time.time() - t0) / len(questions), 1)
    return report, rows


def embedder_prefix(embedder):
    from engine.search.dense import QUERY_PREFIX
    return QUERY_PREFIX.get(embedder.name, "")


def print_table(report):
    groups = sorted({g for r in report.values() for g in r}, key=lambda g: (g != "all", ":" in g, g))
    cols = ["recall@5", "recall@10", "all_gold@10", "mrr"]
    print(f"{'group':28} {'cfg':4} " + " ".join(f"{c:>11}" for c in cols) + "     n")
    for g in groups:
        for name, r in report.items():
            if g in r:
                print(f"{g:28} {name:4} " + " ".join(f"{r[g][c]:>11.3f}" for c in cols) + f" {r[g]['n']:>5}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", type=int, default=500)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    report, rows = evaluate(a.configs, a.hotpot)
    print_table(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # merge: running some configs replaces only those configs' results
    summary_path, rows_path = RESULTS / "retrieval_summary.json", RESULTS / "retrieval_rows.jsonl"
    old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary_path.write_text(json.dumps({**old, **report}, indent=1))
    kept = [r for r in map(json.loads, open(rows_path))] if rows_path.exists() else []
    with open(rows_path, "w") as f:
        for r in [r for r in kept if r["config"] not in report] + rows:
            f.write(json.dumps(r) + "\n")
