"""Retrieval evaluation: no LLM, so it is free, deterministic and runs in CI.

Configs compared (the brief asks for at least two):
  A  fixed chunks,      dense only
  B  fixed chunks,      BM25 only
  C  fixed chunks,      hybrid (RRF)
  D  structural chunks, hybrid (RRF)
  E  structural chunks, hybrid + id/time boost
  F  E + a second hop that follows ids found in the top results (engine/search/chain.py)
  G  F + at most 2 rows of the same file in front (source diversity)

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
}


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
    embedder = Embedder(model)
    qvecs = embedder.passages([embedder_prefix(embedder) + q["question"] for q in questions], batch_size=64)
    retrievers, report, rows = {}, {}, []
    for name in configs:
        strategy, cfg = CONFIGS[name]
        if strategy not in retrievers:
            chunks, vecs, aliases = load_index(strategy, model)
            retrievers[strategy] = (Retriever(chunks, vecs, embedder), aliases)
        retriever, aliases = retrievers[strategy]
        cfg.k = max(k_values) * 3                       # chunks; several chunks may share a document
        per_group = defaultdict(lambda: defaultdict(list))
        t0 = time.time()
        for q, qv in zip(questions, qvecs):
            gold = [aliases.get(g, g) for g in q["gold"]]
            if name in ("F", "G"):
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
    (RESULTS / "retrieval_summary.json").write_text(json.dumps(report, indent=1))
    with open(RESULTS / "retrieval_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
