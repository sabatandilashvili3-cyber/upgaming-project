"""Offline: chunk the full corpus with each strategy and embed it once.

Writes data/cache/<strategy>/{chunks.jsonl, vectors.npy, duplicates.json, stats.json}.
The eval and the database loader both read from here, so the expensive embedding
step runs once per (strategy, model), not once per experiment.

Usage: python -m eval.build_index --strategy structural fixed [--model BAAI/bge-small-en-v1.5]
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from engine.ingest.pipeline import build, iter_hotpot, iter_world
from engine.search.dense import DEFAULT_MODEL, Embedder

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def slug(model):
    return model.split("/")[-1]


def run(strategy, model):
    out = CACHE / strategy / slug(model)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    built = build(list(iter_world()) + list(iter_hotpot()), strategy)
    t1 = time.time()
    with open(out / "chunks.jsonl", "w") as f:
        for c in built.chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    (out / "duplicates.json").write_text(json.dumps(built.duplicates, indent=0))
    emb = Embedder(model)
    vecs = emb.passages([c.text for c in built.chunks], batch_size=128, progress=True)
    np.save(out / "vectors.npy", vecs)
    stats = {"strategy": strategy, "model": model, "records": len(built.records), "chunks": len(built.chunks),
             "exact_duplicates": sum(1 for v in built.duplicates.values() if v[1] == "exact"),
             "near_duplicates": sum(1 for v in built.duplicates.values() if v[1] == "near"),
             "ingest_seconds": round(t1 - t0, 1), "embed_seconds": round(time.time() - t1, 1),
             "dim": int(vecs.shape[1])}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=["structural", "fixed"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    for s in a.strategy:
        run(s, a.model)
