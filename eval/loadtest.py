"""Load and scale test against a running instance (no LLM calls, so no API cost).

    python -m eval.loadtest --base http://127.0.0.1:8000 [--concurrency 1 4 16 32] [--seconds 30]

1. Search throughput: C concurrent clients send /api/search requests (questions sampled from
   the eval sets) for N seconds per concurrency level -> requests/s, p50/p95/p99 latency, errors.
2. Ingestion: upload a generated 2,000-row CSV (as a new file), then the same file with 10% of
   rows changed (update path: only changed rows are re-embedded), then delete it.
   Uses the ADMIN_TOKEN from the environment when the server requires one.
"""
import argparse
import asyncio
import csv
import io
import json
import os
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def sample_queries(n=300, seed=3):
    qs = [json.loads(l)["question"] for l in open(ROOT / "data/corpus/hotpot_questions.jsonl")]
    qs += [json.loads(l)["question"] for l in open(ROOT / "data/corpus/world/world_questions.jsonl")]
    qs += ["CHG-3340", "ledger connection pool", "payment delays in March 2026", "INC-2026-016 duration"]
    return random.Random(seed).sample(qs, min(n, len(qs)))


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


async def search_level(base, concurrency, seconds, queries):
    lat, errors, stop = [], 0, time.time() + seconds
    async with httpx.AsyncClient(base_url=base, timeout=60, verify=False) as client:
        async def worker(i):
            nonlocal errors
            k = i
            while time.time() < stop:
                q = queries[k % len(queries)]; k += concurrency
                t = time.perf_counter()
                try:
                    r = await client.post("/api/search", json={"query": q, "k": 10})
                    r.raise_for_status()
                    lat.append((time.perf_counter() - t) * 1000)
                except Exception:
                    errors += 1
        t0 = time.time()
        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        elapsed = time.time() - t0
    return {"concurrency": concurrency, "requests": len(lat), "errors": errors, "rps": round(len(lat) / elapsed, 1),
            "p50_ms": round(pct(lat, 50)), "p95_ms": round(pct(lat, 95)), "p99_ms": round(pct(lat, 99)),
            "mean_ms": round(statistics.mean(lat)) if lat else None}


def make_csv(rows=2000, changed=0.0, seed=5):
    rnd = random.Random(seed)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["change_id", "timestamp", "service", "type", "summary"])
    for i in range(rows):
        summary = f"Load-test change {i}: adjust {rnd.choice(['pool', 'timeout', 'replicas'])} on service"
        if rnd.random() < changed:
            summary += " (revised)"
        w.writerow([f"LT-{i:05d}", f"2026-09-{1 + i % 28:02d} 10:00 UTC", rnd.choice(["ledger", "auth", "kyc"]),
                    "config", summary])
    return buf.getvalue().encode()


def ingest_test(base):
    headers = {"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")}
    name = "loadtest_changes.csv"
    out = {}
    with httpx.Client(base_url=base, timeout=600, verify=False, headers=headers) as c:
        for label, data in [("new_2000_rows", make_csv()), ("update_10pct", make_csv(changed=0.1)),
                            ("unchanged_reupload", make_csv(changed=0.1))]:
            t = time.perf_counter()
            r = c.post("/api/documents", files={"file": (name, data, "text/csv")})
            r.raise_for_status()
            rep = r.json()
            out[label] = {"seconds": round(time.perf_counter() - t, 1), "added": rep["added"],
                          "updated": rep["updated"], "unchanged": rep["unchanged"], "chunks": rep["chunks"]}
        t = time.perf_counter()
        r = c.post("/api/search", json={"query": "LT-00042", "k": 1})       # forces the BM25 rebuild
        out["first_search_after_upload_s"] = round(time.perf_counter() - t, 1)
        out["found_new_row"] = bool(r.json()) and r.json()[0]["doc_id"].endswith("#LT-00042")
        t = time.perf_counter()
        c.delete("/api/documents", params={"source": f"uploads/{name}"}).raise_for_status()
        out["delete_s"] = round(time.perf_counter() - t, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16, 32])
    ap.add_argument("--seconds", type=int, default=30)
    a = ap.parse_args()
    queries = sample_queries()
    report = {"base": a.base, "search": [], "ingest": None}
    for c in a.concurrency:
        res = asyncio.run(search_level(a.base, c, a.seconds, queries))
        print(res, flush=True)
        report["search"].append(res)
    report["ingest"] = ingest_test(a.base)
    print(report["ingest"])
    (ROOT / "eval/results/loadtest_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
