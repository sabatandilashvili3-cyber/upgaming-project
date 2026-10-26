"""Run an answer-eval configuration as a Langfuse experiment (runs on the server).

    python -m eval.experiment --run opus-high-v2 [--model claude-opus-5-5] [--effort high]
                              [--sets world hotpot redteam] [--limit N] [--yes]

1. Uploads each question set as a Langfuse dataset (idempotent: item ids are stable).
2. Runs every item through the full investigator, inside Langfuse's experiment runner, so each
   item's trace is linked to the dataset run and every metric lands as a score. Runs with
   different --run names are compared side by side in Langfuse (Datasets -> Runs).
3. Also writes eval/results/answers_rows_<run>.jsonl with the draft, evidence and claims,
   so eval.reverify can re-score the run later for free.

One question at a time (max_concurrency=1): the self-hosted Langfuse ran out of heap during
a burst of parallel eval traces once, and it also serves HSAI.
"""
import argparse
import json
import threading
import time

from dotenv import load_dotenv
from langfuse import Evaluation

from engine import observability as obs
from engine.agent.investigator import Investigator
from engine.search.dense import Embedder
from engine.search.retriever import Retriever

from .answers import RESULTS, draft_as_report, load_questions, score, summarise
from .judge import judge_adoption
from .redteam import load_redteam, with_attacks
from .retrieval import load_index

DATASETS = {"world": "meridian-v1", "hotpot": "hotpot-40", "redteam": "redteam-v1"}
EST_COST = 0.10            # USD per question, measured average of the first full run


def question_sets(names):
    qs = load_questions(n_hotpot=40)
    out = {"world": [q for q in qs if q["source"] == "world"], "hotpot": [q for q in qs if q["source"] == "hotpot"]}
    if "redteam" in names:
        out["redteam"] = load_redteam()
    return {n: out[n] for n in names}


def _retry(fn, attempts=4):
    """Langfuse API calls can time out when its server is busy; back off and try again."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:                      # httpx timeouts surface as several exception types
            if i == attempts - 1:
                raise
            wait = 5 * 2 ** i
            print(f"  langfuse call failed ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def upload(lf, name, questions):
    """Create the dataset and any missing items. Idempotent: existing items are skipped."""
    try:
        existing = {it.id for it in _retry(lambda: lf.get_dataset(name)).items}
    except Exception:
        _retry(lambda: lf.create_dataset(name=name, description=f"investigation-engine question set {name}"))
        existing = set()
    for q in questions:
        item_id = f"{name}:{q['qid']}"
        if item_id in existing:
            continue
        _retry(lambda: lf.create_dataset_item(
            dataset_name=name, id=item_id, input={"question": q["question"]},
            expected_output={k: q.get(k) for k in ("answer", "must_include", "expect_abstain", "contradiction",
                                                   "gold_doc_ids", "forbidden")},
            metadata={"qid": q["qid"], "kind": q["kind"], "source": q["source"]}))
    lf.flush()


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--sets", nargs="+", default=["world", "hotpot"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retrieval", choices=["G", "H-routed"], default="H-routed",
                    help="G = without the cross-encoder (the v2 model-comparison runs used G)")
    ap.add_argument("--no-prefetch", action="store_true", help="let the model run the first search itself")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    lf = obs.client()
    if lf is None:
        raise SystemExit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
    sets = question_sets(a.sets)
    n = sum(len(v[: a.limit]) for v in sets.values())
    if not a.yes and input(f"{n} questions on {a.model}/{a.effort}, about ${n * EST_COST:.0f}. Run? [y/N] ") != "y":
        return

    from engine.llm.claude import Claude
    llm = Claude(model=a.model, effort=a.effort)
    chunks, vecs, aliases = load_index("structural", "BAAI/bge-small-en-v1.5")
    embedder = Embedder()
    reranker = None
    if a.retrieval == "H-routed":
        from engine.search.rerank import Reranker
        reranker = Reranker()
    clean = Retriever(chunks, vecs, embedder, reranker)
    rows_path = RESULTS / f"answers_rows_{a.run}.jsonl"
    lock, rows = threading.Lock(), []

    for set_name, questions in sets.items():
        questions = questions[: a.limit]
        ds_name = DATASETS[set_name]
        upload(lf, ds_name, questions)
        by_qid = {q["qid"]: q for q in questions}
        wanted = {f"{ds_name}:{q['qid']}" for q in questions}
        items = [it for it in _retry(lambda: lf.get_dataset(ds_name)).items if it.id in wanted]

        # the attack documents exist only in red-team runs: the live demo and the main eval stay clean
        retriever = with_attacks(chunks, vecs, embedder, reranker) if set_name == "redteam" else clean

        def task(*, item, **kw):
            q = by_qid[item.metadata["qid"]]
            inv = Investigator(llm, retriever, prefetch=not a.no_prefetch).run(q["question"], tags=["eval", a.run, set_name])
            r = inv.report
            row = {"qid": q["qid"], "source": q["source"], "kind": q["kind"], "run": a.run, "model": a.model,
                   "effort": a.effort, "status": inv.status, "error": inv.error, "cost_usd": inv.cost_usd,
                   "latency_ms": inv.latency_ms, "steps": len(inv.steps),
                   "tool_calls": [c for st in inv.steps for c in st["calls"]], "question": q["question"],
                   "report": r, "draft": inv.draft, "evidence": inv.evidence, "claims": inv.claims,
                   "langfuse_trace_id": inv.lf_trace_id,
                   "verified": score(q, r, r.get("sources", []), aliases, r.get("contradictions", []),
                                     [c for st in inv.steps for c in st["calls"]], judge_adoption),
                   "unverified": score(q, draft_as_report(inv.draft), [], aliases, r.get("contradictions", []))}
            row["unverified"]["cite_precision"] = row["unverified"]["cite_recall"] = None
            with lock:
                rows.append(row)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            print(f"[{len(rows)}/{n}] {q['qid']:14} {inv.status:9} correct={row['verified']['answer_correct']} "
                  f"${inv.cost_usd:.3f} {inv.latency_ms / 1000:.0f}s", flush=True)
            return row

        def evaluator(*, input, output, expected_output, metadata, **kw):
            m = output["verified"]
            evals = [Evaluation(name=k, value=float(v)) for k, v in m.items() if v is not None]
            evals += [Evaluation(name="cost_usd", value=output["cost_usd"]),
                      Evaluation(name="latency_s", value=output["latency_ms"] / 1000),
                      Evaluation(name="unverified_answer_correct", value=float(output["unverified"]["answer_correct"]))]
            return evals

        lf.run_experiment(name=a.run, run_name=a.run, description=f"{a.model} effort={a.effort}",
                          data=items, task=task, evaluators=[evaluator], max_concurrency=1,
                          metadata={"model": a.model, "effort": a.effort, "retrieval": a.retrieval,
                                    "prefetch": str(not a.no_prefetch)})
        lf.flush()

    summary = {"verified": summarise(rows, "verified"), "unverified": summarise(rows, "unverified")}
    (RESULTS / f"answers_summary_{a.run}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["verified"]["all"]))


if __name__ == "__main__":
    main()
