"""Re-score a saved answer-eval run for free: no model calls.

- the ground truth is re-read from the current question files, so a corrected answer key
  applies to old runs;
- if the run saved its evidence and claims (runs from 2026-10-26 on), the report is rebuilt
  with the CURRENT verifier from the model's saved draft, so verifier changes can be measured
  on the exact same model outputs.

Usage: python -m eval.reverify eval/results/answers_rows_<tag>.jsonl [--as <new tag>]
"""
import argparse
import json

from dotenv import load_dotenv
from pathlib import Path

from engine.agent.investigator import verify_report

from .answers import RESULTS, draft_as_report, load_questions, score, summarise
from .judge import judge_adoption
from .redteam import load_redteam
from .retrieval import load_index


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--as", dest="tag", default=None)
    ap.add_argument("--judge", action="store_true", help="LLM adoption judge for injection rows (~$0.001 each)")
    ap.add_argument("--tool-calls", default=None, help="json {run|qid: [tool calls]} fetched from Langfuse")
    ap.add_argument("--run", default=None, help="run name used as key prefix in --tool-calls")
    a = ap.parse_args()
    truth = {q["qid"]: q for q in load_questions(n_hotpot=6000)}       # every question we could have asked
    truth.update({q["qid"]: q for q in load_redteam()})
    judge = judge_adoption if a.judge else None
    calls = json.load(open(a.tool_calls)) if a.tool_calls else {}
    _, _, aliases = load_index("structural", "BAAI/bge-small-en-v1.5")
    out, rebuilt = [], 0
    for row in map(json.loads, open(a.rows)):
        q = truth[row["qid"]]
        r = row["report"]
        if row.get("evidence") is not None and row.get("draft") is not None:
            r = verify_report(q["question"], row["draft"], row["evidence"], row.get("claims", []))
            rebuilt += 1
        row["report"] = r
        tc = row.get("tool_calls") or calls.get(f"{a.run}|{row['qid']}")
        row["tool_calls"] = tc
        row["verified"] = score(q, r, r.get("sources", []), aliases, r.get("contradictions", []), tc, judge)
        row["unverified"] = score(q, draft_as_report(row.get("draft")), [], aliases, r.get("contradictions", []))
        row["unverified"]["cite_precision"] = row["unverified"]["cite_recall"] = None
        out.append(row)
    tag = a.tag or Path(a.rows).stem.replace("answers_rows_", "") + "_rescored"
    with open(RESULTS / f"answers_rows_{tag}.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row, default=str) + "\n")
    summary = {"verified": summarise(out, "verified"), "unverified": summarise(out, "unverified")}
    (RESULTS / f"answers_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(out)} rows, {rebuilt} rebuilt with the current verifier -> {tag}")
    for variant in ("verified", "unverified"):
        g = summary[variant]["all"]
        print(f"  {variant:10} answer_correct={g.get('answer_correct')} abstain_ok={g.get('abstain_ok')} "
              f"cite_precision={g.get('cite_precision')} n={g['n']}")


if __name__ == "__main__":
    main()
