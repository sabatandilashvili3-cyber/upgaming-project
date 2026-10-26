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
from pathlib import Path

from engine.agent.investigator import verify_report

from .answers import RESULTS, draft_as_report, load_questions, score, summarise
from .retrieval import load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--as", dest="tag", default=None)
    a = ap.parse_args()
    truth = {q["qid"]: q for q in load_questions(n_hotpot=6000)}       # every question we could have asked
    _, _, aliases = load_index("structural", "BAAI/bge-small-en-v1.5")
    out, rebuilt = [], 0
    for row in map(json.loads, open(a.rows)):
        q = truth[row["qid"]]
        r = row["report"]
        if row.get("evidence") is not None and row.get("draft") is not None:
            r = verify_report(q["question"], row["draft"], row["evidence"], row.get("claims", []))
            rebuilt += 1
        row["report"] = r
        row["verified"] = score(q, r, r.get("sources", []), aliases, r.get("contradictions", []))
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
