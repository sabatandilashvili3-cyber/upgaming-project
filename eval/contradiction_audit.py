"""Contradiction detection on saved runs, re-computed with the CURRENT claims.py. Free: no model calls.

For every world row it recomputes contradictions from the saved claim ledger and labels each one:
  planted  - the generator planted a disagreement on this incident and attribute family
  spurious - anything else (a real but unplanted disagreement, or a false alarm), listed for review

    python -m eval.contradiction_audit eval/results/answers_rows_<run>.jsonl [...]
"""
import json
import sys
from pathlib import Path

from engine.agent.claims import Claim, find_contradictions

from .answers import FAMILY, contradiction_found

WORLD = Path(__file__).resolve().parents[1] / "data" / "corpus" / "world"


def planted():
    out = {}
    for i in json.load(open(WORLD / "truth.json"))["incidents"]:
        for c in i["contradictions"]:
            a = {"suspected_cause": "root_cause"}.get(c["attribute"], c["attribute"])
            out.setdefault(i["incident_id"], set()).add(FAMILY.get(a, a))
    return out


def audit(path, verbose=False):
    qs = {q["qid"]: q for q in map(json.loads, open(WORLD / "world_questions.jsonl"))}
    plant = planted()
    hit = want = spurious = rows = 0
    examples = []
    for row in map(json.loads, open(path)):
        q = qs.get(row["qid"])
        if not q or not row.get("claims"):
            continue
        rows += 1
        found = find_contradictions([Claim(**c) for c in row["claims"]])
        if q.get("contradiction"):
            want += 1
            hit += contradiction_found(q["contradiction"], [{"attribute": c.attribute, "values": c.values} for c in found])
        for c in found:
            fam = FAMILY.get(c.attribute, c.attribute)
            if fam not in plant.get(c.entity, set()):
                spurious += 1
                examples.append(f"{row['qid']} {c.entity} {c.attribute}: " + " | ".join(v["value"][:40] for v in c.values))
    print(f"{Path(path).stem.replace('answers_rows_', ''):22} planted found {hit:.0f}/{want}  "
          f"other contradictions {spurious} (over {rows} rows)")
    if verbose:
        for e in examples:
            print("    ", e)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p != "-v":
            audit(p, "-v" in sys.argv)
