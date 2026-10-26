"""Was the default retrieval config overfitted to the questions it is scored on? Free: saved rows only.

The default was chosen on all 539 questions and reported on the same 539. Here, for many random
50/50 splits: pick the best config on one half (dev), score it on the other half (test). If
choosing on the questions themselves had inflated the numbers, the dev-chosen config would score
clearly lower on its test half than on its dev half, or a different config would win each time.

Score = recall@10 averaged over the two corpora (Meridian and HotpotQA weigh the same).

    python -m eval.split_check [--splits 1000]
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROWS = Path(__file__).parent / "results" / "retrieval_rows.jsonl"


def score(rows, config, qids):
    by = defaultdict(list)
    for q in qids:
        r = rows[config][q]
        by[r["group"].split(":")[0]].append(r["recall@10"])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    a = ap.parse_args()
    rows = defaultdict(dict)
    for r in map(json.loads, open(ROWS)):
        rows[r["config"]][r["qid"]] = r
    configs, qids = sorted(rows), sorted(next(iter(rows.values())))
    full = {c: score(rows, c, qids) for c in configs}
    rng, wins, gaps, test_vs_best = random.Random(0), Counter(), [], []
    for _ in range(a.splits):
        rng.shuffle(qids)
        dev, test = qids[: len(qids) // 2], qids[len(qids) // 2:]
        pick = max(configs, key=lambda c: (score(rows, c, dev), c))
        wins[pick] += 1
        gaps.append(score(rows, pick, dev) - score(rows, pick, test))
        test_vs_best.append(max(score(rows, c, test) for c in configs) - score(rows, pick, test))
    gaps.sort()
    out = {"questions": len(qids), "splits": a.splits,
           "full_set_score": {c: round(v, 3) for c, v in sorted(full.items(), key=lambda x: -x[1])},
           "chosen_on_dev": dict(wins.most_common()),
           "dev_minus_test_mean": round(sum(gaps) / len(gaps), 4),
           "dev_minus_test_p95": round(gaps[int(0.95 * len(gaps))], 4),
           "test_loss_vs_best_on_test_mean": round(sum(test_vs_best) / len(test_vs_best), 4)}
    (ROWS.parent / "split_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
