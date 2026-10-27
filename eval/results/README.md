# Evaluation results

Every number in EVALUATION.md comes from a file here. Per-question rows (`*_rows_*.jsonl`) are
large and git-ignored; the summaries are kept.

| File | What it is |
|---|---|
| `retrieval_summary.json` | retrieval metrics for all 14 search configurations (`python -m eval.retrieval`) |
| `split_check.json` | the overfitting check: config chosen on one half of the questions, scored on the other (`python -m eval.split_check`) |
| `answers_summary_opus-high-v2-v4.json` | **Opus 5.5**, 95 questions, current verifier and contradiction comparison |
| `answers_summary_sonnet-v2-v4.json` | **Sonnet 5.5**, same questions |
| `answers_summary_haiku-v2-v4.json` | **Haiku 5.5**, same questions |
| `answers_summary_heldout-opus-medium.json` | **14 hand-written held-out questions**, live settings (Opus, medium effort) |
| `answers_summary_*-v2-v3.json` | the same three model runs scored with the earlier contradiction comparison: the "before" in EVALUATION.md §4b |
| `answers_summary_latency-medium-hard.json` | the latency experiment: medium effort + prefetch on the hard questions (EVALUATION.md §6b) |
| `answers_summary_baseline-v1.json` | the very first answer run, before the verifier fixes (EVALUATION.md §4) |
| `loadtest_report.json` | the load test on the server: search throughput and upload, update, delete times |
| `server/` | raw copies of the first runs as they came off the server, kept so the early numbers can be re-checked |

**Naming:** `<model>-v2` is the question set and prompt version of the main runs; the last part
is the scoring version (`v3` verifier, `v4` = v3 + contradiction comparison v2), re-scored from the
same saved model outputs with `python -m eval.reverify`, at no API cost.
