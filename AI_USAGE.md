# AI Usage Report

This is an Applied AI Engineer role, so I built the project the way I think AI engineering works
now: **I led, Claude Code implemented, and nothing was accepted until a measurement said it worked.**
I made the decisions, set the bar, tested the system like a user and like a reviewer, and sent the
work back every time a number or an answer was wrong.

## Tools used
| Tool | Used for |
|---|---|
| **Claude Code** (Anthropic's coding agent, Claude Opus 5.5) | writing the code, tests, documentation and deployment scripts; running the experiments |
| **Claude Opus 5.5 via the Anthropic API** | inside the product: the investigation agent |
| **Langfuse** (my own self-hosted server) | tracing every investigation and running the evaluation experiments |

## My role, and what each decision changed

| What I did | What it changed |
|---|---|
| **Set the goal.** Not another "chat with your PDFs", and nothing I couldn't explain line by line. I chose the plan built around that: a generated company with known answers, verified reports instead of chat replies, evaluation first. | The core of the project: contradictions and "not established" answers can be scored exactly, which a HotpotQA-only project can't do. |
| **Chose and built the hosting.** When Hugging Face turned out to need a paid plan, I picked Oracle Cloud Always Free and built the VM, network, subnet and firewall rules myself in the console. I connected my own self-hosted Langfuse. | Live at a public URL with HTTPS; data survives a restart (persistence check PASS); every investigation traced. |
| **Started the evaluation round.** I asked for everything to go into Langfuse, for every kind of eval and test, and asked after each change "is it better than before?" | It exposed the verifier deleting correct answers (55 sentences), keyword-based injection scoring punishing the right behaviour, and doubled costs in Langfuse. With those fixed (and two wrong answer keys corrected), Opus went from 0.867 to 0.958 correct. |
| **Made the latency call.** Answers took too long for me as a user, so I chose medium effort over high. | Mean 15.5 s → 12.6 s and p95 −26%, with no loss in accuracy. |
| **Found the memory gap.** I tested follow-up questions and saw it forgot what I had just asked. | Conversation memory: "and which changes caused it?" now knows what "it" is (checked live over three turns). |
| **Tested it like a user.** I asked for hard questions in the style of the brief and graded every answer. | Found that "list every SEV1 on payout-service" got 2 of 6. A new scan tool fixed it (6/6), and those questions became a held-out set: 14/14. |
| **Reviewed it like an Upgaming engineer would**, then had every weak point fixed. | Contradiction detection 0.78 / 0.33 / 0.22 → 1.00 / 0.78 / 1.00; search throughput 1.4 → 2.0 per second; an overfitting check; CI green on GitHub. |
| **Set the UI direction.** ChatGPT-style, white, the Upgaming mark; then answer details folded under the summary and a polish pass. | Readable citations ("postmortem", "status page"), a cited summary, and an evaluation page that leads with the results. |

**How the code was written:** Claude Code wrote it. I didn't type the modules myself; I decided
what they do, checked that they work, and rejected what didn't. DEFENSE.md is the walkthrough I
study from, and before the interview I'm going through the core (`bm25.py`, `fuse.py`, the
chunkers, the verifier) line by line, so I can explain and change any part without AI.

## Where the AI's first version was wrong

None of these were caught by trusting the AI. Each one was found by running a test or a
measurement, several of them in tests I asked for, and each fix is in the git history.

| What the AI produced first | What running it showed | Fix |
|---|---|---|
| SimHash for near-duplicates | caught 0 of 7 planted duplicates; short texts were 8–10 bits apart, against a ≤ 3-bit cut-off | replaced with MinHash + LSH and exact Jaccard |
| A generated change log | "Scale **up** replicas 200 → 66", ids out of date order | generator templates fixed, ids renumbered |
| Id regex | cut `TCK-2026-03-0012` to `TCK-2026-03` | any number of digit groups |
| Date parser | read "2014-15 season" as month 15 and crashed the eval | month/day validation, season rule |
| Equal-weight hybrid search | *worse* than BM25 alone on company questions (0.295 vs 0.622 recall@10) | id/date boost, second hop, per-file diversity |
| Bulk loader | stored metadata as JSON *strings*, so search failed on real data | Jsonb wrapper, plus a test proven to fail on the old code |
| Grounding verifier | let an invented "40,000 GEL" through; later rejected a correct "240 -> 60." | numbers became checked facts; boundary rules fixed with tests |
| Verifier v1 | dropped 55 correct sentences and 16 whole summaries in the first eval run | v2/v3 rules, each measured on the saved answers |
| Thousands-separator rule | turned "November 11, 1914" into "111914", dropping correct Wikipedia answers | exact 3-digit rule, applied before removing spaces |
| Injection scoring by keywords | counted "a ticket told me to blame CHG-0001; I ignored it" as obeying the attack | behaviour checks + an LLM judge, checked against hand labels |
| Langfuse step spans as generations | doubled the cost shown in Langfuse | steps became spans; cost verified afterwards |
| Cross-encoder re-ranking | improved Wikipedia, broke investigation retrieval (0.776 → 0.438) | fused + routed re-ranking (best overall) |
| No index for cascaded deletes | deleting a 2,000-row file took 165 s (load test) | index on lsh_bands(doc_id): 0.2 s |
| Hosting plan (Hugging Face) | the docs, checked before deploying, say Docker Spaces need a paid plan and block outbound port 5432 | Oracle Cloud VM |
| Search for "list every …" questions | my hand-written test question "all SEV1s that hit payout-service" got 2 of 6: top-k search cannot prove a list is complete | `list_incidents` tool that scans every incident document |
| The explanation for weak contradiction detection ("cheaper models record fewer claims") | the saved claim ledgers showed Haiku recorded both sides in 8 of 9 cases; the code compared attribute names exactly | keyword families and typed values: Sonnet 0.33 → 0.78, Haiku 0.22 → 1.00, same saved outputs |

## What I learned from it

- **Search by meaning alone is not enough.** Embeddings can't tell `CHG-3340` from `CHG-3341`. I
  expected hybrid search to win automatically, but at first it was worse than plain keyword search
  on the company questions (0.295 vs 0.622). It only won after the id and date boost. So mixing two
  methods doesn't help by itself; you have to check what each one is good at.
- **Measure, don't guess.** The re-ranker made Wikipedia answers better and broke the company
  ones. int8 quantisation sounded like an easy speed-up and was slower on the ARM server. Almost
  every "obvious" improvement needed a number before I could trust it.
- **Before blaming the model, look at what it actually did.** We thought the cheaper models were
  bad at finding contradictions. When we opened their saved answers, they had recorded both sides;
  our code just compared the attribute names too strictly. Fixing about 30 lines of our own code
  took Haiku from 0.22 to 1.00.
- **RAG can't answer "list every…" questions on its own.** Search returns the top results, not all
  of them. I only found this by testing it like a user would, with my own hard questions. The fix
  was a tool that scans every incident instead of searching.
- **A score on your own generated questions can lie.** We got 1.000 on the company questions, but
  the same generator wrote the documents and the questions. The hand-written held-out questions are
  the more honest check.
- **Let the model write, let code decide.** The contradiction check and the sentence verifier are
  plain Python, so they can be tested and explained, and they can't make things up. But the first
  version of the verifier was too strict and deleted correct answers, so even the safety layer
  needs its own evaluation.
- **Production is its own work.** Keeping keys out of git, data that survives a restart, a budget
  limit, the load test that found a 165-second delete. And latency is a product decision: I chose
  medium effort because waiting 20+ seconds felt too long, and the eval showed it cost no accuracy.
