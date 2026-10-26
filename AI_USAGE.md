# AI Usage Report

## Tools used
| Tool | Used for |
|---|---|
| **Claude Code** (Anthropic's coding agent, model Claude Opus 5.5) | the plan, most of the code, tests, debugging, documentation, the deployment scripts, and step-by-step help through the Oracle Cloud console |
| **Claude Opus 5.5 via the Anthropic API** | inside the product: the investigation agent |

## How the work was split

**The AI proposed the plan.** It suggested:
- a generated company with known ground truth;
- reports instead of chat answers;
- evaluation first.

**I asked for an approach that would stand out and that I could fully explain, then approved
the plan.** I chose the hosting (Oracle Cloud) when the AI found that Hugging Face Spaces now
need a paid plan. I created the cloud server, network and firewall rules myself in the Oracle
console, with the AI guiding each step.

**The AI wrote the first version of every module and ran the experiments.**

**Every claim in the docs was checked by running the code, not by reasoning about it.** The
evaluation numbers come from scripts in `eval/`, and the bug log in DEFENSE.md §17 lists what
running the code exposed.

## Where AI output was wrong and had to be corrected

These are visible in the git history. Each was found by running or measuring, then fixed.

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

## What I did myself

- **Direction and decisions.** I asked for an approach that would not look like every other
  submission, approved the plan, and made the product calls: Oracle hosting, medium effort instead
  of high when answers took too long, conversation memory after I noticed follow-up questions
  lost their context, and the ChatGPT-style white UI with the Upgaming mark.
- **Infrastructure.** I created the Oracle Cloud VM, network, subnet and firewall rules in the
  console, and provided the API keys and my own self-hosted Langfuse.
- **Testing as a reviewer.** I asked for hard hand-written questions and for a review written as
  an Upgaming engineer would write it. That found the two real gaps fixed last: complete lists and
  contradiction comparison.

> **TO FILL IN before submitting:** any code you rewrote by hand (planned: `engine/search/bm25.py`,
> `engine/search/fuse.py`, `engine/ingest/chunking.py`, then run the tests) and the bug-finding
> drills. Only list what you really did.

## What I learned from it
> **TO FILL IN** in your own words: for example, why hybrid search needed the boost, and why the
> contradiction check is code and not a prompt.
