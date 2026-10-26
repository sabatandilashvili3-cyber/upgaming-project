# How the engine works: a step-by-step guide for the interview

Each section answers four questions, in the order an interviewer usually asks them:
**What does it do? How does it work? Why this way and not another? What could go wrong?**
Numbers come from real runs, not estimates. Code paths are given so you can open the file
and point at the line.

---

## 0. The 60-second pitch

> "Most RAG demos retrieve a few chunks and let the model write an answer. That can't
> tell you when two documents disagree or when the answer simply isn't there. So I built
> it the other way round: search finds evidence, the model only *extracts* structured claims
> from it, and plain Python decides what contradicts what and drops every sentence whose
> facts aren't in its cited source. To measure that honestly I generated a fictional
> fintech, Meridian Pay, where I know every true fact, every planted contradiction and
> every missing piece, and I evaluate on that plus 55,000 HotpotQA paragraphs."

Three ideas to remember: **company world with ground truth**, **report instead of chat
answer**, **evaluation decides**.

---

## 1. The data

### 1a. HotpotQA subset — `data/hotpot_subset.py`
**What:** 6,000 questions sampled from HotpotQA's *distractor validation* split. Each question
comes with 10 Wikipedia paragraphs (2 contain the answer, 8 are distractors). I pool the
paragraphs of all 6,000 questions into **one corpus of 54,937 unique paragraphs**.

**Why pool them:** in the original dataset each question only searches its own 10
paragraphs — far too easy. Pooled, the 2 gold paragraphs must be found among 55k. That's a
realistic search problem and it satisfies the 50,000-record requirement.

**Why the validation split:** it's 27 MB instead of 330 MB, has gold answers and gold
supporting paragraphs, and 7,405 questions are more than enough.

**Reproducibility:** `random.Random(13)` picks the sample; the manifest file saves the seed
and every question id. Same seed, same corpus, every time.

**Fact to quote:** the same paragraph appeared under several questions **4,778 times**. I keep
one copy per title.

### 1b. Meridian Pay, the generated company — `data/world_gen.py`
**What:** a seeded Python script that invents a payments company and writes its documents:

| Document | Format | Count |
|---|---|---|
| Change log (one row per deploy/config change) | CSV, one file per month | 3,305 rows |
| Incident reports, written *during* the incident | PDF | 40 |
| Postmortems, written *after* | Markdown or DOCX | 29 |
| Public status pages | HTML | 40 |
| Alerts, customer tickets | JSON | ~3,500 rows |
| Runbooks (one per service, plus one exact copy) | Markdown | 16 |
| Service catalogue | DOCX | 1 |

It has 15 services with a dependency graph (`payments-api` calls `ledger`, `fraud-scoring`…)
and 6 owning teams.

**What's planted on purpose:**
- **Causes:** 25 of the 40 incidents are caused by a specific change in the change log; 7 have an
  external cause; 8 were never explained.
- **Decoys:** harmless changes on the same service just before each incident, so "the change
  right before" is not automatically the answer.
- **Contradictions:** in 10 incidents two documents disagree (the start time, the duration, or
  the % of payments affected).
- **Superseded suspicions:** in 7 incidents the report written during the incident suspected a
  decoy change; the later postmortem names the real one. A good answer prefers the postmortem.
- **Gaps:** 11 incidents have no postmortem, and 8 were never explained. Asked for their cause,
  the right answer is "unknown".
- **Duplicates:** an exact copy of a runbook, and 7 customer emails that copy a status page
  with one sentence reworded.
- **Prompt injections:** 3 customer tickets with text like *"Ignore all previous
  instructions…"*.

**The headline case is the brief's own example.** In March, payments were delayed for 220
minutes by two changes:
- `CHG-3340` cut the ledger's DB connection pool from 240 to 60;
- `CHG-3357` deployed a slower fraud-scoring model.

The decoy is `CHG-3355`, a UI deploy that was rolled back. The postmortem says 220 minutes;
the status page says 2 hours.

**Why generate data at all (the key argument):** HotpotQA has no contradictions and no "the
answer isn't here" questions, yet the brief requires handling both. Because my script decides
every fact, it also writes **43 questions with exact expected answers**. The questions cover:
- simple facts;
- multi-hop (incident → change → service → owning team);
- multi-document;
- time windows;
- unanswerable questions;
- contradictions;
- prompt injection.

Scoring is deterministic: "does the answer contain `CHG-3340` and `CHG-3357`?" needs no
LLM judge.

**Weakness to admit before they say it:** synthetic text is cleaner than real text. That's why
HotpotQA (real Wikipedia) is in the same index, and the retrieval numbers are reported for
both separately.

**A bug story:** my first generated change log said "Scale **up** replicas 200 → 66", and a
reporting outage was blamed on the bank partner. I read the generated documents myself, found
three realism bugs and fixed them. The habit: look at the output, don't trust the generator.

---

## 2. Reading files — `engine/ingest/loaders.py`

**What:** one function per format turns bytes into `Record`s:

| Format | Library | How headings become `## heading` lines |
|---|---|---|
| PDF | pypdf | not available, so text only, plus page count |
| DOCX | python-docx | from heading styles; list items become `- ` lines; tables become `cell \| cell` rows |
| HTML | BeautifulSoup | from `h1`–`h3`; scripts and navigation dropped; `<meta>` tags kept |
| Markdown | plain text | from `#` lines |
| CSV / JSON / JSONL | built in | **one record per row** |

**Why one record per row:** a change-log row is an independent fact with its own id. If the
whole CSV were one document, "which change caused X" would retrieve a 600-row blob. Row ids
look like `changelog/changes_2026-03.csv#CHG-3340`, so a citation points at the exact row.

**Metadata extracted here, once:**
- every id-like token (`INC-2026-016`, `CHG-3340`, `TCK-2026-03-0012`);
- every date (`2026-03-14`).

Search uses them later to boost and filter.

**Bug the tests caught:** my id regex first allowed only one extra part, so `TCK-2026-03-0012`
became `TCK-2026-03`. The fix is `[A-Z]{2,5}(?:-\d{2,5})+`, meaning "any number of digit
groups".

**Normalisation:** Unicode NFKC (turns look-alike characters into one form), collapses spaces
and blank lines. The same document must always produce the same text, or the content hash in
the next step would change for nothing.

---

## 3. Duplicates — `engine/ingest/dedup.py`

### Exact duplicates
Compute the SHA-256 of the normalised text. Same hash means the same document was uploaded
twice: keep the first, store the second as an *alias* pointing at it, and don't index it again.
**Found 24** (e.g. the copied runbook; HotpotQA articles with identical text under two titles).

### Near duplicates: the interesting part
**Definition:** split the text into overlapping 3-word "shingles"
("payments were delayed", "were delayed for", …). Two texts are near duplicates when
**Jaccard ≥ 0.8**, where Jaccard = shared shingles ÷ all distinct shingles.

**The problem:** comparing every pair of 62k records is about 2 billion comparisons.

**The trick, MinHash + LSH:**
1. Hash every shingle with 64 different hash functions and keep the **minimum** of each, giving
   a 64-number signature.
2. **The key fact:** for one hash function, the chance that two sets have the same minimum
   *equals their Jaccard similarity*. Why: the minimum is like picking a random shingle from
   the union; it's shared exactly when that shingle is in both sets.
3. **LSH banding:** cut the 64 numbers into 16 bands of 4. Two records are compared only if at
   least one band matches exactly.
   - For J = 0.84: P(some band matches) = 1 − (1 − 0.84⁴)¹⁶ ≈ **0.99999**. Real duplicates are
     almost always compared.
   - For J = 0.3: ≈ **0.12**. Most unrelated pairs are never compared.
4. Every candidate pair is checked with the **exact** Jaccard, so LSH only finds candidates and
   never decides.

**Found 70.** All 7 planted emails were caught, at Jaccard 0.83–0.85. All 62k records take
**6 seconds**.

**The story to tell — I measured and changed my mind:** I first used SimHash, the classic
method (64-bit fingerprint, duplicates are ≤ 3 bits apart). It caught **0 of 7**. I measured
why: texts sharing 84% of their shingles were still 8–10 bits apart. The ≤ 3-bit cut-off was
tuned for long web pages, and with only ~35 shingles each change moves many bits. So I switched
to MinHash, which estimates Jaccard directly.

**Policy — near ≠ same:** HotpotQA's "Grammy Award for Best **Female** Rap Solo Performance"
and the **Male** one are 87% similar but are different facts. So near duplicates are **kept
and searchable**, and only merged into one row when both show up in the same result list
(`also_in`). Deleting them would destroy evidence.

**On upload:** the 16 band keys of every stored document live in the `lsh_bands` table, so a
new file is checked against all 62k documents with one indexed lookup.

---

## 4. Chunking: two strategies, compared — `engine/ingest/chunking.py`

**Why chunk at all:** an embedding model compresses a text into one vector. A 10-page document
becomes a blurry average of everything in it, and the model also has an input limit. Smaller
pieces give sharper matches and precise citations.

**Fixed (the baseline):** a sliding window of 200 words, 40 words of overlap (so a sentence cut
at a boundary appears whole in one window). Simple, but blind to structure: a window can start
in the middle of "Impact" without knowing it's about impact, or about which incident.

**Structural (the default):**
- split on headings and paragraphs;
- pack whole paragraphs up to 200 words, **never across a heading**;
- prefix every chunk with a *context line*: `[Postmortem INC-2026-016 > Impact] - Duration:
  220 minutes …`. The chunk now says what it is about, even out of context.
- a paragraph longer than 200 words falls back to fixed windows inside it.

**Why words and not tokens:** independent of the model and cheap to count. One word is about
1.3 tokens, so 200 words fit well inside bge-small's 512-token limit.

**Result:** 63,852 structural chunks vs 63,706 fixed. Nearly the same count, because most
records are short rows, so the difference is in *what's inside* them. The retrieval eval
measures whether that matters (configs C vs D).

---

## 5. Embeddings: meaning as numbers — `engine/search/dense.py`

**What an embedding is:** a model that maps text to a vector (here **384 numbers**) so that texts
with similar meaning point in similar directions. "payments queued" lands near "transactions
delayed" even though they share no words.

**Model:** `BAAI/bge-small-en-v1.5`. Small (33M parameters), strong on retrieval benchmarks for
its size, and it runs on a CPU. That matters because the free hosting has no GPU, and the server
must embed each question with **the same model** used for the documents.

**Normalised vectors:** every vector has length 1, so cosine similarity is just a dot product,
which is fast and what pgvector's cosine index expects.

**Query prefix:** bge was trained with the instruction "Represent this sentence for searching
relevant passages: " in front of *queries only*. Leaving it off measurably hurts retrieval.

**Cost:** all 63,852 chunks were embedded once, offline, on the Mac's GPU in **402 seconds**
(about 160 chunks/s), and cached to disk. No API money spent on embeddings.

---

## 6. Keyword search: BM25, written from the formula — `engine/search/bm25.py`

**Why keyword search at all:** embeddings are bad at exact identifiers. To a sentence model
`CHG-3340` and `CHG-3341` are almost the same string; to an investigation they're different
changes. Keyword search matches them exactly.

**The formula, piece by piece:**

```
score(d, q) = Σ over query terms t:  idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

| Part | Meaning | Intuition |
|---|---|---|
| `tf` | how often t appears in the chunk | more is better, but with diminishing returns |
| `k1 = 1.2` | saturation | the 10th occurrence adds far less than the 2nd |
| `\|d\| / avgdl` | chunk length vs. average | long chunks match more words by chance… |
| `b = 0.75` | length penalty strength | …so they're penalised this much |
| `idf` | rarity | `CHG-3340` appears in 3 chunks and counts a lot; "payment" in thousands and counts little |

**Implementation:** an inverted index, term → (chunk numbers, counts), as numpy arrays. A query
only touches the postings of its own terms, so it never scans all 63k chunks.

**Tokenizer choice:** lower-case, hyphenated ids kept whole **and** split into parts.
`fraud-scoring` is indexed as `fraud-scoring`, `fraud` and `scoring`, so both spellings match.

**Proof it's right:** `tests/test_search.py` computes one score by hand and asserts the code
gives the same number.

---

## 7. Hybrid search: Reciprocal Rank Fusion — `engine/search/fuse.py`

**Problem:** BM25 scores are unbounded (3.2, 17.8…); cosine similarities sit between about
0.3 and 0.9. Adding them would let whichever scale is larger win.

**RRF ignores the scores and uses positions:**

```
rrf(d) = Σ over lists:  1 / (60 + rank)
```

**Worked example (it's a unit test):** a chunk ranked 2nd in both lists gets
2/62 ≈ 0.0323. A chunk ranked 1st in only one list gets 1/61 ≈ 0.0164. **Agreement beats one
first place.**

**Why k = 60:** the value from the original paper (Cormack et al., 2009). It flattens the curve,
so being 1st vs 2nd matters little and appearing in both lists matters a lot.

---

## 8. Investigation-aware boost: ids and dates — `engine/search/boost.py`

**Why:** investigations are about *specific things at specific times*. Embeddings place "March"
and "April" close together, so a March question happily retrieves April incidents.

**How:**
1. Parse the question:
   - ids: `INC-2026-016`;
   - a date window: "in March 2026" → 2026-03-01…31, "on 14 March" → that day, and also
     "2026-03" and "March 3rd, 2026".
2. Re-score each fused result:
   - **×2** if the chunk mentions an id from the question;
   - **×1.5** if it has a date inside the window;
   - **×0.5** if all its dates fall outside the window;
   - chunks with no date (a runbook) are untouched, because they're timeless.

**Bug story:** my first month regex matched "**mar**ket" (as March) and "**dec**line" (as
December), and "may" in "what *may* have caused" as the month of May. The fix matches whole
words only, and a bare "May" counts only after "in/during/since".

---

## 9. Storage: Postgres + pgvector — `engine/store/schema.sql`, `db.py`

**Why one Postgres instead of a separate vector database:** documents, versions, chunks,
vectors, duplicate links, traces and the LLM cache sit in one place with **transactions**. An
upload either fully happens or doesn't: never a document without its chunks. Deletes cascade
through foreign keys.

**HNSW, the vector index:** it builds a layered graph where each vector links to its near
neighbours. A search starts at the top layer and walks downhill towards the query. It's
*approximate*: it can miss the true nearest neighbour, in exchange for millisecond queries
instead of scanning 63k vectors. `ef_search = 100` is how many candidates the walk keeps; higher
means more accurate and slower.

**Updates (a brief requirement):**
- re-uploading a file compares content hashes;
- unchanged records are skipped, so no re-embedding;
- changed ones get `version + 1`, and their old chunks are replaced inside one transaction;
- rows that disappeared from a re-uploaded CSV are deleted.

The tests do exactly this.

**Bulk load:** streams 62k documents and chunks with binary `COPY` (far faster than INSERTs),
then builds the HNSW index once at the end, because updating the graph row by row is much
slower.

**Local development without admin rights:** `pgserver` ships a real Postgres 16 with pgvector
as a pip package. The same tests run against Neon in CI.

---

## 10. Contradictions, decided by code — `engine/agent/claims.py`

**The split that makes this explainable:** the LLM only **extracts** claims, as structured
records through a tool call:
`{entity: INC-2026-016, attribute: duration, value: "220 minutes", chunk_id: …}`. Python then
**decides**:
1. **Normalise the attribute:** "outage length" becomes `duration`.
2. **Normalise by type:**
   - durations: "220 minutes" = "3h 40m" = 220 min;
   - percentages: "34 percent" = "34%";
   - times and dates: compared as timestamps;
   - causes: compared as id sets.
3. **Group by (entity, attribute):** more than one distinct value in a group is a contradiction.
4. **Rank by source authority:** postmortem / change log (4) > runbook (3) > incident report /
   status page (2) > customer email (1) > customer ticket (0).

Step 4 means the report can say *"the postmortem says 220 minutes, the public status page says
120; prefer the postmortem because it outranks the status page"*. With equal authority it says
"cannot choose".

**Why not ask the model "do these contradict?":** models both miss real conflicts and invent
fake ones. Here a contradiction exists only if two different values with two real citations
are in the ledger. That's testable, and it can't hallucinate.

**Prompt injection connection:** the injected ticket claims "the cause was CHG-0001". It becomes
a claim from a *customer ticket* (authority 0) that conflicts with the postmortem (authority 4),
so it's shown as a rejected conflicting claim instead of being obeyed.

---

## 11. The grounding check — `engine/agent/verify.py`

Every sentence in the final report must pass two checks, or it's removed and listed as
"unsupported":
1. **It cites real evidence:** at least one chunk, and every cited chunk was actually
   retrieved in this investigation. The model can't invent a source id.
2. **Its hard facts are present:** every id, date, clock time, percentage and number-with-unit
   must literally appear in its cited chunks. These are exactly the things hallucinations get
   wrong.

**Trade-off to state honestly:** it's strict. A correct paraphrase like "about three and a half
hours" fails if the source says 220 minutes. I chose false drops over false facts, because in an
investigation a confident wrong number is worse than a missing one. The eval measures how often
it drops correct sentences.

---

## 12. Evaluation — `eval/`

**Retrieval eval needs no LLM:** it's free, deterministic and runs in CI on every push. Metrics
are computed on *documents*:

| Metric | Meaning | Example |
|---|---|---|
| **Recall@k** | share of gold documents in the top k | 1 of 2 gold in top 5 = 0.5 |
| **All-gold@k** | 1 only if *every* gold doc is in the top k | multi-hop needs both paragraphs, so this is "answerable" |
| **MRR** | 1 / rank of the first gold doc | first gold at rank 2 = 0.5; rewards evidence on top, since the model reads top-down |

**Five configs** (the brief asks for at least two). Each pair isolates one decision:

| | Chunks | Search |
|---|---|---|
| A | fixed | dense only |
| B | fixed | BM25 only |
| C | fixed | hybrid |
| D | structural | hybrid |
| E | structural | hybrid + boost |
| F | structural | E + a second hop that follows ids (`engine/search/chain.py`) |
| G | structural | F + at most 2 rows of one file in front |

- **A vs B vs C:** does hybrid beat either alone?
- **C vs D:** does the chunking strategy matter?
- **D vs E:** does the boost help?
- **E vs F vs G:** does following ids help, and does limiting rows per file help?

### Retrieval results (539 questions: 500 HotpotQA + 39 Meridian; `python -m eval.retrieval --hotpot 500`)

The whole corpus (Wikipedia + company, ~62k records) is searched for every question.

| Config | | HotpotQA recall@10 | HotpotQA MRR | Meridian recall@5 | Meridian recall@10 | Meridian MRR |
|---|---|---|---|---|---|---|
| A | fixed, dense only | 0.842 | 0.922 | 0.109 | 0.173 | 0.099 |
| B | fixed, BM25 only | 0.764 | 0.787 | 0.186 | 0.622 | 0.198 |
| C | fixed, hybrid | 0.865 | 0.896 | 0.199 | 0.295 | 0.306 |
| D | structural, hybrid | 0.864 | 0.921 | 0.220 | 0.357 | 0.291 |
| E | D + id/time boost | 0.864 | 0.921 | 0.479 | 0.720 | 0.532 |
| F | E + second hop following ids | 0.864 | 0.921 | 0.483 | 0.720 | **0.665** |
| G | F + max 2 rows per file in front | 0.864 | 0.920 | **0.628** | **0.769** | 0.568 |

**What the numbers say (each is a sentence you can say in the interview):**
- **Embeddings alone fail on investigations.** On HotpotQA dense search is strong (MRR 0.92),
  but on Meridian it finds 17% of the evidence: ids and dates are noise to a sentence model.
  BM25 alone finds 62%.
- **Hybrid is not automatically better.** On Meridian, equal-weight RRF (C, 0.295) is *worse*
  than BM25 alone (B, 0.622): when one retriever is useless for a question type, fusion dilutes
  the good one. On Wikipedia hybrid wins (0.865 vs 0.842 dense). The id/time boost (E) is what
  fixes the investigation case: 0.357 → 0.720.
- **Structural chunking helps ranking, not coverage.** C → D: HotpotQA MRR 0.896 → 0.921 (the
  title line tells the model what each paragraph is about); recall@10 is flat.
- **The boost costs Wikipedia nothing.** D → E: HotpotQA identical, because Wikipedia questions
  rarely contain ids or dates, so the boost doesn't fire.
- **Some questions can't be answered by one search, whatever the quality.** "Which change caused
  INC-2026-004?": the answer row CHG-2458 never mentions INC-2026-004; only the postmortem links
  them. The second hop (F) lifts MRR on simple Meridian questions from 0.19 to 0.52.
- **Duplicates of a kind crowd out evidence.** 3-5 complaint tickets per incident all mention its
  id, all get boosted, and fill the top 5. Capping rows per file (G) raises Meridian recall@5
  from 0.48 to 0.63 and multi-hop recall@5 from 0.06 to 0.44, at some MRR cost (0.665 → 0.568).
  **G is the default:** the agent reads all top results, so finding evidence matters more than
  its exact position.
- **Honest caveat:** Meridian groups have 6-8 questions each, so one question moves a group's
  number by 0.12-0.17. The overall Meridian line (n = 39) and HotpotQA (n = 500) are the
  reliable ones. Every table prints n.

**Answer eval (next step):**
- HotpotQA: exact match and F1 against the gold answer.
- Meridian:
  - "does the answer contain every required fact";
  - did it abstain when it should;
  - did it flag the planted contradiction;
  - did it ignore the injection.
- **Citation precision/recall** compares the cited documents with the gold evidence.

**Exact duplicates in eval:** a gold document that was stored as an alias is resolved to its
kept copy, so dedup never counts as a miss.

---

## 13. Tests — `tests/` (38 passing)

- **Algorithms:** BM25 against a hand-computed score, RRF agreement, boost ordering, dense
  nearest neighbour.
- **Ingestion:** every loader format, the id regex, exact + near dedup, both chunkers' edge cases
  (overlap position, never crossing a heading, oversized paragraphs).
- **Database, on real Postgres:** upload → update → unchanged; rows removed from a re-uploaded
  CSV; duplicates detected on upload; delete cascades; HNSW search; a rejected `.exe` writes
  nothing. A fake embedder (hashed word counts) keeps these fast and model-free.
- **Contradictions and verifier:** equal values in different words aren't a conflict; the
  postmortem beats the ticket; invented numbers and invented sources are rejected.

---

## 14. Questions they'll probably ask, with short answers

**"Why not LangChain?"**
The brief asks for an algorithm of my own and says the search can't rely only on ready-made
libraries. More importantly, every step here is ~50 lines I can explain and debug. Libraries
are used where they're the right tool: pypdf, numpy, sentence-transformers, pgvector.

**"BM25 or embeddings — which is better?"**
Neither alone. BM25 wins on ids and rare words; embeddings win on paraphrases. That's why
hybrid exists, and configs A/B/C measure it.

**"What happens with 10 million documents?"**
- BM25 in memory would need a real inverted index service (or Postgres full-text) instead of
  being rebuilt.
- HNSW scales well, but needs RAM and a tuned `ef_search`.
- Embedding becomes a batch pipeline.
- MinHash dedup already scales: it's linear, not pairwise.

**"How do you know the answer isn't hallucinated?"**
- Every sentence must cite retrieved chunks.
- Its ids and numbers must appear in them, or it's removed.
- Contradictions are decided by code.
- The eval measures citation precision.

**"Why is the near-duplicate threshold 0.8?"**
The planted copies measure 0.83–0.85, and the closest real non-duplicate pairs (templated
Wikipedia articles) are why near duplicates are merged, never deleted. A lower threshold would
merge more distinct articles.

**"What's the weakest part?"**
- The grounding check is strict, so it drops correct paraphrases.
- The world data is synthetic.
- BM25 is rebuilt after each upload, which is fine for a demo but not for heavy write traffic.

Say these before they find them.

---

## 15. The investigation loop — `engine/agent/investigator.py`, `tools.py`

**What happens when someone asks a question:**
1. The model gets the question, the system prompt and **5 tools**:
   - `search`: hybrid search (config G) with the automatic second hop;
   - `read_document`: every passage of one document;
   - `timeline`: dated records mentioning a term inside a window;
   - `record_claims`: write facts into the claim ledger;
   - `submit_report`: finish.
2. **The loop:** call the model; if it asked for tools, run them and send all the results back
   in one message; repeat. It stops at `submit_report`, or at **10 steps** at most.
3. **Code finalises the report:**
   - verify every finding, timeline line and the summary;
   - compute contradictions from the ledger;
   - mark the report *answered* only if the model said answerable **and** at least one finding
     survived verification. Otherwise it's *abstained*, with the reasons.

**Evidence aliases:** each passage shown to the model gets a short id (E1, E2…). The model cites
E-ids; code maps them back to real chunk ids. A short id is hard to mistype or invent, and an
E-id that was never shown is rejected by the verifier.

**Prompt injection, three layers:**
1. Retrieved text is wrapped in `<evidence>` tags and the system prompt says it's data, never
   instructions.
2. Every passage carries its source kind and authority, so a customer ticket is visibly a ticket.
3. Even if the model repeats an injected claim, the contradiction logic ranks the ticket below
   the postmortem, and the eval checks the answer doesn't contain the injected change id.

**Why a manual loop instead of the SDK's tool runner:**
- a hard step cap;
- a per-step trace (shown in the UI, stored in `traces`);
- token and cost accounting per question;
- you can explain every line.

**Failure handling — nothing crashes the request:**
- **A failing tool** returns an error result to the model, and the model can try again.
- **A text answer without `submit_report`** gets one reminder.
- **An API failure** (auth, rate limit after the SDK's own retries, network) becomes an `error`
  report with the reason.
- **A safety refusal** is reported as such. Server-side fallback is on, so the API first retries
  on a fallback model.

**Cost control:**
- **Prompt caching:** the system prompt and tool list never change, and each step only appends,
  so later steps re-read the prefix at about a tenth of the price. The trace shows
  `cache_read` tokens per step.
- **On the API side:** an hourly limit per client, plus a daily budget enforced from the
  `traces` table.

**Provider seam (`engine/llm/base.py`):** the agent only knows `start / step / add_tool_results`.
Claude is one adapter. The tests use a scripted fake, so the whole loop is tested with no key
and no cost (`tests/test_investigator.py`).

## 16. The API and the page — `engine/api/app.py`, `web/index.html`

- **Startup:** the port opens immediately. The embedding model and the BM25 index load in a
  background thread, and `/api/health` says "starting" until they're ready.
- **Streaming:** investigations stream as server-sent events, so the page shows each step while
  the model works, then the report.
- **Uploads:** 10 MB cap; the client's path is stripped from the file name. A wrong format
  returns 415; a corrupt file returns 422 with the reason. Uploads and deletes need the admin
  token when one is set.
- **The page is one static HTML file** with no framework: the brief prioritises working
  functionality over visual design. Citation chips open the cited document.

## 17. Bugs found by running it, not by reading it

All of these are in the git history with the fix.

| Bug | How it was found | Fix |
|---|---|---|
| Ticket ids cut short (`TCK-2026-03`) | a unit test | id regex allows any number of digit groups |
| "the 2014-15 season" read as month 15, crashing the eval | the eval crashed on a HotpotQA question | validate month/day ranges; `YYYY-YY` with the next year is a season |
| Bulk-loaded chunk metadata stored as a JSON *string* | the API's search returned 500 on real data | wrap in `Jsonb`. The test was first proven to fail on the old code |
| Unknown document returned as a normal result | a loop test | `ToolError` → `is_error` |
| Verifier let "40,000 GEL" through | my own UI sample report | every 2+ digit number is a checked fact |
| Verifier let "24" pass because "240" was cited | reviewing the previous fix | numbers match whole |
| Generated change log said "Scale up 200 → 66" | reading the generated documents | template fixed; ids and versions renumbered in date order |

## 18. Hosting: why Oracle Cloud — `DEPLOYMENT.md`

- **Hugging Face Spaces was the plan.** Its docs, checked before deploying, say:
  - Docker Spaces now **require a paid plan**;
  - a Space can only open outbound connections on **ports 80, 443 and 8080**, so it could never
    reach Postgres on 5432.
- **Render and Koyeb free tiers give 512 MB.** I **measured** the app at about 550 MB:
  - chunks: 130 MB;
  - BM25: 215 MB;
  - PyTorch model: 130 MB.
- **Oracle's Always Free ARM VM** has 24 GB and ordinary networking. Postgres runs on the same
  machine.

**If asked "how would you fit it in 512 MB?":**
- store BM25 as flat arrays (CSR) instead of a dict of small arrays;
- keep only ids and dates in memory and fetch chunk text from the database for the hits;
- embed queries with ONNX Runtime instead of PyTorch.

That's about 300 MB, then measure that retrieval quality is unchanged.

## 19. Langfuse: observability and experiments — `engine/observability.py`, `eval/experiment.py`

- **One investigation = one trace:**
  - a span per Claude step (stop reason, tool calls);
  - a tool span per call (input, the evidence returned);
  - a `verify` span (kept and dropped sentences, contradictions);
  - scores (status, findings kept, sentences dropped, contradictions, cost).
- **Live traffic and evals stay separate:** live traffic is tagged `production`, eval runs `eval`.
- **Bug found by measuring:** the Langfuse SDK already records every Anthropic call as a
  generation. My own step was also a generation, so **Langfuse showed double the real cost**.
  Steps became plain spans; checked afterwards: $0.0546 in Langfuse vs $0.0556 computed.
- **Evals are Langfuse experiments.** Question sets are datasets with stable item ids, each
  configuration is a run, and every metric is a score, so runs compare side by side.
- **One item at a time:** the self-hosted Langfuse also serves HSAI and once ran out of heap on a
  burst.
- **Cloudflare gotcha:** Python's default `urllib` user-agent gets HTTP 403 "error 1010" from the
  Cloudflare in front of Langfuse. A normal user-agent fixes it.

## 20. What the evaluations changed (each is a story for the interview)

| Finding | How it was found | What changed |
|---|---|---|
| The answer key was wrong for 2 questions (facts no document states) | reading the first failures | ground truth = what documents state; documents proven unchanged (indexed text identical for 6,973 records) |
| The verifier dropped 55 correct sentences and 16 whole summaries | reading every dropped sentence | v2/v3: versions whole, the question's facts given, labelled citation repair, per-sentence summary, ISO dates, units in field names |
| "November 11, 1914" became "111914" | a HotpotQA abstention with both gold paragraphs retrieved | thousands separator = a comma plus *exactly* 3 digits, removed before spaces |
| Injection scoring punished models for reporting attacks | reading the 11 flagged answers | behaviour checks (real tool calls from Langfuse, leaks) + an LLM judge for "adopted vs reported" |
| A web-trained re-ranker wrecked investigation retrieval (0.776 → 0.438) | the retrieval eval split by question set | fuse its order with ours, and use it only for questions with no ids or dates (best on every metric) |
| Search took 1.5 s; deleting a file took 165 s | the load test on the server | re-rank 15 candidates × 256 tokens (0.7 s, no quality loss); index on `lsh_bands(doc_id)` (0.2 s) |

**Model choice, with numbers:**
- Opus 0.958 at 8.6¢; Sonnet 0.937 at 3.3¢; Haiku 0.895 at 0.25¢.
- Opus is the only model that records claims from every source, so it's the only one that
  reliably finds contradictions (0.78 vs 0.33 / 0.22). That's why it's the default.

**Why an LLM judge only for injections:** everything else has an exact answer and is checked by
code. "Did the report adopt the claim or report it?" is a question of meaning. The judge must
quote its evidence, and it was checked against hand labels.

## Still to do
- the short-answer prompt (the bare span, for HotpotQA EM/F1);
- make claim recording mandatory for cheaper models;
- push to GitHub and fill in the AI usage report's own sections.
