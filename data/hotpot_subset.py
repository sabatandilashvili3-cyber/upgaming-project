"""Build a reproducible HotpotQA subset: a paragraph corpus + gold questions.

Source: hotpotqa/hotpot_qa, "distractor" config, validation split (7,405 questions).
Every question ships with 10 Wikipedia intro paragraphs: 2 gold, 8 distractors.
We pool the paragraphs of a seeded sample of questions into one corpus, so each
question's gold paragraphs must be found among ~50k others, not among its own 10.

The same Wikipedia paragraph often appears under several questions. We keep one
copy per title and count the repeats (that count goes into the dedup report).

Usage:  python -m data.hotpot_subset --questions 6000 --seed 13
Writes: data/corpus/hotpot_paragraphs.jsonl   one record per unique paragraph
        data/corpus/hotpot_questions.jsonl    question, answer, type, level, gold titles
        data/corpus/hotpot_manifest.json      seed, counts, question ids (reproducibility)
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "hotpotqa/hotpot_qa"
FILE = "distractor/validation-00000-of-00001.parquet"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "corpus"


def load_rows():
    path = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir=RAW)
    return pq.read_table(path).to_pylist()


def build(n_questions: int, seed: int):
    rows = load_rows()
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_questions, len(rows)))

    paragraphs = {}   # title -> record
    repeats = 0
    questions = []
    for row in sample:
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            text = "".join(sentences).strip()
            if title in paragraphs:
                repeats += 1
                continue
            paragraphs[title] = {
                "doc_id": "hp::" + title,
                "title": title,
                "text": text,
                "source": "hotpotqa",
                "format": "jsonl",
            }
        questions.append({
            "qid": "hp_" + row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],          # bridge | comparison
            "level": row["level"],        # easy | medium | hard
            "gold_doc_ids": sorted({"hp::" + t for t in row["supporting_facts"]["title"]}),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hotpot_paragraphs.jsonl", "w") as f:
        for rec in paragraphs.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(OUT / "hotpot_questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifest = {
        "source": f"{REPO}/{FILE}",
        "seed": seed,
        "questions": len(questions),
        "unique_paragraphs": len(paragraphs),
        "repeated_paragraphs_skipped": repeats,
        "question_ids": [q["qid"] for q in questions],
    }
    (OUT / "hotpot_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    m = build(a.questions, a.seed)
    print({k: v for k, v in m.items() if k != "question_ids"})
