"""Retrieval and answer metrics. Pure functions, unit-tested.

Retrieval is scored on *documents*: a question's gold set is the documents that
contain the evidence, and a ranking is the documents in the order retrieval
surfaced them.

recall@k        share of the gold documents found in the top k.
all_gold@k      1 if every gold document is in the top k, else 0. Multi-hop questions
                need *both* paragraphs, so this is stricter and closer to "answerable".
mrr             1 / rank of the first gold document (0 if none in the list). Rewards
                putting evidence first, which matters because the LLM reads top-down.
"""
import re
import string
from collections import Counter


def recall_at_k(ranking, gold, k):
    if not gold:
        return None
    return len(set(ranking[:k]) & set(gold)) / len(set(gold))


def all_gold_at_k(ranking, gold, k):
    if not gold:
        return None
    return float(set(gold) <= set(ranking[:k]))


def mrr(ranking, gold):
    if not gold:
        return None
    for rank, d in enumerate(ranking, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------- answers
def _norm(s: str) -> str:
    """SQuAD-style normalisation: lower, drop punctuation and articles, squash spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_all(answer: str, required) -> float:
    """World questions: share of required facts present in the answer.

    An entry may be a list of acceptable forms: ["50 minutes", "19:07"] is satisfied by either.
    """
    if not required:
        return None
    a = _norm(answer)
    hit = lambda r: any(_norm(x) in a for x in r) if isinstance(r, list) else _norm(r) in a
    return sum(1 for r in required if hit(r)) / len(required)


def citation_precision(cited, gold):
    """Share of cited documents that are gold evidence (None when the question has no gold set)."""
    if not cited or not gold:
        return None
    return len(set(cited) & set(gold)) / len(set(cited))


def citation_recall(cited, gold):
    """Share of gold evidence documents the answer cites."""
    if not gold:
        return None
    return len(set(cited) & set(gold)) / len(set(gold))
