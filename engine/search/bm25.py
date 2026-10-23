"""Keyword search: Okapi BM25, written from the formula (no search library).

For a query q and a chunk d:

    score(d, q) = sum over terms t in q of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

tf      how often t occurs in d. More is better, but with diminishing returns:
        k1 (1.2) controls how fast the term-frequency curve flattens.
|d|     chunk length in tokens; avgdl the average. b (0.75) controls how much a
        long chunk is penalised for having many words that could match by chance.
idf     rare terms count more: "CHG-3340" appears in 3 chunks, "payment" in thousands.

Tokenizer: lower-case, keep hyphenated identifiers whole ("chg-3340",
"fraud-scoring") AND index their parts, so both "fraud-scoring" and
"fraud scoring" match.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

STOP = set("a an and are as at be by for from has have how in is it its of on or that the this to was were what "
           "when where which who why will with did do does".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str):
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOP:
            continue
        out.append(tok)
        if "-" in tok or "_" in tok:
            out.extend(p for p in re.split(r"[-_.]", tok) if p and p not in STOP)
    return out


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # term -> (chunk indexes, tfs)
        self.lengths = np.zeros(0, dtype=np.float32)

    def build(self, ids, texts):
        self.ids = list(ids)
        lists = defaultdict(lambda: ([], []))
        lengths = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                lists[term][0].append(i)
                lists[term][1].append(tf)
        self.postings = {t: (np.array(ix, dtype=np.int32), np.array(tf, dtype=np.float32))
                         for t, (ix, tf) in lists.items()}
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if len(lengths) else 0.0
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.ids), len(self.postings.get(term, ((), ()))[0])
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, allowed: np.ndarray | None = None):
        """Top-k (chunk_id, score). `allowed` is an optional boolean mask over chunks (filters)."""
        scores = np.zeros(len(self.ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ix, tf = self.postings[term]
            scores[ix] += self.idf(term) * tf * (self.k1 + 1) / (tf + norm[ix])
        if allowed is not None:
            scores[~allowed] = 0
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k] if len(scores) > k else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
