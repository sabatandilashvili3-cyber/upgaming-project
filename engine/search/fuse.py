"""Hybrid search: merge the keyword list and the semantic list into one ranking.

Reciprocal Rank Fusion (Cormack et al., 2009):

    rrf(d) = sum over lists L containing d of  1 / (k + rank_L(d))

with ranks starting at 1 and k = 60.

Why ranks and not scores: BM25 scores are unbounded (3.2, 17.8 ...) while cosine
similarities sit between about 0.3 and 0.9. Adding them would let whichever scale
is larger dominate. Ranks are on the same scale in every list. k = 60 flattens the
curve so that being 1st vs 2nd matters less than appearing in both lists at all:
a chunk ranked 5th by both beats a chunk ranked 1st by only one.
"""


def rrf(ranked_lists, k: int = 60, weights=None):
    """ranked_lists: lists of ids, best first. Returns [(id, score)] best first."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
