"""Evidence chaining: a second retrieval hop that follows identifiers. No LLM involved.

Why it exists (measured, not assumed): for "Which change caused INC-2026-004?" the
answer is the change-log row CHG-2458, and that row never mentions INC-2026-004.
The link lives in the postmortem ("Root cause: CHG-2458 ..."). One search finds the
postmortem; only a second search for CHG-2458 finds the row. So:

  hop 1: search the question
  look:  collect ids mentioned in the top `look` hits that the question did not
         contain, skipping a row's own id (a ticket's TCK number leads nowhere)
  hop 2: search each new id (at most `follow`), most-mentioned first
  merge: RRF over the hop-1 list and every hop-2 list, with hop 1 weighted higher,
         so following ids adds evidence without burying the original results

The LLM agent does the same thing more flexibly (it can follow a service name or a
team, not just an id). This deterministic version is the baseline that shows how
much of multi-hop retrieval needs no model at all.
"""
from collections import Counter

from .boost import analyse
from .fuse import rrf
from .retriever import SearchConfig


def new_ids(hits, question_ids, look=5):
    counts = Counter()
    for h in hits[:look]:
        own = h.doc_id.rsplit("#", 1)[-1] if "#" in h.doc_id else ""
        for i in h.meta.get("ids", []):
            if i not in question_ids and i != own:
                counts[i] += 1
    return [i for i, _ in counts.most_common()]


def chained_search(retriever, question, cfg: SearchConfig, qvec=None, look=5, follow=2, hop_weight=0.5):
    first = retriever.search(question, cfg, qvec=qvec)
    follow_ids = new_ids(first, set(analyse(question).ids), look)[:follow]
    if not follow_ids:
        return first, []
    lists, weights = [[h.chunk_id for h in first]], [1.0]
    by_id = {h.chunk_id: h for h in first}
    for fid in follow_ids:
        hop = retriever.search(fid, SearchConfig(mode=cfg.mode, boost=True, k=5, candidates=cfg.candidates))
        lists.append([h.chunk_id for h in hop])
        weights.append(hop_weight)
        for h in hop:
            by_id.setdefault(h.chunk_id, h)
    merged = [by_id[cid] for cid, _ in rrf(lists, weights=weights)]
    return merged[:cfg.k], follow_ids
