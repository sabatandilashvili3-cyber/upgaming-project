"""Query understanding for investigations: ids and time windows.

Investigation questions name things ("INC-2026-016", "CHG-3340") and periods
("in March 2026", "on 14 March 2026", "2026-03"). Embeddings are poor at both:
"March" and "April" sit close together in vector space, and an id is just noise
to a sentence model. So after fusion we re-score each candidate:

    final = fused * (1 + ID_BOOST * [chunk mentions an id from the query])
                  * (1 + TIME_BOOST * [chunk has a date inside the query's window])

and chunks dated *outside* an explicit window are pushed down (TIME_PENALTY).
Chunks with no date at all are left alone: a runbook is timeless.
"""
import re
from dataclasses import dataclass, field

from ..ingest.loaders import ID_RE

ID_BOOST = 1.0
TIME_BOOST = 0.5
TIME_PENALTY = 0.5

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items()) if m != "may"})
# whole words only, longest first, so "march" wins over "mar" and "market" never matches
MONTH = r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b"


@dataclass
class QueryFacts:
    ids: list = field(default_factory=list)
    window: tuple | None = None          # ("2026-03-01", "2026-03-31") inclusive, or None


def _month_window(y, m, d=None):
    if d:
        day = f"{y:04d}-{m:02d}-{d:02d}"
        return day, day
    last = [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"


def analyse(query: str, default_year: int = 2026) -> QueryFacts:
    facts = QueryFacts(ids=sorted(set(ID_RE.findall(query))))
    q = query.lower()
    year = r"(?:,?\s*(20\d\d))?"
    if m := re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", q):                      # 2026-03-14
        facts.window = _month_window(int(m[1]), int(m[2]), int(m[3]))
    elif m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + MONTH + year, q):       # 14 March 2026
        facts.window = _month_window(int(m[3] or default_year), MONTHS[m[2]], int(m[1]))
    elif m := re.search(MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b" + year, q):       # March 14, 2026
        facts.window = _month_window(int(m[3] or default_year), MONTHS[m[1]], int(m[2]))
    elif m := re.search(MONTH + r"\s*(20\d\d)", q):                                  # March 2026
        facts.window = _month_window(int(m[2]), MONTHS[m[1]])
    elif m := re.search(r"\b(?:in|during|since|throughout|of)\s+" + MONTH, q):         # "in March"
        facts.window = _month_window(default_year, MONTHS[m[1]])
    elif m := re.search(r"\b(20\d\d)-(\d\d)\b", q):                                # 2026-03
        facts.window = _month_window(int(m[1]), int(m[2]))
    return facts


def boost(fused, meta_of, facts: QueryFacts):
    """fused: [(chunk_id, score)]; meta_of(chunk_id) -> chunk metadata. Returns re-sorted list."""
    if not facts.ids and not facts.window:
        return fused
    out = []
    for cid, score in fused:
        meta = meta_of(cid)
        mult = 1.0
        if facts.ids and set(facts.ids) & set(meta.get("ids", [])):
            mult *= 1 + ID_BOOST
        dates = meta.get("dates", [])
        if facts.window and dates:
            lo, hi = facts.window
            if any(lo <= d <= hi for d in dates):
                mult *= 1 + TIME_BOOST
            else:
                mult *= TIME_PENALTY
        out.append((cid, score * mult))
    return sorted(out, key=lambda kv: kv[1], reverse=True)
