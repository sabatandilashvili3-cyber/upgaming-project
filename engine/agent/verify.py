"""Grounding check: does each sentence of the answer actually come from what it cites?

A sentence passes only if:
  1. it cites at least one chunk, and every cited chunk was really retrieved during
     this investigation (the model cannot invent a source id);
  2. every "hard fact" in the sentence appears in at least one of its cited chunks.
     Hard facts are the things a hallucination usually gets wrong: identifiers
     (CHG-3340), numbers with units (220 minutes, 34%), clock times (09:12), dates, and any
     other number of two or more digits (an amount like 40,000 GEL, a count like 240).

Sentences that fail are removed from the report and listed under "unsupported",
so the reader sees what was dropped and why. This is deliberately strict and
mechanical: it never decides a sentence is "probably fine".
"""
import re
from dataclasses import dataclass, field

FACT_PATTERNS = [
    r"\b[A-Z]{2,5}(?:-\d{2,5})+\b",                            # ids
    r"\bv\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+\b",                    # versions, kept whole: v1.9.19 is one fact
    r"\b(?:1[5-9]|20)\d\d-\d\d-\d\d\b",                         # ISO dates, any year from 1500
    r"\b\d{1,2}:\d\d\b",                                       # clock times
    r"\b\d+(?:\.\d+)?\s*%",                                    # percentages
    r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|ms|s)\b",     # quantities with units
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{2,}(?:\.\d+)?\b",       # any other number with 2+ digits
]
FACT_RE = re.compile("|".join(FACT_PATTERNS))


def hard_facts(sentence: str):
    return [m.group(0) for m in FACT_RE.finditer(sentence)]


def _squash(s: str) -> str:
    """Lower-case, drop thousands separators, then spaces: "40,000 GEL" matches "40000 gel".

    Order and exactness matter (both found on HotpotQA): a separator is a comma followed by
    EXACTLY three digits, and it is removed while spaces still exist. Otherwise "November 11,
    1914" became "111914" and "May 8, 2007, 500 locations" became "2007500", and correct
    sentences quoting dates were rejected.
    """
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s.lower())
    return re.sub(r"\s+", "", s)


MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october "
                                     "november december".split(), start=1)}
_MONTH = "|".join(MONTHS)


def iso_dates(text: str) -> str:
    """ISO forms of the natural-language dates in a text: "July 26, 2006" / "26 July 2006" -> 2006-07-26.
    Appended to the evidence, so a timeline line that writes the same date as 2006-07-26 matches."""
    out = []
    for m in re.finditer(rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", text, re.I):
        out.append(f"{m[3]}-{MONTHS[m[1].lower()]:02d}-{int(m[2]):02d}")
    for m in re.finditer(rf"\b(\d{{1,2}})\s+({_MONTH})\s+(\d{{4}})\b", text, re.I):
        out.append(f"{m[3]}-{MONTHS[m[2].lower()]:02d}-{int(m[1]):02d}")
    return " ".join(out)


UNIT_ALIASES = {"ms": ("ms", "millisecond"), "s": ("seconds", "_s ", " s "), "mins": ("min",), "min": ("min",)}


def _present(fact: str, corpus: str) -> bool:
    """Plain numbers must match whole: an invented "24" must not pass because "240" is cited."""
    if re.fullmatch(r"[\d.]+", fact):
        # a digit or a decimal part may not touch it; a sentence's full stop may ("240 -> 60.")
        return re.search(rf"(?<!\d)(?<!\d\.){re.escape(fact)}(?!\d)(?!\.\d)", corpus) is not None
    return fact in corpus


@dataclass
class Verdict:
    supported: bool
    reason: str = ""
    missing: list = field(default_factory=list)


def check(sentence: str, cited_ids, evidence: dict, given: str = "") -> Verdict:
    """evidence: id -> text, for every passage retrieved in this investigation.

    given: text the sentence may restate without a source, i.e. the question itself.
    "Before the incident INC-2026-002" names the incident the user asked about; that is
    not a claim the cited change-log row has to contain.
    """
    if not cited_ids:
        return Verdict(False, "no citation")
    unknown = [c for c in cited_ids if c not in evidence]
    if unknown:
        return Verdict(False, "cites a source that was never retrieved", unknown)
    raw = " ".join(evidence[c] for c in cited_ids) + " " + given
    corpus = _squash(raw + " " + iso_dates(raw))
    missing = [f for f in hard_facts(sentence) if not _present(_squash(f), corpus) and not _unit_ok(f, raw)]
    if missing:
        return Verdict(False, "facts not found in the cited sources", missing)
    return Verdict(True)


def _unit_ok(fact: str, raw: str) -> bool:
    """"5000 ms" is supported by an alert reading "metric: p99_latency_ms, value: 5000": the number is
    there and so is the unit. Only for ms/s, where sources store the unit in the field name; durations
    in minutes/hours must still match as written."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s)", fact.strip().lower())
    if not m:
        return False
    num, unit = m[1], m[2]
    low = raw.lower()
    return bool(re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", low)) and any(a in low for a in UNIT_ALIASES[unit])


def find_support(missing, evidence: dict, exclude=()):
    """A passage the model DID read that contains every missing fact, or None.

    Used for citation repair: "conflicts with the postmortem's 20:27" citing only the status
    page is grounded, the model just forgot to cite the postmortem it read. The repaired
    citation is labelled, so the report shows it was added by the verifier.
    """
    found = find_supports(missing, evidence, exclude, max_extra=1)
    return found[0] if found else None


def find_supports(missing, evidence: dict, exclude=(), max_extra: int = 2):
    """Smallest set (<= max_extra) of passages the model read that together contain every missing
    fact, chosen greedily; [] if they cannot all be covered. A sentence naming two incidents can
    draw on two passages."""
    def has(text, fact):
        return _present(_squash(fact), _squash(text + " " + iso_dates(text))) or _unit_ok(fact, text)
    left, chosen = list(missing), []
    while left and len(chosen) < max_extra:
        best = max(((eid, [f for f in left if has(t, f)]) for eid, t in evidence.items()
                    if eid not in exclude and eid not in chosen), key=lambda x: len(x[1]), default=(None, []))
        if not best[1]:
            break
        chosen.append(best[0])
        left = [f for f in left if f not in best[1]]
    return chosen if not left else []


def split_sentences(text: str):
    """Sentence split that leaves "v1.9.19", "2.4 s" and "e.g." alone."""
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text.strip()) if p.strip()]
