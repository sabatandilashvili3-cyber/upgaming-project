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
    r"\b20\d\d-\d\d-\d\d\b",                                   # dates
    r"\b\d{1,2}:\d\d\b",                                       # clock times
    r"\b\d+(?:\.\d+)?\s*%",                                    # percentages
    r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|ms|s)\b",     # quantities with units
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{2,}(?:\.\d+)?\b",       # any other number with 2+ digits
]
FACT_RE = re.compile("|".join(FACT_PATTERNS))


def hard_facts(sentence: str):
    return [m.group(0) for m in FACT_RE.finditer(sentence)]


def _squash(s: str) -> str:
    """Lower-case, drop spaces and thousands separators: "40,000 GEL" matches "40000 gel"."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", re.sub(r"\s+", "", s.lower()))


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
    corpus = _squash(" ".join(evidence[c] for c in cited_ids) + " " + given)
    missing = [f for f in hard_facts(sentence) if not _present(_squash(f), corpus)]
    if missing:
        return Verdict(False, "facts not found in the cited sources", missing)
    return Verdict(True)


def find_support(missing, evidence: dict, exclude=()):
    """A passage the model DID read that contains every missing fact, or None.

    Used for citation repair: "conflicts with the postmortem's 20:27" citing only the status
    page is grounded, the model just forgot to cite the postmortem it read. The repaired
    citation is labelled, so the report shows it was added by the verifier.
    """
    wanted = [_squash(m) for m in missing]
    for eid, text in evidence.items():
        if eid not in exclude and all(_present(w, _squash(text)) for w in wanted):
            return eid
    return None


def split_sentences(text: str):
    """Sentence split that leaves "v1.9.19", "2.4 s" and "e.g." alone."""
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text.strip()) if p.strip()]
