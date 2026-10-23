"""Grounding check: does each sentence of the answer actually come from what it cites?

A sentence passes only if:
  1. it cites at least one chunk, and every cited chunk was really retrieved during
     this investigation (the model cannot invent a source id);
  2. every "hard fact" in the sentence appears in at least one of its cited chunks.
     Hard facts are the things a hallucination usually gets wrong: identifiers
     (CHG-3340), numbers with units (220 minutes, 34%), clock times (09:12), dates.

Sentences that fail are removed from the report and listed under "unsupported",
so the reader sees what was dropped and why. This is deliberately strict and
mechanical: it never decides a sentence is "probably fine".
"""
import re
from dataclasses import dataclass, field

FACT_PATTERNS = [
    r"\b[A-Z]{2,5}(?:-\d{2,5})+\b",                            # ids
    r"\b20\d\d-\d\d-\d\d\b",                                   # dates
    r"\b\d{1,2}:\d\d\b",                                       # clock times
    r"\b\d+(?:\.\d+)?\s*%",                                    # percentages
    r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|ms|s)\b",     # quantities with units
]
FACT_RE = re.compile("|".join(FACT_PATTERNS))


def hard_facts(sentence: str):
    return [m.group(0) for m in FACT_RE.finditer(sentence)]


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


@dataclass
class Verdict:
    supported: bool
    reason: str = ""
    missing: list = field(default_factory=list)


def check(sentence: str, cited_ids, evidence: dict) -> Verdict:
    """evidence: chunk_id -> chunk text, for every chunk retrieved in this investigation."""
    if not cited_ids:
        return Verdict(False, "no citation")
    unknown = [c for c in cited_ids if c not in evidence]
    if unknown:
        return Verdict(False, "cites a source that was never retrieved", unknown)
    corpus = _squash(" ".join(evidence[c] for c in cited_ids))
    missing = [f for f in hard_facts(sentence) if _squash(f) not in corpus]
    if missing:
        return Verdict(False, "facts not found in the cited sources", missing)
    return Verdict(True)
