"""Contradiction detection, done by code rather than by asking the model "do these disagree?".

The model's only job is extraction: while reading evidence it records claims as
structured records through a tool call:

    {"entity": "INC-2026-016", "attribute": "duration", "value": "220 minutes",
     "chunk_id": "postmortems/INC-2026-016_postmortem.docx::2"}

Then plain Python decides:
  1. normalise entity and attribute names by keyword ("resolved_time", "full recovery" ->
     "end_time"); a window ("12:36 to 15:46") becomes a start and an end claim;
  2. normalise values by type, so "220 minutes", "3h 40m" and "3 hours 40 minutes"
     compare equal, "34%" equals "34 percent", times compare as clock times. A value with
     nothing to compare ("since this morning" as a start time) is left out, not flagged;
  3. group by (entity, attribute); a group with more than one distinct normalised
     value is a contradiction, reported with every value and its sources;
  4. rank the conflicting values by source authority (a postmortem outranks a
     customer ticket), so the report can say which one to believe and why.

This is easy to test and to explain, and it cannot "hallucinate a contradiction":
two different values from two cited chunks either exist in the ledger or they don't.
"""
import re
from dataclasses import dataclass, field

# the model names attributes freely ("resolved_time", "impact_share", "window"), so names are
# matched by keyword, first family wins. Exact alias lists missed most of the cheaper models'
# names: Haiku read both sides of 8 planted contradictions and 2 were compared.
ATTRIBUTE_KEYWORDS = [
    ("affected_services", r"service|component|system"),
    ("root_cause", r"cause"),
    ("window", r"window|start.{0,5}end|period|time ?frame"),
    ("duration", r"duration|how long|length|downtime"),
    ("start_time", r"start|began|begin|detect|onset|first seen"),
    ("end_time", r"\bend|resol|recover|restor|fixed|mitigat"),
    ("impact", r"impact|share|percent|delayed|affected|failing|failed|error rate"),
]
TIME_RE = re.compile(r"\b(\d{1,2}:\d\d)\b")

# how much to trust a document by where it lives; higher wins a contradiction.
# Domain configuration, not logic: an unknown source gets the middle value.
AUTHORITY = [("postmortems/", 4, "postmortem"), ("changelog/", 4, "change log"),
             ("architecture/", 3, "architecture doc"), ("runbooks/", 3, "runbook"),
             ("incidents/", 2, "incident report written during the incident"),
             ("status/", 2, "public status page"), ("alerts/", 2, "alert"),
             ("comms/", 1, "customer email"), ("tickets/", 0, "customer ticket")]


# sources that describe impact in product names ("Payouts"), never service names: their lists
# are a different vocabulary and are not compared with service lists. Domain configuration.
PRODUCT_VOCABULARY = ("status/",)


def authority(doc_id: str):
    for prefix, level, label in AUTHORITY:
        if doc_id.startswith(prefix):
            return level, label
    return 2, "document"


def norm_attribute(a: str) -> str:
    a = a.strip().lower().replace("_", " ")
    for canon, pattern in ATTRIBUTE_KEYWORDS:
        if re.search(pattern, a):
            return canon
    return a


def norm_entity(e: str) -> str:
    return re.sub(r"\s+", " ", e.strip()).upper() if re.match(r"^[A-Za-z]{2,5}-\d", e.strip()) else e.strip().lower()


def _minutes(v: str):
    v = v.lower()
    h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", v)
    m = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", v)
    if h or m:
        return round((float(h[1]) * 60 if h else 0) + (int(m[1]) if m else 0))
    return None


def norm_value(attribute: str, value: str) -> str | None:
    """The comparable form of a value, or None when it has nothing comparable for its type."""
    v = value.strip()
    if attribute == "duration":
        return f"{mins} min" if (mins := _minutes(v)) is not None else None
    if attribute == "impact":
        p = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", v.lower())
        return f"{float(p[1]):g}%" if p else None
    if attribute in ("start_time", "end_time"):
        # clock time only: "04:45 UTC full recovery" and "2026-06-02 04:45 UTC" are the same moment;
        # in "between 18:17 and 20:27" the start is the first time and the end the last
        times = TIME_RE.findall(v)
        return (times[0] if attribute == "start_time" else times[-1]).zfill(5) if times else None
    if attribute == "root_cause":
        ids = sorted(set(re.findall(r"\b[A-Z]{2,5}(?:-\d{2,5})+\b", v)))
        return ", ".join(ids) if ids else None      # "likely related to recent changes" names nothing
    if attribute == "affected_services":
        # "Services showing errors at the time: ledger, fraud-scoring (at 09:20)" -> the names only
        v = re.sub(r"\([^)]*\)", "", v).rsplit(":", 1)[-1]
        return ", ".join(sorted(x.strip(" .").lower() for x in re.split(r",|\band\b", v) if x.strip(" .")))
    return re.sub(r"\s+", " ", v.lower()).strip(" .")


def _expand(c: "Claim"):
    """A window claim ("12:36 UTC to 15:46 UTC") is a start claim and an end claim."""
    if norm_attribute(c.attribute) != "window":
        return [c]
    times = TIME_RE.findall(c.value)
    if len(times) < 2:
        return []
    return [Claim(c.entity, "start_time", times[0], c.chunk_id, c.doc_id),
            Claim(c.entity, "end_time", times[-1], c.chunk_id, c.doc_id)]


def _lists_disagree(values) -> bool:
    """Service lists conflict only when they overlap and neither contains the other. A subset is
    the partial list an early report saw; disjoint lists are different vocabularies (a status
    page names products like "Payouts", a postmortem names services)."""
    sets = [set(v["normalised"].split(", ")) for v in values]
    return any(a & b and not (a <= b or b <= a) for i, a in enumerate(sets) for b in sets[i + 1:])


@dataclass
class Claim:
    entity: str
    attribute: str
    value: str
    chunk_id: str
    doc_id: str = ""

    def key(self):
        return norm_entity(self.entity), norm_attribute(self.attribute)


@dataclass
class Contradiction:
    entity: str
    attribute: str
    values: list = field(default_factory=list)    # [{value, normalised, sources:[{doc_id, chunk_id, authority, kind}]}]
    preferred: str | None = None                  # value backed by the most authoritative source, if unique
    reason: str = ""


def find_contradictions(claims):
    groups = {}
    for c in (x for claim in claims for x in _expand(claim)):
        groups.setdefault(c.key(), []).append(c)
    out = []
    for (entity, attribute), cs in groups.items():
        by_value = {}
        for c in cs:
            nv = norm_value(attribute, c.value)
            if nv is None or (attribute == "affected_services" and (c.doc_id or c.chunk_id).startswith(PRODUCT_VOCABULARY)):
                continue
            slot = by_value.setdefault(nv, {"value": c.value, "normalised": nv, "sources": []})
            level, label = authority(c.doc_id or c.chunk_id)
            if all(s["chunk_id"] != c.chunk_id for s in slot["sources"]):
                slot["sources"].append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "authority": level, "kind": label})
        if len(by_value) < 2 or (attribute == "affected_services" and not _lists_disagree(by_value.values())):
            continue
        values = sorted(by_value.values(), key=lambda s: -max(x["authority"] for x in s["sources"]))
        top = [max(x["authority"] for x in s["sources"]) for s in values]
        con = Contradiction(entity, attribute, values)
        if top[0] > top[1]:
            best = max(values[0]["sources"], key=lambda x: x["authority"])
            con.preferred = values[0]["value"]
            con.reason = f"the {best['kind']} outranks the other sources"
        else:
            con.reason = "sources of equal authority disagree; cannot choose"
        out.append(con)
    return out
