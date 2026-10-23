"""Contradiction detection, done by code rather than by asking the model "do these disagree?".

The model's only job is extraction: while reading evidence it records claims as
structured records through a tool call:

    {"entity": "INC-2026-016", "attribute": "duration", "value": "220 minutes",
     "chunk_id": "postmortems/INC-2026-016_postmortem.docx::2"}

Then plain Python decides:
  1. normalise entity and attribute names ("Duration", "outage length" -> "duration");
  2. normalise values by type, so "220 minutes", "3h 40m" and "3 hours 40 minutes"
     compare equal, "34%" equals "34 percent", times compare as timestamps;
  3. group by (entity, attribute); a group with more than one distinct normalised
     value is a contradiction, reported with every value and its sources;
  4. rank the conflicting values by source authority (a postmortem outranks a
     customer ticket), so the report can say which one to believe and why.

This is easy to test and to explain, and it cannot "hallucinate a contradiction":
two different values from two cited chunks either exist in the ledger or they don't.
"""
import re
from dataclasses import dataclass, field

ATTRIBUTE_ALIASES = {
    "duration": {"duration", "outage duration", "outage length", "length", "how long", "downtime"},
    "start_time": {"start", "start time", "started", "began", "start_time", "detected"},
    "end_time": {"end", "end time", "resolved", "recovered", "end_time"},
    "root_cause": {"root cause", "cause", "caused by", "root_cause", "suspected cause"},
    "impact": {"impact", "payments delayed", "share delayed", "customer impact", "requests affected"},
    "affected_services": {"affected services", "impacted services", "services affected", "affected_services"},
}

# how much to trust a document by where it lives; higher wins a contradiction.
# Domain configuration, not logic: an unknown source gets the middle value.
AUTHORITY = [("postmortems/", 4, "postmortem"), ("changelog/", 4, "change log"),
             ("architecture/", 3, "architecture doc"), ("runbooks/", 3, "runbook"),
             ("incidents/", 2, "incident report written during the incident"),
             ("status/", 2, "public status page"), ("alerts/", 2, "alert"),
             ("comms/", 1, "customer email"), ("tickets/", 0, "customer ticket")]


def authority(doc_id: str):
    for prefix, level, label in AUTHORITY:
        if doc_id.startswith(prefix):
            return level, label
    return 2, "document"


def norm_attribute(a: str) -> str:
    a = a.strip().lower().replace("_", " ")
    for canon, names in ATTRIBUTE_ALIASES.items():
        if a in {n.replace("_", " ") for n in names}:
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


def norm_value(attribute: str, value: str) -> str:
    v = value.strip()
    if attribute == "duration" and (mins := _minutes(v)) is not None:
        return f"{mins} min"
    if attribute == "impact" and (p := re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", v.lower())):
        return f"{float(p[1]):g}%"
    if attribute in ("start_time", "end_time") and (t := re.search(r"(20\d\d-\d\d-\d\d)[ t]*(\d\d:\d\d)", v.lower())):
        return f"{t[1]} {t[2]}"
    if attribute in ("root_cause", "affected_services"):
        ids = sorted(set(re.findall(r"\b[A-Z]{2,5}(?:-\d{2,5})+\b", v)))
        if ids:
            return ", ".join(ids)
        return ", ".join(sorted(x.strip().lower() for x in re.split(r",|\band\b", v) if x.strip()))
    return re.sub(r"\s+", " ", v.lower()).strip(" .")


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
    for c in claims:
        groups.setdefault(c.key(), []).append(c)
    out = []
    for (entity, attribute), cs in groups.items():
        by_value = {}
        for c in cs:
            nv = norm_value(attribute, c.value)
            slot = by_value.setdefault(nv, {"value": c.value, "normalised": nv, "sources": []})
            level, label = authority(c.doc_id or c.chunk_id)
            if all(s["chunk_id"] != c.chunk_id for s in slot["sources"]):
                slot["sources"].append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "authority": level, "kind": label})
        if len(by_value) < 2:
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
