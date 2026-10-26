"""The investigator's tools: what the model can do, and the evidence it has seen.

Every chunk shown to the model gets a short alias (E1, E2, ...) for this investigation.
The model cites aliases; code maps them back to real chunk ids. Short aliases are
hard to mistype or invent, and the verifier rejects any alias that was never shown.

Retrieved text is wrapped in <evidence> tags with its source kind and authority, and
the system prompt says that text inside the tags is data, never instructions: the
first line of defence against prompt injection planted in documents.
"""
import json
from dataclasses import dataclass, field

from ..search.chain import chained_search
from ..search.retriever import SearchConfig
from .claims import Claim, authority, find_contradictions
from ..llm.base import ToolSpec

SEARCH_CFG = SearchConfig(mode="hybrid", boost=True, per_file=2, k=8)
MAX_DOC_CHUNKS = 12


class ToolError(Exception):
    """The model asked for something impossible (unknown document ...). Sent back with is_error."""

MAX_TIMELINE = 25


def _obj(props: dict) -> dict:
    """Strict schema: every property required, nothing else allowed."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


TOOLS = [
    ToolSpec("search", "Hybrid keyword + semantic search over all documents. Returns up to 8 evidence "
             "passages, each with an id like E12. Ids it finds in the top results (changes, incidents) "
             "are followed automatically with a second search.",
             _obj({"query": {"type": "string", "description": "What to look for. Include ids, service "
                                                              "names and dates when you know them."}})),
    ToolSpec("read_document", "Read every passage of one document (for example a postmortem) by its doc id, "
             "as shown in an evidence tag's doc attribute.",
             _obj({"doc_id": {"type": "string"}})),
    ToolSpec("timeline", "List dated records that mention a term inside a date window, oldest first. Use it "
             "to see what changed or happened around an incident.",
             _obj({"term": {"type": "string", "description": "A service name, id or keyword."},
                   "start": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                   "end": {"type": "string", "description": "YYYY-MM-DD, inclusive."}})),
    ToolSpec("record_claims", "Record the key facts you read, one claim per source passage, so that "
             "disagreements between sources can be detected. Record the same fact from every source that "
             "states it, especially times, durations, causes, impact and affected services. Returns any "
             "contradictions found so far.",
             _obj({"claims": {"type": "array", "items": _obj({
                 "entity": {"type": "string", "description": "What the fact is about, e.g. INC-2026-016."},
                 "attribute": {"type": "string", "description": "e.g. duration, start_time, root_cause, impact."},
                 "value": {"type": "string", "description": "The value exactly as the source states it."},
                 "evidence_id": {"type": "string", "description": "The E-id of the passage stating it."}})}})),
    ToolSpec("submit_report", "Finish the investigation. Call exactly once, at the end.",
             _obj({
                 "answerable": {"type": "boolean", "description": "false if the evidence does not answer the "
                                                                  "question; then explain what is missing in unknowns."},
                 "summary": {"type": "string", "description": "Two or three sentences answering the question."},
                 "short_answer": {"type": "string", "description": "The answer in as few words as possible "
                                                                 "(a name, a number, yes/no); empty if not answerable."},
                 "findings": {"type": "array", "items": _obj({
                     "text": {"type": "string", "description": "One factual sentence."},
                     "citations": {"type": "array", "items": {"type": "string"},
                                   "description": "E-ids of the passages that state this."}})},
                 "timeline": {"type": "array", "items": _obj({
                     "time": {"type": "string"}, "event": {"type": "string"},
                     "citations": {"type": "array", "items": {"type": "string"}}})},
                 "unknowns": {"type": "array", "items": {"type": "string"},
                              "description": "What the documents do not establish."}})),
]


@dataclass
class EvidenceBook:
    """Everything shown to the model in one investigation, by alias."""
    by_alias: dict = field(default_factory=dict)       # E3 -> chunk dict
    alias_of: dict = field(default_factory=dict)       # chunk_id -> E3

    def add(self, chunk_id, doc_id, text, meta) -> str:
        if chunk_id not in self.alias_of:
            alias = f"E{len(self.by_alias) + 1}"
            self.alias_of[chunk_id] = alias
            self.by_alias[alias] = {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "meta": meta}
        return self.alias_of[chunk_id]

    def render(self, chunk_id, doc_id, text, meta) -> str:
        alias = self.add(chunk_id, doc_id, text, meta)
        level, kind = authority(doc_id)
        dates = ",".join(meta.get("dates", [])[:3])
        return (f'<evidence id="{alias}" doc="{doc_id}" kind="{kind}" authority="{level}"'
                f'{f" dates={chr(34)}{dates}{chr(34)}" if dates else ""}>\n{text}\n</evidence>')

    def texts(self) -> dict:
        return {a: e["text"] for a, e in self.by_alias.items()}


class Toolbox:
    def __init__(self, retriever, store=None):
        self.retriever, self.store = retriever, store
        self.book = EvidenceBook()
        self.claims: list[Claim] = []

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute one tool call. Returns (content, is_error). Never raises."""
        try:
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return f"Unknown tool {name}.", True
            return fn(**args), False
        except ToolError as e:
            return str(e), True
        except TypeError as e:
            return f"Bad arguments for {name}: {e}", True
        except Exception as e:                      # a tool bug must not kill the investigation
            return f"{name} failed: {type(e).__name__}: {e}", True

    # ------------------------------------------------------------------ tools
    def _t_search(self, query: str) -> str:
        hits, followed = chained_search(self.retriever, query, SEARCH_CFG)
        if not hits:
            return "No results."
        head = f"Followed ids: {', '.join(followed)}\n" if followed else ""
        return head + "\n".join(self.book.render(h.chunk_id, h.doc_id, h.text, h.meta) for h in hits)

    def _t_read_document(self, doc_id: str) -> str:
        chunks = sorted((c for c in self.retriever.chunks.values() if c.doc_id == doc_id),
                        key=lambda c: int(c.chunk_id.rsplit("::", 1)[-1]))
        if not chunks:
            raise ToolError(f"No document with id {doc_id}. Use the doc attribute of an evidence tag.")
        shown = chunks[:MAX_DOC_CHUNKS]
        tail = f"\n({len(chunks) - len(shown)} more passages not shown)" if len(chunks) > len(shown) else ""
        return "\n".join(self.book.render(c.chunk_id, c.doc_id, c.text, c.meta) for c in shown) + tail

    def _t_timeline(self, term: str, start: str, end: str) -> str:
        t = term.lower()
        rows = []
        for c in self.retriever.chunks.values():
            inside = [d for d in c.meta.get("dates", []) if start <= d <= end]
            if inside and t in c.text.lower():
                rows.append((inside[0], c))
        rows.sort(key=lambda r: (r[0], r[1].chunk_id))
        if not rows:
            return f"No dated records mention '{term}' between {start} and {end}."
        more = f"\n({len(rows) - MAX_TIMELINE} more not shown; narrow the window)" if len(rows) > MAX_TIMELINE else ""
        return "\n".join(self.book.render(c.chunk_id, c.doc_id, c.text, c.meta)
                         for _, c in rows[:MAX_TIMELINE]) + more

    def _t_record_claims(self, claims: list) -> str:
        bad = []
        for c in claims:
            ev = self.book.by_alias.get(c["evidence_id"])
            if ev is None:
                bad.append(c["evidence_id"])
                continue
            self.claims.append(Claim(c["entity"], c["attribute"], c["value"], ev["chunk_id"], ev["doc_id"]))
        found = find_contradictions(self.claims)
        out = {"recorded": len(claims) - len(bad), "unknown_evidence_ids": bad,
               "contradictions_so_far": [
                   {"entity": x.entity, "attribute": x.attribute,
                    "values": [{"value": v["value"],
                                "sources": [f'{self.book.alias_of.get(s["chunk_id"], "?")} ({s["kind"]})'
                                            for s in v["sources"]]} for v in x.values],
                    "preferred": x.preferred, "reason": x.reason} for x in found]}
        return json.dumps(out)
