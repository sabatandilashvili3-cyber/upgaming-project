"""The two shapes data takes inside the engine: a Record (one document or one
independent row) and a Chunk (the unit we search over)."""
from dataclasses import dataclass, field


@dataclass
class Record:
    doc_id: str                 # stable id; rows of a CSV/JSON file get "path#row_id"
    source: str                 # file path (relative to the corpus root) or dataset name
    format: str                 # pdf | docx | html | md | csv | json | jsonl
    title: str
    text: str                   # normalised text; headings kept as "## heading" lines
    meta: dict = field(default_factory=dict)   # dates, ids, row fields, page count ...


@dataclass
class Chunk:
    chunk_id: str               # f"{doc_id}::{n}"
    doc_id: str
    text: str                   # what gets embedded and keyword-indexed
    meta: dict = field(default_factory=dict)   # inherited from the record + heading path
