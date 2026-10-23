"""Files -> Records -> dedup -> Chunks. Used by the offline index builder and the API."""
from dataclasses import dataclass
from pathlib import Path

from .chunking import STRATEGIES
from .dedup import find_duplicates
from .loaders import load_path, load_jsonl
from .records import Chunk, Record

DATA = Path(__file__).resolve().parents[2] / "data" / "corpus"
WORLD = DATA / "world"
SKIP = {"truth.json", "world_questions.jsonl"}          # ground truth, never indexed


def iter_world():
    for p in sorted(WORLD.rglob("*")):
        if p.is_file() and p.name not in SKIP:
            yield from load_path(p, WORLD)


def iter_hotpot():
    yield from load_jsonl((DATA / "hotpot_paragraphs.jsonl").read_bytes(), "hotpot")


@dataclass
class Built:
    records: list[Record]
    chunks: list[Chunk]
    duplicates: dict          # doc_id -> (kept_doc_id, kind, similarity)

    def aliases(self):
        """Exact duplicates are not indexed; their ids resolve to the kept copy."""
        return {d: kept for d, (kept, kind, _) in self.duplicates.items() if kind == "exact"}


def build(records, strategy: str, size: int = 200, overlap: int = 40) -> Built:
    records = list(records)
    dupes = find_duplicates(records)
    chunker = STRATEGIES[strategy]
    chunks = []
    for rec in records:
        if dupes.get(rec.doc_id, (None, ""))[1] == "exact":
            continue
        for c in chunker(rec, size=size, overlap=overlap):
            c.meta["source"] = rec.source
            c.meta["format"] = rec.format
            if rec.doc_id in dupes:                      # near duplicate: indexed, but linked
                c.meta["near_duplicate_of"] = dupes[rec.doc_id][0]
            chunks.append(c)
    return Built(records, chunks, dupes)
