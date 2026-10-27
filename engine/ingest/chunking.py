"""Two chunking strategies, compared in the evaluation.

fixed       Slide a window of `size` words with `overlap` words of overlap over the
            text, ignoring structure. Simple and predictable, but a window can start
            mid-section and lose the heading that says what it is about.

structural  Split on headings and blank lines, then pack whole paragraphs into chunks
            of at most `size` words without crossing a heading. Every chunk starts with
            a context line: the document title and the heading it sits under, so
            "Duration: 220 minutes" is stored as
            "[Postmortem INC-2026-016 > Impact] Duration: 220 minutes".
            A paragraph longer than `size` falls back to fixed windows inside it.

Short records (a CSV row, a HotpotQA paragraph) fit in one chunk either way.
Sizes are counted in words, a cheap and model-independent stand-in for tokens.
"""
import re

from .records import Chunk, Record


def _windows(words, size, overlap):
    step = max(1, size - overlap)
    for start in range(0, max(1, len(words) - overlap), step):
        yield words[start:start + size]


def chunk_fixed(rec: Record, size: int = 200, overlap: int = 40):
    words = rec.text.replace("## ", "").split()
    return [Chunk(f"{rec.doc_id}::{n}", rec.doc_id, " ".join(w), {**rec.meta, "title": rec.title})
            for n, w in enumerate(_windows(words, size, overlap))]


def _sections(text):
    """Yield (heading, [paragraphs]) in document order."""
    heading, paras = "", []
    for block in re.split(r"\n\s*\n|\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            first, _, rest = block.partition("\n")
            if paras:
                yield heading, paras
            heading, paras = first[3:].strip(), []
            if rest.strip():
                paras.append(rest.strip())
        else:
            paras.append(block)
    if paras:
        yield heading, paras


def chunk_structural(rec: Record, size: int = 200, overlap: int = 40):
    chunks = []

    def emit(heading, body):
        label = f"{rec.title} > {heading}" if heading and heading != rec.title else rec.title
        chunks.append(Chunk(f"{rec.doc_id}::{len(chunks)}", rec.doc_id, f"[{label}] {body}",
                            {**rec.meta, "title": rec.title, "heading": heading}))

    for heading, paras in _sections(rec.text):
        buf = []
        for p in paras:
            words = p.split()
            if len(words) > size:                       # one huge paragraph: window inside it
                if buf:
                    emit(heading, " ".join(buf))
                    buf = []
                for w in _windows(words, size, overlap):
                    emit(heading, " ".join(w))
                continue
            if len(buf) + len(words) > size and buf:
                emit(heading, " ".join(buf))
                buf = []
            buf.extend(words)
        if buf:
            emit(heading, " ".join(buf))
    if not chunks:                                       # empty or heading-only document
        emit("", rec.text.replace("## ", ""))
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "structural": chunk_structural}
