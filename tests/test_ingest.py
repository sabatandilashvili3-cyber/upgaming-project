import io
import json

import pytest

from engine.ingest.chunking import chunk_fixed, chunk_structural
from engine.ingest.dedup import find_duplicates, jaccard, shingles
from engine.ingest.loaders import UnsupportedFormat, load_bytes, text_meta
from engine.ingest.records import Record


def rec(doc_id, text, title="T"):
    return Record(doc_id, doc_id, "md", title, text, {})


# ---------------------------------------------------------------- loaders
def test_csv_rows_become_records_with_row_ids():
    data = b"change_id,service,summary\nCHG-1,ledger,pool 240 -> 60\nCHG-2,auth,keys\n"
    recs = load_bytes(data, "changelog/changes_2026-03.csv")
    assert [r.doc_id for r in recs] == ["changelog/changes_2026-03.csv#CHG-1", "changelog/changes_2026-03.csv#CHG-2"]
    assert "service: ledger" in recs[0].text and recs[0].meta["row"]["summary"] == "pool 240 -> 60"


def test_json_array_uses_id_field():
    data = json.dumps([{"ticket_id": "TCK-9", "text": "stuck"}]).encode()
    assert load_bytes(data, "tickets.json")[0].doc_id == "tickets.json#TCK-9"


def test_html_keeps_headings_and_meta():
    html = b'<html><head><title>Status</title><meta name="incident" content="INC-2026-016"></head>' \
           b'<body><h1>Card payments degraded</h1><p>Started 2026-03-14 09:12 UTC.</p></body></html>'
    r = load_bytes(html, "status.html")[0]
    assert r.title == "Status" and r.text.startswith("## Card payments degraded")
    assert r.meta["html_meta"]["incident"] == "INC-2026-016" and r.meta["dates"] == ["2026-03-14"]


def test_docx_roundtrip_headings_lists_tables():
    import docx
    d = docx.Document()
    d.add_heading("Impact", level=2)
    d.add_paragraph("Duration: 220 minutes", style="List Bullet")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "ledger", "Payments Core"
    buf = io.BytesIO(); d.save(buf)
    r = load_bytes(buf.getvalue(), "pm.docx")[0]
    assert "## Impact" in r.text and "- Duration: 220 minutes" in r.text and "ledger | Payments Core" in r.text


def test_pdf_text_is_extracted():
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    buf = io.BytesIO()
    SimpleDocTemplate(buf).build([Paragraph("Detected 2026-03-14 on ledger", getSampleStyleSheet()["Normal"])])
    r = load_bytes(buf.getvalue(), "report.pdf")[0]
    assert "ledger" in r.text and r.meta["pages"] == 1


def test_unknown_format_is_rejected_with_a_reason():
    with pytest.raises(UnsupportedFormat, match=".exe"):
        load_bytes(b"MZ", "virus.exe")


def test_text_meta_finds_ids_and_dates():
    m = text_meta("CHG-3340 caused INC-2026-016 on 2026-03-14; ticket TCK-2026-03-0012")
    assert m["ids"] == ["CHG-3340", "INC-2026-016", "TCK-2026-03-0012"] and m["dates"] == ["2026-03-14"]


# ---------------------------------------------------------------- dedup
def test_exact_duplicate_points_at_first_copy():
    d = find_duplicates([rec("a", "same text here"), rec("b", "same text here")])
    assert d == {"b": ("a", "exact", 1.0)}


def test_near_duplicate_found_and_unrelated_kept():
    base = ("Card payments degraded performance started 2026-03-14 09:12 UTC resolved 11:12 UTC some customers "
            "saw delays with card payments and payouts our engineers identified the issue and applied a fix "
            "we apologise for the disruption reference INC-2026-016")
    edited = base.replace("we apologise for", "we are sorry for")
    other = "Runbook for ledger: check connection pools first, then replica counts, then recent changes."
    d = find_duplicates([rec("status", base), rec("email", edited), rec("runbook", other)])
    assert d["email"][0] == "status" and d["email"][1] == "near"
    assert "runbook" not in d
    assert jaccard(shingles(base), shingles(edited)) >= 0.8


# ---------------------------------------------------------------- chunking
PM = "## Summary\nLedger was degraded.\n\n## Impact\n- Duration: 220 minutes\n- Payments delayed: 34%"


def test_structural_chunks_carry_title_and_heading():
    chunks = chunk_structural(rec("pm", PM, title="Postmortem INC-2026-016"))
    assert [c.meta["heading"] for c in chunks] == ["Summary", "Impact"]
    assert chunks[1].text.startswith("[Postmortem INC-2026-016 > Impact] - Duration: 220 minutes")


def test_structural_never_crosses_a_heading():
    chunks = chunk_structural(rec("pm", PM), size=1000)
    assert len(chunks) == 2


def test_fixed_windows_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_fixed(rec("x", text), size=200, overlap=40)
    assert len(chunks) == 2
    assert chunks[1].text.split()[0] == "w160"                  # 200 - 40 = 160


def test_structural_splits_oversized_paragraph():
    long = " ".join(f"w{i}" for i in range(450))
    chunks = chunk_structural(rec("x", "## H\n" + long), size=200, overlap=40)
    assert len(chunks) == 3 and all(c.meta["heading"] == "H" for c in chunks)
