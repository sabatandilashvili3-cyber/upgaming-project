"""The investigation loop end to end, with a scripted model (no API key, no cost)."""
from engine.agent.investigator import Investigator
from engine.ingest.records import Chunk
from engine.llm.base import LLMError, ToolCall, Turn, Usage
from engine.llm.fake import FakeLLM
from engine.search.retriever import Retriever
from tests.test_store import FakeEmbedder

PM, ST = "postmortems/INC-1_postmortem.md", "status/2026-03-14_INC-1.html"
CHUNKS = [
    Chunk(PM + "::0", PM, "[Postmortem INC-1 > Impact] Duration: 220 minutes. Payments delayed: 34%.",
          {"ids": ["INC-1"], "dates": ["2026-03-14"]}),
    Chunk(PM + "::1", PM, "[Postmortem INC-1 > Root cause] CHG-7 (ledger): reduce pool 240 -> 60.",
          {"ids": ["INC-1", "CHG-7"], "dates": ["2026-03-13"]}),
    Chunk(ST + "::0", ST, "[Status INC-1] Started 2026-03-14 09:12 UTC. Resolved 11:12 UTC (120 minutes).",
          {"ids": ["INC-1"], "dates": ["2026-03-14"]}),
]


def retriever():
    emb = FakeEmbedder()
    return Retriever(CHUNKS, emb.passages([c.text for c in CHUNKS]), emb)


def call(name, n=1, **inp):
    return ToolCall(f"c{n}", name, inp)


def tools(*calls):
    return Turn("", list(calls), "tool_use", Usage(100, 20))


def report(answerable=True, findings=(), summary="", unknowns=()):
    return tools(call("submit_report", 9, answerable=answerable, summary=summary, short_answer="",
                      findings=list(findings), timeline=[], unknowns=list(unknowns)))


# read_document lists a document's chunks in order, so aliases are predictable: E1, E2 (postmortem), E3 (status)
READ_BOTH = tools(call("read_document", 1, doc_id=PM), call("read_document", 2, doc_id=ST))
CLAIMS = tools(call("record_claims", 3, claims=[
    {"entity": "INC-1", "attribute": "duration", "value": "220 minutes", "evidence_id": "E1"},
    {"entity": "INC-1", "attribute": "outage length", "value": "120 minutes", "evidence_id": "E3"}]))


def run(script, prefetch=False, **kw):
    llm = FakeLLM(script)
    return Investigator(llm, retriever(), prefetch=prefetch, **kw).run("How long did INC-1 last and what caused it?"), llm


def test_answered_with_verified_findings_and_contradiction():
    inv, _ = run([READ_BOTH, CLAIMS, report(findings=[
        {"text": "INC-1 lasted 220 minutes according to the postmortem.", "citations": ["E1"]},
        {"text": "It was caused by CHG-7 on the ledger.", "citations": ["E2"]}],
        summary="INC-1 lasted 220 minutes; the status page said 120 minutes.")])
    r = inv.report
    assert inv.status == "answered" and len(r["findings"]) == 2 and not r["unsupported"]
    assert r["findings"][1]["citations"][0]["chunk_id"] == PM + "::1"
    (c,) = r["contradictions"]
    assert c["attribute"] == "duration" and c["preferred"] == "220 minutes"
    assert r["sources"] == [PM] and r["summary"].startswith("INC-1 lasted")
    # the summary sentence carries the minimal passages for its facts: 220 (postmortem E1), 120 (status E3)
    assert {c["evidence_id"] for c in r["summary_sentences"][0]["citations"]} == {"E1", "E3"}
    assert inv.usage.input_tokens == 300                        # three model steps counted


def test_invented_number_and_invented_source_are_dropped():
    inv, _ = run([READ_BOTH, report(findings=[
        {"text": "INC-1 lasted 300 minutes.", "citations": ["E1"]},          # number not in E1
        {"text": "CHG-7 caused INC-1.", "citations": ["E99"]},               # never shown
        {"text": "Payments delayed: 34%.", "citations": ["E1"]}])])
    r = inv.report
    assert [f["text"] for f in r["findings"]] == ["Payments delayed: 34%."]
    reasons = {u["text"]: u["reason"] for u in r["unsupported"]}
    assert reasons["INC-1 lasted 300 minutes."] == "facts not found in the cited sources"
    assert reasons["CHG-7 caused INC-1."] == "cites a source that was never retrieved"


def test_summary_with_unverified_fact_is_removed():
    inv, _ = run([READ_BOTH, report(findings=[{"text": "Payments delayed: 34%.", "citations": ["E1"]}],
                                    summary="Delays hit 50% of payments.")])
    assert inv.report["summary"] == "" and inv.report["unsupported"][0]["missing"] == ["50%"]


def test_model_says_not_answerable():
    inv, _ = run([READ_BOTH, report(answerable=False, unknowns=["Revenue loss is never recorded."])])
    assert inv.status == "abstained" and inv.report["unknowns"] == ["Revenue loss is never recorded."]


def test_answerable_but_nothing_verifies_is_abstention():
    inv, _ = run([READ_BOTH, report(findings=[{"text": "It lasted 999 minutes.", "citations": ["E1"]}])])
    assert inv.status == "abstained" and inv.report["unknowns"]


def test_text_answer_gets_one_reminder_then_stops():
    inv, llm = run([Turn("It lasted a while.", [], "end_turn"), Turn("Still no tool.", [], "end_turn")])
    assert inv.status == "abstained" and inv.report["summary"] == "No report was produced."
    assert sum(1 for e in llm.log if e[0] == "user") == 1


def test_api_failure_becomes_an_error_report_not_a_crash():
    inv, _ = run([READ_BOTH, LLMError("rate_limit", "Rate limited after retries", retryable=True)])
    assert inv.status == "error" and inv.error.startswith("rate_limit")


def test_tool_errors_are_returned_to_the_model():
    inv, llm = run([tools(call("read_document", 1, doc_id="nope.md"), call("frobnicate", 2)),
                    report(answerable=False, unknowns=["x"])])
    results = next(e[1] for e in llm.log if e[0] == "results")
    assert all(r.is_error for r in results)
    assert "No document with id nope.md" in results[0].content and "Unknown tool" in results[1].content


def test_step_cap_stops_a_looping_model():
    loop = [tools(call("search", n, query="INC-1")) for n in range(20)]
    inv, llm = run(loop, max_steps=4)
    assert len(inv.steps) == 4 and inv.status == "abstained"
    assert any(e[0] == "user" and "budget" in e[1] for e in llm.log)


def test_search_wraps_evidence_and_marks_authority():
    inv, llm = run([tools(call("search", 1, query="INC-1 duration")), report(answerable=False, unknowns=["x"])])
    content = next(e[1] for e in llm.log if e[0] == "results")[0].content
    assert content.count("<evidence id=") >= 2 and 'kind="postmortem" authority="4"' in content


def test_missing_citation_is_repaired_and_labelled():
    # the 120 minutes is in E3 (status page), which the model read but did not cite
    inv, _ = run([READ_BOTH, report(findings=[
        {"text": "The postmortem says 220 minutes but the status page says 120 minutes.", "citations": ["E1"]}])])
    (f,) = inv.report["findings"]
    assert [c["evidence_id"] for c in f["citations"]] == ["E1", "E3"]
    assert f["citations"][1]["added_by_verifier"] is True


def test_bad_summary_sentence_removed_alone():
    inv, _ = run([READ_BOTH, report(findings=[{"text": "Payments delayed: 34%.", "citations": ["E1"]}],
                                    summary="INC-1 delayed 34% of payments. Revenue fell 40,000 GEL.")])
    assert inv.report["summary"] == "INC-1 delayed 34% of payments."
    assert inv.report["unsupported"][0]["missing"] == ["40,000"]


def test_report_can_be_reverified_offline_from_saved_evidence():
    from engine.agent.investigator import verify_report
    inv, _ = run([READ_BOTH, CLAIMS, report(findings=[{"text": "INC-1 lasted 220 minutes.", "citations": ["E1"]}])])
    again = verify_report(inv.question, inv.draft, inv.evidence, inv.claims)
    assert again["findings"] == inv.report["findings"] and again["contradictions"] == inv.report["contradictions"]



def test_prefetch_hands_the_first_search_to_the_model():
    inv, llm = run([report(answerable=False, unknowns=["x"])], prefetch=True)
    opening = next(e for e in llm.log if e[0] == "start")[2]
    assert "A first search for the question already ran" in opening and "<evidence id=\"E1\"" in opening
    assert inv.evidence                                   # the prefetched passages are citable evidence


# ---------------------------------------------------------------- conversations (follow-up questions)
def test_follow_up_sees_earlier_turns_and_prefetch_follows_their_ids():
    from engine.agent.investigator import prefetch_query
    history = [("How long did INC-2026-016 last?", "INC-1 lasted 220 minutes according to the postmortem.")]
    assert prefetch_query("how much revenue did it generate", history) == "how much revenue did it generate INC-2026-016"
    assert prefetch_query("What about INC-2026-009?", history) == "What about INC-2026-009?"   # own id: untouched
    assert prefetch_query("anything", []) == "anything"
    llm = FakeLLM([report(answerable=False, unknowns=["x"])])
    Investigator(llm, retriever(), prefetch=True).run("how much revenue did it generate", history=history)
    opening = next(e for e in llm.log if e[0] == "start")[2]
    assert "Earlier in this conversation" in opening and "INC-1 lasted 220 minutes" in opening
    assert "it is NOT evidence" in opening and "New question: how much revenue did it generate" in opening


def test_cache_key_depends_on_the_conversation():
    from engine.agent.investigator import cache_key
    a = cache_key("how long did it last?", "m", "medium", "v1", [("about INC-1", "x")])
    b = cache_key("how long did it last?", "m", "medium", "v1", [("about INC-2", "y")])
    assert a != b and a == cache_key("How long did it   last?", "m", "medium", "v1", [("about INC-1", "x")])


def test_list_incidents_is_complete_and_flags_status_only_matches():
    from engine.agent.tools import Toolbox
    docs = [("incidents/INC-2_report.pdf", "[Incident report INC-2] Severity: SEV1. Detected: 2026-03-14 17:07 UTC. "
             "Services showing errors at the time of this report: ledger, payments-api. Initial assessment"),
            ("status/2026-03-14_INC-2.html", "[Status: Balances, Payouts degraded] Started 2026-03-14 17:07 UTC."),
            ("incidents/INC-3_report.pdf", "[Incident report INC-3] Severity: SEV2. Detected: 2026-04-01 10:00 UTC. "
             "Services showing errors at the time of this report: payout-service."),
            ("incidents/INC-4_report.pdf", "[Incident report INC-4] Severity: SEV1. Detected: 2026-05-01 10:00 UTC. "
             "Services showing errors at the time of this report: bank-connector."),
            ("postmortems/INC-4_postmortem.md", "[Postmortem INC-4 > Impact] - Affected services: bank-connector, payout-service")]
    chunks = CHUNKS + [Chunk(d + "::0", d, t, {}) for d, t in docs]
    emb = FakeEmbedder()
    box = Toolbox(Retriever(chunks, emb.passages([c.text for c in chunks]), emb))
    out, err = box.run("list_incidents", {"severity": "SEV1", "service": "payout-service", "start": "", "end": ""})
    assert not err and out.startswith("2 of 4 incidents match")      # INC-1 has no severity: never matches SEV1
    assert "INC-4 | 2026-05-01 | SEV1" in out and "POSSIBLE" in out.split("INC-4")[0]   # INC-2 via its status page
    assert "INC-3 |" not in out                                        # SEV2
    assert "Affected services: bank-connector, payout-service" in out  # the evidence passage itself is shown


def test_list_incidents_reads_free_form_fields():
    """Fields written the way people write them, not this corpus's template."""
    from engine.agent.tools import Toolbox
    docs = [("postmortems/INC-9_postmortem.md", "Postmortem for INC-9. This was a Sev-1 incident that began at "
             "2026-07-02 08:00 UTC. Impacted systems: Ledger and payout-service. The root cause was CHG-77."),
            ("incidents/INC-8_report.pdf", "INC-8 report. severity 2. Detected 2026-07-01 09:00 UTC. "
             "Services affected: payout-service")]
    chunks = CHUNKS + [Chunk(d + "::0", d, t, {}) for d, t in docs]
    emb = FakeEmbedder()
    box = Toolbox(Retriever(chunks, emb.passages([c.text for c in chunks]), emb))
    out, _ = box.run("list_incidents", {"severity": "sev 1", "service": "payout-service", "start": "2026-07-01", "end": ""})
    assert out.startswith("1 of") and "INC-9 | 2026-07-02 | SEV1 | services named: ledger, payout-service" in out
    assert "The root cause was CHG-77" in out
