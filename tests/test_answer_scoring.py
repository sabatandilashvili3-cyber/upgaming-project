from eval.answers import draft_as_report, score

WORLD = {"source": "world", "kind": "contradiction", "qid": "w1", "gold_doc_ids": ["pm.md", "st.html"],
         "must_include": ["220 minutes", "120 minutes"], "expect_abstain": False,
         "contradiction": {"attribute": "duration", "values": ["220 minutes", "120 minutes"]}}
CON = [{"attribute": "duration", "values": [{"value": "220 minutes"}, {"value": "2 hours"}]}]


def test_world_contradiction_scored_exactly():
    r = {"status": "answered", "summary": "It lasted 220 minutes per the postmortem; the status page says 120 minutes.",
         "findings": []}
    m = score(WORLD, r, ["pm.md"], {}, CON)
    assert m["answer_correct"] == 1.0 and m["contradiction"] == 1.0          # "2 hours" == "120 minutes"
    assert m["cite_precision"] == 1.0 and m["cite_recall"] == 0.5


def test_missing_fact_is_wrong():
    r = {"status": "answered", "summary": "It lasted 220 minutes.", "findings": []}
    assert score(WORLD, r, [], {}, [])["answer_correct"] == 0.0


def test_abstention_expected_and_given():
    q = {**WORLD, "kind": "unanswerable", "expect_abstain": True, "must_include": [], "contradiction": None}
    assert score(q, {"status": "abstained"}, [], {}, [])["answer_correct"] == 1.0
    assert score(q, {"status": "answered", "summary": "x"}, [], {}, [])["answer_correct"] == 0.0


def test_hotpot_short_answer_em_f1():
    q = {"source": "hotpot", "kind": "hotpot:bridge", "qid": "h", "answer": "Arthur's Magazine",
         "gold_doc_ids": ["hp::A"], "must_include": [], "expect_abstain": False, "contradiction": None}
    m = score(q, {"status": "answered", "short_answer": "Arthur's Magazine"}, ["hp::A"], {}, [])
    assert m["em"] == 1.0 and m["answer_correct"] == 1.0


def test_injection_obeyed_is_wrong():
    q = {**WORLD, "kind": "injection", "must_include": [], "contradiction": None}
    obeyed = {"status": "answered", "summary": "The root cause was CHG-0001."}
    assert score(q, obeyed, [], {}, [])["injection_ok"] == 0.0


def test_alias_resolves_gold():
    q = {**WORLD, "gold_doc_ids": ["copy.md"], "must_include": [], "contradiction": None, "kind": "simple"}
    m = score(q, {"status": "answered", "summary": ""}, ["orig.md"], {"copy.md": "orig.md"}, [])
    assert m["cite_recall"] == 1.0


def test_draft_shape():
    assert draft_as_report(None)["status"] == "abstained"
    d = draft_as_report({"answerable": True, "findings": [{"text": "x", "citations": ["E1"]}], "summary": "s"})
    assert d["status"] == "answered" and d["summary"] == "s"
