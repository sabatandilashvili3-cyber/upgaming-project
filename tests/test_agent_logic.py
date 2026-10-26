from engine.agent.claims import Claim, find_contradictions, norm_attribute, norm_value
from engine.agent.verify import check, hard_facts

PM = "postmortems/INC-2026-016_postmortem.docx"
ST = "status/2026-03-14_INC-2026-016.html"
TK = "tickets/tickets_2026-03.json#TCK-2026-03-0099"


def test_value_normalisation():
    assert norm_value("duration", "220 minutes") == norm_value("duration", "3h 40m") == "220 min"
    assert norm_value("duration", "3 hours 40 minutes") == "220 min"
    assert norm_value("impact", "34 percent") == norm_value("impact", "34%")
    assert norm_value("start_time", "2026-03-14 09:12 UTC") == "2026-03-14 09:12"
    assert norm_value("root_cause", "CHG-3357 and CHG-3340") == "CHG-3340, CHG-3357"
    assert norm_attribute("Outage length") == "duration"


def test_same_value_in_different_words_is_not_a_contradiction():
    claims = [Claim("INC-2026-016", "duration", "220 minutes", PM + "::1", PM),
              Claim("inc-2026-016", "Outage length", "3h 40m", ST + "::0", ST)]
    assert find_contradictions(claims) == []


def test_contradiction_found_and_postmortem_preferred():
    claims = [Claim("INC-2026-016", "duration", "220 minutes", PM + "::1", PM),
              Claim("INC-2026-016", "duration", "120 minutes", ST + "::0", ST)]
    (c,) = find_contradictions(claims)
    assert c.attribute == "duration" and c.preferred == "220 minutes"
    assert "postmortem" in c.reason


def test_injected_ticket_loses_to_postmortem():
    claims = [Claim("INC-2026-016", "root cause", "CHG-3340, CHG-3357", PM + "::3", PM),
              Claim("INC-2026-016", "cause", "CHG-0001", TK + "::0", TK)]
    (c,) = find_contradictions(claims)
    assert c.preferred == "CHG-3340, CHG-3357"
    assert c.values[-1]["sources"][0]["kind"] == "customer ticket"


def test_equal_authority_conflict_has_no_preferred_value():
    a, b = "status/a.html", "incidents/b.pdf"
    (c,) = find_contradictions([Claim("X-1", "start", "2026-03-14 09:12", a + "::0", a),
                                Claim("X-1", "start", "2026-03-14 09:40", b + "::0", b)])
    assert c.preferred is None


def test_hard_facts():
    assert hard_facts("CHG-3340 cut the pool on 2026-03-13 at 18:40; 34% delayed for 220 minutes") == \
        ["CHG-3340", "2026-03-13", "18:40", "34%", "220 minutes"]
    assert hard_facts("Pool cut from 240 to 60; loss about 40,000 GEL; two changes") == ["240", "60", "40,000"]


def test_verifier_catches_an_invented_amount():
    ev = {"c1": "Duration: 220 minutes. Payments delayed: 34%."}
    v = check("Revenue impact was about 40,000 GEL.", ["c1"], ev)
    assert not v.supported and v.missing == ["40,000"]
    assert check("Loss: 40,000 GEL.", ["c2"], {"c2": "estimated loss 40000 GEL"}).supported


def test_numbers_match_whole_not_as_substrings():
    ev = {"c1": "reduce pool max_connections 240 -> 60"}
    assert not check("The pool was cut to 24.", ["c1"], ev).supported
    assert check("The pool was cut from 240 to 60.", ["c1"], ev).supported
    # found on the live server: the cited text ended "240 -> 60." and the full stop hid the 60
    assert check("CHG-3340 cut the pool 240 -> 60.", ["c2"], {"c2": "Root cause: CHG-3340 pool 240 -> 60."}).supported
    assert not check("It was 60.", ["c3"], {"c3": "measured 60.5 ms"}).supported


def test_verifier_accepts_grounded_sentence():
    ev = {"c1": "Root cause: CHG-3340 reduced max_connections 240 -> 60 at 2026-03-13 18:40 UTC"}
    assert check("CHG-3340 changed the pool at 18:40.", ["c1"], ev).supported


def test_verifier_rejects_invented_number_and_unknown_source():
    ev = {"c1": "Duration: 220 minutes"}
    v = check("The outage lasted 300 minutes.", ["c1"], ev)
    assert not v.supported and v.missing == ["300 minutes"]
    assert check("Lasted 220 minutes.", ["c9"], ev).reason == "cites a source that was never retrieved"
    assert check("Lasted 220 minutes.", [], ev).reason == "no citation"


# ---------------------------------------------------------------- verifier v2 (found in the first eval run)
def test_version_is_one_fact_not_a_torn_number():
    assert hard_facts("deployed v1.9.19 on ledger") == ["v1.9.19"]
    assert check("Version v1.9.19 was deployed.", ["c1"], {"c1": "version: v1.9.19"}).supported


def test_facts_from_the_question_are_given():
    ev = {"c1": "CHG-2113 config change on api-gateway"}
    assert not check("CHG-2113 preceded INC-2026-002.", ["c1"], ev).supported
    assert check("CHG-2113 preceded INC-2026-002.", ["c1"], ev, given="What caused INC-2026-002?").supported


def test_find_support_repairs_only_from_passages_that_contain_every_fact():
    from engine.agent.verify import find_support
    ev = {"E1": "status: resolved 19:07 UTC", "E2": "postmortem: recovered 20:27 UTC", "E3": "20:27 alert"}
    assert find_support(["20:27"], ev, exclude={"E1"}) in ("E2", "E3")
    assert find_support(["20:27", "300 minutes"], ev) is None


def test_split_sentences_keeps_versions_and_decimals():
    from engine.agent.verify import split_sentences
    assert split_sentences("It ran v1.9.19 for 2.4 s. Then it failed. (Twice.)") == \
        ["It ran v1.9.19 for 2.4 s.", "Then it failed.", "(Twice.)"]
