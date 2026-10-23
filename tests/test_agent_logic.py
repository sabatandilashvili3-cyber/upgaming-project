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


def test_verifier_accepts_grounded_sentence():
    ev = {"c1": "Root cause: CHG-3340 reduced max_connections 240 -> 60 at 2026-03-13 18:40 UTC"}
    assert check("CHG-3340 changed the pool at 18:40.", ["c1"], ev).supported


def test_verifier_rejects_invented_number_and_unknown_source():
    ev = {"c1": "Duration: 220 minutes"}
    v = check("The outage lasted 300 minutes.", ["c1"], ev)
    assert not v.supported and v.missing == ["300 minutes"]
    assert check("Lasted 220 minutes.", ["c9"], ev).reason == "cites a source that was never retrieved"
    assert check("Lasted 220 minutes.", [], ev).reason == "no citation"
