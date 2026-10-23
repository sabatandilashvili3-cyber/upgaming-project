from eval.metrics import (all_gold_at_k, citation_precision, citation_recall, contains_all, exact_match, f1, mrr,
                          recall_at_k)


def test_recall_and_all_gold():
    ranking = ["x", "g1", "y", "z", "g2"]
    assert recall_at_k(ranking, ["g1", "g2"], 3) == 0.5
    assert all_gold_at_k(ranking, ["g1", "g2"], 3) == 0.0
    assert all_gold_at_k(ranking, ["g1", "g2"], 5) == 1.0


def test_mrr_uses_first_gold_rank():
    assert mrr(["x", "g", "g2"], ["g", "g2"]) == 0.5
    assert mrr(["x", "y"], ["g"]) == 0.0
    assert mrr(["x"], []) is None


def test_answer_metrics():
    assert exact_match("The Beatles.", "beatles") == 1.0
    assert round(f1("Paris France", "Paris"), 3) == 0.667
    assert contains_all("Caused by CHG-3340 and CHG-3357 on ledger", ["CHG-3340", "CHG-3357", "fraud-scoring"]) == 2 / 3


def test_citation_metrics():
    assert citation_precision(["a", "b"], ["a"]) == 0.5
    assert citation_recall(["a"], ["a", "c"]) == 0.5
    assert citation_precision([], ["a"]) is None
