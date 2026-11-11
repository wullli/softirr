import math

import pytest

from soft_irr.evaluation.distances import BipartiteJaccardDistance, exact_match_distance, jaccard_distance
from soft_irr.evaluation.set_prf import PRFCounts, SetPRFScores, match_labels, prf


def test_prf_counts():
    assert prf(2, 4, 2) == pytest.approx((1.0, 0.5, 2 / 3))


def test_prf_zero_denominators_are_nan():
    p, r, f = prf(0, 0, 0)
    assert math.isnan(p) and math.isnan(r) and math.isnan(f)
    p, r, f = prf(0, 3, 0)
    assert math.isnan(p) and r == 0.0 and f == 0.0


def test_counts_add_up():
    total = PRFCounts(tp_hard=1, tp_soft=1.5, n_true=2, n_pred=3) + PRFCounts(tp_hard=2, tp_soft=2, n_true=2, n_pred=2)
    assert total == PRFCounts(tp_hard=3, tp_soft=3.5, n_true=4, n_pred=5)


def test_exact_distance_matches_equal_labels():
    (matching,) = match_labels([(["fever", "cough", "rash"], ["rash", "nausea", "cough", "pain"])])["none"]
    pairs = {(i, j) for i, j, cost in matching if cost == 0.0}
    assert pairs == {(1, 2), (2, 0)}


def test_simple_distance_is_matched_per_label_pair():
    # Token Jaccard on single labels: "skin rash" ~ "rash" has cost 1/2.
    (matching,) = match_labels([(["dry mouth", "rash"], ["mouth dry", "skin rash"])], jaccard_distance)["none"]
    assert sorted(cost for _, _, cost in matching) == pytest.approx([0.0, 0.5])


def test_bipartite_distance_gives_one_result_per_strategy():
    by_strategy = SetPRFScores.get_by_matching([["dry mouth", "rash"]], [["mouth dry", "skin rash"]], BipartiteJaccardDistance())
    assert set(by_strategy) == {"hungarian", "greedy"}
    hungarian = by_strategy["hungarian"].micro
    assert (hungarian.tp, hungarian.precision, hungarian.recall) == (1, 0.5, 0.5)


def test_soft_f1_equals_one_minus_bipartite_distance():
    dist = BipartiteJaccardDistance()
    a, b = ["dry mouth", "rash", "fatigue"], ["mouth dry", "skin rash"]
    soft = SetPRFScores.get([a], [b], dist, soft=True)
    assert soft.micro.f1 == pytest.approx(1.0 - dist.batch([(a, b)])[0])


def test_identical_labels_take_the_trivial_path():
    scores = SetPRFScores.get([["a b", "c"]], [["a b", "c"]], BipartiteJaccardDistance(), soft=True)
    assert scores.micro.f1 == 1.0


def test_micro_and_macro():
    rater_a = [["a", "b"], ["c"], []]
    rater_b = [["a"], ["c", "d"], None]
    scores = SetPRFScores.get(rater_a, rater_b, exact_match_distance)
    assert (scores.micro.tp, scores.micro.n_true, scores.micro.n_pred) == (2, 3, 3)
    assert scores.n_items == 2  # the item without labels is left out
    assert scores.macro_precision == pytest.approx((1.0 + 0.5) / 2)
    assert scores.macro_recall == pytest.approx((0.5 + 1.0) / 2)
    assert scores.macro_f1 == pytest.approx(2 / 3)


def test_macro_skips_undefined_precision():
    scores = SetPRFScores.get([["a"], ["b"]], [["a"], []])
    assert scores.macro_precision == 1.0
    assert scores.macro_recall == 0.5


def test_match_threshold_controls_hard_true_positives():
    strict = SetPRFScores.get([["rash"]], [["skin rash"]], jaccard_distance)
    lenient = SetPRFScores.get([["rash"]], [["skin rash"]], jaccard_distance, match_threshold=0.6)
    assert (strict.micro.tp, lenient.micro.tp) == (0, 1)


def test_empty_score_counts_agreement_on_no_labels():
    rater_a = [["a", "b"], [], []]
    rater_b = [["a"], [], None]
    excluded = SetPRFScores.get(rater_a, rater_b)
    included = SetPRFScores.get(rater_a, rater_b, empty_score=1.0)
    assert (excluded.n_items, excluded.macro_f1) == (1, pytest.approx(2 / 3))
    assert (included.n_items, included.macro_f1) == (3, pytest.approx((2 / 3 + 1 + 1) / 3))
    assert included.micro == excluded.micro
