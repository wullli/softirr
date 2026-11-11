import pytest

from soft_irr.evaluation.distances import BipartiteJaccardDistance, exact_match_distance
from soft_irr.evaluation.hierarchical_prf import HierarchicalCounts, HierarchicalPRFScores


def _scores(rater_a, rater_b, **kwargs):
    return HierarchicalPRFScores.get(rater_a, rater_b, **kwargs)


def test_identical_inputs_score_one_everywhere():
    item = {"arm a": ["fever", "rash"], "arm b": ["nausea"]}
    s = _scores([item], [item])
    for level in (s.parent, s.conditional, s.end_to_end, s.pooled):
        assert (level.precision, level.recall, level.f1) == (1.0, 1.0, 1.0)
    assert s.recall_coverage == s.precision_coverage == 1.0


def test_unmatched_parent_moves_children_to_fn_fp():
    s = _scores(
        [{"arm a": ["fever", "rash"], "arm b": ["nausea", "pain"]}],
        [{"arm a": ["fever"], "arm c": ["nausea", "pain"]}],
    )
    assert (s.parent.tp, s.parent.n_true, s.parent.n_pred) == (1, 2, 2)
    assert (s.conditional.recall, s.conditional.precision) == (0.5, 1.0)
    assert s.end_to_end.recall == pytest.approx(1 / 4)
    assert s.end_to_end.precision == pytest.approx(1 / 3)
    # Pooled ignores parents: all three predicted children are found.
    assert s.pooled.recall == pytest.approx(3 / 4)
    assert s.pooled.precision == 1.0


def test_coverage_decomposition_multiplies_back():
    s = _scores(
        [{"a": ["x", "y"], "b": ["z"]}, {"c": ["u", "v", "w"]}],
        [{"a": ["x"], "q": ["z", "k"]}, {"c": ["u", "v"]}],
    )
    assert s.end_to_end.recall == pytest.approx(s.recall_coverage * s.conditional.recall)
    assert s.end_to_end.precision == pytest.approx(s.precision_coverage * s.conditional.precision)


def test_bipartite_distance_gives_one_result_per_strategy():
    dist = BipartiteJaccardDistance()
    by_strategy = HierarchicalPRFScores.get_by_matching(
        [{"rai high dose": ["dry mouth", "nausea"]}], [{"high dose rai": ["mouth dry", "skin rash"]}], dist
    )
    assert set(by_strategy) == {"hungarian", "greedy"}
    s = by_strategy["hungarian"]
    assert s.parent.tp == 1 and s.end_to_end.tp == 1
    soft = HierarchicalPRFScores.get(
        [{"rai high dose": ["dry mouth", "nausea"]}], [{"high dose rai": ["mouth dry", "skin rash"]}], dist, soft=True
    )
    assert soft.end_to_end.tp == pytest.approx(1.0)


def test_separate_child_distance():
    # Parents must match exactly; children are matched with token Jaccard.
    s = _scores(
        [{"arm": ["dry mouth"]}],
        [{"arm": ["mouth dry"]}],
        parent_distance_function=exact_match_distance,
        child_distance_function=BipartiteJaccardDistance(),
    )
    assert s.end_to_end.tp == 1


def test_missing_items_and_empty_parents():
    s = _scores([None, {"a": []}], [{}, {"a": []}])
    assert s.parent.tp == 1
    assert s.end_to_end.n_true == 0


def test_counts_add_up_and_round_trip_through_flat_rows():
    counts = HierarchicalPRFScores.item_counts_by_matching(
        [{"a": ["x", "y"]}, {"b": ["z"]}], [{"a": ["x"]}, {"c": ["z"]}]
    )["none"]
    assert [HierarchicalCounts.from_flat(c.flat()) for c in counts] == counts
    assert HierarchicalPRFScores.from_counts(counts) == HierarchicalPRFScores.from_counts([counts[0] + counts[1]])


def test_swapped_counts_equal_scoring_with_raters_exchanged():
    a = [{"a": ["x", "y", "z"], "b": ["w"]}]
    b = [{"a": ["x"], "c": ["v", "u"]}]
    (forward,) = HierarchicalPRFScores.item_counts_by_matching(a, b)["none"]
    (backward,) = HierarchicalPRFScores.item_counts_by_matching(b, a)["none"]
    assert forward.swapped() == backward


def test_end_to_end_counts_match_end_to_end_scores():
    counts = HierarchicalPRFScores.item_counts_by_matching(
        [{"a": ["x", "y"], "b": ["z"]}], [{"a": ["x"], "c": ["z"]}]
    )["none"]
    assert counts[0].end_to_end().scores() == HierarchicalPRFScores.from_counts(counts).end_to_end
