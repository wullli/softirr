
import pytest
from rapidfuzz.distance import LCSseq as rapidfuzz_lcsseq

from soft_irr.evaluation.distances import BipartiteLCSseqDistance, LCSseqDistance

lcsseq_distance = LCSseqDistance()
bipartite_lcsseq_distance = BipartiteLCSseqDistance()


def test_both_empty():
    assert lcsseq_distance("", "") == pytest.approx(0.0)


def test_one_empty():
    assert lcsseq_distance("kitten", "") == pytest.approx(1.0)


def test_equal_strings_zero_distance():
    result = lcsseq_distance("kitten", "kitten")
    assert result == pytest.approx(0.0)


def test_matches_rapidfuzz_normalized_similarity():
    s1, s2 = "kitten", "sitting"
    expected = rapidfuzz_lcsseq.normalized_similarity(s1, s2)
    result = lcsseq_distance(s1, s2)
    assert result == pytest.approx(1.0 - expected)


def test_rewards_subsequence_order_over_substitution():
    close = lcsseq_distance("abc", "acb")
    far = lcsseq_distance("abc", "xyz")
    assert close < far


def test_frozenset_input_is_unpacked():
    result = lcsseq_distance(frozenset({"kitten"}), frozenset({"kitten"}))
    assert result == pytest.approx(0.0)


def test_bipartite_both_empty():
    assert bipartite_lcsseq_distance([], []) == pytest.approx(0.0)


def test_bipartite_one_empty():
    assert bipartite_lcsseq_distance(["fever"], []) == pytest.approx(1.0)


def test_bipartite_equal_sets_zero_distance():
    result = bipartite_lcsseq_distance(["fever", "cough"], ["fever", "cough"])
    assert result == pytest.approx(0.0)


def test_bipartite_edge_cost_uses_lcsseq_normalized_distance():
    a, b = "kitten", "sitting"
    expected_cost = 1.0 - rapidfuzz_lcsseq.normalized_similarity(a, b)
    assert bipartite_lcsseq_distance._edge_cost(a, b) == pytest.approx(expected_cost)
