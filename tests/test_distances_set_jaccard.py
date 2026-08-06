
import pytest

from soft_irr.evaluation.distances import SetJaccardDistance

set_jaccard_distance = SetJaccardDistance()


def test_both_empty():
    assert set_jaccard_distance([], []) == pytest.approx(0.0)


def test_one_empty():
    assert set_jaccard_distance(["fever"], []) == pytest.approx(1.0)


def test_equal_sets_zero_distance():
    assert set_jaccard_distance(["fever", "cough"], ["fever", "cough"]) == pytest.approx(0.0)


def test_disjoint_sets_full_distance():
    assert set_jaccard_distance(["fever"], ["cough"]) == pytest.approx(1.0)


def test_partial_overlap():
    result = set_jaccard_distance(["fever", "cough"], ["fever", "chills"])
    assert result == pytest.approx(1.0 - 1.0 / 3.0)


def test_reduces_to_exact_match_for_singletons():
    assert set_jaccard_distance(["fever"], ["fever"]) == pytest.approx(0.0)
    assert set_jaccard_distance(["fever"], ["cough"]) == pytest.approx(1.0)


def test_does_not_tokenize_multiword_items():
    result = set_jaccard_distance(["diabetes mellitus"], ["mellitus fever"])
    assert result == pytest.approx(1.0)


def test_frozenset_input():
    result = set_jaccard_distance(frozenset({"fever", "cough"}), frozenset({"fever"}))
    assert result == pytest.approx(1.0 - 1.0 / 2.0)
