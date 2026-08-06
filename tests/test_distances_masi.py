
import pytest

from soft_irr.evaluation.distances import JaccardDistance, MASIDistance

masi_distance = MASIDistance()
jaccard_distance = JaccardDistance()


def test_both_empty():
    assert masi_distance([], []) == pytest.approx(0.0)


def test_one_empty():
    assert masi_distance(["fever"], []) == pytest.approx(1.0)


def test_identical_labels_zero_distance():
    assert masi_distance(["fever"], ["fever"]) == pytest.approx(0.0)


def test_disjoint_tokens_full_distance():
    assert masi_distance(["fever"], ["cough"]) == pytest.approx(1.0)


def test_tokenizes_multiword_labels():
    result = masi_distance(["acute kidney failure"], ["chronic kidney failure"])
    assert result < 1.0


def test_subset_gets_monotonicity_credit():
    result = masi_distance(["kidney failure"], ["acute kidney failure"])
    assert result == pytest.approx(1.0 - (2.0 / 3.0) * 0.67)


def test_partial_overlap_uses_intersecting_monotonicity():
    result = masi_distance(["acute renal failure"], ["acute kidney failure"])
    assert result == pytest.approx(1.0 - (2.0 / 4.0) * 0.33)


def test_case_insensitive_like_jaccard():
    assert masi_distance(["Fever"], ["fever"]) == pytest.approx(0.0)


def test_never_looser_than_jaccard():
    pairs = [
        (["acute kidney failure"], ["chronic kidney failure"]),
        (["kidney failure"], ["acute kidney failure"]),
        (["fever", "cough"], ["fever", "chills"]),
        (["diabetes mellitus type 2"], ["type 2 diabetes"]),
    ]
    for a, b in pairs:
        assert masi_distance(a, b) >= jaccard_distance(a, b) - 1e-12


def test_set_input_pools_tokens_across_items():
    result = masi_distance(["fever", "dry cough"], ["fever", "cough"])
    assert result == pytest.approx(1.0 - (2.0 / 3.0) * 0.67)


def test_does_not_reduce_to_exact_match_for_single_labels():
    assert masi_distance(["acute renal failure"], ["acute kidney failure"]) < 1.0


def test_frozenset_input():
    assert masi_distance(frozenset({"fever"}), frozenset({"fever"})) == pytest.approx(0.0)
