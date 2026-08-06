
from __future__ import annotations

import pytest

from soft_irr.evaluation.distances import exact_match_distance
from soft_irr.evaluation.scores import AgreementScores


def test_classic_2x2_contingency_table():
    rater_a = [["Yes"]] * 20 + [["Yes"]] * 5 + [["No"]] * 10 + [["No"]] * 15
    rater_b = [["Yes"]] * 20 + [["No"]] * 5 + [["Yes"]] * 10 + [["No"]] * 15

    scores = AgreementScores.get(rater_a, rater_b, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(0.70, abs=1e-9)
    assert scores.cohens_kappa == pytest.approx(0.4, abs=1e-9)
    assert scores.scotts_pi == pytest.approx(0.1950 / 0.4950, abs=1e-9)


def test_perfect_agreement():
    rater_a = [["A"]] * 3 + [["B"]] * 7
    rater_b = [["A"]] * 3 + [["B"]] * 7

    scores = AgreementScores.get(rater_a, rater_b, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(1.0)
    assert scores.cohens_kappa == pytest.approx(1.0)
    assert scores.scotts_pi == pytest.approx(1.0)


def test_chance_level_agreement_is_zero():
    rater_a = [["X"]] * 50 + [["Y"]] * 50
    rater_b = ([["X"]] * 25 + [["Y"]] * 25) * 2

    scores = AgreementScores.get(rater_a, rater_b, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(0.5, abs=1e-9)
    assert scores.cohens_kappa == pytest.approx(0.0, abs=1e-9)
    assert scores.scotts_pi == pytest.approx(0.0, abs=1e-9)


def test_systematic_disagreement_is_negative():
    rater_a = [["A"]] * 10 + [["B"]] * 10
    rater_b = [["B"]] * 10 + [["A"]] * 10

    scores = AgreementScores.get(rater_a, rater_b, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(0.0, abs=1e-9)
    assert scores.cohens_kappa == pytest.approx(-1.0, abs=1e-9)
    assert scores.scotts_pi == pytest.approx(-1.0, abs=1e-9)


def _mc_labels(n: int) -> list[frozenset[str]]:
    return [frozenset({f"label_{i}"}) for i in range(n)]


def test_mc_chance_pairs_hit_the_target_sample_count_exactly():
    labels = _mc_labels(100)
    pooled = labels + _mc_labels(100)
    target = 300

    kappa_pairs, kappa_weights = AgreementScores.chance_pairs_and_weights(
        labels, labels, mc_target_samples=target
    )
    pi_pairs, pi_weights = AgreementScores.chance_pairs_and_weights(
        pooled, pooled, mc_target_samples=target
    )

    assert len(kappa_pairs) == target
    assert len(pi_pairs) == target, "pi's pooled 2n marginal must not cost more than kappa's"
    for weights in (kappa_weights, pi_weights):
        assert sum(weights) == pytest.approx(1.0)
        assert len(set(weights)) == 1, "Monte Carlo pairs carry equal weight"


def test_mc_chance_pairs_are_deterministic_for_a_given_seed():
    pooled = _mc_labels(100) + _mc_labels(100)

    first, _ = AgreementScores.chance_pairs_and_weights(pooled, pooled, mc_target_samples=250)
    second, _ = AgreementScores.chance_pairs_and_weights(pooled, pooled, mc_target_samples=250)

    assert first == second


def test_mc_chance_pairs_are_not_a_prefix_of_the_shuffles():
    labels = _mc_labels(200)
    pairs, _ = AgreementScores.chance_pairs_and_weights(labels, labels, mc_target_samples=100)

    covered = {a for a, _ in pairs}
    assert len(pairs) == 100
    assert any(a in covered for a in labels[150:]), "sample looks like a prefix, not a draw"


def test_mc_chance_pairs_spread_over_min_shuffles_permutations():
    labels = _mc_labels(100)
    pairs, _ = AgreementScores.chance_pairs_and_weights(labels, labels, mc_target_samples=300)

    partners: dict[frozenset[str], set[frozenset[str]]] = {}
    for a, b in pairs:
        partners.setdefault(a, set()).add(b)
    assert max(len(v) for v in partners.values()) >= 3


def test_explicit_mc_shuffles_skips_the_down_sample():
    labels = _mc_labels(100)
    pairs, _ = AgreementScores.chance_pairs_and_weights(
        labels, labels, mc_shuffles=5, mc_target_samples=50
    )
    assert len(pairs) == 500
