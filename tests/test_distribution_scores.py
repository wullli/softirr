
from __future__ import annotations

import numpy as np
import pytest

from soft_irr.evaluation.distribution_irr import DoDa
from soft_irr.evaluation.distribution_scores import expand_weighted_sample, sigma_and_ks
from soft_irr.evaluation.scores import AgreementScores


def test_matches_reference_implementation_when_unweighted():
    rng = np.random.default_rng(0)
    observed = rng.beta(2, 5, size=120)
    expected = rng.beta(5, 2, size=200)

    sigma, ks = sigma_and_ks(observed, expected)
    reference = DoDa(observed, expected)

    assert sigma == pytest.approx(reference.get_sigma())
    assert ks == pytest.approx(reference.get_ks())


def test_exact_weights_expand_in_proportion_to_the_cross_product():
    a_labels = [frozenset({"x"})] * 3 + [frozenset({"y"})] * 1
    b_labels = [frozenset({"x"})] * 2 + [frozenset({"y"})] * 2
    _, weights = AgreementScores.chance_pairs_and_weights(a_labels, b_labels)

    values = np.arange(len(weights), dtype=float)
    expanded = expand_weighted_sample(values, weights, cap=10_000)

    multiplicities = np.array([np.sum(expanded == v) for v in values], dtype=float)
    assert multiplicities / multiplicities.sum() == pytest.approx(np.asarray(weights))


def test_uniform_weights_expand_to_one_each():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    weights = [0.25] * 4

    expanded = expand_weighted_sample(values, weights, cap=10_000)

    assert sorted(expanded) == pytest.approx(sorted(values))


def test_weighting_shifts_sigma_towards_the_heavy_pairs():
    observed = np.full(20, 0.3)
    expected = np.array([0.1, 0.9])

    heavy_low = sigma_and_ks(observed, expected, weights=[0.99, 0.01])[0]
    heavy_high = sigma_and_ks(observed, expected, weights=[0.01, 0.99])[0]

    assert heavy_high > heavy_low


def test_perfect_agreement_scores_high():
    observed = np.zeros(40)
    expected = np.linspace(0.2, 1.0, 60)

    sigma, ks = sigma_and_ks(observed, expected)

    assert sigma == pytest.approx(1.0)
    assert ks > 0.9


def test_chance_level_agreement_scores_low():
    rng = np.random.default_rng(1)
    pool = rng.uniform(0, 1, size=400)
    observed, expected = pool[:200], pool[200:]

    sigma, ks = sigma_and_ks(observed, expected)

    assert sigma < 0.2
    assert ks < 0.2


@pytest.mark.parametrize(
    "observed,expected",
    [
        (np.full(10, 0.5), np.full(10, 0.5)),
        (np.full(10, 0.0), np.full(10, 1.0)),
        (np.linspace(0, 1, 10), np.full(10, 0.4)),
    ],
)
def test_constant_samples_do_not_raise(observed, expected):
    sigma, ks = sigma_and_ks(observed, expected)

    assert np.isfinite(sigma) and np.isfinite(ks)
    assert 0.0 <= sigma <= 1.0
    assert 0.0 <= ks <= 1.0


def test_expansion_is_capped_and_deterministic():
    rng = np.random.default_rng(2)
    values = rng.uniform(0, 1, size=500)
    weights = np.full(500, 1 / 500)
    cap = 1_000

    first = expand_weighted_sample(values, weights * 1_000_000, cap=cap)
    second = expand_weighted_sample(values, weights * 1_000_000, cap=cap)

    assert len(first) <= cap + len(values)
    assert np.array_equal(first, second)
