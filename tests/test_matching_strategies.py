
from __future__ import annotations

import pytest

from soft_irr.evaluation.distances import exact_match_distance, jaccard_distance
from soft_irr.evaluation.distances.base import NO_MATCHING, matching_strategies
from soft_irr.evaluation.distances.bipartite_string import BipartiteLevenshteinDistance
from soft_irr.evaluation.distances.single_wrapper import _SingleBipartiteWrapper
from soft_irr.evaluation.scores import AgreementScores, MultiRaterAgreementScores
from soft_irr.experiments.synthetic_irr import _summary_key

RATER_A = [["myocardial infarction", "cough"], ["fever"], ["rash", "itch"], ["headache", "nausea"]]
RATER_B = [["heart attack", "coughing"], ["pyrexia"], ["itchy rash", "rash"], ["head ache", "nauseous"]]


class _CountingLevenshtein(BipartiteLevenshteinDistance):

    def __init__(self) -> None:
        super().__init__()
        self.n_batch_calls = 0
        self.n_edge_costs = 0

    def batch(self, pairs, batch_size: int = 32):
        self.n_batch_calls += 1
        return super().batch(pairs, batch_size)

    def _edge_cost(self, a: str, b: str) -> float:
        self.n_edge_costs += 1
        return super()._edge_cost(a, b)


def test_distances_under_rescoring_needs_no_new_inference():
    fn = _CountingLevenshtein()
    hungarian = fn.batch([(frozenset(a), frozenset(b)) for a, b in zip(RATER_A, RATER_B)])
    edge_costs_after_batch = fn.n_edge_costs

    greedy = fn.distances_under("greedy")

    assert fn.n_edge_costs == edge_costs_after_batch
    assert fn.distances_under("hungarian") == hungarian
    assert all(g >= h - 1e-12 for g, h in zip(greedy, hungarian))


def test_distances_under_rejects_unknown_strategy():
    fn = BipartiteLevenshteinDistance()
    fn.batch([(frozenset({"fever"}), frozenset({"pyrexia"}))])
    with pytest.raises(ValueError, match="unknown matching strategy"):
        fn.distances_under("stable-marriage")


@pytest.mark.parametrize("scorer", ["two_rater", "multi_rater"])
def test_scoring_every_strategy_costs_one_batch(scorer):
    fn = _CountingLevenshtein()
    if scorer == "two_rater":
        scores = AgreementScores.get_by_matching(
            RATER_A, RATER_B, distance_function=fn, chance_distance_function=fn,
        )
    else:
        scores = MultiRaterAgreementScores.get_by_matching(
            [[a, b] for a, b in zip(RATER_A, RATER_B)], distance_function=fn,
            chance_distance_function=fn,
        )

    assert set(scores) == {"hungarian", "greedy"}
    assert fn.n_batch_calls == 2


def test_get_matches_the_hungarian_variant():
    fn = BipartiteLevenshteinDistance()
    by_matching = AgreementScores.get_by_matching(
        RATER_A, RATER_B, distance_function=fn, chance_distance_function=fn,
    )
    assert AgreementScores.get(RATER_A, RATER_B, distance_function=fn, chance_distance_function=fn) == (
        by_matching["hungarian"]
    )

    items = [[a, b] for a, b in zip(RATER_A, RATER_B)]
    multi = MultiRaterAgreementScores.get_by_matching(
        items, distance_function=fn, chance_distance_function=fn,
    )
    assert MultiRaterAgreementScores.get(
        items, distance_function=fn, chance_distance_function=fn
    ) == multi["hungarian"]


def test_strategies_are_scored_independently():
    fn = BipartiteLevenshteinDistance()
    scores = AgreementScores.get_by_matching(
        RATER_A, RATER_B, distance_function=fn, chance_distance_function=fn,
    )
    hungarian, greedy = scores["hungarian"], scores["greedy"]

    ae = {
        name: (s.average_agreement - s.cohens_kappa) / (1.0 - s.cohens_kappa)
        for name, s in (("hungarian", hungarian), ("greedy", greedy))
    }
    assert ae["greedy"] <= ae["hungarian"] + 1e-12


@pytest.mark.parametrize("scorer", ["two_rater", "multi_rater"])
def test_every_strategy_reports_the_distribution_metrics(scorer):
    fn = BipartiteLevenshteinDistance()
    if scorer == "two_rater":
        scores = AgreementScores.get_by_matching(
            RATER_A, RATER_B, distance_function=fn, chance_distance_function=fn,
        )
    else:
        scores = MultiRaterAgreementScores.get_by_matching(
            [[a, b] for a, b in zip(RATER_A, RATER_B)], distance_function=fn,
            chance_distance_function=fn,
        )

    for score in scores.values():
        assert 0.0 <= score.sigma <= 1.0
        assert 0.0 <= score.ks <= 1.0


def test_measures_without_a_matching_get_a_single_strategy():
    assert matching_strategies(exact_match_distance) == (NO_MATCHING,)
    assert matching_strategies(jaccard_distance) == (NO_MATCHING,)
    assert matching_strategies(BipartiteLevenshteinDistance()) == ("hungarian", "greedy")

    scores = AgreementScores.get_by_matching(RATER_A, RATER_B, distance_function=exact_match_distance)
    assert set(scores) == {NO_MATCHING}


def test_single_label_wrapper_reports_no_matching():
    wrapper = _SingleBipartiteWrapper(BipartiteLevenshteinDistance())
    assert matching_strategies(wrapper) == (NO_MATCHING,)

    single_a = [["fever"], ["rash"], ["cough"], ["headache"]]
    single_b = [["pyrexia"], ["itchy rash"], ["coughing"], ["head ache"]]
    scores = AgreementScores.get_by_matching(
        single_a, single_b, distance_function=wrapper, chance_distance_function=wrapper,
    )
    assert set(scores) == {NO_MATCHING}


def test_summary_key_only_annotates_non_default_strategies():
    assert _summary_key({"measure": "jaccard", "matching": "none"}) == "jaccard"
    assert _summary_key({"measure": "levenshtein_bipartite", "matching": "hungarian"}) == (
        "levenshtein_bipartite"
    )
    assert _summary_key({"measure": "levenshtein_bipartite", "matching": "greedy"}) == (
        "levenshtein_bipartite (greedy)"
    )
    assert _summary_key({"measure": "jaccard"}) == "jaccard"
