
from __future__ import annotations

from typing import Any

import pytest

from soft_irr.evaluation.distances import (
    BipartiteLevenshteinDistance,
    BipartiteNormalizedIndelDistance,
    levenshtein_distance,
    normalized_indel_similarity,
)
from soft_irr.evaluation.distances.base import trivial_distance
from soft_irr.evaluation.distances.llm_local import _BipartiteLocalLLMBase
from soft_irr.evaluation.scores import AgreementScores


@pytest.mark.parametrize(
    ("label1", "label2", "expected"),
    [
        (frozenset(), frozenset(), 0.0),
        (frozenset(["fever"]), frozenset(["fever"]), 0.0),
        (frozenset(), frozenset(["fever"]), 1.0),
        (frozenset(["fever"]), frozenset(), 1.0),
        (frozenset(["fever"]), frozenset(["pyrexia"]), None),
    ],
)
def test_trivial_distance_truth_table(label1, label2, expected):
    strs_a = sorted(label1)
    strs_b = sorted(label2)
    assert trivial_distance(label1, label2, strs_a, strs_b) == expected


def test_trivial_distance_covers_labels_with_no_strings():
    assert trivial_distance([1], ["fever"], [], ["fever"]) == 1.0


@pytest.mark.parametrize("cls", [BipartiteLevenshteinDistance, BipartiteNormalizedIndelDistance])
def test_string_measures_short_circuit_like_the_model_backed_ones(cls):
    dist = cls()
    results = dist.batch(
        [
            (frozenset(), frozenset()),
            (frozenset(["fever"]), frozenset(["fever"])),
            (frozenset(["fever"]), frozenset()),
            (frozenset(["fever"]), frozenset(["pyrexia"])),
        ]
    )
    assert results[:3] == [0.0, 0.0, 1.0]
    assert 0.0 < results[3] <= 1.0
    assert dist.last_cost_matrices[:3] == [None, None, None]
    assert dist.last_cost_matrices[3] is not None


def test_string_measures_keep_their_matching_strategies():
    dist = BipartiteLevenshteinDistance()
    hungarian = dist.batch([(frozenset(["fever", "cough"]), frozenset(["pyrexia", "coughing"]))])
    assert dist.distances_under("hungarian") == hungarian
    assert dist.distances_under("greedy")[0] >= hungarian[0] - 1e-12


def test_levenshtein_and_indel_stay_distinct_metrics():
    a, b = "cat", "cots"
    assert levenshtein_distance(a, b) == pytest.approx(2 / 4)
    assert normalized_indel_similarity(a, b) == pytest.approx(3 / 7)


class _FakeLocalLLM(_BipartiteLocalLLMBase):

    def __init__(self) -> None:
        super().__init__(model_name="fake")
        self.prompts: list[str] = []

    def _load(self) -> None:
        self._tokenizer = object()
        self._model = object()

    def _build_prompt(self, x: str, y: str) -> str:
        return f"{x}|{y}"

    def _infer(self, prompts: list[str], batch_size: int) -> list[float]:
        self.prompts.extend(prompts)
        return [0.25] * len(prompts)


def test_local_llm_skips_pairs_that_never_reach_a_cost_matrix():
    dist = _FakeLocalLLM()
    dist.batch(
        [
            (frozenset(["fever"]), frozenset(["fever"])),
            (frozenset(), frozenset(["fever"])),
            (frozenset(["fever"]), frozenset(["pyrexia"])),
        ]
    )
    assert dist.last_n_prompts == 2
    assert sorted(dist.prompts) == ["fever|pyrexia", "pyrexia|fever"]


def test_local_llm_caches_costs_across_batches():
    dist = _FakeLocalLLM()
    pair = (frozenset(["fever"]), frozenset(["pyrexia"]))
    first = dist.batch([pair])
    n_prompts = len(dist.prompts)
    second = dist.batch([pair])
    assert len(dist.prompts) == n_prompts
    assert first == second


def _chance_kwargs(**overrides: Any) -> dict:
    return AgreementScores.chance_kwargs(**overrides)


def test_chance_kwargs_drops_unset_values():
    assert _chance_kwargs() == {}
    assert _chance_kwargs(max_exact_chance_pairs=50) == {"max_exact_pairs": 50}
    assert _chance_kwargs(mc_target_samples=200, mc_shuffles=2, mc_min_shuffles=1) == {
        "mc_target_samples": 200,
        "mc_shuffles": 2,
        "mc_min_shuffles": 1,
    }


def test_chance_kwargs_are_accepted_by_chance_pairs_and_weights():
    labels = [frozenset([f"c{i}"]) for i in range(20)]
    pairs, weights = AgreementScores.chance_pairs_and_weights(
        labels, labels, **_chance_kwargs(max_exact_chance_pairs=10, mc_target_samples=15, mc_min_shuffles=1)
    )
    assert len(pairs) == 15
    assert sum(weights) == pytest.approx(1.0)
