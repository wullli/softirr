
from __future__ import annotations

from typing import Any

import pytest

from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance

PAIRS = [(frozenset({"a", "b"}), frozenset({"c", "d"})), (frozenset({"a"}), frozenset({"e"}))]


class _CountingDistance(BipartiteMatchingDistance):

    is_pair_cacheable = True

    def __init__(self) -> None:
        super().__init__(model_name="fake")
        self.prepared: list[tuple[str, str]] = []
        self._costs: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        pass

    def _prepare(self, parsed: list[tuple[Any, Any, list[str], list[str]]], batch_size: int) -> None:
        for _, _, strs_a, strs_b in parsed:
            for a in strs_a:
                for b in strs_b:
                    if self._is_cached(a, b) or (a, b) in self._costs:
                        continue
                    self.prepared.append((a, b))
                    self._costs[(a, b)] = 0.5

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        return self._costs.get((a, b), self._costs.get((b, a), 1.0))


def test_without_recovered_costs_everything_is_inferred():
    dist_fn = _CountingDistance()
    dist_fn.batch(PAIRS)
    assert sorted(dist_fn.prepared) == [("a", "c"), ("a", "d"), ("a", "e"), ("b", "c"), ("b", "d")]


def test_fully_recovered_costs_infer_nothing_and_are_used_verbatim():
    cached = {("a", "c"): 0.0, ("a", "d"): 0.9, ("b", "c"): 0.9, ("b", "d"): 0.0, ("a", "e"): 0.3}
    dist_fn = _CountingDistance()
    dist_fn._pair_cache = cached

    assert dist_fn.batch(PAIRS) == pytest.approx([0.0, 0.3])
    assert dist_fn.prepared == []
    assert dist_fn.last_cost_matrices[1] == [[0.3]]


def test_partially_recovered_costs_infer_only_the_missing_pairs():
    dist_fn = _CountingDistance()
    dist_fn._pair_cache = {("a", "c"): 0.0, ("a", "d"): 0.9, ("b", "c"): 0.9, ("b", "d"): 0.0}
    dist_fn.batch(PAIRS)
    assert dist_fn.prepared == [("a", "e")]


def test_lookup_ignores_the_order_the_strings_arrive_in():
    dist_fn = _CountingDistance()
    dist_fn._pair_cache = {("a", "z"): 0.25}
    assert dist_fn.batch([(frozenset({"z"}), frozenset({"a"}))]) == [0.25]
    assert dist_fn.prepared == []


def test_pairs_the_measure_short_circuits_never_consult_the_cache():
    dist_fn = _CountingDistance()
    dist_fn._pair_cache = {("a", "a"): 0.9}
    assert dist_fn.batch([(frozenset({"a"}), frozenset({"a"})), (frozenset(), frozenset({"b"}))]) == [0.0, 1.0]
    assert dist_fn.last_cost_matrices == [None, None]


def test_string_distance_cache_still_takes_precedence():
    dist_fn = _CountingDistance()
    dist_fn._pair_cache = {("a", "c"): 0.0, ("a", "d"): 0.9, ("b", "c"): 0.9, ("b", "d"): 0.0}
    dist_fn._string_distance_cache = {("a", "c"): 1.0, ("a", "d"): 0.0, ("b", "c"): 0.0, ("b", "d"): 1.0}
    assert dist_fn.batch([PAIRS[0]]) == [0.0]
    assert dist_fn.prepared == []
