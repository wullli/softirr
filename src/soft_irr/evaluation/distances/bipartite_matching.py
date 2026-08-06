
from abc import ABC, abstractmethod
from typing import Any

from scipy.optimize import linear_sum_assignment

from soft_irr.evaluation.distances.base import (
    BipartiteMatchingState,
    DistanceFunction,
    _as_matching_str_list,
    _free_torch_memory,
    trivial_distance,
)
from soft_irr.evaluation.distances.pair_cache import canonical


class BipartiteMatchingDistance(BipartiteMatchingState, DistanceFunction, ABC):

    is_bipartite = True

    is_pair_cacheable = False

    enable_reasoning = False

    def __init__(self, model_name: str, match_threshold: float = 1 / 3) -> None:
        self.model_name = model_name
        self._pipeline: Any = None
        self._string_distance_cache: dict[tuple[str, str], float] | None = None
        self._pair_cache: dict[tuple[str, str], float] | None = None
        self._pair_cost_sink: Any = None
        self._fresh_costs: dict[tuple[str, str], float] = {}
        self._reset_matching_state()
        self.last_n_prompts: int = 0
        self.match_threshold = match_threshold

    @abstractmethod
    def _load(self) -> None:
        ...

    @abstractmethod
    def _prepare(
        self,
        parsed: list[tuple[Any, Any, list[str], list[str]]],
        batch_size: int,
    ) -> None:
        ...

    @abstractmethod
    def _cost(self, a: str, b: str) -> float:
        ...

    def _is_cached(self, a: str, b: str) -> bool:
        return self._pair_cache is not None and canonical(a, b) in self._pair_cache

    def _matrix_cost(self, a: str, b: str) -> float:
        key = canonical(a, b)
        cached = None if self._pair_cache is None else self._pair_cache.get(key)
        if cached is not None:
            return cached
        cost = self._cost(a, b)
        if self._pair_cost_sink is not None:
            self._fresh_costs[key] = cost
        return cost

    def _flush_fresh_costs(self) -> None:
        if self._pair_cost_sink is not None and self._fresh_costs:
            self._pair_cost_sink(self._fresh_costs)
            if self._pair_cache is None:
                self._pair_cache = {}
            self._pair_cache.update(self._fresh_costs)
        self._fresh_costs = {}

    def release(self) -> None:
        self._pipeline = None
        _free_torch_memory()

    def __call__(self, label1: Any, label2: Any) -> float:
        return self.batch([(label1, label2)])[0]

    def batch(
        self,
        pairs: list[tuple[Any, Any]],
        batch_size: int = 32,
    ) -> list[float]:
        self._reset_matching_state()
        if not pairs:
            return []

        parsed: list[tuple[Any, Any, list[str], list[str]]] = [
            (l1, l2, _as_matching_str_list(l1), _as_matching_str_list(l2)) for l1, l2 in pairs
        ]

        if self._string_distance_cache is None:
            self._prepare(parsed, batch_size)

        results: list[float] = []
        for label1, label2, strs_a, strs_b in parsed:
            trivial = trivial_distance(label1, label2, strs_a, strs_b)
            if trivial is not None:
                results.append(trivial)
                self._record_trivial(
                    [(i, i) for i in range(len(strs_a))] if label1 == label2 else None
                )
                continue
            n_a, n_b = len(strs_a), len(strs_b)
            cost = [[self._matrix_cost(a, b) for b in strs_b] for a in strs_a]
            row_ind, col_ind = linear_sum_assignment(cost)
            self._record_matching(cost, row_ind, col_ind)
            results.append(self.matching_cost(cost, self.last_assignments[-1], n_a, n_b))

        self.last_distances = list(results)
        self._flush_fresh_costs()
        return results
