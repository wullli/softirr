
import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Hashable

_MATCHING_ORDER_KEY = b"soft-irr-label-order"


def _as_str_list(label: Any) -> list[str]:
    if isinstance(label, (frozenset, set)):
        return sorted(s for s in label if isinstance(s, str))
    if isinstance(label, list):
        return [s for s in label if isinstance(s, str)]
    return [str(label)]


def matching_order(label: Any) -> list[int]:
    strings = _as_str_list(label)
    return sorted(
        range(len(strings)),
        key=lambda i: hashlib.blake2b(
            strings[i].encode("utf-8"), digest_size=16, key=_MATCHING_ORDER_KEY
        ).digest(),
    )


def _as_matching_str_list(label: Any) -> list[str]:
    strings = _as_str_list(label)
    return [strings[i] for i in matching_order(label)]


def _unpack_single(label: Any) -> str:
    parts = _as_str_list(label)
    return " ".join(parts) if parts else ""


def trivial_distance(
    label1: Any, label2: Any, strs_a: list[str], strs_b: list[str]
) -> float | None:
    if not label1 and not label2:
        return 0.0
    if label1 == label2:
        return 0.0
    if not label1 or not label2:
        return 1.0
    if not strs_a or not strs_b:
        return 1.0
    return None


def _free_torch_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


NO_MATCHING = "none"


def matching_strategies(distance_function: Any) -> tuple[str, ...]:
    strategies = getattr(distance_function, "MATCHING_STRATEGIES", None)
    if strategies is None or not hasattr(distance_function, "distances_under"):
        return (NO_MATCHING,)
    return tuple(strategies)


class BipartiteMatchingState:

    is_bipartite = True

    MATCHING_STRATEGIES = ("hungarian", "greedy")

    def _reset_matching_state(self) -> None:
        self.last_cost_matrices: list[list[list[float]] | None] = []
        self.last_assignments: list[list[tuple[int, int]]] = []
        self.last_greedy_assignments: list[list[tuple[int, int]]] = []
        self.last_confident_assignments: list[list[tuple[int, int]]] = []
        self.last_distances: list[float] = []

    def _record_trivial(
        self,
        assignment: list[tuple[int, int]] | None = None,
    ) -> None:
        self.last_cost_matrices.append(None)
        self.last_assignments.append(assignment or [])
        self.last_greedy_assignments.append(assignment or [])
        self.last_confident_assignments.append(assignment or [])

    @staticmethod
    def _greedy_assignment(cost: Any) -> list[tuple[int, int]]:
        n_a, n_b = len(cost), len(cost[0])
        flat = sorted((cost[i][j], i, j) for i in range(n_a) for j in range(n_b))
        used_a: set[int] = set()
        used_b: set[int] = set()
        matched: list[tuple[int, int]] = []
        for _, i, j in flat:
            if i not in used_a and j not in used_b:
                matched.append((i, j))
                used_a.add(i)
                used_b.add(j)
                if len(matched) == min(n_a, n_b):
                    break
        return matched

    def _record_matching(
        self,
        cost: Any,
        row_ind: Any,
        col_ind: Any,
    ) -> None:
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        cost_list = [[float(c) for c in row] for row in cost]
        self.last_cost_matrices.append(cost_list)
        self.last_assignments.append(pairs)
        self.last_greedy_assignments.append(self._greedy_assignment(cost_list))
        self.last_confident_assignments.append(
            [(r, c) for r, c in pairs if cost[r][c] < self.match_threshold]
        )

    @staticmethod
    def matching_cost(
        cost: Any,
        pairs: list[tuple[int, int]],
        n_a: int,
        n_b: int,
    ) -> float:
        matched_quality = sum(1.0 - cost[r][c] for r, c in pairs)
        return 1.0 - 2.0 * matched_quality / (n_a + n_b)

    def score_with_assignment(self, index: int, pairs: list[tuple[int, int]]) -> float | None:
        cost = self.last_cost_matrices[index]
        if cost is None:
            return None
        n_a, n_b = len(cost), len(cost[0])
        return self.matching_cost(cost, pairs, n_a, n_b)

    def distances_under(self, strategy: str) -> list[float]:
        if strategy not in self.MATCHING_STRATEGIES:
            raise ValueError(
                f"unknown matching strategy {strategy!r}, expected one of {self.MATCHING_STRATEGIES}"
            )
        if strategy == "hungarian":
            return list(self.last_distances)
        rescored: list[float] = []
        for i, distance in enumerate(self.last_distances):
            cost = self.last_cost_matrices[i]
            if cost is None:
                rescored.append(distance)
            else:
                rescored.append(
                    self.matching_cost(cost, self.last_greedy_assignments[i], len(cost), len(cost[0]))
                )
        return rescored


class DirectionalEntailmentCache:

    _BOTH_DIRECTIONS = True
    is_pair_cacheable = True
    _string_distance_cache: dict[tuple[str, str], float] | None
    _is_cached: Callable[[str, str], bool]

    def _init_cost_cache(self) -> None:
        self._infer_cost_cache: dict[tuple[str, str], float] = {}

    def _pending_keys(
        self, parsed: list[tuple[Any, Any, list[str], list[str]]]
    ) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        keys: list[tuple[str, str]] = []
        for label1, label2, strs_a, strs_b in parsed:
            if trivial_distance(label1, label2, strs_a, strs_b) is not None:
                continue
            for a in strs_a:
                for b in strs_b:
                    if self._is_cached(a, b):
                        continue
                    directions = ((a, b), (b, a)) if self._BOTH_DIRECTIONS else ((a, b),)
                    for key in directions:
                        if key not in self._infer_cost_cache and key not in seen:
                            seen.add(key)
                            keys.append(key)
        return keys

    def _store_cost(self, key: tuple[str, str], result: Any) -> None:
        if self._BOTH_DIRECTIONS:
            self._infer_cost_cache[key] = result
            return
        x, y = key
        forward, backward = result
        self._infer_cost_cache[(x, y)] = forward
        self._infer_cost_cache[(y, x)] = backward

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        forward = self._infer_cost_cache.get((a, b), 1.0)
        backward = self._infer_cost_cache.get((b, a), 1.0)
        return 1.0 - math.sqrt((1.0 - forward) * (1.0 - backward))


class DistanceFunction(ABC):

    is_bipartite: bool = False

    @abstractmethod
    def __call__(self, label1: Hashable, label2: Hashable) -> float:
        ...

    def batch(
        self,
        pairs: list[tuple[Any, Any]],
        batch_size: int = 32,
    ) -> list[float]:
        return [self(label1, label2) for label1, label2 in pairs]

    def release(self) -> None:
        pass
