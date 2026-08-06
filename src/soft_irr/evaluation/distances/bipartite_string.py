
from abc import ABC
from typing import Any

from rapidfuzz.distance import Indel as rapidfuzz_indel
from rapidfuzz.distance import LCSseq as rapidfuzz_lcsseq
from rapidfuzz.distance import Levenshtein as rapidfuzz_levenshtein

from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance
from soft_irr.evaluation.distances.simple import jaccard_distance


class _BipartiteStringDistance(BipartiteMatchingDistance, ABC):

    _metric: Any = None

    def __init__(self, match_threshold: float = 1 / 3) -> None:
        super().__init__(model_name="", match_threshold=match_threshold)
        self._edge_cost_cache: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        pass

    def _prepare(
        self,
        parsed: list[tuple[Any, Any, list[str], list[str]]],
        batch_size: int,
    ) -> None:
        pass

    def _edge_cost(self, a: str, b: str) -> float:
        return 1.0 - self._metric.normalized_similarity(a, b)

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        key = (a, b)
        cost = self._edge_cost_cache.get(key)
        if cost is None:
            cost = self._edge_cost(a, b)
            self._edge_cost_cache[key] = cost
        return cost


class BipartiteJaccardDistance(_BipartiteStringDistance):

    def _edge_cost(self, a: str, b: str) -> float:
        return jaccard_distance(a, b)


bipartite_jaccard_distance = BipartiteJaccardDistance()


class BipartiteLevenshteinDistance(_BipartiteStringDistance):

    _metric = rapidfuzz_levenshtein


bipartite_levenshtein_distance = BipartiteLevenshteinDistance()


class BipartiteNormalizedIndelDistance(_BipartiteStringDistance):

    _metric = rapidfuzz_indel


bipartite_normalized_indel_distance = BipartiteNormalizedIndelDistance()


class BipartiteLCSseqDistance(_BipartiteStringDistance):

    _metric = rapidfuzz_lcsseq


bipartite_lcsseq_distance = BipartiteLCSseqDistance()
