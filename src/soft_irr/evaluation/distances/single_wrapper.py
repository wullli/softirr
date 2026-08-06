
from typing import Any

from soft_irr.evaluation.distances.base import DistanceFunction
from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance
from soft_irr.evaluation.distances.embedding import (
    bipartite_embedding_distance,
    bipartite_qwen3_embedding_distance,
)
from soft_irr.evaluation.distances.llm_local import (
    bipartite_medgemma_log_prob_distance,
    bipartite_medgemma_verbalized_prob_distance,
    bipartite_qwen35_log_prob_distance,
    bipartite_qwen35_verbalized_prob_distance,
)
from soft_irr.evaluation.distances.nli import (
    bipartite_nli_distance,
    bipartite_nli_multilingual_distance,
)


class _SingleBipartiteWrapper(DistanceFunction):

    def __init__(self, base: BipartiteMatchingDistance) -> None:
        self._base = base

    def __call__(self, label1: Any, label2: Any) -> float:
        return self.batch([(label1, label2)])[0]

    def batch(self, pairs: list[tuple[Any, Any]], batch_size: int = 32) -> list[float]:
        return self._base.batch(pairs, batch_size=batch_size)

    def release(self) -> None:
        self._base.release()

    @property
    def last_n_prompts(self) -> int:
        return self._base.last_n_prompts

    @property
    def last_assignments(self) -> list[list[tuple[int, int]]]:
        return self._base.last_assignments

    @property
    def last_confident_assignments(self) -> list[list[tuple[int, int]]]:
        return self._base.last_confident_assignments

    @property
    def last_cost_matrices(self) -> list[list[list[float]] | None]:
        return self._base.last_cost_matrices

    def score_with_assignment(self, index: int, pairs: list[tuple[int, int]]) -> float | None:
        return self._base.score_with_assignment(index, pairs)


single_logprob_qwen35 = _SingleBipartiteWrapper(bipartite_qwen35_log_prob_distance)
single_verbalized_qwen35 = _SingleBipartiteWrapper(bipartite_qwen35_verbalized_prob_distance)
single_logprob_medgemma = _SingleBipartiteWrapper(bipartite_medgemma_log_prob_distance)
single_verbalized_medgemma = _SingleBipartiteWrapper(bipartite_medgemma_verbalized_prob_distance)
single_embedding_pubmedbert = _SingleBipartiteWrapper(bipartite_embedding_distance)
single_embedding_qwen3 = _SingleBipartiteWrapper(bipartite_qwen3_embedding_distance)
single_nli_mednli = _SingleBipartiteWrapper(bipartite_nli_distance)
single_nli_multilingual = _SingleBipartiteWrapper(bipartite_nli_multilingual_distance)
