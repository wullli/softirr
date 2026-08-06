
import importlib

from soft_irr.evaluation.distances.base import DistanceFunction
from soft_irr.evaluation.distances.simple import (
    ExactMatchDistance,
    JaccardDistance,
    LCSseqDistance,
    LevenshteinDistance,
    MASIDistance,
    NormalizedIndelSimilarity,
    SetJaccardDistance,
    exact_match_distance,
    jaccard_distance,
    lcsseq_distance,
    levenshtein_distance,
    masi_distance,
    normalized_indel_similarity,
    set_jaccard_distance,
)
from soft_irr.evaluation.distances.bipartite_string import (
    BipartiteJaccardDistance,
    BipartiteLCSseqDistance,
    BipartiteLevenshteinDistance,
    BipartiteNormalizedIndelDistance,
    bipartite_jaccard_distance,
    bipartite_lcsseq_distance,
    bipartite_levenshtein_distance,
    bipartite_normalized_indel_distance,
)
from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance

_LAZY: dict[str, str] = {
    "BipartiteNLIDistance": "soft_irr.evaluation.distances.nli",
    "bipartite_nli_distance": "soft_irr.evaluation.distances.nli",
    "bipartite_nli_multilingual_distance": "soft_irr.evaluation.distances.nli",
    "BipartiteEmbeddingDistance": "soft_irr.evaluation.distances.embedding",
    "BipartiteQwen3EmbeddingDistance": "soft_irr.evaluation.distances.embedding",
    "bipartite_embedding_distance": "soft_irr.evaluation.distances.embedding",
    "bipartite_qwen3_embedding_distance": "soft_irr.evaluation.distances.embedding",
    "BipartiteLogProbDistance": "soft_irr.evaluation.distances.llm_remote",
    "BipartiteVerbalizedProbDistance": "soft_irr.evaluation.distances.llm_remote",
    "bipartite_log_prob_distance": "soft_irr.evaluation.distances.llm_remote",
    "bipartite_verbalized_prob_distance": "soft_irr.evaluation.distances.llm_remote",
    "BipartiteLocalLogProbDistance": "soft_irr.evaluation.distances.llm_local",
    "BipartiteLocalVerbalizedProbDistance": "soft_irr.evaluation.distances.llm_local",
    "bipartite_medgemma_log_prob_distance": "soft_irr.evaluation.distances.llm_local",
    "bipartite_medgemma_verbalized_prob_distance": "soft_irr.evaluation.distances.llm_local",
    "bipartite_qwen35_log_prob_distance": "soft_irr.evaluation.distances.llm_local",
    "bipartite_qwen35_verbalized_prob_distance": "soft_irr.evaluation.distances.llm_local",
    "single_embedding_pubmedbert": "soft_irr.evaluation.distances.single_wrapper",
    "single_embedding_qwen3": "soft_irr.evaluation.distances.single_wrapper",
    "single_logprob_medgemma": "soft_irr.evaluation.distances.single_wrapper",
    "single_logprob_qwen35": "soft_irr.evaluation.distances.single_wrapper",
    "single_nli_mednli": "soft_irr.evaluation.distances.single_wrapper",
    "single_nli_multilingual": "soft_irr.evaluation.distances.single_wrapper",
    "single_verbalized_medgemma": "soft_irr.evaluation.distances.single_wrapper",
    "single_verbalized_qwen35": "soft_irr.evaluation.distances.single_wrapper",
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DistanceFunction",
    "ExactMatchDistance",
    "exact_match_distance",
    "JaccardDistance",
    "jaccard_distance",
    "SetJaccardDistance",
    "set_jaccard_distance",
    "MASIDistance",
    "masi_distance",
    "LevenshteinDistance",
    "levenshtein_distance",
    "NormalizedIndelSimilarity",
    "normalized_indel_similarity",
    "LCSseqDistance",
    "lcsseq_distance",
    "BipartiteJaccardDistance",
    "bipartite_jaccard_distance",
    "BipartiteLevenshteinDistance",
    "bipartite_levenshtein_distance",
    "BipartiteNormalizedIndelDistance",
    "bipartite_normalized_indel_distance",
    "BipartiteLCSseqDistance",
    "bipartite_lcsseq_distance",
    "BipartiteMatchingDistance",
    "BipartiteNLIDistance",
    "bipartite_nli_distance",
    "bipartite_nli_multilingual_distance",
    "BipartiteEmbeddingDistance",
    "bipartite_embedding_distance",
    "BipartiteQwen3EmbeddingDistance",
    "bipartite_qwen3_embedding_distance",
    "BipartiteLogProbDistance",
    "bipartite_log_prob_distance",
    "BipartiteVerbalizedProbDistance",
    "bipartite_verbalized_prob_distance",
    "BipartiteLocalLogProbDistance",
    "bipartite_medgemma_log_prob_distance",
    "bipartite_qwen35_log_prob_distance",
    "BipartiteLocalVerbalizedProbDistance",
    "bipartite_medgemma_verbalized_prob_distance",
    "bipartite_qwen35_verbalized_prob_distance",
    "single_logprob_qwen35",
    "single_verbalized_qwen35",
    "single_logprob_medgemma",
    "single_verbalized_medgemma",
    "single_embedding_pubmedbert",
    "single_embedding_qwen3",
    "single_nli_mednli",
    "single_nli_multilingual",
]
