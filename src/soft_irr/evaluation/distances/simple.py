
from typing import Any, Hashable

import nltk
from nltk.metrics.distance import jaccard_distance as nltk_jaccard_distance
from nltk.metrics.distance import masi_distance as nltk_masi_distance
from nltk.tokenize import word_tokenize
from rapidfuzz.distance import Indel as rapidfuzz_indel
from rapidfuzz.distance import LCSseq as rapidfuzz_lcsseq
from rapidfuzz.distance import Levenshtein as rapidfuzz_levenshtein

from soft_irr.evaluation.distances.base import DistanceFunction, _as_str_list, _unpack_single

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class ExactMatchDistance(DistanceFunction):

    def __call__(self, label1: Hashable, label2: Hashable) -> float:
        return 0.0 if label1 == label2 else 1.0


exact_match_distance = ExactMatchDistance()


class JaccardDistance(DistanceFunction):

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(word_tokenize(text.lower()))

    def __call__(self, label1: Any, label2: Any) -> float:
        s1, s2 = _unpack_single(label1), _unpack_single(label2)
        a, b = self._tokens(s1), self._tokens(s2)
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        return nltk_jaccard_distance(a, b)


jaccard_distance = JaccardDistance()


class SetJaccardDistance(DistanceFunction):

    def __call__(self, label1: Any, label2: Any) -> float:
        s1 = frozenset(_as_str_list(label1))
        s2 = frozenset(_as_str_list(label2))
        union = s1 | s2
        if not union:
            return 0.0
        return 1.0 - len(s1 & s2) / len(union)


set_jaccard_distance = SetJaccardDistance()


class MASIDistance(DistanceFunction):

    def __call__(self, label1: Any, label2: Any) -> float:
        s1, s2 = _unpack_single(label1), _unpack_single(label2)
        a, b = JaccardDistance._tokens(s1), JaccardDistance._tokens(s2)
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        return nltk_masi_distance(a, b)


masi_distance = MASIDistance()


class _RapidfuzzDistance(DistanceFunction):

    _metric: Any

    def __call__(self, label1: Any, label2: Any) -> float:
        s1, s2 = _unpack_single(label1), _unpack_single(label2)
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        return 1.0 - self._metric.normalized_similarity(s1, s2)


class LevenshteinDistance(_RapidfuzzDistance):

    _metric = rapidfuzz_levenshtein


levenshtein_distance = LevenshteinDistance()


class NormalizedIndelSimilarity(_RapidfuzzDistance):

    _metric = rapidfuzz_indel


normalized_indel_similarity = NormalizedIndelSimilarity()


class LCSseqDistance(_RapidfuzzDistance):

    _metric = rapidfuzz_lcsseq


lcsseq_distance = LCSseqDistance()
