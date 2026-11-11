"""Item-level precision, recall and F1 of a rater against a reference rater.

For each item, the labels of both raters are paired by a matching. Bipartite distances use
their own matching per strategy (see :func:`matching_strategies`); any other distance is
applied to each pair of labels and the pairs are matched optimally (Hungarian). A pair is
a hard true positive when its cost is below ``match_threshold``; the soft true-positive
count is the summed match quality ``1 - cost`` over the matching, so the soft F1 of one
item equals ``1 - distance`` of a bipartite distance.
"""

import math
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from soft_irr.evaluation.distances import DistanceFunction, exact_match_distance
from soft_irr.evaluation.distances.base import BipartiteMatchingState, matching_order, matching_strategies

DEFAULT_MATCH_THRESHOLD = 1 / 3

_ASSIGNMENTS_BY_STRATEGY = {"hungarian": "last_assignments", "greedy": "last_greedy_assignments"}

Matching = list[tuple[int, int, float]]
"""``(index in reference labels, index in rater labels, cost)`` per matched pair."""


def _labels(labels: Any) -> list[str]:
    return [] if labels is None else [str(label) for label in labels]


def match_labels(
    pairs: list[tuple[list[str], list[str]]],
    distance_function: DistanceFunction = exact_match_distance,
) -> dict[str, list[Matching]]:
    """Matching per strategy, for each ``(reference labels, rater labels)`` pair."""
    if isinstance(distance_function, BipartiteMatchingState):
        distance_function.batch(pairs)  # type: ignore[attr-defined]
        out: dict[str, list[Matching]] = {}
        for strategy in matching_strategies(distance_function):
            assignments = getattr(distance_function, _ASSIGNMENTS_BY_STRATEGY[strategy])
            matchings: list[Matching] = []
            for (a, b), cost, assignment in zip(pairs, distance_function.last_cost_matrices, assignments):
                # The distance matches a permuted label order; map back to list indices.
                order_a, order_b = matching_order(a), matching_order(b)
                matchings.append(
                    [(order_a[r], order_b[c], 0.0 if cost is None else float(cost[r][c])) for r, c in assignment]
                )
            out[strategy] = matchings
        return out

    matchings = []
    for a, b in pairs:
        if not a or not b:
            matchings.append([])
            continue
        cost = np.array([[distance_function(x, y) for y in b] for x in a], dtype=float)
        rows, cols = linear_sum_assignment(cost)
        matchings.append([(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols)])
    return {matching_strategies(distance_function)[0]: matchings}


def prf(tp: float, n_true: float, n_pred: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts; NaN where a denominator is zero."""
    nan = float("nan")
    precision = tp / n_pred if n_pred else nan
    recall = tp / n_true if n_true else nan
    f1 = 2.0 * tp / (n_true + n_pred) if (n_true + n_pred) else nan
    return precision, recall, f1


class PRFScores(BaseModel):

    precision: float
    recall: float
    f1: float
    tp: float
    n_true: float
    n_pred: float


class PRFCounts(BaseModel):
    """True-positive and label counts; counts of several items add up."""

    tp_hard: float = 0.0
    tp_soft: float = 0.0
    n_true: float = 0.0
    n_pred: float = 0.0

    @classmethod
    def from_matching(cls, n_true: int, n_pred: int, matching: Matching, match_threshold: float) -> "PRFCounts":
        return cls(
            tp_hard=sum(cost < match_threshold for _, _, cost in matching),
            tp_soft=sum(1.0 - cost for _, _, cost in matching),
            n_true=n_true,
            n_pred=n_pred,
        )

    def __add__(self, other: "PRFCounts") -> "PRFCounts":
        return PRFCounts(
            tp_hard=self.tp_hard + other.tp_hard,
            tp_soft=self.tp_soft + other.tp_soft,
            n_true=self.n_true + other.n_true,
            n_pred=self.n_pred + other.n_pred,
        )

    def swapped(self) -> "PRFCounts":
        """Counts with reference and rater exchanged (precision and recall swap)."""
        return self.model_copy(update={"n_true": self.n_pred, "n_pred": self.n_true})

    def scores(self, soft: bool = False) -> PRFScores:
        tp = self.tp_soft if soft else self.tp_hard
        precision, recall, f1 = prf(tp, self.n_true, self.n_pred)
        return PRFScores(precision=precision, recall=recall, f1=f1, tp=tp, n_true=self.n_true, n_pred=self.n_pred)


def _threshold(distance_function: DistanceFunction, match_threshold: float | None) -> float:
    if match_threshold is not None:
        return match_threshold
    return getattr(distance_function, "match_threshold", DEFAULT_MATCH_THRESHOLD)


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


class SetPRFScores(BaseModel):
    """Micro-averaged (pooled counts) and macro-averaged (mean over items) P/R/F1 of
    ``rater_b`` against the reference ``rater_a``.

    Items without labels from either rater add nothing to the micro counts. In the macro
    averages they are left out by default (``empty_score=None``); with ``empty_score=1.0``
    they count as full agreement, so agreeing on "no labels" is rewarded."""

    micro: PRFScores
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n_items: int

    @staticmethod
    def item_counts_by_matching(
        rater_a: list[list[str]],
        rater_b: list[list[str]],
        distance_function: DistanceFunction = exact_match_distance,
        match_threshold: float | None = None,
    ) -> dict[str, list[PRFCounts]]:
        """Per-item counts per matching strategy."""
        pairs = [(_labels(a), _labels(b)) for a, b in zip(rater_a, rater_b)]
        threshold = _threshold(distance_function, match_threshold)
        return {
            strategy: [PRFCounts.from_matching(len(a), len(b), m, threshold) for (a, b), m in zip(pairs, matchings)]
            for strategy, matchings in match_labels(pairs, distance_function).items()
        }

    @classmethod
    def from_counts(
        cls, counts: list[PRFCounts], soft: bool = False, empty_score: float | None = None
    ) -> "SetPRFScores":
        informative = [c for c in counts if c.n_true or c.n_pred]
        per_item = [c.scores(soft) for c in informative]
        precision = [s.precision for s in per_item]
        recall = [s.recall for s in per_item]
        f1 = [s.f1 for s in per_item]
        if empty_score is not None:
            n_empty = len(counts) - len(informative)
            for values in (precision, recall, f1):
                values.extend([empty_score] * n_empty)
        return cls(
            micro=sum(informative, PRFCounts()).scores(soft),
            macro_precision=_nanmean(precision),
            macro_recall=_nanmean(recall),
            macro_f1=_nanmean(f1),
            n_items=len(f1),
        )

    @classmethod
    def get(
        cls,
        rater_a: list[list[str]],
        rater_b: list[list[str]],
        distance_function: DistanceFunction = exact_match_distance,
        match_threshold: float | None = None,
        soft: bool = False,
        empty_score: float | None = None,
    ) -> "SetPRFScores":
        return cls.get_by_matching(rater_a, rater_b, distance_function, match_threshold, soft, empty_score)[
            matching_strategies(distance_function)[0]
        ]

    @classmethod
    def get_by_matching(
        cls,
        rater_a: list[list[str]],
        rater_b: list[list[str]],
        distance_function: DistanceFunction = exact_match_distance,
        match_threshold: float | None = None,
        soft: bool = False,
        empty_score: float | None = None,
    ) -> dict[str, "SetPRFScores"]:
        by_strategy = cls.item_counts_by_matching(rater_a, rater_b, distance_function, match_threshold)
        return {strategy: cls.from_counts(counts, soft, empty_score) for strategy, counts in by_strategy.items()}
