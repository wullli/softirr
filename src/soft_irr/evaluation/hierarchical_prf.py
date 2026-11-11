"""Two-level (parent → child) precision, recall and F1.

Each rater annotates an item with parent labels, each carrying child labels, e.g.
treatment arms with their adverse events: ``{"low dose": ["nausea"], "high dose": []}``.
Parents are matched first; children are matched only inside aligned parent pairs, so the
children of unmatched parents count as false negatives (reference) or false positives
(rater). Scores are reported at four levels:

- ``parent``: the parent labels.
- ``conditional``: children, restricted to aligned parent pairs.
- ``end_to_end``: children over all parents, i.e. including the parent-matching errors.
- ``pooled``: children pooled per item, ignoring the parents.

End-to-end recall factors exactly into ``recall_coverage`` (share of reference children
under aligned parents) times conditional recall; precision factors the same way.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from soft_irr.evaluation.distances import DistanceFunction, exact_match_distance
from soft_irr.evaluation.distances.base import matching_strategies
from soft_irr.evaluation.set_prf import PRFCounts, PRFScores, _labels, _threshold, match_labels

NestedLabels = dict[str, list[str]]
"""Parent label → child labels, for one item and one rater."""


class HierarchicalCounts(BaseModel):
    """Counts of one item; counts of several items add up."""

    parent: PRFCounts = PRFCounts()
    child_aligned: PRFCounts = PRFCounts()
    """Child true positives within aligned parent pairs, and the children of aligned parents."""
    child_n_true: float = 0.0
    child_n_pred: float = 0.0
    """All children, including those of unaligned parents."""
    pooled: PRFCounts = PRFCounts()

    def __add__(self, other: "HierarchicalCounts") -> "HierarchicalCounts":
        return HierarchicalCounts(
            parent=self.parent + other.parent,
            child_aligned=self.child_aligned + other.child_aligned,
            child_n_true=self.child_n_true + other.child_n_true,
            child_n_pred=self.child_n_pred + other.child_n_pred,
            pooled=self.pooled + other.pooled,
        )

    def end_to_end(self) -> PRFCounts:
        """Child counts over all parents: true positives within aligned parents, all children."""
        return self.child_aligned.model_copy(update={"n_true": self.child_n_true, "n_pred": self.child_n_pred})

    def swapped(self) -> "HierarchicalCounts":
        """Counts with reference and rater exchanged, e.g. to score two raters both ways."""
        return HierarchicalCounts(
            parent=self.parent.swapped(),
            child_aligned=self.child_aligned.swapped(),
            child_n_true=self.child_n_pred,
            child_n_pred=self.child_n_true,
            pooled=self.pooled.swapped(),
        )

    def flat(self) -> dict[str, float]:
        """Counts as a flat ``{"parent.tp_hard": ...}`` dict, e.g. for a table row."""
        out: dict[str, float] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, dict):
                out.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                out[key] = value
        return out

    @classmethod
    def from_flat(cls, row: dict) -> "HierarchicalCounts":
        """Inverse of :meth:`flat`; extra keys are ignored."""
        nested: dict = {}
        for name, field in cls.model_fields.items():
            if field.annotation is PRFCounts:
                nested[name] = {k: row[f"{name}.{k}"] for k in PRFCounts.model_fields}
            else:
                nested[name] = row[name]
        return cls.model_validate(nested)


def _union(labels: NestedLabels) -> list[str]:
    return sorted({child for children in labels.values() for child in _labels(children)})


class HierarchicalPRFScores(BaseModel):
    """Micro-averaged P/R/F1 of ``rater_b`` against the reference ``rater_a``."""

    parent: PRFScores
    conditional: PRFScores
    end_to_end: PRFScores
    pooled: PRFScores
    recall_coverage: float
    precision_coverage: float

    @staticmethod
    def item_counts_by_matching(
        rater_a: Sequence[NestedLabels | None],
        rater_b: Sequence[NestedLabels | None],
        parent_distance_function: DistanceFunction = exact_match_distance,
        child_distance_function: DistanceFunction | None = None,
        match_threshold: float | None = None,
    ) -> dict[str, list[HierarchicalCounts]]:
        """Per-item counts per matching strategy of ``parent_distance_function``. Children
        use the same strategy when ``child_distance_function`` offers it, else its default.
        ``child_distance_function`` defaults to the parent distance; parents are aligned
        when their matching cost is below ``match_threshold`` (default: the distance's)."""
        child_df = child_distance_function if child_distance_function is not None else parent_distance_function
        a_items = [dict(a or {}) for a in rater_a]
        b_items = [dict(b or {}) for b in rater_b]
        parents = [(list(a), list(b)) for a, b in zip(a_items, b_items)]
        parent_threshold = _threshold(parent_distance_function, match_threshold)
        child_threshold = _threshold(child_df, match_threshold)
        child_strategies = matching_strategies(child_df)

        pooled_pairs = [(_union(a), _union(b)) for a, b in zip(a_items, b_items)]
        pooled_by_strategy = match_labels(pooled_pairs, child_df)

        out: dict[str, list[HierarchicalCounts]] = {}
        for strategy, parent_matchings in match_labels(parents, parent_distance_function).items():
            child_strategy = strategy if strategy in child_strategies else child_strategies[0]
            aligned = [[(i, j) for i, j, cost in m if cost < parent_threshold] for m in parent_matchings]
            child_pairs = [
                (_labels(a[pa[i]]), _labels(b[pb[j]]))
                for a, b, (pa, pb), al in zip(a_items, b_items, parents, aligned)
                for i, j in al
            ]
            child_matchings = iter(match_labels(child_pairs, child_df)[child_strategy])
            pooled_matchings = pooled_by_strategy[child_strategy]

            counts: list[HierarchicalCounts] = []
            for k, (a, b, (pa, pb), al) in enumerate(zip(a_items, b_items, parents, aligned)):
                child_aligned = PRFCounts()
                for i, j in al:
                    ca, cb = _labels(a[pa[i]]), _labels(b[pb[j]])
                    child_aligned += PRFCounts.from_matching(len(ca), len(cb), next(child_matchings), child_threshold)
                pooled_a, pooled_b = pooled_pairs[k]
                counts.append(
                    HierarchicalCounts(
                        parent=PRFCounts.from_matching(len(pa), len(pb), parent_matchings[k], parent_threshold),
                        child_aligned=child_aligned,
                        child_n_true=sum(len(_labels(c)) for c in a.values()),
                        child_n_pred=sum(len(_labels(c)) for c in b.values()),
                        pooled=PRFCounts.from_matching(
                            len(pooled_a), len(pooled_b), pooled_matchings[k], child_threshold
                        ),
                    )
                )
            out[strategy] = counts
        return out

    @classmethod
    def from_counts(cls, counts: list[HierarchicalCounts], soft: bool = False) -> "HierarchicalPRFScores":
        """Scores of summed counts. ``soft`` uses the soft true-positive counts."""
        total = sum(counts, HierarchicalCounts())
        conditional = total.child_aligned.scores(soft)
        nan = float("nan")
        return cls(
            parent=total.parent.scores(soft),
            conditional=conditional,
            end_to_end=total.end_to_end().scores(soft),
            pooled=total.pooled.scores(soft),
            recall_coverage=total.child_aligned.n_true / total.child_n_true if total.child_n_true else nan,
            precision_coverage=total.child_aligned.n_pred / total.child_n_pred if total.child_n_pred else nan,
        )

    @classmethod
    def get(
        cls,
        rater_a: Sequence[NestedLabels | None],
        rater_b: Sequence[NestedLabels | None],
        parent_distance_function: DistanceFunction = exact_match_distance,
        child_distance_function: DistanceFunction | None = None,
        match_threshold: float | None = None,
        soft: bool = False,
    ) -> "HierarchicalPRFScores":
        return cls.get_by_matching(
            rater_a, rater_b, parent_distance_function, child_distance_function, match_threshold, soft
        )[matching_strategies(parent_distance_function)[0]]

    @classmethod
    def get_by_matching(
        cls,
        rater_a: Sequence[NestedLabels | None],
        rater_b: Sequence[NestedLabels | None],
        parent_distance_function: DistanceFunction = exact_match_distance,
        child_distance_function: DistanceFunction | None = None,
        match_threshold: float | None = None,
        soft: bool = False,
    ) -> dict[str, "HierarchicalPRFScores"]:
        by_strategy = cls.item_counts_by_matching(
            rater_a, rater_b, parent_distance_function, child_distance_function, match_threshold
        )
        return {strategy: cls.from_counts(counts, soft) for strategy, counts in by_strategy.items()}
