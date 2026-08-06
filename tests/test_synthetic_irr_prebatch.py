
from __future__ import annotations

from typing import Any

from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance
from soft_irr.evaluation.scores import AgreementScores
from soft_irr.experiments.load_taxonomies import ConceptEntry
from soft_irr.experiments.synthetic_irr import _generate_populations, _prebatch_populations


class _RecordingDistance(BipartiteMatchingDistance):

    def __init__(self) -> None:
        super().__init__(model_name="fake")
        self.seen_pairs: set[tuple[str, str]] = set()

    def _load(self) -> None:
        pass

    def _prepare(self, parsed: list[tuple[Any, Any, list[str], list[str]]], batch_size: int) -> None:
        for _, _, strs_a, strs_b in parsed:
            for a in strs_a:
                for b in strs_b:
                    self.seen_pairs.add((a, b))
                    self.seen_pairs.add((b, a))

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        return 0.0 if a == b else 1.0


def _make_pool() -> list[ConceptEntry]:
    return [
        ConceptEntry(code=str(i), preferred_term=f"Concept {i}", synonyms=[f"c{i}s1", f"c{i}s2", f"c{i}s3"])
        for i in range(20)
    ]


def test_prebatch_populations_covers_every_pair_agreementscores_get_needs():
    pool = _make_pool()
    concept_by_code = {c.code: c for c in pool}
    populations = _generate_populations(
        "single", pool, n_samples=25, seeds=[1, 2, 3, 4, 5],
        sibling_index=None, concept_by_code=concept_by_code,
    )

    dist_fn = _RecordingDistance()
    timing = _prebatch_populations(dist_fn, populations, "fake")
    assert timing is not None
    pairs_after_prebatch = set(dist_fn.seen_pairs)
    assert pairs_after_prebatch

    for _, _, rater_a, rater_b, _, _, _ in populations:
        AgreementScores.get(rater_a, rater_b, distance_function=dist_fn, chance_distance_function=dist_fn)

    assert dist_fn.seen_pairs == pairs_after_prebatch, (
        "AgreementScores.get() requested pair(s) that prebatch never queued -- prebatch is "
        "missing some pair(s) it needs (e.g. Scott's pi's pooled chance-agreement term)."
    )


def test_prebatch_populations_set_mode_covers_every_pair_too():
    pool = _make_pool()
    concept_by_code = {c.code: c for c in pool}
    populations = _generate_populations(
        "set", pool, n_samples=15, seeds=[10, 11, 12],
        sibling_index=None, concept_by_code=concept_by_code,
        min_set_size=1, max_set_size=3,
    )

    dist_fn = _RecordingDistance()
    _prebatch_populations(dist_fn, populations, "fake")
    pairs_after_prebatch = set(dist_fn.seen_pairs)

    for _, _, rater_a, rater_b, _, _, _ in populations:
        AgreementScores.get(rater_a, rater_b, distance_function=dist_fn, chance_distance_function=dist_fn)

    assert dist_fn.seen_pairs == pairs_after_prebatch
