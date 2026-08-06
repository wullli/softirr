
from __future__ import annotations

from unittest.mock import MagicMock

from soft_irr.evaluation.distances.nli import BipartiteNLIDistance

TRIVIAL_PAIRS = [
    (frozenset(), frozenset()),
    (frozenset({"fever"}), frozenset({"fever"})),
    (frozenset({"fever"}), frozenset()),
]


def test_trivial_pairs_issue_no_prompts_and_load_no_model():
    dist = BipartiteNLIDistance()
    assert dist.batch(TRIVIAL_PAIRS) == [0.0, 0.0, 1.0]
    assert dist.last_n_prompts == 0
    assert dist._pipeline is None


def test_non_trivial_pairs_still_prompt_both_directions():
    dist = BipartiteNLIDistance()
    dist._load = MagicMock(side_effect=lambda: setattr(dist, "_pipeline", _fake_pipeline()))
    dist.batch([*TRIVIAL_PAIRS, (frozenset({"fever"}), frozenset({"pyrexia"}))])
    assert dist.last_n_prompts == 2


def _fake_pipeline():
    return MagicMock(side_effect=lambda chunk, batch_size: [[{"label": "entailment", "score": 0.9}]] * len(chunk))
