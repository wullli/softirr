from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from soft_irr.evaluation.distances.s1 import BipartiteS1Distance

TRIVIAL_PAIRS = [
    (frozenset(), frozenset()),
    (frozenset({"fever"}), frozenset({"fever"})),
    (frozenset({"fever"}), frozenset()),
]


def _fake_predictor(probs: dict[tuple[str, str], float]):
    """predict_batch stub answering P(true) per (text, hypothesis) from `probs`."""

    def predict_batch(requests, batch_size):
        out = []
        for text, questions in requests:
            hypothesis = questions["entails"]["instructions"].split('"')[1]
            out.append({"answers": {"entails": {"type": "noul", "noul": probs[(text, hypothesis)]}}})
        return out

    predictor = MagicMock()
    predictor.predict_batch = MagicMock(side_effect=predict_batch)
    return predictor


def _with_predictor(dist: BipartiteS1Distance, probs: dict[tuple[str, str], float]) -> BipartiteS1Distance:
    dist._load = MagicMock(side_effect=lambda: setattr(dist, "_pipeline", _fake_predictor(probs)))
    return dist


def test_trivial_pairs_issue_no_prompts_and_load_no_model():
    dist = BipartiteS1Distance()
    assert dist.batch(TRIVIAL_PAIRS) == [0.0, 0.0, 1.0]
    assert dist.last_n_prompts == 0
    assert dist._pipeline is None


def test_non_trivial_pair_asks_both_directions_and_uses_geometric_mean():
    dist = _with_predictor(BipartiteS1Distance(), {("fever", "pyrexia"): 0.9, ("pyrexia", "fever"): 0.4})
    [distance] = dist.batch([(frozenset({"fever"}), frozenset({"pyrexia"}))])
    assert dist.last_n_prompts == 2
    assert distance == pytest.approx(1.0 - math.sqrt(0.9 * 0.4))


def test_question_names_the_hypothesis():
    question = BipartiteS1Distance()._question("pyrexia")["entails"]
    assert question["type"] == "noul"
    assert '"pyrexia"' in question["instructions"]
    assert "criteria" not in question


def test_custom_question_text_and_criteria():
    criteria = {"false": "different event", "true": "same event"}
    dist = BipartiteS1Distance(
        question_template='Same event as "{hypothesis}"?', text_template="Event: {premise}", criteria=criteria
    )
    dist._pipeline = MagicMock()
    dist._pipeline.predict_batch.return_value = [{"answers": {"entails": {"noul": 0.7}}}]
    assert dist._entailment_probs([("fever", "pyrexia")], batch_size=8) == [0.7]
    [(text, questions)] = dist._pipeline.predict_batch.call_args.args[0]
    assert text == "Event: fever"
    assert questions["entails"] == {"type": "noul", "instructions": 'Same event as "pyrexia"?', "criteria": criteria}
