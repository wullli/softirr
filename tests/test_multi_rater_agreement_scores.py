
from __future__ import annotations

import pytest

from soft_irr.evaluation.distances import exact_match_distance
from soft_irr.evaluation.scores import AgreementScores, MultiRaterAgreementScores

_FLEISS_1971_TABLE: dict[int, list[int]] = {
    1: [0, 0, 0, 0, 14],
    2: [0, 2, 6, 4, 2],
    3: [0, 0, 3, 5, 6],
    4: [0, 3, 9, 2, 0],
    5: [2, 2, 8, 1, 1],
    6: [7, 7, 0, 0, 0],
    7: [3, 2, 6, 3, 0],
    8: [2, 5, 3, 2, 2],
    9: [6, 5, 2, 1, 0],
    10: [0, 2, 2, 3, 7],
}
_FLEISS_1971_N_RATERS = 14


def _fleiss_1971_raters() -> list[list[list[str]]]:
    raters = []
    for counts in _FLEISS_1971_TABLE.values():
        item = [[f"cat{cat_idx + 1}"] for cat_idx, n in enumerate(counts) for _ in range(n)]
        assert len(item) == _FLEISS_1971_N_RATERS
        raters.append(item)
    return raters


def test_fleiss_1971_worked_example():
    scores = MultiRaterAgreementScores.get(_fleiss_1971_raters(), distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(0.378, abs=5e-4)
    assert scores.fleiss_kappa == pytest.approx(0.210, abs=5e-4)


def test_reduces_to_agreement_scores_when_every_item_has_two_raters():
    rater_a = [["Yes"]] * 20 + [["Yes"]] * 5 + [["No"]] * 10 + [["No"]] * 15
    rater_b = [["Yes"]] * 20 + [["No"]] * 5 + [["Yes"]] * 10 + [["No"]] * 15

    two_rater = AgreementScores.get(rater_a, rater_b, distance_function=exact_match_distance)
    multi_rater = MultiRaterAgreementScores.get(
        [[a, b] for a, b in zip(rater_a, rater_b)], distance_function=exact_match_distance,
    )

    assert multi_rater.average_agreement == pytest.approx(two_rater.average_agreement, abs=1e-9)
    assert multi_rater.cohens_kappa == pytest.approx(two_rater.cohens_kappa, abs=1e-9)
    assert multi_rater.fleiss_kappa == pytest.approx(two_rater.scotts_pi, abs=1e-9)


def test_perfect_agreement_with_variable_rater_counts():
    raters = [
        [["A"], ["A"]],
        [["B"], ["B"], ["B"]],
        [["A"], ["A"], ["A"], ["A"]],
    ]

    scores = MultiRaterAgreementScores.get(raters, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(1.0)
    assert scores.cohens_kappa == pytest.approx(1.0)
    assert scores.fleiss_kappa == pytest.approx(1.0)


def test_variable_rater_count_hand_computed_example():
    raters = [
        [["A"], ["A"], ["B"]],
        [["A"], ["B"], ["B"]],
        [["A"], ["B"]],
        [["B"], ["A"]],
    ]

    scores = MultiRaterAgreementScores.get(raters, distance_function=exact_match_distance)

    assert scores.average_agreement == pytest.approx(1.0 / 6.0, abs=1e-9)
    assert scores.cohens_kappa == pytest.approx(-1.0 / 6.0, abs=1e-9)
    assert scores.fleiss_kappa == pytest.approx(-2.0 / 3.0, abs=1e-9)
