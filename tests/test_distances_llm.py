
import math
from unittest.mock import MagicMock, patch

import pytest

from soft_irr.evaluation.distances import (
    BipartiteLogProbDistance,
    BipartiteVerbalizedProbDistance,
)


def _litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_batch_completion(mock_litellm: MagicMock, response: MagicMock) -> None:
    mock_litellm.batch_completion.side_effect = lambda **kwargs: [response] * len(kwargs["messages"])


@pytest.fixture()
def verbalized():
    return BipartiteVerbalizedProbDistance()


@pytest.fixture()
def logprob():
    return BipartiteLogProbDistance()


def _logprob_response(p_yes: float, p_no: float) -> MagicMock:
    def _tok(token: str, logprob: float) -> MagicMock:
        t = MagicMock()
        t.token = token
        t.logprob = logprob
        return t

    content_item = MagicMock()
    content_item.top_logprobs = [
        _tok(" yes", math.log(p_yes + 1e-12)),
        _tok(" no",  math.log(p_no  + 1e-12)),
    ]
    logprobs_obj = MagicMock()
    logprobs_obj.content = [content_item]
    choice = MagicMock()
    choice.logprobs = logprobs_obj
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_verbalized_both_empty(verbalized):
    assert verbalized(frozenset(), frozenset()) == pytest.approx(0.0)


def test_verbalized_equal_sets(verbalized):
    s = frozenset(["headache"])
    assert verbalized(s, s) == pytest.approx(0.0)


def test_verbalized_true_empty(verbalized):
    assert verbalized(frozenset(), frozenset(["headache"])) == pytest.approx(1.0)


def test_verbalized_pred_empty(verbalized):
    assert verbalized(frozenset(["headache"]), frozenset()) == pytest.approx(1.0)


def test_verbalized_high_score(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.9,0.9"))
        result = verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert result == pytest.approx(0.1, abs=1e-9)


def test_verbalized_low_score(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.1,0.1"))
        result = verbalized(frozenset(["headache"]), frozenset(["nausea"]))
    assert result == pytest.approx(0.9, abs=1e-9)


def test_verbalized_score_clamped_above_one(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("1.5,1.5"))
        result = verbalized(frozenset(["a"]), frozenset(["b"]))
    assert 0.0 <= result <= 1.0


def test_verbalized_score_clamped_below_zero(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("-0.3,-0.3"))
        result = verbalized(frozenset(["a"]), frozenset(["b"]))
    assert 0.0 <= result <= 1.0


def test_verbalized_string_distance_cache_bypasses_api(verbalized):
    verbalized._string_distance_cache = {("headache", "cephalgia"): 0.2}
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        result = verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
        mock_litellm.batch_completion.assert_not_called()
    assert result == pytest.approx(0.2, abs=1e-9)


def test_verbalized_duplicate_pairs_call_api_once(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.8,0.8"))
        verbalized.batch([
            (frozenset(["headache"]), frozenset(["cephalgia"])),
            (frozenset(["headache"]), frozenset(["cephalgia"])),
        ])
    assert mock_litellm.batch_completion.call_count == 1
    assert len(mock_litellm.batch_completion.call_args.kwargs["messages"]) == 1


def test_verbalized_directions_parsed_from_single_joint_response(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.9,0.3"))
        result = verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert mock_litellm.batch_completion.call_count == 1
    assert len(mock_litellm.batch_completion.call_args.kwargs["messages"]) == 1
    assert result == pytest.approx(1.0 - math.sqrt(0.9 * 0.3), abs=1e-9)


def test_verbalized_costs_cached_across_batch_calls(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.7,0.7"))
        verbalized.batch([(frozenset(["a"]), frozenset(["b"]))])
        verbalized.batch([(frozenset(["a"]), frozenset(["b"]))])
    assert mock_litellm.batch_completion.call_count == 1


def test_verbalized_last_assignments_length(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.9,0.9"))
        verbalized.batch([
            (frozenset(["a"]), frozenset(["b"])),
            (frozenset(["x"]), frozenset(["y"])),
        ])
    assert len(verbalized.last_assignments) == 2


def test_verbalized_multi_label_calls_cross_product(verbalized):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _litellm_response("0.5,0.5"))
        verbalized(frozenset(["a", "b"]), frozenset(["c", "d"]))
    assert mock_litellm.batch_completion.call_count == 1
    assert len(mock_litellm.batch_completion.call_args.kwargs["messages"]) == 4


def test_logprob_both_empty(logprob):
    assert logprob(frozenset(), frozenset()) == pytest.approx(0.0)


def test_logprob_equal_sets(logprob):
    s = frozenset(["headache"])
    assert logprob(s, s) == pytest.approx(0.0)


def test_logprob_true_empty(logprob):
    assert logprob(frozenset(), frozenset(["headache"])) == pytest.approx(1.0)


def test_logprob_pred_empty(logprob):
    assert logprob(frozenset(["headache"]), frozenset()) == pytest.approx(1.0)


def test_logprob_high_yes_prob_gives_low_distance(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.9, p_no=0.1))
        result = logprob(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert result == pytest.approx(0.1, abs=1e-5)


def test_logprob_equal_probs_give_half_distance(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.5, p_no=0.5))
        result = logprob(frozenset(["a"]), frozenset(["b"]))
    assert result == pytest.approx(0.5, abs=1e-5)


def test_logprob_high_no_prob_gives_high_distance(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.05, p_no=0.95))
        result = logprob(frozenset(["headache"]), frozenset(["nausea"]))
    assert result > 0.8


@pytest.mark.parametrize("p_yes,p_no", [(0.99, 0.01), (0.01, 0.99), (0.5, 0.5)])
def test_logprob_distance_in_unit_interval(logprob, p_yes, p_no):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=p_yes, p_no=p_no))
        result = logprob(frozenset(["x"]), frozenset(["y"]))
    assert 0.0 <= result <= 1.0


def test_logprob_string_distance_cache_bypasses_api(logprob):
    logprob._string_distance_cache = {("headache", "cephalgia"): 0.15}
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        result = logprob(frozenset(["headache"]), frozenset(["cephalgia"]))
        mock_litellm.batch_completion.assert_not_called()
    assert result == pytest.approx(0.15, abs=1e-9)


def test_logprob_duplicate_pairs_call_api_once(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.8, p_no=0.2))
        logprob.batch([
            (frozenset(["a"]), frozenset(["b"])),
            (frozenset(["a"]), frozenset(["b"])),
        ])
    assert mock_litellm.batch_completion.call_count == 1
    assert len(mock_litellm.batch_completion.call_args.kwargs["messages"]) == 2


def test_logprob_mutual_entailment_combines_both_directions(logprob):
    def _side_effect(**kwargs):
        responses = []
        for messages in kwargs["messages"]:
            content = messages[-1]["content"]
            if 'Does "b" semantically entail "a"' in content:
                responses.append(_logprob_response(p_yes=0.9, p_no=0.1))
            else:
                responses.append(_logprob_response(p_yes=0.1, p_no=0.9))
        return responses

    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        mock_litellm.batch_completion.side_effect = _side_effect
        result = logprob(frozenset(["a"]), frozenset(["b"]))
    assert result == pytest.approx(1.0 - math.sqrt(0.9 * 0.1), abs=1e-5)


def test_logprob_costs_cached_across_batch_calls(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.7, p_no=0.3))
        logprob.batch([(frozenset(["a"]), frozenset(["b"]))])
        logprob.batch([(frozenset(["a"]), frozenset(["b"]))])
    assert mock_litellm.batch_completion.call_count == 1


def test_logprob_last_assignments_length(logprob):
    with patch("soft_irr.evaluation.distances.llm_remote.litellm") as mock_litellm:
        _mock_batch_completion(mock_litellm, _logprob_response(p_yes=0.9, p_no=0.1))
        logprob.batch([
            (frozenset(["a"]), frozenset(["b"])),
            (frozenset(["x"]), frozenset(["y"])),
        ])
    assert len(logprob.last_assignments) == 2
