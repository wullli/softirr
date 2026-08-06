
import math
from unittest.mock import MagicMock

import pytest
import torch

from soft_irr.evaluation.distances.llm_local import (
    BipartiteLocalLogProbDistance,
    BipartiteLocalVerbalizedProbDistance,
)

_YES_ID = 100
_NO_ID = 200
_VOCAB_SIZE = 300
_PROMPT_LEN = 10


class _FakeInputs(dict):
    def to(self, device):
        return self


def _make_tokenizer(decode_text: str = '0.9, "x_entails_y": 0.9}') -> MagicMock:
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token_id = 2

    def _encode(t, add_special_tokens=True):
        if t.lower() == "yes":
            return [_YES_ID]
        if t.lower() == "no":
            return [_NO_ID]
        return []

    tok.encode.side_effect = _encode
    tok.apply_chat_template.return_value = "PROMPT"
    tok.decode.return_value = decode_text

    def _call(texts, **kwargs):
        batch = len(texts)
        return _FakeInputs({
            "input_ids": torch.zeros(batch, _PROMPT_LEN, dtype=torch.long),
            "attention_mask": torch.ones(batch, _PROMPT_LEN, dtype=torch.long),
        })

    tok.side_effect = _call
    return tok


def _make_logprob_model(
    pairs: list[tuple[float, float]] | None = None,
    p_yes: float = 0.9,
    p_no: float = 0.1,
) -> MagicMock:
    if pairs is None:
        pairs = [(p_yes, p_no)] * 100

    idx = [0]

    def _forward(**kwargs):
        batch = kwargs["input_ids"].shape[0]
        vocab = torch.zeros(batch, 1, _VOCAB_SIZE)
        for i in range(batch):
            py, pn = pairs[idx[0] + i]
            vocab[i, 0, _YES_ID] = math.log(py + 1e-12)
            vocab[i, 0, _NO_ID] = math.log(pn + 1e-12)
        idx[0] += batch
        out = MagicMock()
        out.logits = vocab
        return out

    model = MagicMock()
    model.device = "cpu"
    model.side_effect = _forward
    return model


def _make_verbalized_model() -> MagicMock:
    def _generate(**kwargs):
        batch = kwargs["input_ids"].shape[0]
        prompt_len = kwargs["input_ids"].shape[1]
        return torch.zeros(batch, prompt_len + 5, dtype=torch.long)

    model = MagicMock()
    model.device = "cpu"
    model.generate.side_effect = _generate
    return model


def _inject_logprob(
    dist: BipartiteLocalLogProbDistance, **kwargs
) -> tuple[MagicMock, MagicMock]:
    tok = _make_tokenizer()
    model = _make_logprob_model(**kwargs)
    dist._tokenizer = tok
    dist._model = model
    return tok, model


def _inject_verbalized(
    dist: BipartiteLocalVerbalizedProbDistance,
    decode_text: str = '0.9, "x_entails_y": 0.9}',
) -> tuple[MagicMock, MagicMock]:
    tok = _make_tokenizer(decode_text)
    model = _make_verbalized_model()
    dist._tokenizer = tok
    dist._model = model
    return tok, model


@pytest.fixture()
def local_logprob():
    return BipartiteLocalLogProbDistance(model_name="test-model")


@pytest.fixture()
def local_verbalized():
    return BipartiteLocalVerbalizedProbDistance(model_name="test-model")


# --- pure function tests ---


def test_cost_from_logits_high_yes():
    logits = torch.zeros(_VOCAB_SIZE)
    logits[_YES_ID] = math.log(0.9)
    logits[_NO_ID] = math.log(0.1)
    assert BipartiteLocalLogProbDistance._cost_from_logits(logits, [_YES_ID], [_NO_ID]) == pytest.approx(0.1, abs=1e-5)


def test_cost_from_logits_equal():
    logits = torch.zeros(_VOCAB_SIZE)
    logits[_YES_ID] = math.log(0.5)
    logits[_NO_ID] = math.log(0.5)
    assert BipartiteLocalLogProbDistance._cost_from_logits(logits, [_YES_ID], [_NO_ID]) == pytest.approx(0.5, abs=1e-5)


def test_cost_from_logits_no_yes_ids():
    logits = torch.zeros(_VOCAB_SIZE)
    logits[_NO_ID] = math.log(0.9)
    assert BipartiteLocalLogProbDistance._cost_from_logits(logits, [], [_NO_ID]) == 1.0


def test_cost_from_logits_no_no_ids():
    logits = torch.zeros(_VOCAB_SIZE)
    logits[_YES_ID] = math.log(0.9)
    assert BipartiteLocalLogProbDistance._cost_from_logits(logits, [_YES_ID], []) == 0.0


def test_cost_from_logits_neither():
    logits = torch.zeros(_VOCAB_SIZE)
    assert BipartiteLocalLogProbDistance._cost_from_logits(logits, [], []) == 0.5


def test_find_answer_step_found():
    d = BipartiteLocalLogProbDistance(model_name="test-model")
    tok = MagicMock()

    def _decode(tokens, skip_special_tokens=False):
        return '"entails": "' if len(tokens) >= 3 else "thinking..."

    tok.decode.side_effect = _decode
    d._tokenizer = tok
    assert d._find_answer_step(torch.zeros(5, dtype=torch.long)) == 3


def test_find_answer_step_not_found():
    d = BipartiteLocalLogProbDistance(model_name="test-model")
    tok = MagicMock()
    tok.decode.return_value = "no answer here"
    d._tokenizer = tok
    assert d._find_answer_step(torch.zeros(5, dtype=torch.long)) is None


# --- BipartiteLocalLogProbDistance ---


def test_local_logprob_both_empty(local_logprob):
    assert local_logprob(frozenset(), frozenset()) == pytest.approx(0.0)


def test_local_logprob_equal_sets(local_logprob):
    s = frozenset(["headache"])
    assert local_logprob(s, s) == pytest.approx(0.0)


def test_local_logprob_true_empty(local_logprob):
    assert local_logprob(frozenset(), frozenset(["headache"])) == pytest.approx(1.0)


def test_local_logprob_pred_empty(local_logprob):
    assert local_logprob(frozenset(["headache"]), frozenset()) == pytest.approx(1.0)


def test_local_logprob_high_yes_prob_gives_low_distance(local_logprob):
    _inject_logprob(local_logprob, p_yes=0.9, p_no=0.1)
    result = local_logprob(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert result == pytest.approx(0.1, abs=1e-5)


def test_local_logprob_equal_probs_give_half_distance(local_logprob):
    _inject_logprob(local_logprob, p_yes=0.5, p_no=0.5)
    result = local_logprob(frozenset(["a"]), frozenset(["b"]))
    assert result == pytest.approx(0.5, abs=1e-5)


def test_local_logprob_high_no_prob_gives_high_distance(local_logprob):
    _inject_logprob(local_logprob, p_yes=0.05, p_no=0.95)
    result = local_logprob(frozenset(["headache"]), frozenset(["nausea"]))
    assert result > 0.8


@pytest.mark.parametrize("p_yes,p_no", [(0.99, 0.01), (0.01, 0.99), (0.5, 0.5)])
def test_local_logprob_distance_in_unit_interval(local_logprob, p_yes, p_no):
    _inject_logprob(local_logprob, p_yes=p_yes, p_no=p_no)
    result = local_logprob(frozenset(["x"]), frozenset(["y"]))
    assert 0.0 <= result <= 1.0


def test_local_logprob_string_distance_cache_bypasses_model(local_logprob):
    local_logprob._string_distance_cache = {("headache", "cephalgia"): 0.15}
    result = local_logprob(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert local_logprob._model is None
    assert result == pytest.approx(0.15, abs=1e-9)


def test_local_logprob_duplicate_pairs_call_model_once(local_logprob):
    tok, _ = _inject_logprob(local_logprob, p_yes=0.8, p_no=0.2)
    local_logprob.batch([
        (frozenset(["a"]), frozenset(["b"])),
        (frozenset(["a"]), frozenset(["b"])),
    ])
    assert tok.call_count == 1


def test_local_logprob_costs_cached_across_batch_calls(local_logprob):
    tok, _ = _inject_logprob(local_logprob, p_yes=0.7, p_no=0.3)
    local_logprob.batch([(frozenset(["a"]), frozenset(["b"]))])
    local_logprob.batch([(frozenset(["a"]), frozenset(["b"]))])
    assert tok.call_count == 1


def test_local_logprob_last_assignments_length(local_logprob):
    _inject_logprob(local_logprob, p_yes=0.9, p_no=0.1)
    local_logprob.batch([
        (frozenset(["a"]), frozenset(["b"])),
        (frozenset(["x"]), frozenset(["y"])),
    ])
    assert len(local_logprob.last_assignments) == 2


def test_local_logprob_mutual_entailment_combines_both_directions(local_logprob):
    _inject_logprob(local_logprob, pairs=[(0.9, 0.1), (0.1, 0.9)])
    result = local_logprob(frozenset(["a"]), frozenset(["b"]))
    assert result == pytest.approx(1.0 - math.sqrt(0.9 * 0.1), abs=1e-5)


# --- BipartiteLocalVerbalizedProbDistance ---


def test_local_verbalized_both_empty(local_verbalized):
    assert local_verbalized(frozenset(), frozenset()) == pytest.approx(0.0)


def test_local_verbalized_equal_sets(local_verbalized):
    s = frozenset(["headache"])
    assert local_verbalized(s, s) == pytest.approx(0.0)


def test_local_verbalized_true_empty(local_verbalized):
    assert local_verbalized(frozenset(), frozenset(["headache"])) == pytest.approx(1.0)


def test_local_verbalized_pred_empty(local_verbalized):
    assert local_verbalized(frozenset(["headache"]), frozenset()) == pytest.approx(1.0)


def test_local_verbalized_high_score(local_verbalized):
    _inject_verbalized(local_verbalized, '0.9, "x_entails_y": 0.9}')
    result = local_verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert result == pytest.approx(0.1, abs=1e-9)


def test_local_verbalized_low_score(local_verbalized):
    _inject_verbalized(local_verbalized, '0.1, "x_entails_y": 0.1}')
    result = local_verbalized(frozenset(["headache"]), frozenset(["nausea"]))
    assert result == pytest.approx(0.9, abs=1e-9)


def test_local_verbalized_score_clamped_above_one(local_verbalized):
    _inject_verbalized(local_verbalized, '1.5, "x_entails_y": 1.5}')
    result = local_verbalized(frozenset(["a"]), frozenset(["b"]))
    assert 0.0 <= result <= 1.0


def test_local_verbalized_score_clamped_below_zero(local_verbalized):
    _inject_verbalized(local_verbalized, '-0.3, "x_entails_y": -0.3}')
    result = local_verbalized(frozenset(["a"]), frozenset(["b"]))
    assert 0.0 <= result <= 1.0


def test_local_verbalized_string_distance_cache_bypasses_model(local_verbalized):
    local_verbalized._string_distance_cache = {("headache", "cephalgia"): 0.2}
    result = local_verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert local_verbalized._model is None
    assert result == pytest.approx(0.2, abs=1e-9)


def test_local_verbalized_duplicate_pairs_call_model_once(local_verbalized):
    tok, _ = _inject_verbalized(local_verbalized, '0.8, "x_entails_y": 0.8}')
    local_verbalized.batch([
        (frozenset(["headache"]), frozenset(["cephalgia"])),
        (frozenset(["headache"]), frozenset(["cephalgia"])),
    ])
    assert tok.call_count == 1


def test_local_verbalized_costs_cached_across_batch_calls(local_verbalized):
    tok, _ = _inject_verbalized(local_verbalized, '0.7, "x_entails_y": 0.7}')
    local_verbalized.batch([(frozenset(["a"]), frozenset(["b"]))])
    local_verbalized.batch([(frozenset(["a"]), frozenset(["b"]))])
    assert tok.call_count == 1


def test_local_verbalized_last_assignments_length(local_verbalized):
    _inject_verbalized(local_verbalized, '0.9, "x_entails_y": 0.9}')
    local_verbalized.batch([
        (frozenset(["a"]), frozenset(["b"])),
        (frozenset(["x"]), frozenset(["y"])),
    ])
    assert len(local_verbalized.last_assignments) == 2


def test_local_verbalized_multi_label_calls_cross_product(local_verbalized):
    tok, _ = _inject_verbalized(local_verbalized, '0.5, "x_entails_y": 0.5}')
    local_verbalized(frozenset(["a", "b"]), frozenset(["c", "d"]))
    assert tok.call_count == 1
    assert len(tok.call_args[0][0]) == 4


def test_local_verbalized_directions_parsed_from_single_joint_response(local_verbalized):
    tok, _ = _inject_verbalized(local_verbalized, '0.9, "x_entails_y": 0.3}')
    result = local_verbalized(frozenset(["headache"]), frozenset(["cephalgia"]))
    assert tok.call_count == 1
    assert len(tok.call_args[0][0]) == 1
    assert result == pytest.approx(1.0 - math.sqrt(0.9 * 0.3), abs=1e-9)
