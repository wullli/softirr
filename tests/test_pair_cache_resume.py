from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from soft_irr.evaluation.distances.llm_local import (
    BipartiteLocalLogProbDistance,
    BipartiteLocalVerbalizedProbDistance,
)
from soft_irr.evaluation.distances.pair_cache import (
    PairCostWriter,
    _iter_cost_records,
    attach_writer,
    canonical,
    load_costs,
)

_ITEMS = {("a", f"b{i}"): i / 100 for i in range(200)}


def _torn_sidecar(path, items=_ITEMS, cut=200):
    with PairCostWriter(path) as writer:
        writer.write("nli_mednli", False, items)
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - cut)


@pytest.fixture()
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.mark.parametrize("cut", range(8, 400, 7))
def test_a_torn_member_followed_by_an_appended_one_does_not_fail_the_load(results_dir, cut):
    path = results_dir / "paircosts_run.jsonl.gz"
    _torn_sidecar(path, cut=cut)
    record = {"measure": "nli_mednli", "reasoning": False, "x": "p", "y": "q", "cost": 0.5}
    with gzip.open(path, "at") as f:
        f.write((json.dumps(record) + "\n") * 50)

    load_costs(results_dir, ["nli_mednli"])


@pytest.mark.parametrize("cut", [8, 50, 200, 400])
def test_appending_to_a_torn_sidecar_keeps_its_prefix_and_the_new_costs(results_dir, cut):
    path = results_dir / "paircosts_run.jsonl.gz"
    _torn_sidecar(path, cut=cut)
    before = load_costs(results_dir, ["nli_mednli"])["nli_mednli"]

    with PairCostWriter(path) as writer:
        writer.write("nli_mednli", False, {("p", "q"): 0.5})

    after = load_costs(results_dir, ["nli_mednli"])["nli_mednli"]
    assert after == {**before, ("p", "q"): 0.5}


def test_reopening_an_intact_sidecar_keeps_its_records(results_dir):
    path = results_dir / "paircosts_run.jsonl.gz"
    with PairCostWriter(path) as writer:
        writer.write("nli_mednli", False, _ITEMS)
    original = list(_iter_cost_records(path))

    PairCostWriter(path).close()

    assert list(_iter_cost_records(path)) == original
    assert list(results_dir.iterdir()) == [path]


def test_written_costs_survive_a_writer_that_is_never_closed(results_dir):
    path = results_dir / "paircosts_run.jsonl.gz"
    writer = PairCostWriter(path)
    writer.write("nli_mednli", False, _ITEMS)

    assert len(list(_iter_cost_records(path))) == len(_ITEMS)
    writer.close()


class _StubLogProb(BipartiteLocalLogProbDistance):
    _CHECKPOINT_PROMPTS = 4

    def __init__(self) -> None:
        super().__init__(model_name="test-model")
        self.infer_calls: list[int] = []
        self.written_per_call: list[int] = []
        self.sink: list[tuple[str, str]] = []
        self.die_after_calls: int | None = None

    def _load(self) -> None:
        self._tokenizer, self._model = object(), object()

    def _build_prompt(self, x: str, y: str) -> str:
        return f"{x}|{y}"

    def _infer(self, prompts: list[str], batch_size: int) -> list[Any]:
        if self.die_after_calls is not None and len(self.infer_calls) == self.die_after_calls:
            raise KeyboardInterrupt
        self.infer_calls.append(len(prompts))
        self.written_per_call.append(len(self.sink))
        return [0.2] * len(prompts)


class _StubVerbalized(BipartiteLocalVerbalizedProbDistance):
    _CHECKPOINT_PROMPTS = 3

    def __init__(self) -> None:
        super().__init__(model_name="test-model")
        self.infer_calls: list[int] = []

    def _load(self) -> None:
        self._tokenizer, self._model = object(), object()

    def _build_prompt(self, x: str, y: str) -> str:
        return f"{x}|{y}"

    def _infer(self, prompts: list[str], batch_size: int) -> list[Any]:
        self.infer_calls.append(len(prompts))
        return [(0.2, 0.4)] * len(prompts)


_PAIRS = [
    (frozenset({"a", "b"}), frozenset({"c", "d"})),
    (frozenset({"a", "e"}), frozenset({"f"})),
]
_STRING_PAIRS = {canonical(x, y) for x, y in [
    ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("a", "f"), ("e", "f"),
]}


def test_local_llm_inference_writes_costs_block_by_block():
    dist_fn = _StubLogProb()
    dist_fn._pair_cost_sink = dist_fn.sink.extend

    dist_fn.batch(_PAIRS)

    assert all(n <= dist_fn._CHECKPOINT_PROMPTS for n in dist_fn.infer_calls)
    assert len(dist_fn.infer_calls) > 1
    assert dist_fn.written_per_call[-1] > 0
    assert sorted(dist_fn.sink) == sorted(_STRING_PAIRS)


def test_an_interrupted_inference_has_already_written_its_finished_blocks():
    dist_fn = _StubLogProb()
    dist_fn.die_after_calls = 2
    dist_fn._pair_cost_sink = dist_fn.sink.extend

    with pytest.raises(KeyboardInterrupt):
        dist_fn.batch(_PAIRS)

    assert 0 < len(dist_fn.sink) < len(_STRING_PAIRS)
    assert len(set(dist_fn.sink)) == len(dist_fn.sink)


def test_checkpointed_costs_match_the_unchecked_ones():
    expected = _StubLogProb().batch(_PAIRS)

    dist_fn = _StubLogProb()
    dist_fn._pair_cost_sink = dist_fn.sink.extend
    assert dist_fn.batch(_PAIRS) == expected


def test_one_prompt_per_pair_measures_checkpoint_every_pair():
    dist_fn = _StubVerbalized()
    sink: list[tuple[str, str]] = []
    dist_fn._pair_cost_sink = sink.extend

    dist_fn.batch(_PAIRS)

    assert all(n <= dist_fn._CHECKPOINT_PROMPTS for n in dist_fn.infer_calls)
    assert sorted(sink) == sorted(_STRING_PAIRS)


def test_checkpointed_costs_round_trip_through_the_sidecar(results_dir):
    path = results_dir / "paircosts_run.jsonl.gz"
    dist_fn = _StubLogProb()
    with PairCostWriter(path) as writer:
        attach_writer({"logprob_qwen35": dist_fn}, writer)
        dist_fn.batch(_PAIRS)

    records = list(_iter_cost_records(path))
    assert sorted((r["x"], r["y"]) for r in records) == sorted(_STRING_PAIRS)
