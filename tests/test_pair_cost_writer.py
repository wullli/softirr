
from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance
from soft_irr.evaluation.distances.pair_cache import PairCostWriter, attach_writer, load_costs

PAIRS = [(frozenset({"a", "b"}), frozenset({"c", "d"})), (frozenset({"a"}), frozenset({"e"}))]


class _Fake(BipartiteMatchingDistance):
    is_pair_cacheable = True

    def __init__(self) -> None:
        super().__init__(model_name="fake")

    def _load(self) -> None:
        pass

    def _prepare(self, parsed: list[tuple[Any, Any, list[str], list[str]]], batch_size: int) -> None:
        pass

    def _cost(self, a: str, b: str) -> float:
        return 0.25 if a < b else 0.75


def _read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def test_every_freshly_computed_edge_cost_is_written(tmp_path):
    path = tmp_path / "paircosts_run.jsonl.gz"
    dist_fn = _Fake()
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)

    records = _read(path)
    assert {(r["x"], r["y"]) for r in records} == {
        ("a", "c"), ("a", "d"), ("a", "e"), ("b", "c"), ("b", "d"),
    }
    assert {r["measure"] for r in records} == {"nli_mednli"}
    assert all(r["reasoning"] is False for r in records)


def test_costs_already_in_the_cache_are_not_rewritten(tmp_path):
    path = tmp_path / "paircosts_run.jsonl.gz"
    dist_fn = _Fake()
    dist_fn._pair_cache = {("a", "c"): 0.1, ("a", "d"): 0.1, ("b", "c"): 0.1, ("b", "d"): 0.1}
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)

    assert {(r["x"], r["y"]) for r in _read(path)} == {("a", "e")}


def test_trivial_pairs_are_not_written(tmp_path):
    path = tmp_path / "paircosts_run.jsonl.gz"
    dist_fn = _Fake()
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch([(frozenset({"a"}), frozenset({"a"})), (frozenset(), frozenset({"b"}))])
    assert _read(path) == []


def test_entries_are_written_once_across_batches(tmp_path):
    path = tmp_path / "paircosts_run.jsonl.gz"
    dist_fn = _Fake()
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)
        dist_fn.batch(PAIRS)
    assert len(_read(path)) == 5


def test_a_rerun_that_computes_nothing_does_not_wipe_the_sidecar(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "paircosts_synthetic_icd11_set_random.jsonl.gz"

    cold = _Fake()
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": cold}, writer)
        cold.batch(PAIRS)
    first = load_costs(results, ["nli_mednli"])["nli_mednli"]
    assert len(first) == 5

    warm = _Fake()
    warm._pair_cache = dict(first)
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": warm}, writer)
        warm.batch(PAIRS)

    assert load_costs(results, ["nli_mednli"])["nli_mednli"] == first


def test_appended_runs_accumulate_with_the_newest_value_winning(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "paircosts_synthetic_icd11_set_random.jsonl.gz"
    for cost in (0.1, 0.9):
        dist_fn = _Fake()
        dist_fn._cost = lambda a, b, c=cost: c
        with PairCostWriter(path) as writer:
            attach_writer({"nli_mednli": dist_fn}, writer)
            dist_fn.batch([PAIRS[1]])
    assert load_costs(results, ["nli_mednli"])["nli_mednli"] == {("a", "e"): 0.9}


def test_uncacheable_measures_get_no_writer(tmp_path):
    from soft_irr.evaluation.distances.simple import exact_match_distance

    with PairCostWriter(tmp_path / "p.jsonl.gz") as writer:
        assert attach_writer({"exact": exact_match_distance}, writer) == 0


def test_a_written_sidecar_reloads_without_any_predictions_file(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    dist_fn = _Fake()
    with PairCostWriter(results / "paircosts_synthetic_icd11_set_random.jsonl.gz") as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)

    costs = load_costs(results, ["nli_mednli"])
    assert costs["nli_mednli"] == {
        ("a", "c"): 0.25, ("a", "d"): 0.25, ("a", "e"): 0.25, ("b", "c"): 0.25, ("b", "d"): 0.25,
    }


def test_reasoning_runs_are_kept_apart_in_the_sidecar(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    plain, reasoning = _Fake(), _Fake()
    reasoning.enable_reasoning = True
    with PairCostWriter(results / "paircosts_a.jsonl.gz") as w:
        attach_writer({"nli_mednli": plain}, w)
        plain.batch([PAIRS[1]])
    with PairCostWriter(results / "paircosts_b.jsonl.gz") as w:
        attach_writer({"nli_mednli": reasoning}, w)
        reasoning.batch([PAIRS[0]])

    assert set(load_costs(results, ["nli_mednli"], reasoning=False)["nli_mednli"]) == {("a", "e")}
    assert set(load_costs(results, ["nli_mednli"], reasoning=True)["nli_mednli"]) == {
        ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"),
    }


def test_a_none_path_disables_writing(tmp_path):
    dist_fn = _Fake()
    with PairCostWriter(None) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)
    assert list(tmp_path.iterdir()) == []


def test_a_truncated_sidecar_yields_its_complete_prefix(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    path = results / "paircosts_synthetic_icd11_set_random.jsonl.gz"
    dist_fn = _Fake()
    with PairCostWriter(path) as writer:
        attach_writer({"nli_mednli": dist_fn}, writer)
        dist_fn.batch(PAIRS)
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - 8)

    recovered = load_costs(results, ["nli_mednli"]).get("nli_mednli", {})
    assert 0 < len(recovered) <= 5


@pytest.mark.parametrize("module", ["synthetic_irr", "real_irr"])
def test_measures_are_wired_to_a_writer(module):
    import importlib

    source = importlib.import_module(f"soft_irr.experiments.{module}").__file__
    with open(source) as f:
        text = f.read()
    assert "attach_writer(" in text and "PairCostWriter(" in text
