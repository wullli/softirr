
from __future__ import annotations

import csv
import gzip
import json
import os

import pytest

from soft_irr.evaluation.distances.nli import BipartiteNLIDistance
from soft_irr.evaluation.distances.pair_cache import attach, load_costs
from soft_irr.evaluation.distances.simple import exact_match_distance
from soft_irr.evaluation.distances.single_wrapper import _SingleBipartiteWrapper

_KEY = {
    "dataset": "icd11",
    "mode": "set",
    "negative_mode": "random",
    "agreement_level": 0.5,
    "seed": 52,
    "sample_index": 0,
}


def _write_run(results_dir, suffix, measure, rater_a, rater_b, cost, reasoning=False, mtime=None):
    tag = f"synthetic_icd11_set_random{suffix}"
    with gzip.open(results_dir / f"costs_{tag}.jsonl.gz", "wt") as f:
        f.write(json.dumps({**_KEY, "measure": measure, "cost": cost}) + "\n")
    with open(results_dir / f"predictions_{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*_KEY, "measure", "reasoning", "rater_a", "rater_b"])
        writer.writeheader()
        writer.writerow({
            **_KEY, "measure": measure, "reasoning": reasoning,
            "rater_a": json.dumps(rater_a), "rater_b": json.dumps(rater_b),
        })
    if mtime is not None:
        for name in (f"costs_{tag}.jsonl.gz", f"predictions_{tag}.csv"):
            os.utime(results_dir / name, (mtime, mtime))


@pytest.fixture()
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def test_matrix_cells_map_to_their_string_pairs(results_dir):
    _write_run(results_dir, "", "nli_mednli", ["a", "b"], ["c", "d"], [[0.1, 0.2], [0.3, 0.4]])
    stats: dict[str, int] = {}
    costs = load_costs(results_dir, stats=stats)

    assert costs["nli_mednli"] == {
        ("a", "c"): 0.1, ("a", "d"): 0.2, ("b", "c"): 0.3, ("b", "d"): 0.4,
    }
    assert stats["unmatched"] == 0
    assert stats["shape_mismatch"] == 0


def test_lookup_is_symmetric(results_dir):
    _write_run(results_dir, "", "nli_mednli", ["z"], ["a"], [[0.25]])
    assert load_costs(results_dir)["nli_mednli"] == {("a", "z"): 0.25}


def test_single_wrapper_measures_share_the_base_entry(results_dir):
    _write_run(results_dir, "", "nli_mednli_single", ["a"], ["c"], [[0.6]])
    assert load_costs(results_dir) == {"nli_mednli": {("a", "c"): 0.6}}


def test_uncacheable_measures_are_skipped(results_dir):
    _write_run(results_dir, "", "levenshtein_bipartite", ["a"], ["c"], [[0.6]])
    _write_run(results_dir, "_draft", "embedding_qwen3", ["a"], ["e"], [[0.6]])
    assert load_costs(results_dir) == {}


def test_shape_mismatch_is_skipped_not_mis_stored(results_dir):
    _write_run(results_dir, "", "nli_mednli", ["a", "b"], ["c"], [[0.1, 0.2]])
    stats: dict[str, int] = {}
    assert load_costs(results_dir, stats=stats) == {}
    assert stats["shape_mismatch"] == 1


def test_newer_files_win_over_older_ones(results_dir):
    _write_run(results_dir, "", "nli_mednli", ["a"], ["c"], [[0.1]], mtime=1_000_000)
    _write_run(results_dir, "_draft", "nli_mednli", ["a"], ["c"], [[0.9]], mtime=2_000_000)
    assert load_costs(results_dir)["nli_mednli"] == {("a", "c"): 0.9}


def test_reasoning_runs_do_not_reuse_non_reasoning_costs(results_dir):
    _write_run(results_dir, "", "logprob_qwen35", ["a"], ["c"], [[0.1]], reasoning=False)
    _write_run(results_dir, "_reasoning", "logprob_qwen35", ["a"], ["e"], [[0.9]], reasoning=True)
    assert load_costs(results_dir, reasoning=False) == {"logprob_qwen35": {("a", "c"): 0.1}}
    assert load_costs(results_dir, reasoning=True) == {"logprob_qwen35": {("a", "e"): 0.9}}


def test_cost_file_without_predictions_is_reported(results_dir):
    _write_run(results_dir, "", "nli_mednli", ["a"], ["c"], [[0.1]])
    (results_dir / "predictions_synthetic_icd11_set_random.csv").unlink()
    stats: dict[str, int] = {}
    assert load_costs(results_dir, stats=stats) == {}
    assert stats["missing_predictions"] == 1


def test_empty_results_directory_is_not_an_error(results_dir):
    assert load_costs(results_dir) == {}


def test_attach_unwraps_single_wrappers_and_skips_uncacheable_measures():
    nli = BipartiteNLIDistance()
    entries = {("a", "c"): 0.5}
    measures = {
        "nli_mednli": nli,
        "nli_mednli_single": _SingleBipartiteWrapper(nli),
        "exact": exact_match_distance,
    }
    assert attach(measures, {"nli_mednli": entries}) == 1
    assert nli._pair_cache is entries
    assert not hasattr(exact_match_distance, "_pair_cache")


def test_attach_leaves_measures_with_no_recovered_costs_alone():
    nli = BipartiteNLIDistance()
    assert attach({"nli_mednli": nli}, {"logprob_qwen35": {("a", "c"): 0.5}}) == 0
    assert nli._pair_cache is None


@pytest.mark.parametrize("module", ["synthetic_irr", "real_irr"])
def test_no_cache_flag_turns_the_cache_off(monkeypatch, module):
    import importlib
    import sys

    parse_args = importlib.import_module(f"soft_irr.experiments.{module}")._parse_args
    base = ["--dataset", "icd11"] if module == "synthetic_irr" else ["--dataset", "reflacx"]

    monkeypatch.setattr(sys, "argv", [module, *base])
    assert parse_args().use_pair_cache is True
    monkeypatch.setattr(sys, "argv", [module, *base, "--no-cache"])
    assert parse_args().use_pair_cache is False
