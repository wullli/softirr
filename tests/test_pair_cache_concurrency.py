
from __future__ import annotations

import csv
import gzip
import json

import pytest

from soft_irr.evaluation.distances.pair_cache import load_costs

_KEY = {
    "dataset": "icd11", "mode": "set", "negative_mode": "random",
    "agreement_level": 0.5, "seed": 52,
}
_TAG = "synthetic_icd11_set_random"


def _record(sample_index, rater_a, rater_b, cost):
    return {**_KEY, "measure": "nli_mednli", "sample_index": sample_index, "cost": cost}


@pytest.fixture()
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def _write_predictions(results_dir, rows, truncate_bytes=0):
    path = results_dir / f"predictions_{_TAG}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[*_KEY, "measure", "reasoning", "sample_index", "rater_a", "rater_b"]
        )
        writer.writeheader()
        for i, (a, b) in enumerate(rows):
            writer.writerow({
                **_KEY, "measure": "nli_mednli", "reasoning": False, "sample_index": i,
                "rater_a": json.dumps(a), "rater_b": json.dumps(b),
            })
    if truncate_bytes:
        with open(path, "r+b") as f:
            f.truncate(path.stat().st_size - truncate_bytes)
    return path


def _write_costs(results_dir, records, truncate_bytes=0):
    path = results_dir / f"costs_{_TAG}.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    if truncate_bytes:
        with open(path, "r+b") as f:
            f.truncate(path.stat().st_size - truncate_bytes)
    return path


def test_a_cost_file_still_being_written_yields_its_complete_prefix(results_dir):
    _write_predictions(results_dir, [(["a"], ["c"]), (["b"], ["d"])])
    _write_costs(results_dir, [_record(0, ["a"], ["c"], [[0.1]]), _record(1, ["b"], ["d"], [[0.2]])],
                 truncate_bytes=12)

    costs = load_costs(results_dir)
    assert costs.get("nli_mednli", {}).get(("a", "c")) == 0.1


def test_a_truncated_predictions_row_does_not_fail_the_load(results_dir):
    _write_predictions(results_dir, [(["a"], ["c"]), (["b"], ["d"])], truncate_bytes=10)
    _write_costs(results_dir, [_record(0, ["a"], ["c"], [[0.1]]), _record(1, ["b"], ["d"], [[0.2]])])

    stats: dict[str, int] = {}
    costs = load_costs(results_dir, stats=stats)
    assert costs.get("nli_mednli", {}).get(("a", "c")) == 0.1
    assert stats["unreadable"] == 0


def test_costs_written_before_their_predictions_row_are_counted_not_guessed(results_dir):
    _write_predictions(results_dir, [(["a"], ["c"])])
    _write_costs(results_dir, [_record(0, ["a"], ["c"], [[0.1]]), _record(1, ["b"], ["d"], [[0.2]])])

    stats: dict[str, int] = {}
    costs = load_costs(results_dir, stats=stats)
    assert costs["nli_mednli"] == {("a", "c"): 0.1}
    assert stats["unmatched"] == 1


def test_an_empty_half_created_cost_file_is_skipped(results_dir):
    _write_predictions(results_dir, [(["a"], ["c"])])
    (results_dir / f"costs_{_TAG}.jsonl.gz").write_bytes(b"")
    assert load_costs(results_dir) == {}
