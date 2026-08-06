
from __future__ import annotations

import math

import pytest

from soft_irr.experiments.synthetic_irr import (
    _significance_rows_for_metric,
    _significance_vs_best,
)

MAE = "mae_avg_agreement"


def _rows(measure_errors: dict[str, dict[int, float]]) -> list[dict]:
    rows = []
    for measure, errors in measure_errors.items():
        for seed, err in errors.items():
            rows.append({"measure": measure, "seed": seed, MAE: err})
    return rows


def test_best_measure_is_lowest_mean_error():
    rows = _rows({
        "good": {1: 0.1, 2: 0.1, 3: 0.1},
        "bad": {1: 0.9, 2: 0.9, 3: 0.9},
    })
    best, best_mean, results = _significance_vs_best(rows, MAE)
    assert best == "good"
    assert best_mean == pytest.approx(0.1)
    assert [name for name, _, _ in results] == ["bad"]


def test_best_measure_excluded_from_results():
    rows = _rows({
        "good": {1: 0.1, 2: 0.1},
        "mid": {1: 0.5, 2: 0.5},
        "bad": {1: 0.9, 2: 0.9},
    })
    _, _, results = _significance_vs_best(rows, MAE)
    names = [name for name, _, _ in results]
    assert "good" not in names
    assert set(names) == {"mid", "bad"}


def test_identical_distributions_yield_p_value_one():
    rows = _rows({
        "best": {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8},
        "same": {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8},
    })
    _, _, results = _significance_vs_best(rows, MAE, num_rounds=1000, seed=0)
    [(name, mean_err, p_value)] = results
    assert name == "same"
    assert mean_err == results[0][1]
    assert p_value == 1.0


def test_clearly_different_distributions_are_significant():
    rows = _rows({
        "best": {i: 0.05 for i in range(20)},
        "worse": {i: 0.95 for i in range(20)},
    })
    _, _, results = _significance_vs_best(rows, MAE, num_rounds=1000, seed=0)
    [(name, _, p_value)] = results
    assert name == "worse"
    assert p_value < 0.05


def test_pairing_uses_shared_seeds_not_row_order():
    rows = _rows({
        "best": {1: 0.1, 2: 0.2, 3: 0.3},
        "other": {3: 0.9, 1: 0.1, 5: 0.5},
    })
    _, _, results = _significance_vs_best(rows, MAE, num_rounds=200, seed=0)
    [(name, mean_err, p_value)] = results
    assert name == "other"
    assert math.isclose(mean_err, (0.9 + 0.1 + 0.5) / 3)
    assert not math.isnan(p_value)


def test_fewer_than_two_shared_populations_yields_nan_p_value():
    rows = _rows({
        "best": {1: 0.1, 2: 0.2, 3: 0.3},
        "sparse": {1: 0.4},
    })
    _, _, results = _significance_vs_best(rows, MAE)
    [(name, _, p_value)] = results
    assert name == "sparse"
    assert math.isnan(p_value)


def test_no_shared_populations_yields_nan_p_value():
    rows = _rows({
        "best": {1: 0.1, 2: 0.2},
        "disjoint": {3: 0.4, 4: 0.5},
    })
    _, _, results = _significance_vs_best(rows, MAE)
    [(name, _, p_value)] = results
    assert name == "disjoint"
    assert math.isnan(p_value)


def test_deterministic_with_same_seed():
    rows = _rows({
        "best": {i: 0.1 + 0.01 * i for i in range(15)},
        "other": {i: 0.3 + 0.02 * i for i in range(15)},
    })
    _, _, results_1 = _significance_vs_best(rows, MAE, num_rounds=500, seed=7)
    _, _, results_2 = _significance_vs_best(rows, MAE, num_rounds=500, seed=7)
    assert results_1 == results_2


def test_single_measure_has_no_results():
    rows = _rows({"only": {1: 0.1, 2: 0.2}})
    best, best_mean, results = _significance_vs_best(rows, MAE)
    assert best == "only"
    assert best_mean == pytest.approx(0.15)
    assert results == []


def test_significance_rows_for_metric_prints_and_returns_csv_rows(capsys):
    rows = _rows({
        "best": {i: 0.05 for i in range(20)},
        "worse": {i: 0.95 for i in range(20)},
    })
    sig_rows = _significance_rows_for_metric(
        rows, dataset="meddra", mode="set", negative_mode="random",
        metric_label="avg_agreement", mae_field=MAE,
    )
    out = capsys.readouterr().out
    assert "best measure 'best'" in out
    assert "worse" in out
    assert "significant" in out

    assert len(sig_rows) == 2
    best_row = next(r for r in sig_rows if r["is_best"])
    worse_row = next(r for r in sig_rows if not r["is_best"])
    assert best_row["measure"] == "best"
    assert best_row["best_measure"] == "best"
    assert math.isnan(best_row["p_value"])
    assert best_row["significant"] is False
    assert worse_row["measure"] == "worse"
    assert worse_row["best_measure"] == "best"
    assert worse_row["p_value"] < 0.05
    assert worse_row["significant"] is True
    for row in sig_rows:
        assert row["dataset"] == "meddra"
        assert row["mode"] == "set"
        assert row["negative_mode"] == "random"
        assert row["metric"] == "avg_agreement"
