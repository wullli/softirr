
from __future__ import annotations

import math

import pytest

from soft_irr.evaluation.distances import exact_match_distance
from soft_irr.experiments.real_irr import _RESULT_FIELDS, run_experiment


@pytest.fixture
def rows_by_hash() -> dict[str, list[tuple[str, frozenset[str], str]]]:
    return {
        "img1": [
            ("red scaly plaque", frozenset({"psoriasis"}), "srcA"),
            ("scaly red plaque", frozenset({"psoriasis"}), "srcB"),
            ("itchy patch", frozenset({"eczema"}), "srcC"),
        ],
        "img2": [
            ("pigmented lesion", frozenset({"melanoma"}), "srcA"),
            ("dark mole", frozenset({"nevus"}), "srcB"),
        ],
        "img3": [
            ("blistering rash", frozenset({"eczema"}), "srcA"),
            ("blistering rash", frozenset({"eczema"}), "srcB"),
        ],
    }


def test_result_fields_cover_sigma_and_ks():
    for field in ("true_sigma", "true_ks", "sigma", "ks", "mae_sigma", "mape_sigma", "mae_ks", "mape_ks"):
        assert field in _RESULT_FIELDS


def test_run_experiment_reports_finite_sigma_and_ks(rows_by_hash):
    rows, _ = run_experiment(
        rows_by_hash=rows_by_hash,
        category_field="disease_label",
        measures={"exact": exact_match_distance},
        dataset="derm1",
    )

    assert rows
    for row in rows:
        assert set(row) == set(_RESULT_FIELDS)
        for field in ("true_sigma", "true_ks", "sigma", "ks"):
            assert math.isfinite(row[field]), field
            assert 0.0 <= row[field] <= 1.0
        assert row["mae_sigma"] == pytest.approx(abs(row["sigma"] - row["true_sigma"]))
        assert row["mae_ks"] == pytest.approx(abs(row["ks"] - row["true_ks"]))


def test_chance_writer_records_the_pooled_chance_sample(rows_by_hash, tmp_path):
    import csv

    from soft_irr.experiments.real_irr import _CHANCE_FIELDS

    path = tmp_path / "chance.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CHANCE_FIELDS)
        writer.writeheader()
        run_experiment(
            rows_by_hash=rows_by_hash,
            category_field="disease_label",
            measures={"exact": exact_match_distance},
            dataset="derm1",
            chance_writer=writer,
        )

    with open(path, newline="", encoding="utf-8") as f:
        chance_rows = list(csv.DictReader(f))

    assert chance_rows
    assert set(chance_rows[0]) == set(_CHANCE_FIELDS)
    assert {r["phash"] for r in chance_rows} <= set(rows_by_hash)
    assert sum(float(r["weight"]) for r in chance_rows) == pytest.approx(1.0)
    for row in chance_rows:
        assert 0.0 <= float(row["distance"]) <= 1.0
