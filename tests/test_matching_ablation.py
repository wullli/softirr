
from __future__ import annotations

import gzip
import json
import random

import pytest

from soft_irr.experiments.matching_ablation import (
    MATRIX_KEY,
    greedy_assignment,
    hungarian_assignment,
    is_decisive,
    load_matrices,
    load_true_pairs,
    run,
    score,
)

IDENTITY = ([0, 1], [0, 1])


class TestAssignments:
    def test_greedy_matches_the_measures_own_implementation(self):
        from soft_irr.evaluation.distances.base import BipartiteMatchingState

        rng = random.Random(0)
        for _ in range(200):
            n_cols = rng.randint(2, 5)
            cost = [[round(rng.random(), 3) for _ in range(n_cols)] for _ in range(rng.randint(2, 5))]
            assert set(greedy_assignment(cost)) == set(BipartiteMatchingState._greedy_assignment(cost))

    def test_hungarian_never_costs_more_than_greedy(self):
        rng = random.Random(1)
        for _ in range(300):
            n_cols = rng.randint(2, 6)
            cost = [[rng.random() for _ in range(n_cols)] for _ in range(rng.randint(2, 6))]
            h = sum(cost[i][j] for i, j in hungarian_assignment(cost))
            g = sum(cost[i][j] for i, j in greedy_assignment(cost))
            assert h <= g + 1e-12

    def test_greedy_can_be_beaten_by_hungarian(self):
        cost = [[0.0, 0.4], [0.1, 1.0]]
        assert set(greedy_assignment(cost)) == {(0, 0), (1, 1)}
        assert set(hungarian_assignment(cost)) == {(0, 1), (1, 0)}


class TestScore:
    def test_recall_counts_recovered_true_pairs(self):
        cost = [[0.0, 0.9], [0.9, 0.0]]
        out = score(cost, [(0, 0), (1, 1)], *IDENTITY)
        assert out == {"n_true": 2, "recall_hungarian": 1.0, "recall_greedy": 1.0}

    def test_recall_is_zero_when_the_measure_prefers_the_wrong_pairing(self):
        cost = [[0.9, 0.1], [0.2, 0.9]]
        out = score(cost, [(0, 0), (1, 1)], *IDENTITY)
        assert out["recall_hungarian"] == 0.0
        assert out["recall_greedy"] == 0.0

    def test_partial_truth_scores_only_the_pairs_that_exist(self):
        cost = [[0.0, 0.9], [0.9, 0.5]]
        assert score(cost, [(0, 0)], *IDENTITY)["recall_hungarian"] == 1.0

    def test_permuting_a_matrix_with_one_optimum_changes_nothing(self):
        cost = [[0.0, 0.9, 0.9], [0.9, 0.0, 0.9], [0.9, 0.9, 0.0]]
        truth = [(0, 0), (1, 1), (2, 2)]
        rng = random.Random(0)
        for _ in range(50):
            rows = rng.sample(range(3), 3)
            cols = rng.sample(range(3), 3)
            assert score(cost, truth, rows, cols) == score(cost, truth, [0, 1, 2], [0, 1, 2])

    def test_permuting_a_tied_matrix_moves_the_answer(self):
        cost = [[0.5, 0.5], [0.5, 0.5]]
        truth = [(0, 0), (1, 1)]
        seen = {
            score(cost, truth, rows, cols)["recall_hungarian"]
            for rows in ([0, 1], [1, 0]) for cols in ([0, 1], [1, 0])
        }
        assert len(seen) > 1

    def test_pairs_are_reported_in_original_indices(self):
        cost = [[0.0, 0.9], [0.9, 0.0]]
        assert score(cost, [(0, 0), (1, 1)], [1, 0], [1, 0])["recall_hungarian"] == 1.0

    def test_handles_rectangular_matrices(self):
        cost = [[0.0, 0.9, 0.9], [0.9, 0.9, 0.0]]
        assert score(cost, [(0, 0), (1, 2)], [0, 1], [0, 1, 2])["recall_hungarian"] == 1.0


def _write_run(tmp_path, records, predictions):
    costs = tmp_path / "costs_synthetic_icd11_set_random_uniform.jsonl.gz"
    with gzip.open(costs, "wt") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    header = ",".join(MATRIX_KEY) + ",true_assignment\n"
    body = "".join(
        ",".join(str(row[field]) for field in MATRIX_KEY) + f',"{json.dumps(row["true"])}"\n'
        for row in predictions
    )
    (tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv").write_text(header + body)
    return costs


def _record(sample_index, cost, measure="m"):
    return {"dataset": "icd11", "negative_mode": "random", "measure": measure,
            "agreement_level": "0.5", "seed": "1", "sample_index": sample_index, "cost": cost}


def _prediction(sample_index, true, measure="m"):
    return {"dataset": "icd11", "negative_mode": "random", "measure": measure,
            "agreement_level": "0.5", "seed": "1", "sample_index": sample_index, "true": true}


class TestLoading:
    def test_true_pairs_are_keyed_by_matrix(self, tmp_path):
        _write_run(tmp_path, [], [_prediction(0, [[0, 0], [1, 1]])])
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        assert truths[("icd11", "random", "m", "0.5", "1", "0")] == [(0, 0), (1, 1)]

    def test_matrices_are_paired_with_their_own_truth(self, tmp_path):
        costs = _write_run(
            tmp_path,
            [_record(0, [[0.1, 0.9], [0.9, 0.1]]), _record(1, [[0.2, 0.8], [0.8, 0.2]])],
            [_prediction(0, [[0, 0]]), _prediction(1, [[1, 1]])],
        )
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        loaded = {key[-1]: (cost, true) for key, cost, true in load_matrices(costs, truths)}
        assert loaded["0"][1] == [(0, 0)]
        assert loaded["1"][0] == [[0.2, 0.8], [0.8, 0.2]]

    def test_short_circuited_samples_have_no_matrix_and_are_skipped(self, tmp_path):
        costs = _write_run(tmp_path, [_record(0, None), _record(1, [])], [_prediction(0, [[0, 0]])])
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        assert list(load_matrices(costs, truths)) == []

    def test_samples_without_true_pairs_are_skipped(self, tmp_path):
        costs = _write_run(tmp_path, [_record(0, [[0.1, 0.9], [0.9, 0.1]])], [_prediction(0, [])])
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        assert list(load_matrices(costs, truths)) == []

    def test_measure_filter(self, tmp_path):
        costs = _write_run(
            tmp_path,
            [_record(0, [[0.1, 0.9], [0.9, 0.1]], "a"), _record(1, [[0.1, 0.9], [0.9, 0.1]], "b")],
            [_prediction(0, [[0, 0]], "a"), _prediction(1, [[0, 0]], "b")],
        )
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        assert [key[2] for key, _, _ in load_matrices(costs, truths, ["a"])] == ["a"]

    def test_torn_final_record_stops_the_read_without_raising(self, tmp_path):
        costs = _write_run(tmp_path, [], [_prediction(0, [[0, 0]])])
        with gzip.open(costs, "wt") as fh:
            fh.write(json.dumps(_record(0, [[0.1, 0.9], [0.9, 0.1]])) + "\n")
            fh.write('{"measure": "m", "cost"')
        truths = load_true_pairs(tmp_path / "predictions_synthetic_icd11_set_random_uniform.csv")
        assert len(list(load_matrices(costs, truths))) == 1


class TestRun:
    def _setup(self, tmp_path, cost):
        _write_run(tmp_path, [_record(0, cost)], [_prediction(0, [[0, 0], [1, 1]])])

    def test_emits_one_row_per_matrix_and_permutation(self, tmp_path):
        self._setup(tmp_path, [[0.1, 0.9], [0.9, 0.1]])
        rows = run(tmp_path, permutations=5)
        assert len(rows) == 5
        assert [row["permutation"] for row in rows] == [0, 1, 2, 3, 4]

    def test_permutation_zero_is_the_runs_own_layout(self, tmp_path):
        cost = [[0.5, 0.5], [0.5, 0.5]]
        self._setup(tmp_path, cost)
        rows = run(tmp_path, permutations=1)
        assert rows[0]["recall_hungarian"] == score(cost, [(0, 0), (1, 1)], *IDENTITY)["recall_hungarian"]

    def test_rows_carry_the_matrix_key(self, tmp_path):
        self._setup(tmp_path, [[0.1, 0.9], [0.9, 0.1]])
        row = run(tmp_path, permutations=1)[0]
        assert all(field in row for field in MATRIX_KEY)
        assert row["dataset"] == "icd11" and row["negative_mode"] == "random"

    def test_is_deterministic_under_a_fixed_seed(self, tmp_path):
        self._setup(tmp_path, [[0.5, 0.5], [0.5, 0.5]])
        assert run(tmp_path, permutations=8, seed=3) == run(tmp_path, permutations=8, seed=3)

    def test_dataset_filter(self, tmp_path):
        self._setup(tmp_path, [[0.1, 0.9], [0.9, 0.1]])
        assert run(tmp_path, 1, datasets=["mesh"]) == []
        assert run(tmp_path, 1, datasets=["icd11"])

    def test_runs_without_a_predictions_table_are_skipped(self, tmp_path):
        with gzip.open(tmp_path / "costs_synthetic_icd11_set_random_uniform.jsonl.gz", "wt") as fh:
            fh.write(json.dumps(_record(0, [[0.1, 0.9], [0.9, 0.1]])) + "\n")
        assert run(tmp_path, permutations=1) == []

    def test_permutations_reveal_instability_on_a_tied_matrix(self, tmp_path):
        self._setup(tmp_path, [[0.5, 0.5], [0.5, 0.5]])
        rows = run(tmp_path, permutations=40, seed=0)
        assert len({row["recall_hungarian"] for row in rows}) > 1

    def test_permutations_leave_a_decisive_matrix_alone(self, tmp_path):
        self._setup(tmp_path, [[0.0, 0.9], [0.9, 0.0]])
        rows = run(tmp_path, permutations=40, seed=0)
        assert {row["recall_hungarian"] for row in rows} == {1.0}
        assert {row["recall_greedy"] for row in rows} == {1.0}


def test_recall_is_never_out_of_range(tmp_path):
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(2, 5)
        cost = [[round(rng.choice([0.0, 0.25, 0.5, 1.0]), 3) for _ in range(n)] for _ in range(n)]
        truth = [(i, i) for i in range(rng.randint(1, n))]
        out = score(cost, truth, rng.sample(range(n), n), rng.sample(range(n), n))
        assert 0.0 <= out["recall_hungarian"] <= 1.0
        assert 0.0 <= out["recall_greedy"] <= 1.0


@pytest.mark.parametrize("permutations", [1, 3])
def test_permutation_count_is_respected(tmp_path, permutations):
    _write_run(tmp_path, [_record(0, [[0.1, 0.9], [0.9, 0.1]])], [_prediction(0, [[0, 0]])])
    assert len(run(tmp_path, permutations=permutations)) == permutations


class TestDecisive:

    def test_distinct_costs_with_one_clear_optimum(self):
        assert is_decisive([[0.0, 0.9], [0.8, 0.1]])

    def test_all_equal_costs_are_not_decisive(self):
        assert not is_decisive([[0.5, 0.5], [0.5, 0.5]])

    def test_two_optimal_assignments_of_equal_cost(self):
        assert not is_decisive([[0.1, 0.3], [0.1, 0.3]])

    def test_greedy_tie_at_the_first_pick_is_not_decisive(self):
        cost = [[0.1, 0.1, 0.9], [0.9, 0.9, 0.2], [0.9, 0.9, 0.8]]
        assert not is_decisive(cost)

    def test_ties_greedy_never_has_to_choose_between_do_not_disqualify(self):
        assert is_decisive([[0.0, 0.9], [0.9, 0.9]])

    def test_agrees_with_permuting_the_layout(self, tmp_path):
        rng = random.Random(0)
        checked = 0
        for _ in range(300):
            n = rng.randint(2, 4)
            cost = [[rng.choice([0.0, 0.25, 0.5, 1.0]) for _ in range(n)] for _ in range(n)]
            truth = [(i, i) for i in range(n)]
            if not is_decisive(cost):
                continue
            checked += 1
            base = score(cost, truth, list(range(n)), list(range(n)))
            for _ in range(10):
                rows, cols = rng.sample(range(n), n), rng.sample(range(n), n)
                assert score(cost, truth, rows, cols) == base
        assert checked > 20

    def test_unstable_matrices_are_the_ones_permutation_moves(self, tmp_path):
        cost = [[0.5, 0.5], [0.5, 0.5]]
        _write_run(tmp_path, [_record(0, cost)], [_prediction(0, [[0, 0], [1, 1]])])
        rows = run(tmp_path, permutations=30)
        assert not any(row["decisive"] for row in rows)
        assert len({row["recall_hungarian"] for row in rows}) > 1

    def test_decisive_flag_is_written_per_row(self, tmp_path):
        _write_run(tmp_path, [_record(0, [[0.0, 0.9], [0.8, 0.1]])],
                   [_prediction(0, [[0, 0], [1, 1]])])
        rows = run(tmp_path, permutations=3)
        assert all(row["decisive"] for row in rows)
