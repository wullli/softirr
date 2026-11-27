import numpy as np
import pandas as pd
import pytest

from soft_irr.evaluation.significance import (
    holm,
    paired_deltas_vs_reference,
    sign_flip_pvalue,
)


def test_sign_flip_zero_differences_give_p_one():
    assert sign_flip_pvalue(np.zeros(8), n_perm=500, rng=np.random.default_rng(0)) == 1.0


def test_sign_flip_consistent_shift_is_significant():
    p = sign_flip_pvalue(np.full(12, 0.2), n_perm=5_000, rng=np.random.default_rng(0))
    assert p < 0.01


def test_sign_flip_symmetric_noise_is_not_significant():
    diff = np.array([0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.3, -0.3])
    p = sign_flip_pvalue(diff, n_perm=5_000, rng=np.random.default_rng(0))
    assert p > 0.5


def test_sign_flip_p_is_never_below_monte_carlo_floor():
    p = sign_flip_pvalue(np.full(30, 1.0), n_perm=99, rng=np.random.default_rng(0))
    assert p >= 1 / 100


def test_sign_flip_is_reproducible_with_seed():
    diff = np.array([0.1, 0.3, -0.05, 0.2, 0.15])
    a = sign_flip_pvalue(diff, n_perm=1_000, rng=np.random.default_rng(3))
    b = sign_flip_pvalue(diff, n_perm=1_000, rng=np.random.default_rng(3))
    assert a == b


def test_holm_known_values():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_clips_at_one():
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


def _frame(n_seeds=12, shift=0.2):
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(n_seeds):
        base_a = 0.3 + rng.normal(0, 0.02)
        base_b = 0.4 + rng.normal(0, 0.02)
        noise = 0.05 if seed % 2 else -0.05
        rows += [
            {"dataset": "icd11", "mode": "single", "agreement_level": 0.5, "seed": seed,
             "measure": "a", "negative_mode": "random", "mae_kappa": base_a},
            {"dataset": "icd11", "mode": "single", "agreement_level": 0.5, "seed": seed,
             "measure": "a", "negative_mode": "sibling", "mae_kappa": base_a + shift},
            {"dataset": "icd11", "mode": "single", "agreement_level": 0.5, "seed": seed,
             "measure": "b", "negative_mode": "random", "mae_kappa": base_b},
            {"dataset": "icd11", "mode": "single", "agreement_level": 0.5, "seed": seed,
             "measure": "b", "negative_mode": "sibling", "mae_kappa": base_b + noise},
        ]
    return pd.DataFrame(rows)


def test_paired_deltas_flags_only_the_shifted_measure():
    table = paired_deltas_vs_reference(_frame(), n_perm=2_000)
    flagged = set(zip(table.loc[table["significant"], "measure"], table.loc[table["significant"], "negative_mode"]))
    assert flagged == {("a", "sibling")}


def test_paired_deltas_reports_expected_columns_and_values():
    table = paired_deltas_vs_reference(_frame(), n_perm=2_000)
    assert list(table.columns) == [
        "measure", "negative_mode", "n_pairs", "mean_delta", "p_value", "p_adjusted", "significant"
    ]
    row = table.set_index(["measure", "negative_mode"]).loc[("a", "sibling")]
    assert row["n_pairs"] == 12
    assert row["mean_delta"] == pytest.approx(0.2)


def test_paired_deltas_excludes_reference_rows():
    table = paired_deltas_vs_reference(_frame(), n_perm=200)
    assert "random" not in set(table["negative_mode"])


def test_paired_deltas_applies_holm_across_the_whole_frame():
    table = paired_deltas_vs_reference(_frame(), n_perm=2_000)
    np.testing.assert_allclose(table["p_adjusted"], holm(table["p_value"]))


def test_paired_deltas_rejects_duplicate_pairing_keys():
    frame = _frame()
    with pytest.raises(ValueError, match="duplicate"):
        paired_deltas_vs_reference(pd.concat([frame, frame.iloc[:1]]))


def test_paired_deltas_tests_each_matching_separately():
    hungarian = _frame().assign(matching="hungarian")
    greedy = _frame(shift=0.0).assign(matching="greedy")
    table = paired_deltas_vs_reference(pd.concat([hungarian, greedy], ignore_index=True), n_perm=2_000)
    assert list(table.columns) == [
        "measure", "matching", "negative_mode", "n_pairs", "mean_delta", "p_value", "p_adjusted", "significant"
    ]
    assert len(table) == 4
    assert (table["n_pairs"] == 12).all()
    flagged = set(zip(table.loc[table["significant"], "measure"], table.loc[table["significant"], "matching"]))
    assert flagged == {("a", "hungarian")}


def test_paired_deltas_keeps_rows_with_missing_arm_values():
    single = _frame().assign(matching=np.nan, reasoning=False)
    table = paired_deltas_vs_reference(single, n_perm=2_000)
    assert len(table) == 2
    assert table["matching"].isna().all()
    flagged = set(zip(table.loc[table["significant"], "measure"], table.loc[table["significant"], "negative_mode"]))
    assert flagged == {("a", "sibling")}


def test_paired_deltas_constant_arm_matches_no_arm_p_values():
    plain = paired_deltas_vs_reference(_frame(), n_perm=500, seed=3)
    with_arm = paired_deltas_vs_reference(_frame().assign(matching="hungarian"), n_perm=500, seed=3)
    pd.testing.assert_frame_equal(plain, with_arm.drop(columns="matching"))


def test_paired_deltas_skips_contrasts_with_fewer_than_two_pairs():
    table = paired_deltas_vs_reference(_frame(n_seeds=1), n_perm=200)
    assert table.empty


def test_paired_deltas_without_reference_returns_empty_table():
    frame = _frame()
    table = paired_deltas_vs_reference(frame[frame["negative_mode"] != "random"], n_perm=200)
    assert table.empty
    assert "significant" in table.columns


def test_paired_deltas_on_empty_frame_returns_empty_table():
    assert paired_deltas_vs_reference(pd.DataFrame()).empty


def test_paired_deltas_is_reproducible_with_seed():
    a = paired_deltas_vs_reference(_frame(), n_perm=500, seed=7)
    b = paired_deltas_vs_reference(_frame(), n_perm=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
