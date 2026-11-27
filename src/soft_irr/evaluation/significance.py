from collections.abc import Sequence

import numpy as np
import pandas as pd

PAIRING_KEYS = ("dataset", "mode", "agreement_level", "seed")
# Columns that split one measure into separately tested variants (set-mode matching strategy,
# ablation arms). Present ones are tested per variant and reported next to "measure".
ARM_KEYS = ("matching", "chance_measure", "reasoning")
RESULT_COLUMNS = ["measure", "negative_mode", "n_pairs", "mean_delta", "p_value", "p_adjusted", "significant"]


def sign_flip_pvalue(diff, n_perm: int = 10_000, rng: np.random.Generator | None = None) -> float:
    diff = np.asarray(diff, dtype=float)
    rng = rng if rng is not None else np.random.default_rng()
    observed = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = np.abs((flips * diff).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (n_perm + 1))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def paired_deltas_vs_reference(
    frame: pd.DataFrame,
    metric: str = "mae_kappa",
    ref: str = "random",
    alpha: float = 0.05,
    n_perm: int = 10_000,
    seed: int = 0,
    keys: Sequence[str] = PAIRING_KEYS,
    arm_keys: Sequence[str] = ARM_KEYS,
) -> pd.DataFrame:
    """Sign-flip test of each negative mode against ``ref``, per measure (and per arm), Holm-corrected.

    Rows are paired on ``keys``. Each combination of measure and the ``arm_keys`` columns present in
    ``frame`` (e.g. hungarian vs. greedy matching) is tested separately; missing arm values count
    as their own arm.
    """
    keys = [k for k in keys if k in frame.columns]
    arms = [k for k in arm_keys if k in frame.columns and k not in keys]
    columns = [RESULT_COLUMNS[0], *arms, *RESULT_COLUMNS[1:]]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    group_keys = ["measure", *arms]
    if frame.duplicated([*keys, *group_keys, "negative_mode"]).any():
        raise ValueError(f"duplicate rows for pairing keys {keys} + {group_keys} + negative_mode")

    rng = np.random.default_rng(seed)
    rows = []
    for group, msub in frame.groupby(group_keys, dropna=False, sort=True):
        wide = msub.pivot(index=keys, columns="negative_mode", values=metric)
        if ref not in wide.columns:
            continue
        for neg_mode in wide.columns.drop(ref):
            pair = wide[[ref, neg_mode]].dropna()
            if len(pair) < 2:
                continue
            diff = (pair[neg_mode] - pair[ref]).to_numpy()
            rows.append({
                **dict(zip(group_keys, group)),
                "negative_mode": neg_mode,
                "n_pairs": len(diff),
                "mean_delta": diff.mean(),
                "p_value": sign_flip_pvalue(diff, n_perm, rng),
            })

    table = pd.DataFrame(rows, columns=columns[:-2])
    table["p_adjusted"] = holm(table["p_value"])
    table["significant"] = table["p_adjusted"] < alpha
    return table
