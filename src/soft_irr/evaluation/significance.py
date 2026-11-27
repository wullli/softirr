from collections.abc import Sequence

import numpy as np
import pandas as pd

PAIRING_KEYS = ("dataset", "mode", "agreement_level", "seed")
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
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    keys = [k for k in keys if k in frame.columns]
    if frame.duplicated([*keys, "measure", "negative_mode"]).any():
        raise ValueError(f"duplicate rows for pairing keys {keys} + measure + negative_mode")

    rng = np.random.default_rng(seed)
    rows = []
    for measure, msub in frame.groupby("measure"):
        wide = msub.pivot(index=keys, columns="negative_mode", values=metric)
        if ref not in wide.columns:
            continue
        for neg_mode in wide.columns.drop(ref):
            pair = wide[[ref, neg_mode]].dropna()
            if len(pair) < 2:
                continue
            diff = (pair[neg_mode] - pair[ref]).to_numpy()
            rows.append({
                "measure": measure,
                "negative_mode": neg_mode,
                "n_pairs": len(diff),
                "mean_delta": diff.mean(),
                "p_value": sign_flip_pvalue(diff, n_perm, rng),
            })

    table = pd.DataFrame(rows, columns=RESULT_COLUMNS[:5])
    table["p_adjusted"] = holm(table["p_value"])
    table["significant"] = table["p_adjusted"] < alpha
    return table
