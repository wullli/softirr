
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import LinAlgError

from soft_irr.evaluation.distribution_irr import DoDa

DEFAULT_SIGMA_THRESHOLD = 0.05
DEFAULT_MAX_EXPECTED_SAMPLE = 100_000


def expand_weighted_sample(
    values: ArrayLike, weights: Sequence[float] | np.ndarray, cap: int = DEFAULT_MAX_EXPECTED_SAMPLE
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    if len(values) == 0:
        return values

    counts = np.rint(weights / weights.min()).astype(int)
    total = int(counts.sum())
    if total > cap:
        counts = np.maximum(1, np.rint(counts * cap / total).astype(int))
    return np.repeat(values, counts)


def sigma_and_ks(
    observed: ArrayLike,
    expected: ArrayLike,
    weights: Sequence[float] | np.ndarray | None = None,
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
    max_expected_sample: int = DEFAULT_MAX_EXPECTED_SAMPLE,
) -> tuple[float, float]:
    observed = np.asarray(observed, dtype=float)
    if weights is not None:
        expected = expand_weighted_sample(expected, weights, cap=max_expected_sample)
    else:
        expected = np.asarray(expected, dtype=float)

    doda = DoDa(observed, expected)
    try:
        sigma = doda.get_sigma(thresh=sigma_threshold)
    except (LinAlgError, ValueError):
        sigma = doda.get_sigma(thresh=sigma_threshold, use_kde=False)
    return sigma, doda.get_ks()
