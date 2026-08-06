
from unittest.mock import patch

import pytest

from soft_irr.evaluation.distances import (
    BipartiteJaccardDistance,
    BipartiteLCSseqDistance,
    BipartiteLevenshteinDistance,
    BipartiteNormalizedIndelDistance,
)

CLASSES = [
    BipartiteJaccardDistance,
    BipartiteLevenshteinDistance,
    BipartiteNormalizedIndelDistance,
    BipartiteLCSseqDistance,
]


@pytest.mark.parametrize("cls", CLASSES)
def test_edge_cost_computed_once_per_unique_pair(cls):
    dist = cls()
    with patch.object(dist, "_edge_cost", wraps=dist._edge_cost) as spy:
        dist.batch([(["fever", "cough"], ["high temperature", "coughing"])])
        first_call_count = spy.call_count
        dist.batch([(["fever", "cough"], ["high temperature", "coughing"])])
        second_call_count = spy.call_count

    assert first_call_count == 4
    assert second_call_count == first_call_count


@pytest.mark.parametrize("cls", CLASSES)
def test_repeated_batch_call_gives_identical_result(cls):
    dist = cls()
    pair = (["fever", "cough"], ["high temperature", "coughing"])
    result1 = dist.batch([pair])[0]
    result2 = dist.batch([pair])[0]
    assert result1 == pytest.approx(result2, abs=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
def test_edge_cost_cache_shared_across_overlapping_pairs(cls):
    dist = cls()
    dist.batch([(["fever"], ["high temperature"])])
    with patch.object(dist, "_edge_cost", wraps=dist._edge_cost) as spy:
        dist.batch([(["fever", "cough"], ["high temperature"])])
    assert spy.call_count == 1
    assert len(dist._edge_cost_cache) == 2
