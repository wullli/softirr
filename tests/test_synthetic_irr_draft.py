import argparse

from soft_irr.experiments.synthetic_irr import (
    _DRAFT_MAX_EXACT_CHANCE_PAIRS,
    chance_settings,
    output_suffix,
)


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        draft=False, chance_measure=None, max_exact_chance_pairs=None, enable_reasoning=False
    )
    return argparse.Namespace(**{**defaults, **kwargs})


def test_draft_keeps_chance_measure_unset():
    name, cap = chance_settings(_args(draft=True))
    assert name is None
    assert cap == _DRAFT_MAX_EXACT_CHANCE_PAIRS


def test_draft_respects_explicit_chance_measure_and_cap():
    name, cap = chance_settings(
        _args(draft=True, chance_measure="levenshtein_bipartite", max_exact_chance_pairs=7)
    )
    assert name == "levenshtein_bipartite"
    assert cap == 7


def test_non_draft_defaults_to_self_scored_chance():
    assert chance_settings(_args()) == (None, None)


def test_draft_suffix():
    assert output_suffix(_args(draft=True), "single") == "_draft"
    assert output_suffix(_args(draft=True, chance_measure="jaccard"), "single") == "_chance-jaccard"
    assert output_suffix(_args(), "single") == ""


def test_set_mode_suffix_marks_the_uniform_agreement_levels():
    assert output_suffix(_args(), "set") == "_uniform"
    assert output_suffix(_args(draft=True), "set") == "_uniform_draft"
    assert output_suffix(_args(draft=True, chance_measure="jaccard"), "set") == "_uniform_chance-jaccard"
    assert output_suffix(_args(enable_reasoning=True), "set") == "_uniform_reasoning"
