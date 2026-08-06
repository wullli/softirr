
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys

import pytest

from soft_irr.experiments.load_taxonomies import (
    ConceptEntry,
    build_hierarchy_index,
    build_lexical_index,
    build_sibling_index,
)
from soft_irr.experiments.synthetic_irr import (
    _generate_populations,
    _parse_args,
    _generate_set_pairs,
    _generate_single_pairs,
    _n_shared,
)


def make_pool(with_parents: bool = False) -> list[ConceptEntry]:
    parents = (
        {"A": ["P1"], "B": ["P1"], "C": ["P1"], "D": ["P2"], "E": ["P2"], "F": ["P2"]}
        if with_parents else {}
    )
    return [
        ConceptEntry(code="A", preferred_term="Concept A", synonyms=["a1", "a2"],
                     parent_codes=parents.get("A", [])),
        ConceptEntry(code="B", preferred_term="Concept B", synonyms=["b1"],
                     parent_codes=parents.get("B", [])),
        ConceptEntry(code="C", preferred_term="Concept C", synonyms=["c1", "c2", "c3"],
                     parent_codes=parents.get("C", [])),
        ConceptEntry(code="D", preferred_term="Concept D", synonyms=["d1"],
                     parent_codes=parents.get("D", [])),
        ConceptEntry(code="E", preferred_term="Concept E", synonyms=["e1", "e2"],
                     parent_codes=parents.get("E", [])),
        ConceptEntry(code="F", preferred_term="Concept F", synonyms=["f1"],
                     parent_codes=parents.get("F", [])),
    ]


def make_large_pool(n: int = 50) -> list[ConceptEntry]:
    return [
        ConceptEntry(code=f"C{i}", preferred_term=f"Concept {i}", synonyms=[f"s{i}"])
        for i in range(n)
    ]


def by_term(pool: list[ConceptEntry]) -> dict[str, ConceptEntry]:
    return {c.preferred_term: c for c in pool}


def by_code(pool: list[ConceptEntry]) -> dict[str, ConceptEntry]:
    return {c.code: c for c in pool}


def by_synonym(pool: list[ConceptEntry]) -> dict[str, ConceptEntry]:
    return {syn: c for c in pool for syn in c.synonyms}


def test_build_hierarchy_index_rejects_unknown_direction():
    with pytest.raises(ValueError):
        build_hierarchy_index({}, direction="grandparent")


def test_build_hierarchy_index_parent_falls_back_to_parent_terms_when_parent_not_in_pool():
    pool = make_pool(with_parents=True)
    concepts = {c.code: c for c in pool}
    hierarchy_index = build_hierarchy_index(
        concepts, direction="parent", parent_terms={"P1": "Parent One", "P2": "Parent Two"},
    )
    for code, expected_title in [("A", "Parent One"), ("D", "Parent Two")]:
        [candidate] = hierarchy_index[code]
        assert candidate.code in ("P1", "P2")
        assert candidate.preferred_term == expected_title
        assert candidate.synonyms == [expected_title]


def test_build_hierarchy_index_parent_drops_parent_with_no_terms_or_pool_match():
    pool = [ConceptEntry(code="A", preferred_term="Concept A", synonyms=["a1"], parent_codes=["P1"])]
    concepts = {c.code: c for c in pool}
    assert build_hierarchy_index(concepts, direction="parent") == {"A": []}


def test_build_hierarchy_index_parent_resolves_real_parent_via_hierarchy_key():
    pool = [
        ConceptEntry(code="X", preferred_term="Ancestor", synonyms=["anc1", "anc2"], hierarchy_key=["T1"]),
        ConceptEntry(code="Y", preferred_term="Child of X", synonyms=["child1"], parent_codes=["T1"]),
        ConceptEntry(code="Z", preferred_term="Unrelated", synonyms=["z1"]),
    ]
    concepts = {c.code: c for c in pool}
    hierarchy_index = build_hierarchy_index(concepts, direction="parent")
    assert hierarchy_index["Y"] == [concepts["X"]]
    assert hierarchy_index["X"] == []
    assert hierarchy_index["Z"] == []


def test_build_hierarchy_index_children_resolves_real_child_via_hierarchy_key():
    pool = [
        ConceptEntry(code="X", preferred_term="Ancestor", synonyms=["anc1", "anc2"], hierarchy_key=["T1"]),
        ConceptEntry(code="Y", preferred_term="Child of X", synonyms=["child1"], parent_codes=["T1"]),
        ConceptEntry(code="Z", preferred_term="Unrelated", synonyms=["z1"]),
    ]
    concepts = {c.code: c for c in pool}
    hierarchy_index = build_hierarchy_index(concepts, direction="children")
    assert hierarchy_index["X"] == [concepts["Y"]]
    assert hierarchy_index["Y"] == []
    assert hierarchy_index["Z"] == []


def test_build_hierarchy_index_children_is_always_empty_for_single_depth_pools():
    pool = make_pool(with_parents=True)
    concepts = {c.code: c for c in pool}
    hierarchy_index = build_hierarchy_index(concepts, direction="children")
    assert all(candidates == [] for candidates in hierarchy_index.values())


def test_build_lexical_index_ranks_by_string_similarity_and_excludes_self():
    pool = [
        ConceptEntry(code="A", preferred_term="Hypertension", synonyms=["a1"]),
        ConceptEntry(code="B", preferred_term="Hypotension", synonyms=["b1"]),
        ConceptEntry(code="C", preferred_term="Bronchitis", synonyms=["c1"]),
    ]
    concepts = {c.code: c for c in pool}
    lexical_index = build_lexical_index(concepts, top_k=1)
    assert lexical_index["A"] == [concepts["B"]]
    assert lexical_index["B"] == [concepts["A"]]
    assert lexical_index["C"] == [concepts["A"]] or lexical_index["C"] == [concepts["B"]]
    for candidates in lexical_index.values():
        assert len(candidates) == 1


def test_build_lexical_index_reranks_by_synonym_not_just_preferred_term():
    pool = [
        ConceptEntry(code="A", preferred_term="Dichloroethane", synonyms=["ethylidene chloride"]),
        ConceptEntry(code="B", preferred_term="Dichloromethane", synonyms=["xyz totally unrelated word"]),
        ConceptEntry(code="C", preferred_term="Zzzzz completely unrelated name", synonyms=["ethylidene chlorine"]),
    ]
    concepts = {c.code: c for c in pool}
    lexical_index = build_lexical_index(concepts, top_k=1)
    assert lexical_index["A"] == [concepts["C"]]


def test_build_lexical_index_caps_at_top_k():
    pool = [
        ConceptEntry(code=str(i), preferred_term=f"Concept {i}", synonyms=[f"c{i}"])
        for i in range(6)
    ]
    concepts = {c.code: c for c in pool}
    lexical_index = build_lexical_index(concepts, top_k=3)
    for code, candidates in lexical_index.items():
        assert len(candidates) == 3
        assert code not in {c.code for c in candidates}


def _mesh_xml(n_descriptors: int = 12) -> str:
    records = "".join(
        f"""<DescriptorRecord DescriptorClass="1">
             <DescriptorUI>D{i:04d}</DescriptorUI>
             <DescriptorName><String>Concept {i}</String></DescriptorName>
             <ConceptList><Concept PreferredConceptYN="Y"><TermList>
               <Term RecordPreferredTermYN="N"><String>synonym {i}</String></Term>
             </TermList></Concept></ConceptList>
             <TreeNumberList>
               <TreeNumber>A{i:02d}.111.236</TreeNumber>
               <TreeNumber>B{i:02d}.222</TreeNumber>
               <TreeNumber>C{i:02d}.333.444</TreeNumber>
               <TreeNumber>D{i:02d}.555.666</TreeNumber>
             </TreeNumberList>
           </DescriptorRecord>"""
        for i in range(n_descriptors)
    )
    return f"<DescriptorRecordSet>{records}</DescriptorRecordSet>"


def test_load_mesh_parent_codes_are_reproducible_across_hash_seeds(tmp_path):
    xml_path = tmp_path / "desc.xml"
    xml_path.write_text(_mesh_xml(), encoding="utf-8")

    script = (
        "import json,sys;"
        "from soft_irr.experiments.load_taxonomies import load_mesh;"
        f"c=load_mesh({str(xml_path)!r});"
        "print(json.dumps({k: v.parent_codes for k, v in c.items()}))"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True, check=True,
        ).stdout
        for seed in ("1", "2")
    ]
    assert json.loads(outputs[0]) == json.loads(outputs[1])
    assert any(len(v) > 1 for v in json.loads(outputs[0]).values()), "fixture must exercise ordering"


def test_single_pairs_shape():
    pool = make_pool()
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(pool, 10, 0.5, random.Random(0))
    assert len(rater_a) == 10
    assert len(rater_b) == 10
    assert len(rater_a_concepts) == 10
    assert len(rater_b_concepts) == 10
    assert all(len(t) == 1 for t in rater_a)
    assert all(len(p) == 1 for p in rater_b)
    assert all(len(p) == 1 for p in rater_a_concepts)
    assert all(len(p) == 1 for p in rater_b_concepts)


def test_single_pairs_zero_n_returns_empty():
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(make_pool(), 0, 0.5, random.Random(0))
    assert rater_a == []
    assert rater_b == []
    assert rater_a_concepts == []
    assert rater_b_concepts == []


def test_single_pairs_full_agreement_is_always_a_synonym():
    pool = make_pool()
    syns = by_synonym(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(pool, 50, 1.0, random.Random(1))
    for [a_term], [b_term], [a_concept], [b_concept] in zip(rater_a, rater_b, rater_a_concepts, rater_b_concepts):
        assert a_term in syns
        assert b_term in syns
        assert a_concept == b_concept


def test_single_pairs_full_disagreement_is_always_a_different_concept_random():
    pool = make_pool()
    syns = by_synonym(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(pool, 50, 0.0, random.Random(2))
    for [a_term], [b_term], [a_concept], [b_concept] in zip(rater_a, rater_b, rater_a_concepts, rater_b_concepts):
        assert a_term in syns
        assert b_term in syns
        assert a_concept != b_concept


def test_single_pairs_disagreement_uses_siblings_when_provided():
    pool = make_pool(with_parents=True)
    sibling_index = build_sibling_index({c.code: c for c in pool})
    concept_by_code = by_code(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(
        pool, 50, 0.0, random.Random(3),
        sibling_index=sibling_index, concept_by_code=concept_by_code,
    )
    terms = by_term(pool)
    for [a_concept], [b_concept] in zip(rater_a_concepts, rater_b_concepts):
        a_code = terms[a_concept].code
        b_code = terms[b_concept].code
        assert b_code in sibling_index[a_code]


def test_single_pairs_disagreement_uses_negative_index_when_provided():
    pool = make_pool(with_parents=True)
    hierarchy_index = build_hierarchy_index(
        {c.code: c for c in pool}, direction="parent",
        parent_terms={"P1": "Parent One", "P2": "Parent Two"},
    )
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(
        pool, 50, 0.0, random.Random(3), negative_index=hierarchy_index,
    )
    terms = by_term(pool)
    for [a_concept], [b_concept] in zip(rater_a_concepts, rater_b_concepts):
        a_code = terms[a_concept].code
        eligible_terms = {c.preferred_term for c in hierarchy_index[a_code]}
        assert b_concept in eligible_terms


def test_single_pairs_deterministic_with_same_seed():
    pool = make_pool()
    rater_a_1, rater_b_1, rater_a_concepts_1, rater_b_concepts_1 = _generate_single_pairs(pool, 20, 0.4, random.Random(123))
    rater_a_2, rater_b_2, rater_a_concepts_2, rater_b_concepts_2 = _generate_single_pairs(pool, 20, 0.4, random.Random(123))
    assert rater_a_1 == rater_a_2
    assert rater_b_1 == rater_b_2
    assert rater_a_concepts_1 == rater_a_concepts_2
    assert rater_b_concepts_1 == rater_b_concepts_2


@pytest.mark.parametrize("pool_size", [0, 1])
def test_set_pairs_rejects_pool_smaller_than_min_set_size(pool_size):
    pool = make_pool()[:pool_size]
    with pytest.raises(ValueError, match="min_set_size"):
        _generate_set_pairs(pool, 5, 0.5, random.Random(0), min_set_size=2, max_set_size=5)


def test_single_pairs_rejects_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        _generate_single_pairs([], 5, 0.5, random.Random(0))


def test_set_pairs_shape():
    pool = make_pool()
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 20, 0.5, random.Random(0), min_set_size=2, max_set_size=5,
    )
    assert len(rater_a) == 20
    assert len(rater_b) == 20
    assert len(rater_a_concepts) == 20
    assert len(rater_b_concepts) == 20
    assert all(2 <= len(t) <= 5 for t in rater_a)
    assert all(2 <= len(p) <= 5 for p in rater_b)
    assert all(2 <= len(p) <= 5 for p in rater_a_concepts)
    assert all(2 <= len(p) <= 5 for p in rater_b_concepts)
    assert all(len(t) == len(tc) for t, tc in zip(rater_a, rater_a_concepts))
    assert all(len(p) == len(pc) for p, pc in zip(rater_b, rater_b_concepts))


def test_set_pairs_sizes_can_mismatch_below_full_agreement():
    pool = make_pool()
    rater_a, rater_b, _, _ = _generate_set_pairs(
        pool, 20, 0.5, random.Random(0), min_set_size=2, max_set_size=5,
    )
    assert any(len(t) != len(p) for t, p in zip(rater_a, rater_b))


def test_set_pairs_sizes_always_match_at_full_agreement():
    pool = make_pool()
    rater_a, rater_b, _, _ = _generate_set_pairs(
        pool, 20, 1.0, random.Random(0), min_set_size=2, max_set_size=5,
    )
    assert all(len(t) == len(p) for t, p in zip(rater_a, rater_b))


def test_set_pairs_sampled_concepts_are_distinct():
    pool = make_pool()
    _, _, rater_a_concepts, _ = _generate_set_pairs(pool, 10, 0.5, random.Random(4), min_set_size=2, max_set_size=5)
    for a_concept_set in rater_a_concepts:
        assert len(set(a_concept_set)) == len(a_concept_set)


def test_set_pairs_full_agreement_is_exact_set_equality():
    pool = make_pool()
    syns = by_synonym(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 30, 1.0, random.Random(5), min_set_size=2, max_set_size=5,
    )
    for a_set, b_set, a_concept_set, b_concept_set in zip(rater_a, rater_b, rater_a_concepts, rater_b_concepts):
        assert len(a_set) == len(b_set) == len(a_concept_set) == len(b_concept_set)
        assert set(a_concept_set) == set(b_concept_set)
        for a_term, b_term in zip(a_set, b_set):
            assert a_term in syns
            assert b_term in syns


def test_set_pairs_full_disagreement_every_item_is_a_mismatch_random():
    pool = make_pool()
    syns = by_synonym(pool)
    _, rater_b, _, rater_b_concepts = _generate_set_pairs(
        pool, 20, 0.0, random.Random(6), min_set_size=2, max_set_size=5,
    )
    for b_set, b_concept_set in zip(rater_b, rater_b_concepts):
        for b_term, b_concept in zip(b_set, b_concept_set):
            assert b_term in syns
            assert syns[b_term].preferred_term == b_concept


def test_set_pairs_disagreement_uses_siblings_when_provided():
    pool = make_pool(with_parents=True)
    sibling_index = build_sibling_index({c.code: c for c in pool})
    concept_by_code = by_code(pool)
    terms = by_term(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 20, 0.0, random.Random(7), min_set_size=2, max_set_size=3,
        sibling_index=sibling_index, concept_by_code=concept_by_code,
    )
    for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
        a_codes = {terms[t].code for t in a_concept_set}
        eligible_sibling_codes = set().union(*(sibling_index[c] for c in a_codes))
        for b_concept in b_concept_set:
            assert terms[b_concept].code in eligible_sibling_codes


def test_set_pairs_disagreement_uses_negative_index_when_provided():
    pool = make_pool(with_parents=True)
    hierarchy_index = build_hierarchy_index(
        {c.code: c for c in pool}, direction="parent",
        parent_terms={"P1": "Parent One", "P2": "Parent Two"},
    )
    terms = by_term(pool)
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 20, 0.0, random.Random(7), min_set_size=2, max_set_size=3,
        negative_index=hierarchy_index,
    )
    for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
        a_codes = {terms[t].code for t in a_concept_set}
        eligible_terms = set().union(*(
            {c.preferred_term for c in hierarchy_index[code]} for code in a_codes
        ))
        for b_concept in b_concept_set:
            assert b_concept in eligible_terms


@pytest.mark.parametrize(
    "set_size, agreement_level",
    [(5, 0.3), (5, 0.1), (4, 0.5), (3, 1 / 3), (6, 0.9)],
)
def test_set_pairs_partial_agreement_hits_the_target_jaccard(set_size, agreement_level):
    pool = make_large_pool()
    exact_n_agree = agreement_level * 2 * set_size / (1 + agreement_level)
    _, _, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 400, agreement_level, random.Random(8), min_set_size=set_size, max_set_size=set_size,
    )
    jaccards = []
    for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
        n_agree = len(set(a_concept_set) & set(b_concept_set))
        assert math.floor(exact_n_agree) <= n_agree <= math.ceil(exact_n_agree)
        assert b_concept_set[:n_agree] == a_concept_set[:n_agree]
        for a_concept, b_concept in zip(a_concept_set[n_agree:], b_concept_set[n_agree:]):
            assert b_concept != a_concept
        jaccards.append(n_agree / (2 * set_size - n_agree))
    assert sum(jaccards) / len(jaccards) == pytest.approx(agreement_level, abs=0.03)


def test_set_pairs_negatives_are_disjoint_from_rater_a_concepts():
    pool = make_large_pool()
    for agreement_level in (0.0, 0.25, 0.5, 0.75):
        _, _, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
            pool, 300, agreement_level, random.Random(11), min_set_size=2, max_set_size=5,
        )
        for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
            n_agree = len(set(a_concept_set) & set(b_concept_set))
            assert b_concept_set[:n_agree] == a_concept_set[:n_agree]
            assert not set(b_concept_set[n_agree:]) & set(a_concept_set)


def test_set_pairs_rater_b_concepts_are_distinct():
    pool = make_large_pool()
    for agreement_level in (0.0, 0.5, 0.9):
        _, _, _, rater_b_concepts = _generate_set_pairs(
            pool, 300, agreement_level, random.Random(12), min_set_size=2, max_set_size=5,
        )
        for b_concept_set in rater_b_concepts:
            assert len(set(b_concept_set)) == len(b_concept_set)


@pytest.mark.parametrize("agreement_level", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_set_pairs_realized_overlap_is_the_target_jaccard_solution_rounded(agreement_level):
    pool = make_large_pool()
    _, _, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 500, agreement_level, random.Random(13), min_set_size=2, max_set_size=5,
    )
    for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
        a_set, b_set = set(a_concept_set), set(b_concept_set)
        exact = agreement_level * (len(a_set) + len(b_set)) / (1.0 + agreement_level)
        cap = min(len(a_set), len(b_set))
        assert min(math.floor(exact), cap) <= len(a_set & b_set) <= min(math.ceil(exact), cap)


@pytest.mark.parametrize("target_jaccard", [0.043, 0.3, 0.617, 0.85, 0.914, 0.978])
def test_n_shared_mixes_counts_so_the_realized_jaccard_is_unbiased(target_jaccard):
    rng = random.Random(17)
    exact = target_jaccard * 10 / (1.0 + target_jaccard)
    draws = [_n_shared(target_jaccard, 5, 5, rng) for _ in range(4000)]
    assert set(draws) == {math.floor(exact), math.ceil(exact)}
    jaccards = [k / (10 - k) for k in draws]
    assert sum(jaccards) / len(jaccards) == pytest.approx(target_jaccard, abs=0.02)


@pytest.mark.parametrize("agreement_level", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])
def test_set_pairs_mean_jaccard_tracks_the_agreement_level(agreement_level):
    pool = make_large_pool(200)
    _, _, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 2000, agreement_level, random.Random(15), min_set_size=1, max_set_size=10,
    )
    jaccards = []
    for a_concept_set, b_concept_set in zip(rater_a_concepts, rater_b_concepts):
        a_set, b_set = set(a_concept_set), set(b_concept_set)
        jaccards.append(len(a_set & b_set) / len(a_set | b_set))
    assert sum(jaccards) / len(jaccards) == pytest.approx(agreement_level, abs=0.03)


def _cli_defaults(monkeypatch) -> dict[str, int]:
    monkeypatch.setattr(sys, "argv", ["synthetic_irr", "--dataset", "icd11", "--mode", "set"])
    args = _parse_args()
    return {
        "min_set_size": args.min_set_size,
        "max_set_size": args.max_set_size,
        "n_populations": args.n_populations,
        "n_samples": args.n_samples,
        "base_seed": args.base_seed,
    }


def test_generate_populations_set_mode_true_kappa_covers_the_unit_interval(monkeypatch):
    cli = _cli_defaults(monkeypatch)
    pool = make_large_pool(200)
    populations = _generate_populations(
        "set", pool, n_samples=cli["n_samples"],
        seeds=[cli["base_seed"] + i for i in range(cli["n_populations"])],
        sibling_index=None, concept_by_code=by_code(pool),
        min_set_size=cli["min_set_size"], max_set_size=cli["max_set_size"],
    )
    kappas = []
    for population in populations:
        agreement_level, _, _, _, _, _, truth = population
        true_cohens_kappa = truth.cohens_kappa
        assert true_cohens_kappa == pytest.approx(agreement_level, abs=0.1)
        kappas.append(true_cohens_kappa)
    quartiles = [sum(1 for k in kappas if lo <= k < hi) for lo, hi in
                 [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]]
    assert all(count >= 2 for count in quartiles), quartiles


def test_set_pairs_degrades_gracefully_when_pool_too_small_for_disjoint_negatives():
    pool = make_pool()
    rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
        pool, 30, 0.0, random.Random(14), min_set_size=2, max_set_size=5,
    )
    assert all(2 <= len(p) <= 5 for p in rater_b)
    assert all(len(p) == len(pc) for p, pc in zip(rater_b, rater_b_concepts))
    assert len(rater_a) == len(rater_b) == len(rater_a_concepts) == 30


def test_set_pairs_deterministic_with_same_seed():
    pool = make_pool()
    rater_a_1, rater_b_1, rater_a_concepts_1, rater_b_concepts_1 = _generate_set_pairs(
        pool, 10, 0.4, random.Random(99), min_set_size=2, max_set_size=5,
    )
    rater_a_2, rater_b_2, rater_a_concepts_2, rater_b_concepts_2 = _generate_set_pairs(
        pool, 10, 0.4, random.Random(99), min_set_size=2, max_set_size=5,
    )
    assert rater_a_1 == rater_a_2
    assert rater_b_1 == rater_b_2
    assert rater_a_concepts_1 == rater_a_concepts_2
    assert rater_b_concepts_1 == rater_b_concepts_2


def test_generate_populations_single_mode_count_and_shape():
    pool = make_pool()
    seeds = [1, 2]
    populations = _generate_populations(
        "single", pool, n_samples=7, seeds=seeds,
        sibling_index=None, concept_by_code=by_code(pool),
    )
    assert len(populations) == len(seeds)
    for (
        agreement_level, seed, rater_a, rater_b, rater_a_concepts, rater_b_concepts, truth,
    ) in populations:
        assert len(rater_a) == 7
        assert len(rater_b) == 7
        assert all(len(t) == 1 for t in rater_a)
        assert 0.0 <= agreement_level <= 1.0
        assert seed in seeds
        assert 0.0 <= truth.average_agreement <= 1.0
        assert -1.0 <= truth.cohens_kappa <= 1.0
        assert -1.0 <= truth.scotts_pi <= 1.0
        assert 0.0 <= truth.sigma <= 1.0
        assert 0.0 <= truth.ks <= 1.0


def test_generate_populations_set_mode_count_and_shape():
    pool = make_pool()
    seeds = [10]
    populations = _generate_populations(
        "set", pool, n_samples=5, seeds=seeds,
        sibling_index=None, concept_by_code=by_code(pool),
        min_set_size=2, max_set_size=5,
    )
    assert len(populations) == len(seeds)
    for (
        agreement_level, seed, rater_a, rater_b, rater_a_concepts, rater_b_concepts, truth,
    ) in populations:
        assert 0.0 <= agreement_level <= 1.0
        assert all(2 <= len(t) <= 5 for t in rater_a)
        assert all(2 <= len(p) <= 5 for p in rater_b)
        assert 0.0 <= truth.average_agreement <= 1.0
        assert -1.0 <= truth.cohens_kappa <= 1.0
        assert -1.0 <= truth.scotts_pi <= 1.0
        assert 0.0 <= truth.sigma <= 1.0
        assert 0.0 <= truth.ks <= 1.0


def test_generate_populations_reproducible_across_calls():
    pool = make_pool()
    args = dict(
        mode="single", pool=pool,
        n_samples=5, seeds=[1, 2],
        sibling_index=None, concept_by_code=by_code(pool),
    )
    populations_1 = _generate_populations(**args)
    populations_2 = _generate_populations(**args)
    assert populations_1 == populations_2


def test_generate_populations_different_seeds_produce_different_data():
    pool = make_pool()
    populations = _generate_populations(
        "single", pool, n_samples=20, seeds=[1, 2],
        sibling_index=None, concept_by_code=by_code(pool),
    )
    (_, _, rater_a_1, rater_b_1, _, _, _), (_, _, rater_a_2, rater_b_2, _, _, _) = populations
    assert (rater_a_1, rater_b_1) != (rater_a_2, rater_b_2)
