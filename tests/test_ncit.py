import json

from soft_irr.experiments.load_taxonomies import (
    ConceptEntry,
    build_hierarchy_index,
    build_lexical_index,
    lexical_name,
    load_ncit,
    load_ncit_ancestor_hierarchy,
)
from soft_irr.preprocess_ncit import (
    build_definitions,
    template_frame,
    clean_definition,
    parse_owl,
    strip_trailing_tags,
)

_MARKERS = ("CONCEPT DROP MARKER",)
LONG = "A benign or malignant growth of tissue caused by uncontrolled cell division"


def test_trailing_citation_after_sentence_end_is_removed():
    cleaned, rules, tags = clean_definition("Cells with neuronal and myoblast differentiation. (INHAND)")
    assert cleaned == "Cells with neuronal and myoblast differentiation."
    assert tags == ["INHAND"]
    assert rules == ["trailing_tags"]


def test_acronym_before_final_period_is_kept():
    text = "A codelist of the Study Data Tabulation Model (SDTM)."
    assert clean_definition(text)[0] == text


def test_keep_list_tag_is_kept():
    text = "Have you ever used tobacco products. (Past 3 Months)"
    assert clean_definition(text)[0] == text


def test_two_trailing_tags_are_both_removed():
    cleaned, tags = strip_trailing_tags("Tumor confined to the uterus. (from AJCC 8th Ed.) (NCI04)")
    assert cleaned.rstrip() == "Tumor confined to the uterus."
    assert tags == ["NCI04", "from AJCC 8th Ed."]


def test_dash_year_is_removed_before_trailing_citation():
    cleaned, rules, tags = clean_definition("Referred to as rickets. (Diagnostic Surgical Pathology, 3rd ed.) --2003")
    assert cleaned == "Referred to as rickets."
    assert rules == ["dash_year", "trailing_tags"]


def test_note_citation_and_cross_reference_are_removed():
    cleaned, rules, _ = clean_definition(
        "The process of validating questionnaires. NOTE: Several types exist. "
        "[Guyatt et al., 1993] See also validation, compare to reliability."
    )
    assert cleaned == "The process of validating questionnaires."
    assert rules == ["notes_cross_refs"]


def test_square_citation_is_removed_but_content_bracket_is_kept():
    cleaned, rules, _ = clean_definition(
        "An attack of the immune system on the body. [Janeway CA, Immunobiology, 2001] It targets self."
    )
    assert cleaned == "An attack of the immune system on the body. It targets self."
    assert rules == ["square_citations"]
    text = "A cofactor of carbamoyl-phosphate synthase [ammonia] in mitochondria."
    assert clean_definition(text)[0] == text


def test_url_in_brackets_is_removed():
    cleaned, rules, _ = clean_definition(
        "A process to find facts that guide future studies. (https://mrctcenter.org/glossaryterm/x/)"
    )
    assert cleaned == "A process to find facts that guide future studies."
    assert "urls" in rules


def test_bridg_field_labels_are_cut():
    cleaned, rules, _ = clean_definition("A human being.  EXAMPLE(S): a patient  OTHER NAME(S):  NOTE(S):")
    assert cleaned == "A human being."
    assert rules == ["field_labels"]


def test_inline_source_acronym_and_also_known_as_are_kept():
    text = "A standard of the Clinical Data Interchange Standards Consortium (CDISC), also known as SDTM."
    assert clean_definition(text)[0] == text


_NS = 'xmlns:owl="http://www.w3.org/2002/07/owl#" ' \
      'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ' \
      'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" ' \
      'xmlns="http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"'
_URI = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"


def _class(code, name, parents=(), sty="Disease or Syndrome", status=None, defined_parents=(), role="R100"):
    subs = "".join(f'<rdfs:subClassOf rdf:resource="{_URI}{p}"/>' for p in parents)
    if defined_parents:
        members = "".join(f'<rdf:Description rdf:about="{_URI}{p}"/>' for p in defined_parents)
        subs += (
            '<owl:equivalentClass><owl:Class><owl:intersectionOf rdf:parseType="Collection">'
            f'{members}<owl:Restriction><owl:onProperty rdf:resource="{_URI}{role}"/>'
            f'<owl:someValuesFrom rdf:resource="{_URI}C99"/></owl:Restriction>'
            "</owl:intersectionOf></owl:Class></owl:equivalentClass>"
        )
    st = f"<P310>{status}</P310>" if status else ""
    return f'<owl:Class rdf:about="{_URI}{code}">{subs}<P106>{sty}</P106><P108>{name}</P108>{st}</owl:Class>'


def _axiom(code, prop, text, source, attribution=None):
    attr = f"<P381>{attribution}</P381>" if attribution else ""
    return (
        f'<owl:Axiom><owl:annotatedSource rdf:resource="{_URI}{code}"/>'
        f'<owl:annotatedProperty rdf:resource="{_URI}{prop}"/>'
        f"<owl:annotatedTarget>{text}</owl:annotatedTarget><P378>{source}</P378>{attr}</owl:Axiom>"
    )


def _write_owl(tmp_path):
    body = "".join([
        _class("C1", "Root", sty="Conceptual Entity"),
        _class("C2", "Good Concept", parents=["C1"]),
        _axiom("C2", "P325", f"{LONG}. (NCI-GLOSS citation)", "NCI-GLOSS", "Based on NCI"),
        _axiom("C2", "P97", "An abnormal mass of tissue arising from cells that divide too often. (NCI05)", "NCI"),
        _axiom("C2", "P325", f"{LONG} in the body", "CDISC"),
        _axiom("C2", "P325", "Too short text", "FDA"),
        _axiom("C2", "P325", "Some dataset field describing a patient record value", "DICOM"),
        _axiom("C2", "P325", "Old text here and more words. (This definition may be outdated - see the DesignNote.)",
               "NICHD"),
        _class("C3", "Single Definition", parents=["C2"]),
        _axiom("C3", "P97", "Only one definition exists for this concept here", "NCI"),
        _class("C4", "Retired Concept", parents=["C2"], status="Retired_Concept"),
        _axiom("C4", "P97", "First definition of a retired concept in thesaurus", "NCI"),
        _axiom("C4", "P325", "Second wholly different retired wording for testing", "FDA"),
        _class("C5", "Grade 2 Rash, CTCAE", parents=["C2"]),
        _axiom("C5", "P97", "First definition of a grading criterion concept", "NCI"),
        _axiom("C5", "P325", "Second wholly different grading text for testing", "FDA"),
        _class("C6", "Leaf Concept", defined_parents=["C3"], sty="Body Part, Organ, or Organ Component"),
        _axiom("C6", "P97", "The first organ definition used in the leaf test", "NCI"),
        _axiom("C6", "P325", "A second and distinct anatomical explanation of it", "UBERON"),
        _class("C7", "Calcitonin Measurement", parents=["C2"], sty="Laboratory Procedure"),
        _axiom("C7", "P97", "The determination of the amount of calcitonin present", "NCI"),
        _axiom("C7", "P325", "A measurement of calcitonin hormone within some specimen", "CDISC"),
        _class("C8", "CDISC Questionnaire Test Code Terminology", parents=["C2"], sty="Intellectual Product"),
        _axiom("C8", "P97", "Test codes of questionnaire questions for a scale", "NCI"),
        _axiom("C8", "P325", "Some other questionnaire scale test code wording", "CDISC"),
        _class("C9", "Stage I Liver Cancer", defined_parents=["C2"], role="R88"),
        _axiom("C9", "P97", "Liver cancer confined to the liver without spread", "NCI"),
        _axiom("C9", "P325", "A stage of cancer in which tumor has not spread beyond organ", "NCI-GLOSS"),
        _class("C11", "Flagged Concept", parents=["C1"]),
        _axiom("C11", "P97", "A normal looking definition text. CONCEPT DROP MARKER for the test.", "NCI"),
        _axiom("C11", "P325", "A second normal definition of the flagged concept", "CDISC"),
        _class("C10", "Stage IA Liver Cancer", parents=["C9"]),
        _axiom("C10", "P97", "A single small liver tumor confined to the liver", "NCI"),
        _axiom("C10", "P325", "Early cancer stage with one small tumor present only", "NCI-GLOSS"),
    ])
    path = tmp_path / "Thesaurus.owl"
    path.write_text(f'<?xml version="1.0"?><rdf:RDF {_NS}>{body}</rdf:RDF>', encoding="utf-8")
    return path


def test_build_definitions_applies_all_filters(tmp_path):
    concepts, definitions = parse_owl(_write_owl(tmp_path))
    output, log, tag_counts, summary, _ = build_definitions(
        concepts, definitions, concept_drop_markers=_MARKERS
    )

    hierarchy = output.pop("_hierarchy")
    assert set(output) == {"C2", "C6"}
    c2 = output["C2"]
    assert [d["source"] for d in c2["definitions"]] == ["NCI", "NCI-GLOSS"]
    assert c2["definitions"][0]["text"].endswith("divide too often.")
    assert c2["definitions"][1] == {"text": f"{LONG}.", "source": "NCI-GLOSS", "attribution": "Based on NCI"}
    assert c2["parents"] == ["C1"]
    assert "C4" not in hierarchy
    assert hierarchy["C6"] == ["C3"]

    rules = {(row["code"], row["source"], row["rule"]) for row in log}
    assert ("C2", "DICOM", "drop_source") in rules
    assert ("C2", "NICHD", "drop_marker") in rules
    assert ("C2", "FDA", "drop_short") in rules
    assert ("C2", "CDISC", "drop_duplicate") in rules
    assert not any(row["code"] in {"C4", "C5", "C7", "C8", "C9", "C10", "C11"} for row in log)
    assert hierarchy["C7"] == ["C2"]
    assert tag_counts["NCI#"] == 1
    assert summary["concepts_out"] == 2


def test_load_ncit_and_semantic_type_filter(tmp_path):
    concepts, definitions = parse_owl(_write_owl(tmp_path))
    output, *_ = build_definitions(concepts, definitions, concept_drop_markers=_MARKERS)
    path = tmp_path / "definitions.json"
    path.write_text(json.dumps(output), encoding="utf-8")

    loaded = load_ncit(path)
    assert set(loaded) == {"C2", "C6"}
    assert loaded["C2"].preferred_term == "Good Concept (C2)"
    assert len(loaded["C2"].synonyms) == 2
    assert loaded["C6"].parent_codes == ["C3"]
    assert loaded["C6"].hierarchy_key == ["C6"]

    assert set(load_ncit(path, {"Disease or Syndrome"})) == {"C2"}
    assert load_ncit_ancestor_hierarchy(path)["C3"] == ["C2"]


def test_parent_index_traverses_ancestors_outside_pool_without_name_fallback():
    concepts = {
        "C2": ConceptEntry("C2", "Good", ["def a", "def b"], parent_codes=["C1"], hierarchy_key=["C2"]),
        "C6": ConceptEntry("C6", "Leaf", ["def c", "def d"], parent_codes=["C3"], hierarchy_key=["C6"]),
    }
    index = build_hierarchy_index(
        concepts, direction="parent", parent_terms={},
        ancestor_parent_codes={"C3": ["C2"], "C2": ["C1"]}, hops=2,
    )
    assert [c.code for c in index["C6"]] == ["C2"]
    assert index["C2"] == []


def test_lexical_name_strips_only_the_own_code_suffix():
    assert lexical_name(ConceptEntry("C2", "Depression (C2)", ["x"])) == "Depression"
    assert lexical_name(ConceptEntry("D1", "Depression (C2)", ["x"])) == "Depression (C2)"


def test_lexical_index_ignores_code_suffix():
    names = {"C11111": "Fever", "C22222": "Fever Grade", "C11112": "Liver Cyst", "C33333": "Fevers"}
    plain = {c: ConceptEntry(c, n, [f"definition of {n}"]) for c, n in names.items()}
    coded = {c: ConceptEntry(c, f"{n} ({c})", [f"definition of {n}"]) for c, n in names.items()}
    assert _codes(build_lexical_index(coded, top_k=3)) == _codes(build_lexical_index(plain, top_k=3))


def _codes(index):
    return {c: [e.code for e in v] for c, v in index.items()}


def test_template_frame_masks_name_words_and_numbers():
    assert template_frame("Any bacterial organism assigned to the species Klebsiella pneumoniae.", "Klebsiella pneumoniae") \
        == "any bacterial organism assigned to the species <N> ."
    assert template_frame("Saturated fatty acid with 16 carbons.", "Palmitic Acid") \
        == "saturated fatty <N> with <N> carbons ."


def test_templated_definitions_are_dropped(tmp_path):
    body = "".join(
        _class(code, name, parents=["C1"], sty="Bacterium")
        + _axiom(code, "P97", f"A rod shaped bacterium {extra} found in soil samples", "NCI")
        + _axiom(code, "P325", f"Any bacterial organism that can be assigned to the species {name}.", "CDISC")
        + _axiom(code, "P325", other, "FDA")
        for code, name, extra, other in [
            ("C21", "Alpha bacillus", "with spores", "Microbe causing wound infections in farm workers"),
            ("C22", "Beta coccus", "without flagella", "Round cells that cluster together on agar plates"),
            ("C23", "Gamma vibrio", "that is curved", "Comma shaped organism living in brackish coastal water"),
        ]
    )
    path = tmp_path / "Thesaurus.owl"
    path.write_text(f'<?xml version="1.0"?><rdf:RDF {_NS}>{_class("C1", "Root")}{body}</rdf:RDF>', encoding="utf-8")
    concepts, definitions = parse_owl(path)
    output, log, _, summary, templates = build_definitions(concepts, definitions, template_min_count=3)
    output.pop("_hierarchy")

    assert set(output) == {"C21", "C22", "C23"}
    assert all("CDISC" not in [d["source"] for d in c["definitions"]] for c in output.values())
    assert summary["dropped_template"] == 3
    assert sum(row["rule"] == "drop_template" for row in log) == 3
    assert templates[0]["source"] == "CDISC" and templates[0]["count"] == 3
