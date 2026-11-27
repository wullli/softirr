from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OWL = "data/ncit/Thesaurus.owl"
DEFAULT_OUTPUT = "data/ncit/definitions.json"
DEFAULT_LOG = "data/ncit/cleaning_log.csv"
DEFAULT_TAGS = "data/ncit/trailing_tags.csv"
DEFAULT_TEMPLATES = "data/ncit/templates.csv"

_EVS = "{http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#}"
_OWL = "{http://www.w3.org/2002/07/owl#}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
_EVS_PREFIX = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"

DEFINITION_PROPERTIES = {"P97": "DEFINITION", "P325": "ALT_DEFINITION"}
INACTIVE_STATUSES = {"Retired_Concept", "Obsolete_Concept"}
EXCLUDED_SEMANTIC_TYPES = frozenset({"Intellectual Product", "Laboratory Procedure"})
EXCLUDED_ROLES = frozenset({"R88"})

SOURCE_ALLOWLIST = frozenset({
    "NCI", "NCI-GLOSS", "NICHD", "FDA", "CTCAE", "CTCAE 5.0", "ACC/AHA", "UBERON", "CRCH", "MMHCC",
    "CTEP", "CDISC", "CDISC-GLOSS", "WHO", "ICH", "MRCT-Ctr", "EDQM-HC", "PQCMC", "CareLex",
    "FDA-NIH-MoRE", "BLOODPAC", "UWDA142", "UMD2001",
})

TRAILING_TAG_KEEP = frozenset({
    "Lifetime", "Past # Months", "Past # Month", "Since Study Start", "Prior to Study Entry", "mean score > #",
})

DROP_MARKERS = (
    "This definition may be outdated",
    "FACIT Measurement System, which is owned exclusively by",
)

CONCEPT_DROP_MARKERS = ("does not comply with Executive Order 14168",)

MIN_TOKENS = 5
DEDUP_JACCARD = 0.7
MIN_DEFINITIONS = 2
TEMPLATE_MIN_COUNT = 10

_FIELD_LABELS = re.compile(r"\s*\b(?:EXAMPLE\(S\)|NOTE\(S\)|OTHER NAME\(S\)):.*$", re.IGNORECASE | re.DOTALL)
_NOTES_AND_CROSS_REFS = re.compile(r"\s*(?:\bNOTE:|\bSee also\b|\b[Cc]ontrast with\b|\b[Cc]ompare to\b).*$", re.DOTALL)
_SQUARE_CITATION = re.compile(r"\s*\[[^\[\]]*\b(?:1[89]|20)\d{2}\b[^\[\]]*\]|(?<=[.!?])\s*\[[^\[\]]*\]")
_URL = re.compile(r"(?:https?://|www\.)[^\s\)\]]+[^\s\)\]\.,;]")
_EMPTY_BRACKETS = re.compile(r"\s*(?:\(\s*[.,;:]?\s*\)|\[\s*[.,;:]?\s*\])")
_BRACKET_GROUP = r"[\(\[](?:[^\(\)\[\]]*(?:[\(\[][^\(\)\[\]]*[\)\]][^\(\)\[\]]*)*)[\)\]]"
_TRAILING_TAG_RUN = re.compile(rf"(?<=[.!?])(?P<run>(?:\s*{_BRACKET_GROUP}\s*\.?)+)\s*$")
_GROUP = re.compile(rf"\s*({_BRACKET_GROUP})\s*\.?")
_TRAILING_DASH_YEAR = re.compile(r"\s+--\s*\d{4}\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def normalize_tag(tag: str) -> str:
    return re.sub(r"\d+", "#", tag.strip())


def strip_field_labels(text: str) -> str:
    return _FIELD_LABELS.sub("", text)


def strip_notes_and_cross_refs(text: str) -> str:
    return _NOTES_AND_CROSS_REFS.sub("", text)


def strip_square_citations(text: str) -> str:
    return _SQUARE_CITATION.sub("", text)


def strip_urls(text: str) -> str:
    return _EMPTY_BRACKETS.sub("", _URL.sub("", text))


def strip_trailing_tags(text: str, keep: frozenset[str] = TRAILING_TAG_KEEP) -> tuple[str, list[str]]:
    match = _TRAILING_TAG_RUN.search(text)
    if match is None:
        return text, []
    groups = _GROUP.findall(match.group("run"))
    removed: list[str] = []
    while groups and normalize_tag(groups[-1][1:-1]) not in keep:
        removed.append(groups.pop()[1:-1].strip())
    if not removed:
        return text, []
    return text[: match.start()] + "".join(f" {g}" for g in groups), removed


def clean_definition(text: str, keep: frozenset[str] = TRAILING_TAG_KEEP) -> tuple[str, list[str], list[str]]:
    rules: list[str] = []
    steps = (
        ("field_labels", strip_field_labels),
        ("notes_cross_refs", strip_notes_and_cross_refs),
        ("square_citations", strip_square_citations),
        ("urls", strip_urls),
        ("dash_year", lambda t: _TRAILING_DASH_YEAR.sub("", t)),
    )
    for rule, step in steps:
        new = step(text)
        if new != text:
            rules.append(rule)
            text = new
    text, tags = strip_trailing_tags(text, keep)
    if tags:
        rules.append("trailing_tags")
    return " ".join(text.split()), rules, tags


def template_frame(text: str, name: str) -> str:
    name_tokens = set(tokens(name))
    frame: list[str] = []
    for word in re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text):
        lower = word.lower()
        token = "<N>" if lower in name_tokens or lower.rstrip("s") in name_tokens or lower.isdigit() else lower
        if not (token == "<N>" and frame and frame[-1] == "<N>"):
            frame.append(token)
    return " ".join(frame)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


@dataclass
class RawConcept:
    name: str = ""
    semantic_types: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    roles: set[str] = field(default_factory=set)


@dataclass
class RawDefinition:
    code: str
    kind: str
    text: str
    source: str | None
    attribution: str | None


def _local(uri: str | None) -> str | None:
    return uri[len(_EVS_PREFIX):] if uri and uri.startswith(_EVS_PREFIX) else None


def _resource(el: ET.Element | None) -> str | None:
    return _local(el.get(_RDF + "resource")) if el is not None else None


def _named_parents(cls: ET.Element) -> list[str]:
    parents = [_resource(sub) for sub in cls.findall(_RDFS + "subClassOf")]
    parents += [
        _local(member.get(_RDF + "about"))
        for member in cls.findall(f"{_OWL}equivalentClass/{_OWL}Class/{_OWL}intersectionOf/{_RDF}Description")
    ]
    return list(dict.fromkeys(p for p in parents if p is not None))


def descendants_with_roles(concepts: dict[str, RawConcept], roles: frozenset[str]) -> set[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for code, concept in concepts.items():
        for parent in concept.parents:
            children[parent].append(code)
    found = {code for code, concept in concepts.items() if concept.roles & roles}
    frontier = list(found)
    while frontier:
        for child in children[frontier.pop()]:
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def parse_owl(path: str | Path) -> tuple[dict[str, RawConcept], list[RawDefinition]]:
    concepts: dict[str, RawConcept] = {}
    definitions: list[RawDefinition] = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == _OWL + "Class":
            code = _local(el.get(_RDF + "about"))
            if code is not None:
                concepts[code] = RawConcept(
                    name=(el.findtext(_EVS + "P108") or "").strip(),
                    semantic_types=[s.text for s in el.findall(_EVS + "P106") if s.text],
                    statuses=[s.text for s in el.findall(_EVS + "P310") if s.text],
                    parents=_named_parents(el),
                    roles={
                        role for prop in el.iter(_OWL + "onProperty") if (role := _resource(prop)) is not None
                    },
                )
                el.clear()
        elif el.tag == _OWL + "Axiom":
            prop = _resource(el.find(_OWL + "annotatedProperty"))
            code = _resource(el.find(_OWL + "annotatedSource"))
            if prop in DEFINITION_PROPERTIES and code is not None:
                definitions.append(RawDefinition(
                    code=code,
                    kind=DEFINITION_PROPERTIES[prop],
                    text=(el.findtext(_OWL + "annotatedTarget") or "").strip(),
                    source=el.findtext(_EVS + "P378"),
                    attribution=el.findtext(_EVS + "P381"),
                ))
            el.clear()
    return concepts, definitions


def is_active(concept: RawConcept) -> bool:
    return not INACTIVE_STATUSES & set(concept.statuses)


def build_definitions(
    concepts: dict[str, RawConcept],
    definitions: list[RawDefinition],
    source_allowlist: frozenset[str] = SOURCE_ALLOWLIST,
    keep_tags: frozenset[str] = TRAILING_TAG_KEEP,
    excluded_roles: frozenset[str] = EXCLUDED_ROLES,
    concept_drop_markers: tuple[str, ...] = CONCEPT_DROP_MARKERS,
    template_min_count: int = TEMPLATE_MIN_COUNT,
) -> tuple[dict, list[dict], Counter, Counter, list[dict]]:
    log: list[dict] = []
    tag_counts: Counter = Counter()
    summary: Counter = Counter()

    def _log(d: RawDefinition, rule: str, cleaned: str = "") -> None:
        log.append({"code": d.code, "source": d.source, "rule": rule, "original": d.text, "cleaned": cleaned})

    active = {code for code, c in concepts.items() if is_active(c)}
    without_ctcae = {code for code in active if "ctcae" not in concepts[code].name.lower()}
    without_types = {
        code for code in without_ctcae if not EXCLUDED_SEMANTIC_TYPES & set(concepts[code].semantic_types)
    }
    without_roles = without_types - descendants_with_roles(concepts, excluded_roles)
    flagged = {d.code for d in definitions if any(marker in d.text for marker in concept_drop_markers)}
    eligible = without_roles - flagged
    summary["concepts_active"] = len(active)
    summary["concepts_without_ctcae_name"] = len(without_ctcae)
    summary["concepts_without_excluded_semantic_types"] = len(without_types)
    summary["concepts_without_excluded_roles"] = len(without_roles)
    summary["concepts_without_concept_markers"] = len(eligible)

    by_code: dict[str, list[tuple[RawDefinition, str]]] = defaultdict(list)
    for d in sorted(definitions, key=lambda d: d.kind != "DEFINITION"):
        if d.code not in eligible:
            continue
        summary["definitions_in"] += 1
        if d.source not in source_allowlist:
            summary["dropped_source"] += 1
            _log(d, "drop_source")
            continue
        if any(marker in d.text for marker in DROP_MARKERS):
            summary["dropped_marker"] += 1
            _log(d, "drop_marker")
            continue
        cleaned, rules, tags = clean_definition(d.text, keep_tags)
        tag_counts.update(normalize_tag(t) for t in tags)
        if rules:
            summary["cleaned"] += 1
            _log(d, "+".join(rules), cleaned)
        if len(tokens(cleaned)) < MIN_TOKENS:
            summary["dropped_short"] += 1
            _log(d, "drop_short", cleaned)
            continue
        kept = by_code[d.code]
        if any(jaccard(cleaned, text) >= DEDUP_JACCARD for _, text in kept):
            summary["dropped_duplicate"] += 1
            _log(d, "drop_duplicate", cleaned)
            continue
        kept.append((d, cleaned))

    frames = {
        (code, i): (d.source, template_frame(text, concepts[code].name))
        for code, kept in by_code.items()
        for i, (d, text) in enumerate(kept)
    }
    frame_counts = Counter(frames.values())
    examples: dict[tuple, str] = {}
    output: dict = {}
    for code, kept in sorted(by_code.items()):
        defs = []
        for i, (d, text) in enumerate(kept):
            key = frames[(code, i)]
            if frame_counts[key] >= template_min_count:
                summary["dropped_template"] += 1
                _log(d, "drop_template", text)
                examples.setdefault(key, text)
                continue
            defs.append({"text": text, "source": d.source, "attribution": d.attribution})
        if len(defs) >= MIN_DEFINITIONS:
            output[code] = {
                "name": concepts[code].name,
                "semantic_types": concepts[code].semantic_types,
                "parents": concepts[code].parents,
                "definitions": defs,
            }
    templates = [
        {"source": source, "frame": frame, "count": frame_counts[(source, frame)], "example": example}
        for (source, frame), example in sorted(examples.items(), key=lambda item: -frame_counts[item[0]])
    ]
    summary["concepts_out"] = len(output)
    summary["definitions_out"] = sum(len(c["definitions"]) for c in output.values())
    output["_hierarchy"] = {code: concepts[code].parents for code in sorted(active) if concepts[code].parents}
    return output, log, tag_counts, summary, templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cleaned multi-definition concepts from the NCIt OWL release.")
    parser.add_argument("--owl", default=DEFAULT_OWL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--tags", default=DEFAULT_TAGS)
    parser.add_argument("--templates", default=DEFAULT_TEMPLATES)
    args = parser.parse_args()

    print(f"Parsing {args.owl} ...")
    concepts, definitions = parse_owl(args.owl)
    print(f"  {len(concepts):,} classes, {len(definitions):,} definition axioms")
    output, log, tag_counts, summary, templates = build_definitions(concepts, definitions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "source", "rule", "original", "cleaned"])
        writer.writeheader()
        writer.writerows(log)
    with open(args.tags, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "count"])
        writer.writerows(tag_counts.most_common())
    with open(args.templates, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "frame", "count", "example"])
        writer.writeheader()
        writer.writerows(templates)

    for key, value in summary.items():
        print(f"  {key}: {value:,}")
    print(f"Saved {args.output}, {args.log}, {args.tags}, {args.templates}")


if __name__ == "__main__":
    main()
