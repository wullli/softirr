
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Indel


@dataclass
class ConceptEntry:
    code: str
    preferred_term: str
    synonyms: list[str]
    parent_codes: list[str] = field(default_factory=list)
    hierarchy_key: list[str] = field(default_factory=list)


def _icd11_category_parent_codes(rows: list[dict]) -> dict[str, str]:
    category_codes = {
        row["Code"].strip() for row in rows if row.get("ClassKind") == "category" and row.get("Code", "").strip()
    }
    parent_of: dict[str, str] = {}
    for row in rows:
        if row.get("ClassKind") != "category":
            continue
        code = row.get("Code", "").strip()
        if not code:
            continue
        if "." in code:
            dot_parent = code.rsplit(".", 1)[0]
            if dot_parent in category_codes:
                parent_of[code] = dot_parent
                continue
        groupings = [row.get(f"Grouping{i}", "").strip() for i in range(1, 6)]
        block = next((g for g in reversed(groupings) if g), None)
        if block:
            parent_of[code] = block
    return parent_of


def load_icd11(path: str | Path, synonyms_json: str | Path | None = None) -> dict[str, ConceptEntry]:
    synonyms: dict[str, list[str]] = {}
    if synonyms_json is not None and Path(synonyms_json).exists():
        with open(synonyms_json, encoding="utf-8") as f:
            synonyms = json.load(f)

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    parent_of = _icd11_category_parent_codes(rows)

    concepts: dict[str, ConceptEntry] = {}
    for row in rows:
        if row.get("ClassKind") != "category":
            continue
        code = row.get("Code", "").strip()
        if not code:
            continue
        title = row["Title"].strip().strip('"').lstrip("- ").strip()
        syns = synonyms.get(code, [])
        if not syns:
            continue
        parent = parent_of.get(code)
        concepts[code] = ConceptEntry(
            code=code,
            preferred_term=title,
            synonyms=syns,
            parent_codes=[parent] if parent else [],
            hierarchy_key=[code],
        )

    return concepts


def load_meddra(path: str | Path) -> dict[str, ConceptEntry]:
    from meddra_graph.meddra_loader import MedDRALoader

    data = MedDRALoader.load(Path(path) / "MedAscii")

    pts = {
        code: term["pt_name"]
        for code, term in data.terms.items()
        if term["term_type"] == "pt"
    }
    llt_by_pt: dict[str, list[str]] = {code: [] for code in pts}
    for term in data.terms.values():
        if term["term_type"] != "llt" or term["llt_currency"] != "Y":
            continue
        pt_code = term["pt_code"]
        if pt_code not in llt_by_pt:
            continue
        llt_name = term["llt_name"]
        if llt_name != pts[pt_code]:
            llt_by_pt[pt_code].append(llt_name)

    pt_to_hlts: dict[str, list[str]] = defaultdict(list)
    hlt_pt_path = Path(path) / "MedAscii" / "hlt_pt.asc"
    if hlt_pt_path.exists():
        with open(hlt_pt_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("$")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    pt_to_hlts[parts[1]].append(parts[0])

    return {
        code: ConceptEntry(
            code=code,
            preferred_term=name,
            synonyms=llt_by_pt[code],
            parent_codes=pt_to_hlts.get(code, []),
        )
        for code, name in pts.items()
        if llt_by_pt[code]
    }


def load_mesh(path: str | Path) -> dict[str, ConceptEntry]:
    concepts: dict[str, ConceptEntry] = {}
    tree = ET.parse(path)

    for record in tree.getroot().iter("DescriptorRecord"):
        if record.get("DescriptorClass") != "1":
            continue
        ui_el = record.find("DescriptorUI")
        name_el = record.find("DescriptorName/String")
        if ui_el is None or name_el is None:
            continue

        preferred_term = (name_el.text or "").strip()
        synonyms: list[str] = []
        for concept in record.iter("Concept"):
            if concept.get("PreferredConceptYN") != "Y":
                continue
            for term in concept.iter("Term"):
                if term.get("RecordPreferredTermYN") == "Y":
                    continue
                t = (term.findtext("String") or "").strip()
                if t and t != preferred_term:
                    synonyms.append(t)

        tree_numbers = [
            (tn.text or "").strip()
            for tn in record.findall("TreeNumberList/TreeNumber")
            if tn.text
        ]
        parent_tree_numbers = sorted({
            ".".join(tn.split(".")[:-1])
            for tn in tree_numbers
            if "." in tn
        })

        if synonyms:
            ui = (ui_el.text or "").strip()
            concepts[ui] = ConceptEntry(
                code=ui,
                preferred_term=preferred_term,
                synonyms=synonyms,
                parent_codes=parent_tree_numbers,
                hierarchy_key=tree_numbers,
            )

    return concepts


def build_sibling_index(concepts: dict[str, ConceptEntry]) -> dict[str, list[str]]:
    parent_to_codes: dict[str, list[str]] = defaultdict(list)
    for code, entry in concepts.items():
        for parent in entry.parent_codes:
            parent_to_codes[parent].append(code)

    sibling_index: dict[str, list[str]] = {}
    for code, entry in concepts.items():
        sibling_set: set[str] = set()
        for parent in entry.parent_codes:
            sibling_set.update(parent_to_codes[parent])
        sibling_set.discard(code)
        sibling_index[code] = sorted(sibling_set)
    return sibling_index


def load_meddra_ancestor_names(path: str | Path) -> dict[str, str]:
    from meddra_graph.meddra_loader import MedDRALoader

    data = MedDRALoader.load(Path(path) / "MedAscii")
    return {
        code: term[f"{term['term_type']}_name"]
        for code, term in data.terms.items()
        if term["term_type"] in ("hlt", "hlgt", "soc")
    }


def load_meddra_ancestor_hierarchy(path: str | Path) -> dict[str, list[str]]:

    def _load_pairs(filename: str) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = defaultdict(list)
        file_path = Path(path) / "MedAscii" / filename
        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("$")
                    if len(parts) >= 2 and parts[0] and parts[1]:
                        mapping[parts[1]].append(parts[0])
        return mapping

    hierarchy: dict[str, list[str]] = {}
    hierarchy.update(_load_pairs("hlgt_hlt.asc"))
    hierarchy.update(_load_pairs("soc_hlgt.asc"))
    return hierarchy


def load_icd11_ancestor_titles(path: str | Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("ClassKind") == "block":
                code = row.get("BlockId", "").strip()
            elif row.get("ClassKind") == "category":
                code = row.get("Code", "").strip()
            else:
                continue
            if not code:
                continue
            titles[code] = row["Title"].strip().strip('"').lstrip("- ").strip()
    return titles


def load_icd11_ancestor_hierarchy(path: str | Path) -> dict[str, list[str]]:
    block_parents: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        groupings = [row.get(f"Grouping{i}", "").strip() for i in range(1, 6)]
        non_empty = [g for g in groupings if g]
        for parent, child in zip(non_empty, non_empty[1:]):
            block_parents[child].add(parent)
    hierarchy: dict[str, list[str]] = {code: sorted(parents) for code, parents in block_parents.items()}
    for code, parent in _icd11_category_parent_codes(rows).items():
        hierarchy[code] = [parent]
    return hierarchy


def build_hierarchy_index(
    concepts: dict[str, ConceptEntry],
    direction: str,
    parent_terms: dict[str, str] | None = None,
    ancestor_parent_codes: dict[str, list[str]] | None = None,
    hops: int = 1,
) -> dict[str, list[ConceptEntry]]:
    if direction not in ("parent", "children"):
        raise ValueError(f"direction must be 'parent' or 'children', got {direction!r}")

    hierarchy_index: dict[str, list[ConceptEntry]] = {}

    if direction == "parent":
        parent_terms = parent_terms or {}
        ancestor_parent_codes = ancestor_parent_codes or {}
        code_by_hierarchy_key = {
            key: code for code, entry in concepts.items() for key in entry.hierarchy_key
        }
        for code, entry in concepts.items():
            candidates: list[ConceptEntry] = []
            seen_codes: set[str] = set()
            visited_ancestor_codes: set[str] = set()
            frontier = list(entry.parent_codes)
            hop = 0
            while frontier and hop < hops:
                next_frontier: list[str] = []
                for ancestor_code in frontier:
                    if ancestor_code in visited_ancestor_codes:
                        continue
                    visited_ancestor_codes.add(ancestor_code)
                    matched_code = code_by_hierarchy_key.get(ancestor_code)
                    if matched_code is not None:
                        resolved = concepts[matched_code]
                        if resolved.code not in seen_codes:
                            seen_codes.add(resolved.code)
                            candidates.append(resolved)
                        next_frontier.extend(resolved.parent_codes)
                    elif ancestor_code in parent_terms:
                        if ancestor_code not in seen_codes:
                            seen_codes.add(ancestor_code)
                            title = parent_terms[ancestor_code]
                            candidates.append(ConceptEntry(code=ancestor_code, preferred_term=title, synonyms=[title]))
                    next_frontier.extend(ancestor_parent_codes.get(ancestor_code, []))
                frontier = next_frontier
                hop += 1
            hierarchy_index[code] = candidates
    else:
        parent_to_children: dict[str, list[str]] = defaultdict(list)
        for code, entry in concepts.items():
            for parent in entry.parent_codes:
                parent_to_children[parent].append(code)
        for code, entry in concepts.items():
            candidates = []
            for key in entry.hierarchy_key:
                for child_code in parent_to_children.get(key, []):
                    if child_code != code:
                        candidates.append(concepts[child_code])
            hierarchy_index[code] = candidates

    return hierarchy_index


def build_lexical_index(
    concepts: dict[str, ConceptEntry], top_k: int = 10, candidate_pool_size: int | None = None,
) -> dict[str, list[ConceptEntry]]:
    candidate_pool_size = candidate_pool_size or max(top_k * 5, 50)
    codes = list(concepts.keys())
    terms = [concepts[c].preferred_term for c in codes]
    n = len(codes)
    chunk_size = 500

    coarse_candidates: dict[str, list[str]] = {}
    for start in range(0, n, chunk_size):
        chunk_codes = codes[start:start + chunk_size]
        chunk_terms = terms[start:start + chunk_size]
        sims = process.cdist(chunk_terms, terms, scorer=Indel.normalized_similarity, workers=-1)
        for i, code in enumerate(chunk_codes):
            row = sims[i]
            self_idx = start + i
            k = min(candidate_pool_size + 1, n)
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(-row[top])]
            coarse_candidates[code] = [codes[j] for j in top if j != self_idx][:candidate_pool_size]

    lexical_index: dict[str, list[ConceptEntry]] = {}
    for code, cand_codes in coarse_candidates.items():
        if not cand_codes:
            lexical_index[code] = []
            continue
        forms_a = [concepts[code].preferred_term, *concepts[code].synonyms]
        cand_forms: list[str] = []
        boundaries = [0]
        for cand_code in cand_codes:
            cand = concepts[cand_code]
            cand_forms.extend([cand.preferred_term, *cand.synonyms])
            boundaries.append(len(cand_forms))
        sims = process.cdist(forms_a, cand_forms, scorer=Indel.normalized_similarity, workers=-1)
        mean_over_a = sims.mean(axis=0)
        scored = [
            (float(mean_over_a[boundaries[j]:boundaries[j + 1]].mean()), cand_codes[j])
            for j in range(len(cand_codes))
        ]
        scored.sort(key=lambda x: -x[0])
        lexical_index[code] = [concepts[cc] for _, cc in scored[:top_k]]
    return lexical_index
