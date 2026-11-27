# NCI Thesaurus (NCIt) Data

**Source:** NCI Thesaurus, NCI Enterprise Vocabulary Services (EVS)

**Download:** https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/Thesaurus.OWL.zip
- Unzip to `data/ncit/Thesaurus.owl` (about 750 MB)
- Version used in experiments: 26.10d

**Why NCIt:** many concepts have more than one independent definition: the NCI definition (`P97`) and alternative definitions (`P325`) from other sources such as NCI-GLOSS, NICHD, FDA or CDISC. The source of each definition is in `P378`. In the synthetic experiments, the definitions of one concept take the role of synonyms. This gives a long-form version of the short-form ICD-11, MedDRA and MeSH experiments.

**Preprocessing:**

```bash
python -m soft_irr.preprocess_ncit --owl data/ncit/Thesaurus.owl
```

Steps, in order:
1. Keep active concepts only. Remove `Retired_Concept` and `Obsolete_Concept`.
2. Remove concepts that have "CTCAE" in the name. These are grading criteria or copies of a base concept.
3. Remove concepts with a semantic type in `EXCLUDED_SEMANTIC_TYPES`: Intellectual Product (CDISC code lists, questionnaire items, trial documents) and Laboratory Procedure (definitions that follow a template and differ only in the analyte name). Also remove staging concepts: concepts that use a role in `EXCLUDED_ROLES` (`R88` `Disease_Is_Stage`) in their logical definition, and all their descendants. An example is "Stage IIIA Adult Liver Cancer AJCC v7". Also remove every concept in which any definition contains a marker in `CONCEPT_DROP_MARKERS`. This is a fixed NCIt policy notice that is added to some definitions and is not part of the definition text.
4. Keep only definitions from sources in `SOURCE_ALLOWLIST`.
5. Remove definitions that contain a marker in `DROP_MARKERS` ("This definition may be outdated", FACIT copyright text).
6. Clean the text:
   - Cut BRIDG field labels (`EXAMPLE(S):`, `NOTE(S):`, `OTHER NAME(S):`).
   - Remove URLs.
   - Remove a ` --<year>` suffix.
   - Remove bracketed tags at the end that come after a full stop, `!` or `?`, for example "(from AJCC 8th Ed.)" or "(NCI04)". Tags in `TRAILING_TAG_KEEP` are kept.
   - Do not change source names inside the text, or "also known as" clauses.
7. Remove definitions with fewer than 5 tokens.
8. Deduplicate within each concept. Definitions with a token Jaccard of 0.7 or more count as duplicates. The NCI `P97` definition is kept first.
9. Keep concepts with 2 or more definitions.

**Outputs:**
- `definitions.json`:
  - `{code: {name, semantic_types, parents, definitions: [{text, source, attribution}]}}`
  - plus `_hierarchy`, the `rdfs:subClassOf` parents of all active concepts. Parent-mode negatives use it.
- `cleaning_log.csv`: one row for each changed or removed definition. Columns: `code, source, rule, original, cleaned`.
- `trailing_tags.csv`: how often each removed trailing tag occurs. Digits are shown as `#`. Use it to review `TRAILING_TAG_KEEP`.

**Running the experiment:**

```bash
python -m soft_irr.experiments.synthetic_irr --dataset ncit --negative-mode sibling
python -m soft_irr.experiments.synthetic_irr --dataset ncit --ncit-semantic-types clinical
```

`--ncit-semantic-types` takes NCIt semantic types (`P106`). `clinical` expands to `NCIT_CLINICAL_SEMANTIC_TYPES` in `load_taxonomies.py`. In parent mode, ancestors that are not in the pool are traversed, but they are never used as negatives. This means every negative is a definition and never a short name.
