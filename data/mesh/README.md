# MeSH Data

**Source:** Medical Subject Headings (MeSH), National Library of Medicine (NLM)

**Download:** https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/
- Download `desc{YEAR}.xml` (descriptor records, ~250 MB) — this is the primary file needed
- Optionally: `supp{YEAR}.xml` (supplementary concept records with additional synonyms)

**Expected file:** `desc2025.xml` (or current year's release)

**Notes for experiments:**
- Each `<DescriptorRecord>` contains:
  - `<DescriptorName>/<String>` — the preferred term
  - `<ConceptList>/<Concept>/<TermList>/<Term>/<String>` — synonyms (Terms within the preferred Concept where `ConceptPreferredTermYN=Y` or `RecordPreferredTermYN=N`)
  - `<TreeNumberList>` — MeSH tree location (use to filter by branch, e.g. `C` for Diseases, `A` for Anatomy)
- Filter `<DescriptorClass>` = `1` (topical descriptors) to exclude publication types and geographic terms
- Synonyms within the *preferred concept* (`PreferredConceptYN=Y`) are the closest equivalents; terms from non-preferred concepts represent related but distinct meanings