# ICD-11 Data

**Source:** WHO ICD-11 Mortality and Morbidity Statistics (MMS) linearization

**Download:** https://icd.who.int/browse/2025-01/mms/en
- Navigate to *Downloads* → *Linearization files* → select language (English) and release
- Or use the ICD-11 API: https://icd.who.int/icdapi

**Expected file:** `LinearizationMiniOutput-MMS-en.txt`
- Tab-separated, columns include: `Foundation URI`, `Code`, `Title`, `ClassKind`, `isLeaf`
- Version used in experiments: 2026 Apr 15

**Fetching synonyms:**

The linearization file only contains preferred titles. Synonyms and inclusions are fetched from a local `whoicd/icd-api` Docker container (no credentials needed):

```bash
python src/soft_irr/fetch_icd11_synonyms.py --docker
```

This starts the container automatically, fetches all ~36k category codes (leaf and non-leaf), saves to `data/icd11/synonyms.json`, and stops the container on exit. Resumable if interrupted — re-run with the same command.

To use a container you started manually:
```bash
docker run -d -p 8080:80 --env acceptLicense=true whoicd/icd-api
python src/soft_irr/fetch_icd11_synonyms.py
```

Pass the result to the experiment script:
```bash
python -m soft_irr.experiments.synthetic_irr \
    --dataset icd11 --data-path data/icd11/LinearizationMiniOutput-MMS-en.txt \
    --icd11-synonyms data/icd11/synonyms.json
```

**Notes:**
- All `category` entries (leaf and non-leaf) with at least one fetched synonym/inclusion term are used in experiments; non-leaf categories without any (the API sometimes returns none) are dropped, same as leaves.
- Downloads at https://icd.who.int/dev11/downloads only provide simplified linearizations without synonyms