# MedDRA Data

**Source:** Medical Dictionary for Regulatory Activities (MedDRA), distributed by MSSO

**Download:** https://www.meddra.org/software-packages
- Requires a valid MedDRA licence (MSSO subscription)
- Select *ASCII format* for the version matching the release used

**Version used in experiments:** 28.0 (English)

**Directory structure after download:**
```
meddra/
├── MedAscii/          # Pipe-delimited ASCII files (primary)
│   ├── llt.asc        # Lowest Level Terms — synonyms of PTs
│   ├── pt.asc         # Preferred Terms — one per concept
│   ├── hlt.asc        # High Level Terms
│   ├── hlgt.asc       # High Level Group Terms
│   ├── soc.asc        # System Organ Classes
│   ├── hlt_pt.asc     # HLT↔PT hierarchy links
│   ├── hlgt_hlt.asc   # HLGT↔HLT links
│   └── soc_hlgt.asc   # SOC↔HLGT links
└── SeqAscii/          # Sequential ASCII variant (same content)
```

**Notes for experiments:**
- Each PT in `pt.asc` is the preferred term; its LLTs in `llt.asc` (linked by `pt_code`) serve as synonyms
- Filter to LLTs where `llt_currency = 'Y'` (non-obsolete)
- Fields in `llt.asc`: `llt_code|llt_name|pt_code|llt_whoart_code|llt_harts_code|llt_costart_sym|llt_icd9_code|llt_icd9cm_code|llt_icd10_code|llt_currency|llt_jart_code`
- Fields in `pt.asc`: `pt_code|pt_name|null_field|pt_soc_code|pt_whoart_code|pt_harts_code|pt_costart_sym|pt_icd9_code|pt_icd9cm_code|pt_icd10_code|pt_jart_code`