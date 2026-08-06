# Measuring Consensus in Unstructured Biomedical Text-Annotations

[![Tests](https://github.com/wullli/softirr/actions/workflows/tests.yml/badge.svg)](https://github.com/wullli/softirr/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/wullli/softirr/branch/main/graph/badge.svg)](https://codecov.io/gh/wullli/softirr)

Soft inter-rater reliability (IRR) metrics for medical NLP evaluation: distance functions
between annotation label sets, plus Cohen's kappa / average-agreement scoring built on top of
them.

## Requirements

- Python >= 3.10
- git (for the `meddra_graph` submodule, used by the MedDRA taxonomy loader)
- Optional: an API key in `.env` (e.g. `OPENAI_API_KEY`) for the LLM-based distance measures

## Installation

```bash
git clone --recurse-submodules <repo-url>
cd soft-irr
pip install -e ".[dev]"
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## Usage

### Distance between two label sets

```python
from soft_irr.evaluation.distances import exact_match_distance, jaccard_distance

rater_a_label = ["diabetes"]
rater_b_label = ["diabetes mellitus"]

print(exact_match_distance(rater_a_label, rater_b_label))  # 1.0 (no exact string match)
print(jaccard_distance(rater_a_label, rater_b_label))       # 0.5 (partial token overlap)
```

### Agreement score over a dataset

```python
from soft_irr.evaluation.distances import jaccard_distance
from soft_irr.evaluation.scores import AgreementScores

rater_a = [["diabetes"], ["asthma"], ["migraine"]]
rater_b = [["diabetes mellitus"], ["asthma"], ["headache"]]

scores = AgreementScores.get(rater_a, rater_b, distance_function=jaccard_distance)
print(scores.average_agreement)  # 0.5
print(scores.cohens_kappa)       # 0.4375
```
