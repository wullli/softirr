# Consensus Measures for Unstructured Biomedical Text Annotations

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

### Precision, recall and F1 against a reference

`SetPRFScores` pairs the labels of each item and scores `rater_b` against the reference
`rater_a`. Bipartite distances use their own matching (one result per strategy via
`get_by_matching`); other distances are applied to each pair of labels and matched
optimally. A pair is a true positive when its cost is below `match_threshold`; with
`soft=True` it counts its match quality `1 - cost` instead.

```python
from soft_irr.evaluation.distances import BipartiteJaccardDistance
from soft_irr.evaluation.set_prf import SetPRFScores

reference = [["dry mouth", "nausea"], ["fatigue"]]
predicted = [["mouth dry"], ["fatigue", "headache"]]

scores = SetPRFScores.get(reference, predicted, distance_function=BipartiteJaccardDistance())
print(scores.micro.precision, scores.micro.recall, scores.micro.f1)  # 0.666... 0.666... 0.666...
```

### Two-level (parent → child) precision, recall and F1

For nested labels, e.g. adverse events per treatment arm, `HierarchicalPRFScores` first
matches the parents, then the children only inside aligned parents. Children of
unmatched parents become false negatives or false positives, so `end_to_end` includes
the parent errors; `conditional` restricts to aligned parents and `pooled` ignores them.

```python
from soft_irr.evaluation.hierarchical_prf import HierarchicalPRFScores

reference = [{"low dose": ["nausea", "neck pain"], "high dose": ["dry mouth"]}]
predicted = [{"low dose": ["nausea"], "high activity": ["dry mouth"]}]

scores = HierarchicalPRFScores.get(reference, predicted)
print(scores.parent.f1)          # 0.5
print(scores.conditional.recall) # 0.5
print(scores.end_to_end.recall)  # 0.333...
print(scores.pooled.recall)      # 0.666...
```
