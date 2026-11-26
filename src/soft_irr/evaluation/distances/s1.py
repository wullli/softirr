import importlib.util
from typing import Any

from tqdm import tqdm

from soft_irr.evaluation.distances.nli import BipartiteNLIDistance

_S1_PUBMEDBERT = "pritamdeka/S1-PubMedBERT"
# Pinned: the model ships its own inference code (sbert2s1.py), reviewed at this commit.
_S1_PUBMEDBERT_REVISION = "dd6b2917665669d6008b76cf05d59900bcc0df80"


class BipartiteS1Distance(BipartiteNLIDistance):
    """Directional entailment from an sbert2s1 typed-decision model, asked as a yes/no ("noul") question.

    P(a entails b) is the model's calibrated P(true) for `question_template` filled with b, asked about
    `text_template` filled with a; the pair cost is the same geometric mean of both directions as
    BipartiteNLIDistance. `criteria` optionally describes the {"false": ..., "true": ...} answers.
    """

    DEFAULT_QUESTION = 'Does this text imply that "{hypothesis}" holds?'

    def __init__(
        self,
        model_name: str = _S1_PUBMEDBERT,
        revision: str = _S1_PUBMEDBERT_REVISION,
        match_threshold: float = 1 / 3,
        question_template: str = DEFAULT_QUESTION,
        text_template: str = "{premise}",
        criteria: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self.revision = revision
        self.question_template = question_template
        self.text_template = text_template
        self.criteria = criteria

    def _load(self) -> None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.model_name, "sbert2s1.py", revision=self.revision)
        spec = importlib.util.spec_from_file_location("sbert2s1", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load sbert2s1 inference code from {path}")
        sbert2s1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sbert2s1)
        self._pipeline = sbert2s1.load(self.model_name, revision=self.revision)

    def _question(self, hypothesis: str) -> dict[str, Any]:
        question: dict[str, Any] = {
            "type": "noul",
            "instructions": self.question_template.format(hypothesis=hypothesis),
        }
        if self.criteria is not None:
            question["criteria"] = self.criteria
        return {"entails": question}

    def _entailment_probs(self, keys: list[tuple[str, str]], batch_size: int) -> list[float]:
        probs: list[float] = []
        for i in tqdm(range(0, len(keys), batch_size), desc="  S1 inference", leave=False, unit="batch"):
            requests = [
                (self.text_template.format(premise=premise), self._question(hypothesis))
                for premise, hypothesis in keys[i : i + batch_size]
            ]
            out = self._pipeline.predict_batch(requests, batch_size=batch_size)
            probs.extend(o["answers"]["entails"]["noul"] for o in out)
        return probs


bipartite_s1_pubmedbert_distance = BipartiteS1Distance()
