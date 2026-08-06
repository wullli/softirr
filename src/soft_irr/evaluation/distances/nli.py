
import math
from typing import Any

from tqdm import tqdm

from soft_irr.evaluation.distances.base import trivial_distance
from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance


class BipartiteNLIDistance(BipartiteMatchingDistance):

    is_pair_cacheable = True

    def __init__(
        self,
        model_name: str = "pritamdeka/PubMedBERT-MNLI-MedNLI",
        match_threshold: float = 1 / 3,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._entailment_cache: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            max_length=512,
            truncation=True,
        )

    def _prepare(
        self,
        parsed: list[tuple[Any, Any, list[str], list[str]]],
        batch_size: int,
    ) -> None:
        nli_inputs: list[dict[str, str]] = []
        input_keys: list[tuple[str, str]] = []
        queued: set[tuple[str, str]] = set()

        for label1, label2, strs_a, strs_b in parsed:
            if trivial_distance(label1, label2, strs_a, strs_b) is not None:
                continue
            for a in strs_a:
                for b in strs_b:
                    if self._is_cached(a, b):
                        continue
                    for premise, hypothesis in ((a, b), (b, a)):
                        key = (premise, hypothesis)
                        if key not in self._entailment_cache and key not in queued:
                            queued.add(key)
                            input_keys.append(key)
                            nli_inputs.append({"text": premise, "text_pair": hypothesis})

        self.last_n_prompts = len(nli_inputs)
        if not nli_inputs:
            return
        if self._pipeline is None:
            self._load()
        if self._pipeline is None:
            raise ValueError("Failed to load NLI model pipeline.")
        raw: list[Any] = []
        for i in tqdm(
            range(0, len(nli_inputs), batch_size), desc="  NLI inference", leave=False, unit="batch"
        ):
            chunk = nli_inputs[i : i + batch_size]
            raw.extend(self._pipeline(chunk, batch_size=batch_size))
        for key, result in zip(input_keys, raw):
            entailment = next(
                (r["score"] for r in result if r["label"].lower() == "entailment"), 0.0
            )
            self._entailment_cache[key] = entailment

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        forward = self._entailment_cache.get((a, b), 0.0)
        backward = self._entailment_cache.get((b, a), 0.0)
        return 1.0 - math.sqrt(forward * backward)


bipartite_nli_distance = BipartiteNLIDistance()

bipartite_nli_multilingual_distance = BipartiteNLIDistance(
    model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)
