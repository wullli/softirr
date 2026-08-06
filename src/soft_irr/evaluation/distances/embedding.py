
from typing import Any

import numpy as np
from tqdm import tqdm

from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance


class BipartiteEmbeddingDistance(BipartiteMatchingDistance):

    def __init__(
        self,
        model_name: str = "NeuML/pubmedbert-base-embeddings",
        match_threshold: float = 0.9,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._embedding_cache: dict[str, Any] = {}

    def _load(self) -> None:
        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(
            "feature-extraction",
            model=self.model_name,
            truncation=True,
            max_length=128,
        )

    def _get_embeddings(self, texts: list[str], batch_size: int) -> Any:
        if self._pipeline is None:
            self._load()
        if self._pipeline is None:
            raise ValueError("Failed to load embedding model pipeline.")
        raw: list[Any] = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc="  embedding inference", leave=False, unit="batch"
        ):
            chunk = texts[i : i + batch_size]
            raw.extend(self._pipeline(chunk, batch_size=batch_size))
        pooled = []
        for r in raw:
            arr = np.array(r)
            if arr.ndim == 3:
                arr = arr[0]
            pooled.append(arr.mean(axis=0))
        embs = np.stack(pooled)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return embs / norms

    def _prepare(
        self,
        parsed: list[tuple[Any, Any, list[str], list[str]]],
        batch_size: int,
    ) -> None:
        all_strs = sorted({s for _, _, ts, ps in parsed for s in ts + ps})
        uncached = [s for s in all_strs if s not in self._embedding_cache]
        self.last_n_prompts = len(uncached)
        if uncached:
            embs = self._get_embeddings(uncached, batch_size)
            for s, emb in zip(uncached, embs):
                self._embedding_cache[s] = emb

    def _cost(self, a: str, b: str) -> float:
        if self._string_distance_cache is not None:
            return self._string_distance_cache.get((a, b), 1.0)
        return float(1.0 - np.dot(self._embedding_cache[a], self._embedding_cache[b]))


bipartite_embedding_distance = BipartiteEmbeddingDistance()


class BipartiteQwen3EmbeddingDistance(BipartiteEmbeddingDistance):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        match_threshold: float = 0.9,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="right")
        self._model = AutoModel.from_pretrained(self.model_name)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def _get_embeddings(self, texts: list[str], batch_size: int) -> Any:
        import torch
        import torch.nn.functional as F
        if self._tokenizer is None or self._model is None:
            self._load()
        all_embs: list[Any] = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc="  embedding inference", leave=False, unit="batch"
        ):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            embs = outputs.last_hidden_state[torch.arange(len(batch)), last_idx]
            embs = F.normalize(embs, p=2, dim=1).to(torch.float32).cpu().numpy()
            all_embs.append(embs)
        return np.concatenate(all_embs, axis=0)

    def release(self) -> None:
        self._tokenizer = None
        self._model = None
        super().release()


bipartite_qwen3_embedding_distance = BipartiteQwen3EmbeddingDistance()
