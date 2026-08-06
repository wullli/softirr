
from abc import ABC, abstractmethod
from typing import Any

import litellm
from tqdm import tqdm

from soft_irr.evaluation.distances.base import DirectionalEntailmentCache
from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance

_DEFAULT_LLM = "huggingface/Qwen/Qwen3.5-35B-A3B:novita"


class _BipartiteLLMBase(DirectionalEntailmentCache, BipartiteMatchingDistance, ABC):

    def __init__(
        self,
        model_name: str = _DEFAULT_LLM,
        match_threshold: float = 1 / 3,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._init_cost_cache()

    def _load(self) -> None:
        pass

    @abstractmethod
    def _build_messages(self, x: str, y: str) -> list[dict[str, str]]:
        ...

    @abstractmethod
    def _completion_kwargs(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def _parse_response(self, response: Any) -> Any:
        ...

    def _prepare(
        self,
        parsed: list[tuple[Any, Any, list[str], list[str]]],
        batch_size: int,
    ) -> None:
        unique_pairs = self._pending_keys(parsed)
        self.last_n_prompts = len(unique_pairs)
        if not unique_pairs:
            return

        for i in tqdm(
            range(0, len(unique_pairs), batch_size), desc="  LLM inference", leave=False, unit="batch"
        ):
            chunk = unique_pairs[i : i + batch_size]
            messages_list = [self._build_messages(x, y) for x, y in chunk]
            responses = litellm.batch_completion(
                model=self.model_name,
                messages=messages_list,
                **self._completion_kwargs(),
            )
            for key, resp in zip(chunk, responses):
                if isinstance(resp, Exception):
                    raise resp
                self._store_cost(key, self._parse_response(resp))


class BipartiteLogProbDistance(_BipartiteLLMBase):

    _YES_TOKENS: frozenset[str] = frozenset({"yes", " yes", "Yes", " Yes", "YES"})
    _NO_TOKENS: frozenset[str] = frozenset({"no", " no", "No", " No", "NO"})
    _TOP_LOGPROBS: int = 2

    def __init__(
        self,
        model_name: str = _DEFAULT_LLM,
        match_threshold: float = 1 / 3,
        extra_body_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._extra_body_overrides = extra_body_overrides

    def _build_messages(self, x: str, y: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    f'Does "{y}" semantically entail "{x}"? '
                    f'Answer "yes" or "no":'
                ),
            }
        ]

    def _completion_kwargs(self) -> dict[str, Any]:
        extra_body: dict[str, Any] = (
            self._extra_body_overrides
            if self._extra_body_overrides is not None
            else {
                "logprobs": True,
                "top_logprobs": self._TOP_LOGPROBS,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        return {
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self._TOP_LOGPROBS,
            "extra_body": extra_body,
        }

    def _parse_response(self, response: Any) -> float:
        import math

        first = response.choices[0].logprobs.content[0]
        top_lps = first.top_logprobs or []
        yes_lp = next((lp.logprob for lp in top_lps if lp.token in self._YES_TOKENS), None)
        no_lp = next((lp.logprob for lp in top_lps if lp.token in self._NO_TOKENS), None)
        if yes_lp is None and first.token in self._YES_TOKENS:
            yes_lp = first.logprob
        if no_lp is None and first.token in self._NO_TOKENS:
            no_lp = first.logprob

        if yes_lp is None and no_lp is None:
            return 0.5
        if yes_lp is None:
            return 1.0
        if no_lp is None:
            return 0.0
        p_yes = math.exp(yes_lp)
        p_no = math.exp(no_lp)
        return 1.0 - p_yes / (p_yes + p_no)


bipartite_log_prob_distance = BipartiteLogProbDistance()


class BipartiteVerbalizedProbDistance(_BipartiteLLMBase):

    _BOTH_DIRECTIONS = False

    def _build_messages(self, x: str, y: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert in semantic textual entailment. Respond with only two "
                    "decimal numbers separated by a comma, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Consider two annotations, X: "{x}" and Y: "{y}". '
                    f'Rate from 0.0 (no relation) to 1.0 (complete semantic entailment) both '
                    f'directions: how much does Y entail X, and how much '
                    f'does X entail Y. Output exactly two decimal numbers separated by a comma, '
                    f'in the form "<Y entails X>,<X entails Y>", nothing else.'
                ),
            },
        ]

    def _completion_kwargs(self) -> dict[str, Any]:
        return {"temperature": 0}

    def _parse_response(self, response: Any) -> tuple[float, float]:
        content = response.choices[0].message.content.strip()
        parts = content.split(",")
        if len(parts) != 2:
            return 0.5, 0.5
        try:
            y_entails_x = max(0.0, min(1.0, float(parts[0].strip())))
            x_entails_y = max(0.0, min(1.0, float(parts[1].strip())))
            return 1.0 - y_entails_x, 1.0 - x_entails_y
        except ValueError:
            return 0.5, 0.5


bipartite_verbalized_prob_distance = BipartiteVerbalizedProbDistance()
