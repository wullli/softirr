
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from tqdm import tqdm

from soft_irr.evaluation.distances.base import DirectionalEntailmentCache
from soft_irr.evaluation.distances.bipartite_matching import BipartiteMatchingDistance

logger = logging.getLogger(__name__)

_MEDGEMMA_LOCAL = "google/medgemma-1.5-4b-it"
_QWEN35_LOCAL = "Qwen/Qwen3.5-9B"


class _BipartiteLocalLLMBase(DirectionalEntailmentCache, BipartiteMatchingDistance, ABC):

    _loaded_models: dict[tuple[str, bool], tuple[Any, Any]] = {}

    _PROMPT_MAX_LENGTH = 1024

    def __init__(
        self,
        model_name: str,
        match_threshold: float = 1 / 3,
        enable_reasoning: bool = False,
        thought_skip_prefill: str = "",
    ) -> None:
        super().__init__(model_name, match_threshold)
        self._tokenizer: Any = None
        self._model: Any = None
        self._init_cost_cache()
        self.enable_reasoning = enable_reasoning
        self.thought_skip_prefill = thought_skip_prefill
        self.fp16 = False

    def set_reasoning(self, enable_reasoning: bool) -> None:
        self.enable_reasoning = enable_reasoning

    def set_precision(self, fp16: bool) -> None:
        self.fp16 = fp16

    _REASONING_BATCH_SIZE = 4

    def _effective_batch_size(self, batch_size: int) -> int:
        return min(batch_size, self._REASONING_BATCH_SIZE) if self.enable_reasoning else batch_size

    @property
    def _is_gemma(self) -> bool:
        return "gemma" in self.model_name

    @property
    def _load_key(self) -> tuple[str, bool]:
        return (self.model_name, self.fp16 and self._is_gemma)

    def _load(self) -> None:
        cached = _BipartiteLocalLLMBase._loaded_models.get(self._load_key)
        if cached is not None:
            self._tokenizer, self._model = cached
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        free_bytes, _ = torch.cuda.mem_get_info()
        gpu_budget_gib = free_bytes / (1024**3) - 4
        use_fp16 = not self._is_gemma or self.fp16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto",
            max_memory={0: f"{gpu_budget_gib:.1f}GiB", "cpu": "0GiB"},
            low_cpu_mem_usage=True,
            attn_implementation="eager" if (self._is_gemma and self.fp16) else "sdpa",
        )
        self._model.eval()
        _BipartiteLocalLLMBase._loaded_models[self._load_key] = (self._tokenizer, self._model)

    def release(self) -> None:
        _BipartiteLocalLLMBase._loaded_models.pop(self._load_key, None)
        self._tokenizer = None
        self._model = None
        super().release()

    @abstractmethod
    def _build_prompt(self, x: str, y: str) -> str:
        ...

    @abstractmethod
    def _infer(self, prompts: list[str], batch_size: int) -> list[Any]:
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
        if self._tokenizer is None or self._model is None:
            self._load()
        prompts = [self._build_prompt(x, y) for x, y in unique_pairs]
        for key, result in zip(unique_pairs, self._infer(prompts, batch_size)):
            self._store_cost(key, result)


class BipartiteLocalLogProbDistance(_BipartiteLocalLLMBase):

    _YES_TOKENS: list[str] = ["yes", "Yes", "YES"]
    _NO_TOKENS: list[str] = ["no", "No", "NO"]
    _ANSWER_KEY_RE = re.compile(r'"entails"\s*:\s*"')
    _VALUE_SLOT_PREFILL = '{"entails": "'
    _MAX_NEW_TOKENS_REASONING = 2048

    def _build_prompt(self, x: str, y: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an expert in semantic textual entailment. Respond with only a "
                    "single JSON object, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Does "{y}" semantically entail "{x}"? Respond with exactly one JSON '
                    f'object with a single key "entails" whose value is either "yes" or "no".'
                ),
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_reasoning,
        )
        if not self.enable_reasoning:
            prompt += self.thought_skip_prefill + self._VALUE_SLOT_PREFILL
        return prompt

    def _yes_no_ids(self) -> tuple[list[int], list[int]]:
        yes_ids = [
            self._tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self._YES_TOKENS
            if self._tokenizer.encode(t, add_special_tokens=False)
        ]
        no_ids = [
            self._tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self._NO_TOKENS
            if self._tokenizer.encode(t, add_special_tokens=False)
        ]
        return yes_ids, no_ids

    @staticmethod
    def _cost_from_logits(logits, yes_ids: list[int], no_ids: list[int]) -> float:
        yes_lp = max((logits[i].item() for i in yes_ids), default=None)
        no_lp = max((logits[i].item() for i in no_ids), default=None)
        if yes_lp is None and no_lp is None:
            return 0.5
        if yes_lp is None:
            return 1.0
        if no_lp is None:
            return 0.0
        diff = no_lp - yes_lp
        if diff >= 0:
            return 1.0 / (1.0 + math.exp(-diff))
        ediff = math.exp(diff)
        return ediff / (1.0 + ediff)

    def _find_answer_step(self, new_tokens_row: Any) -> int | None:
        for t in range(len(new_tokens_row)):
            prefix_text = self._tokenizer.decode(new_tokens_row[:t], skip_special_tokens=True)
            if self._ANSWER_KEY_RE.search(prefix_text):
                return t
        return None

    def _infer(self, prompts: list[str], batch_size: int) -> list[float]:
        import torch

        yes_ids, no_ids = self._yes_no_ids()
        costs: list[float] = []
        batch_size = self._effective_batch_size(batch_size)
        for start in tqdm(
            range(0, len(prompts), batch_size), desc="  logprob inference", leave=False, unit="batch"
        ):
            chunk = prompts[start : start + batch_size]
            inputs = self._tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=self._PROMPT_MAX_LENGTH, padding=True,
            ).to(self._model.device)

            if not self.enable_reasoning:
                with torch.no_grad():
                    logits = self._model(**inputs, logits_to_keep=1).logits[:, -1, :]
                for row in logits:
                    costs.append(self._cost_from_logits(row, yes_ids, no_ids))
                continue

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._MAX_NEW_TOKENS_REASONING,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = outputs.sequences[:, prompt_len:]
            for row_idx, row in enumerate(new_tokens):
                anchor_step = self._find_answer_step(row)
                if anchor_step is None:
                    logger.warning(
                        'LogProb reasoning mode: never reached "entails": " within %d '
                        'tokens for %s -- falling back to cost 0.5',
                        self._MAX_NEW_TOKENS_REASONING, self.model_name,
                    )
                    costs.append(0.5)
                    continue
                costs.append(
                    self._cost_from_logits(outputs.scores[anchor_step][row_idx], yes_ids, no_ids)
                )
        return costs


class BipartiteLocalVerbalizedProbDistance(_BipartiteLocalLLMBase):

    _BOTH_DIRECTIONS = False
    _MAX_NEW_TOKENS_REASONING = 2048
    _MAX_NEW_TOKENS_PREFILL = 20
    _JSON_KEYS_RE = re.compile(
        r'"y_entails_x"\s*:\s*(-?\d+(?:\.\d+)?).*?"x_entails_y"\s*:\s*(-?\d+(?:\.\d+)?)',
        re.DOTALL,
    )
    _VALUE_SLOT_PREFILL = '{"y_entails_x": '

    def _build_prompt(self, x: str, y: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an expert in semantic textual entailment. Respond with only a "
                    "single JSON object, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Consider two annotations, X: "{x}" and Y: "{y}". '
                    f'Rate from 0.0 (contradiction) to 1.0 (complete semantic entailment) both '
                    f'directions. Respond with exactly one JSON object with two keys: '
                    f'"y_entails_x" and "x_entails_y", each a float between 0.0 and 1.0, and '
                    f'nothing else.'
                ),
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_reasoning,
        )
        if not self.enable_reasoning:
            prompt += self.thought_skip_prefill + self._VALUE_SLOT_PREFILL
        return prompt

    def _infer(self, prompts: list[str], batch_size: int) -> list[tuple[float, float]]:
        import torch

        max_new_tokens = (
            self._MAX_NEW_TOKENS_REASONING if self.enable_reasoning else self._MAX_NEW_TOKENS_PREFILL
        )
        results: list[tuple[float, float]] = []
        batch_size = self._effective_batch_size(batch_size)
        for start in tqdm(
            range(0, len(prompts), batch_size), desc="  verbalized inference", leave=False, unit="batch"
        ):
            chunk = prompts[start : start + batch_size]
            inputs = self._tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=self._PROMPT_MAX_LENGTH, padding=True,
            ).to(self._model.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            for row in new_tokens:
                text = self._tokenizer.decode(row, skip_special_tokens=True).strip()
                full_text = text if self.enable_reasoning else (self._VALUE_SLOT_PREFILL + text)
                match = self._JSON_KEYS_RE.search(full_text)
                if not match:
                    logger.warning(
                        "Verbalized-prob parse failed for %s (JSON keys not found in %d "
                        "generated tokens): %r -- falling back to (0.5, 0.5)",
                        self.model_name, len(row), text,
                    )
                    results.append((0.5, 0.5))
                    continue
                y_entails_x = max(0.0, min(1.0, float(match.group(1))))
                x_entails_y = max(0.0, min(1.0, float(match.group(2))))
                results.append((1.0 - y_entails_x, 1.0 - x_entails_y))
        return results


_MEDGEMMA_THOUGHT_SKIP = "<unused94>\n<unused95>"

bipartite_medgemma_log_prob_distance = BipartiteLocalLogProbDistance(
    model_name=_MEDGEMMA_LOCAL, thought_skip_prefill=_MEDGEMMA_THOUGHT_SKIP
)
bipartite_medgemma_verbalized_prob_distance = BipartiteLocalVerbalizedProbDistance(
    model_name=_MEDGEMMA_LOCAL, thought_skip_prefill=_MEDGEMMA_THOUGHT_SKIP
)
bipartite_qwen35_log_prob_distance = BipartiteLocalLogProbDistance(model_name=_QWEN35_LOCAL)
bipartite_qwen35_verbalized_prob_distance = BipartiteLocalVerbalizedProbDistance(
    model_name=_QWEN35_LOCAL
)
