
import math
import random
from collections import Counter, defaultdict

import numpy as np
from nltk import AnnotationTask
from pydantic import BaseModel

from soft_irr.evaluation.distances import DistanceFunction, set_jaccard_distance
from soft_irr.evaluation.distances.base import NO_MATCHING, matching_strategies
from soft_irr.evaluation.distribution_scores import sigma_and_ks

DEFAULT_MAX_EXACT_CHANCE_PAIRS = 1000
DEFAULT_MC_MIN_SHUFFLES = 3
DEFAULT_MC_TARGET_SAMPLES = 300 
DEFAULT_MC_SEED = 0


class _SoftAeAnnotationTask(AnnotationTask):

    def __init__(self, data, distance, chance_agreement_kappa: float, chance_agreement_pi: float):
        super().__init__(data=data, distance=distance)
        self._chance_agreement_kappa = chance_agreement_kappa
        self._chance_agreement_pi = chance_agreement_pi

    def Ae_kappa(self, cA, cB) -> float:
        return self._chance_agreement_kappa

    def pi(self) -> float:
        return self._chance_corrected_agreement(self.avg_Ao(), self._chance_agreement_pi)

    @staticmethod
    def _chance_corrected_agreement(avg_Ao: float, Ae: float) -> float:
        if Ae >= 1.0:
            return 1.0 if avg_Ao >= 1.0 else float("-inf")
        return (avg_Ao - Ae) / (1.0 - Ae)


class AgreementScores(BaseModel):

    cohens_kappa: float
    scotts_pi: float
    average_agreement: float
    sigma: float
    ks: float

    @staticmethod
    def _get_annotation_data(
        labels: list[list[str]], annotator: str = "rater_a"
    ) -> list[tuple[str, str, frozenset[str]]]:
        annotation_data = []
        for i, labels_i in enumerate(labels):
            label_set = frozenset(labels_i if labels_i is not None else [])
            annotation_data.append((annotator, f"item_{i}", label_set))
        return annotation_data

    @staticmethod
    def chance_pairs_and_weights(
        rater_a_labels: list[frozenset[str]],
        rater_b_labels: list[frozenset[str]],
        max_exact_pairs: int = DEFAULT_MAX_EXACT_CHANCE_PAIRS,
        mc_shuffles: int | None = None,
        mc_target_samples: int | None = None,
        mc_min_shuffles: int = DEFAULT_MC_MIN_SHUFFLES,
        seed: int = DEFAULT_MC_SEED,
    ) -> tuple[list[tuple[frozenset[str], frozenset[str]]], list[float]]:
        n = len(rater_a_labels)
        rater_a_counts = Counter(rater_a_labels)
        rater_b_counts = Counter(rater_b_labels)

        if len(rater_a_counts) * len(rater_b_counts) <= max_exact_pairs:
            pairs = [(a, b) for a in rater_a_counts for b in rater_b_counts]
            weights = [(rater_a_counts[a] / n) * (rater_b_counts[b] / n) for a, b in pairs]
            return pairs, weights

        if mc_shuffles is None:
            target = DEFAULT_MC_TARGET_SAMPLES if mc_target_samples is None else mc_target_samples
            mc_shuffles = max(mc_min_shuffles, math.ceil(target / n))
        else:
            target = None

        rng = random.Random(seed)
        pairs = []
        for _ in range(mc_shuffles):
            shuffled = rater_b_labels[:]
            rng.shuffle(shuffled)
            pairs.extend(zip(rater_a_labels, shuffled))
        if target is not None and len(pairs) > target:
            pairs = rng.sample(pairs, target)
        weights = [1.0 / len(pairs)] * len(pairs)
        return pairs, weights

    @staticmethod
    def chance_kwargs(
        max_exact_chance_pairs: int | None = None,
        mc_target_samples: int | None = None,
        mc_shuffles: int | None = None,
        mc_min_shuffles: int | None = None,
    ) -> dict[str, int]:
        named = {
            "max_exact_pairs": max_exact_chance_pairs,
            "mc_target_samples": mc_target_samples,
            "mc_shuffles": mc_shuffles,
            "mc_min_shuffles": mc_min_shuffles,
        }
        return {key: value for key, value in named.items() if value is not None}

    @staticmethod
    def _batch_by_strategy(
        distance_function: DistanceFunction,
        pairs: list[tuple[frozenset[str], frozenset[str]]],
        strategies: tuple[str, ...],
    ) -> dict[str, list[float]]:
        values = distance_function.batch(pairs)
        own = matching_strategies(distance_function)
        return {
            strategy: values if strategy == NO_MATCHING or strategy not in own
            else distance_function.distances_under(strategy)
            for strategy in strategies
        }

    @classmethod
    def get(
        cls,
        rater_a: list[list[str]],
        rater_b: list[list[str]],
        distance_function: DistanceFunction = set_jaccard_distance,
        chance_distance_function: DistanceFunction | None = None,
        max_exact_chance_pairs: int | None = None,
        mc_target_samples: int | None = None,
        mc_shuffles: int | None = None,
        mc_min_shuffles: int | None = None,
    ) -> "AgreementScores":
        by_matching = cls.get_by_matching(
            rater_a,
            rater_b,
            distance_function=distance_function,
            chance_distance_function=chance_distance_function,
            max_exact_chance_pairs=max_exact_chance_pairs,
            mc_target_samples=mc_target_samples,
            mc_shuffles=mc_shuffles,
            mc_min_shuffles=mc_min_shuffles,
        )
        return by_matching[matching_strategies(distance_function)[0]]

    @classmethod
    def get_by_matching(
        cls,
        rater_a: list[list[str]],
        rater_b: list[list[str]],
        distance_function: DistanceFunction = set_jaccard_distance,
        chance_distance_function: DistanceFunction | None = None,
        max_exact_chance_pairs: int | None = None,
        mc_target_samples: int | None = None,
        mc_shuffles: int | None = None,
        mc_min_shuffles: int | None = None,
    ) -> dict[str, "AgreementScores"]:
        strategies = matching_strategies(distance_function)

        annotation_data = cls._get_annotation_data(rater_a, "rater_a")
        annotation_data.extend(cls._get_annotation_data(rater_b, "rater_b"))

        rater_a_by_item = {entry[1]: entry[2] for entry in annotation_data if entry[0] == "rater_a"}
        rater_b_by_item = {entry[1]: entry[2] for entry in annotation_data if entry[0] == "rater_b"}
        paired = [(rater_a_by_item[k], rater_b_by_item[k]) for k in rater_a_by_item if k in rater_b_by_item]
        all_pairs = list(dict.fromkeys([(a, b) for a, b in paired] + [(b, a) for a, b in paired]))
        dist_values = cls._batch_by_strategy(distance_function, all_pairs, strategies)
        caches: dict[str, dict[tuple, float]] = {
            strategy: dict(zip(all_pairs, values)) for strategy, values in dist_values.items()
        }

        rater_a_labels = [a for a, _ in paired]
        rater_b_labels = [b for _, b in paired]
        pooled_labels = rater_a_labels + rater_b_labels

        cap = cls.chance_kwargs(
            max_exact_chance_pairs, mc_target_samples, mc_shuffles, mc_min_shuffles
        )
        kappa_chance_pairs, kappa_chance_weights = cls.chance_pairs_and_weights(
            rater_a_labels, rater_b_labels, **cap
        )
        pi_chance_pairs, pi_chance_weights = cls.chance_pairs_and_weights(pooled_labels, pooled_labels, **cap)

        chance_fn = chance_distance_function if chance_distance_function is not None else distance_function
        all_chance_pairs = list(dict.fromkeys(kappa_chance_pairs + pi_chance_pairs))
        chance_dist_values = cls._batch_by_strategy(chance_fn, all_chance_pairs, strategies)

        scores: dict[str, AgreementScores] = {}
        for strategy in strategies:
            cache = caches[strategy]
            chance_cache = dict(zip(all_chance_pairs, chance_dist_values[strategy]))
            chance_agreement_kappa = sum(
                weight * (1.0 - chance_cache[pair])
                for weight, pair in zip(kappa_chance_weights, kappa_chance_pairs)
            )
            chance_agreement_pi = sum(
                weight * (1.0 - chance_cache[pair]) for weight, pair in zip(pi_chance_weights, pi_chance_pairs)
            )

            task = _SoftAeAnnotationTask(
                annotation_data,
                distance=lambda l1, l2, _cache=cache: _cache[l1, l2],
                chance_agreement_kappa=chance_agreement_kappa,
                chance_agreement_pi=chance_agreement_pi,
            )
            sigma, ks = sigma_and_ks(
                [cache[pair] for pair in paired],
                [chance_cache[pair] for pair in kappa_chance_pairs],
                weights=kappa_chance_weights,
            )
            scores[strategy] = AgreementScores(
                cohens_kappa=task.kappa_pairwise("rater_a", "rater_b"),
                scotts_pi=task.pi(),
                average_agreement=task.avg_Ao(),
                sigma=sigma,
                ks=ks,
            )

        return scores


class MultiRaterAgreementScores(BaseModel):

    cohens_kappa: float
    fleiss_kappa: float
    average_agreement: float
    sigma: float
    ks: float

    @staticmethod
    def _chance_corrected(observed: float, expected: float) -> float:
        if math.isclose(expected, 1.0):
            if math.isclose(observed, 1.0):
                return 1.0
            raise ValueError(
                f"Expected agreement is 1.0 but observed agreement is {observed:.4f}. "
                "This indicates a distance function that violates distance(l, l) = 0, "
                "or otherwise undefined coefficient semantics."
            )
        return (observed - expected) / (1.0 - expected)

    @classmethod
    def get(
        cls,
        raters: list[list[list[str]]],
        distance_function: DistanceFunction = set_jaccard_distance,
        chance_distance_function: DistanceFunction | None = None,
    ) -> "MultiRaterAgreementScores":
        return cls.get_by_matching(
            raters,
            distance_function=distance_function,
            chance_distance_function=chance_distance_function,
        )[matching_strategies(distance_function)[0]]

    @classmethod
    def get_by_matching(
        cls,
        raters: list[list[list[str]]],
        distance_function: DistanceFunction = set_jaccard_distance,
        chance_distance_function: DistanceFunction | None = None,
    ) -> dict[str, "MultiRaterAgreementScores"]:
        strategies = matching_strategies(distance_function)
        chance_fn = chance_distance_function if chance_distance_function is not None else distance_function
        items: list[list[frozenset]] = [[frozenset(r) for r in item] for item in raters]

        item_pairs = [
            (item_idx, s, t, item[s], item[t])
            for item_idx, item in enumerate(items)
            for s in range(len(item))
            for t in range(s + 1, len(item))
        ]

        all_pairs = list(dict.fromkeys((a, b) for _, _, _, a, b in item_pairs))
        dist_values = AgreementScores._batch_by_strategy(distance_function, all_pairs, strategies)
        dist_caches: dict[str, dict[tuple, float]] = {
            strategy: dict(zip(all_pairs, values)) for strategy, values in dist_values.items()
        }

        by_slot_pair: dict[tuple[int, int], list[tuple[frozenset, frozenset]]] = defaultdict(list)
        for _, s, t, a, b in item_pairs:
            by_slot_pair[(s, t)].append((a, b))

        pooled_labels = [label for item in items for label in item]
        slot_pair_keys = list(by_slot_pair.keys())
        chance_requests = [(pooled_labels, pooled_labels)] + [
            ([a for a, _ in by_slot_pair[key]], [b for _, b in by_slot_pair[key]]) for key in slot_pair_keys
        ]

        request_pairs: list[list[tuple[frozenset, frozenset]]] = []
        request_weights: list[list[float]] = []
        all_chance_pairs: list[tuple[frozenset, frozenset]] = []
        for a_labels, b_labels in chance_requests:
            pairs, weights = AgreementScores.chance_pairs_and_weights(a_labels, b_labels)
            request_pairs.append(pairs)
            request_weights.append(weights)
            all_chance_pairs.extend(pairs)
        all_chance_pairs = list(dict.fromkeys(all_chance_pairs))
        chance_dist_values = AgreementScores._batch_by_strategy(chance_fn, all_chance_pairs, strategies)

        scores: dict[str, MultiRaterAgreementScores] = {}
        for strategy in strategies:
            dist_cache = dist_caches[strategy]
            chance_cache = dict(zip(all_chance_pairs, chance_dist_values[strategy]))

            po_by_item: dict[int, list[float]] = defaultdict(list)
            for item_idx, _, _, a, b in item_pairs:
                po_by_item[item_idx].append(1.0 - dist_cache[(a, b)])
            average_agreement = float(np.mean([np.mean(v) for v in po_by_item.values()]))

            chance_agreements = [
                sum(w * (1.0 - chance_cache[p]) for w, p in zip(weights, pairs))
                for pairs, weights in zip(request_pairs, request_weights)
            ]
            pooled_chance_agreement, *slot_pair_chance_agreements = chance_agreements

            kappas = []
            for key, chance_agreement in zip(slot_pair_keys, slot_pair_chance_agreements):
                pairs = by_slot_pair[key]
                ao = sum(1.0 - dist_cache[p] for p in pairs) / len(pairs)
                kappas.append(cls._chance_corrected(ao, chance_agreement))

            sigma, ks = sigma_and_ks(
                [dist_cache[(a, b)] for _, _, _, a, b in item_pairs],
                [chance_cache[pair] for pair in request_pairs[0]],
                weights=request_weights[0],
            )
            scores[strategy] = MultiRaterAgreementScores(
                cohens_kappa=float(np.mean(kappas)),
                fleiss_kappa=cls._chance_corrected(average_agreement, pooled_chance_agreement),
                average_agreement=average_agreement,
                sigma=sigma,
                ks=ks,
            )

        return scores

